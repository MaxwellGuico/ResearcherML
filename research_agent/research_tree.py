"""Persistent hypothesis graph and experiment-lineage planner memory."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping, Sequence

from .logger import ResearchLogger
from .planner import CapabilityAction, ResearchDirection


def hypothesis_id(hypothesis: str) -> str:
    """Return a stable identity for materially identical normalized hypothesis text."""
    normalized = re.sub(r"\s+", " ", hypothesis.strip().lower())
    return f"hyp_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]}"


class ResearchTree:
    """Records transitions append-only and publishes a rebuildable graph snapshot."""

    def __init__(self, logger: ResearchLogger) -> None:
        self.logger = logger
        self.store = logger.store

    def record_planner_batch(
        self,
        metadata: Mapping[str, Any],
        *,
        incumbent_experiment_id: str,
    ) -> None:
        slate = self._slate(metadata)
        candidates = [item for item in slate.get("candidates", []) if isinstance(item, Mapping)]
        if not candidates:
            return
        advices = self._advices(metadata)
        selected_ids = {
            str(item.get("selected_candidate_id")) for item in advices
            if item.get("selected_candidate_id")
        }
        deferred_ids = {
            str(candidate_id) for item in advices
            for candidate_id in item.get("deferred_candidate_ids", [])
        }
        batch_id = str(
            metadata.get("planner", {}).get("response_id")
            or metadata.get("ideator", {}).get("response_id")
            or metadata.get("response_id")
            or f"batch_{len(self.store.read_research_tree_events()) + 1:03d}"
        )
        if any(
            event.get("event_type") == "hypothesis_considered" and event.get("batch_id") == batch_id
            for event in self.store.read_research_tree_events()
        ):
            return
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            hypothesis = str(candidate.get("hypothesis", "")).strip()
            if not hypothesis:
                continue
            status = (
                "selected" if candidate_id in selected_ids
                else "deferred" if candidate_id in deferred_ids
                else "proposed"
            )
            self.logger.record_research_tree_event({
                "event_type": "hypothesis_considered",
                "hypothesis_id": hypothesis_id(hypothesis),
                "hypothesis": hypothesis,
                "candidate_id": candidate_id,
                "domain": candidate.get("domain"),
                "expected_mechanism": candidate.get("expected_mechanism"),
                "required_capabilities": list(candidate.get("required_capabilities", [])),
                "lineage_parent_id": candidate.get("lineage_parent_id"),
                "lineage_action": candidate.get("lineage_action"),
                "evidence_reference": candidate.get("evidence_reference"),
                "status": status,
                "batch_id": batch_id,
                "parent_experiment_id": incumbent_experiment_id,
            })

    def bind_experiment(self, direction: ResearchDirection, proposal: Any) -> None:
        source = next(
            (
                event for event in reversed(self.store.read_research_tree_events())
                if direction.selected_candidate_id
                and event.get("candidate_id") == direction.selected_candidate_id
                and event.get("event_type") == "hypothesis_considered"
            ),
            {},
        )
        self.logger.record_research_tree_event({
            "event_type": "experiment_bound",
            "hypothesis_id": hypothesis_id(direction.hypothesis),
            "hypothesis": direction.hypothesis,
            "domain": direction.direction_id,
            "experiment_id": proposal.experiment_id,
            "parent_experiment_id": proposal.parent_experiment_id or "baseline",
            "direction_id": direction.direction_id,
            "search_region_id": proposal.search_region_id,
            "search_strategy": proposal.search_strategy,
            "portfolio_role": proposal.portfolio_role,
            "source_hypothesis_id": source.get("hypothesis_id"),
            "lineage_parent_id": source.get("hypothesis_id") or proposal.parent_experiment_id or "baseline",
            "lineage_action": (
                "continue"
                if source.get("hypothesis_id") == hypothesis_id(direction.hypothesis)
                else "refine"
            ),
            "evidence_reference": source.get("hypothesis_id") or proposal.parent_experiment_id or "baseline",
            "status": "running",
        })

    def record_capability_action(
        self,
        action: CapabilityAction,
        record: Mapping[str, Any],
        *,
        incumbent_experiment_id: str,
    ) -> None:
        self.logger.record_research_tree_event({
            "event_type": "capability_branch_recorded",
            "hypothesis_id": hypothesis_id(action.hypothesis),
            "hypothesis": action.hypothesis,
            "parent_experiment_id": incumbent_experiment_id,
            "action": action.action,
            "action_id": record.get("action_id"),
            "capability_gap_id": action.capability_gap_id,
            "status": record.get("status"),
        })

    def record_outcome(self, experiment_id: str, *, incumbent_experiment_id: str) -> None:
        record = next(
            (item for item in reversed(self.store.read_iterations()) if item.get("experiment_id") == experiment_id),
            None,
        )
        if record is None:
            return
        self.logger.record_research_tree_event({
            "event_type": "experiment_completed",
            "hypothesis_id": hypothesis_id(str(record.get("hypothesis", ""))),
            "hypothesis": record.get("hypothesis"),
            "experiment_id": experiment_id,
            "parent_experiment_id": record.get("parent_experiment_id") or "baseline",
            "decision": record.get("decision"),
            "primary": record.get("metrics", {}).get("primary"),
            "delta_primary": record.get("delta_primary"),
            "terminal_reason": record.get("terminal_reason"),
            "status": record.get("decision"),
            "became_incumbent": incumbent_experiment_id == experiment_id,
        })

    def refresh(self, state: Any) -> dict[str, Any]:
        self._backfill_historical_planner_slates()
        events = self.store.read_research_tree_events()
        iterations = self.store.read_iterations()
        hypotheses: dict[str, dict[str, Any]] = {}
        experiment_to_hypothesis: dict[str, str] = {}
        for event in events:
            identifier = event.get("hypothesis_id")
            if not identifier:
                continue
            node = hypotheses.setdefault(str(identifier), self._new_hypothesis_node(event))
            node["last_seen"] = event.get("timestamp")
            node["visits"] += event.get("event_type") == "hypothesis_considered"
            if event.get("status"):
                node["status"] = event["status"]
            parent_experiment_id = event.get("parent_experiment_id")
            if parent_experiment_id and parent_experiment_id not in node["parent_experiment_ids"]:
                node["parent_experiment_ids"].append(parent_experiment_id)
            if event.get("event_type") == "hypothesis_considered" and event.get("status") == "deferred":
                node["deferred_count"] += 1
            if event.get("experiment_id"):
                experiment_id = str(event["experiment_id"])
                experiment_to_hypothesis[experiment_id] = str(identifier)
                if experiment_id not in node["experiment_ids"]:
                    node["experiment_ids"].append(experiment_id)
            if event.get("action_id") and event.get("action_id") not in node["capability_action_ids"]:
                node["capability_action_ids"].append(event["action_id"])

        experiment_nodes: list[dict[str, Any]] = []
        for item in iterations:
            experiment_id = str(item.get("experiment_id"))
            identifier = experiment_to_hypothesis.get(
                experiment_id, hypothesis_id(str(item.get("hypothesis", "")))
            )
            node = hypotheses.setdefault(identifier, self._new_hypothesis_node(item))
            if experiment_id not in node["experiment_ids"]:
                node["experiment_ids"].append(experiment_id)
            node["status"] = item.get("decision") or node["status"]
            primary = item.get("metrics", {}).get("primary")
            if _finite_number(primary):
                node["best_primary"] = max(
                    float(primary),
                    float(node["best_primary"]) if node["best_primary"] is not None else float("-inf"),
                )
            experiment_nodes.append({
                "experiment_id": experiment_id,
                "hypothesis_id": identifier,
                "parent_experiment_id": item.get("parent_experiment_id") or "baseline",
                "direction_id": item.get("direction_id"),
                "search_region_id": item.get("search_region_id"),
                "search_strategy": item.get("search_strategy"),
                "portfolio_role": item.get("portfolio_role"),
                "changed_factors": list(item.get("changed_factors", [])),
                "decision": item.get("decision"),
                "primary": float(primary) if _finite_number(primary) else None,
                "delta_primary": item.get("delta_primary"),
                "terminal_reason": item.get("terminal_reason"),
            })

        ancestry = self._ancestry(experiment_nodes, state.current_best_experiment_id)
        candidates = self._continuation_candidates(
            hypotheses, experiment_nodes, state.current_best_experiment_id, state.current_best_primary
        )
        snapshot = {
            "schema_version": 1,
            "incumbent": {
                "experiment_id": state.current_best_experiment_id,
                "primary": state.current_best_primary,
                "ancestry": ancestry,
            },
            "hypotheses": sorted(hypotheses.values(), key=lambda item: (item.get("first_seen") or "", item["hypothesis_id"])),
            "experiments": experiment_nodes,
            "edges": [
                {
                    "from": item["parent_experiment_id"],
                    "to": item["experiment_id"],
                    "type": "experiment_branch",
                }
                for item in experiment_nodes
            ] + [
                {
                    "from": event["source_hypothesis_id"],
                    "to": event["hypothesis_id"],
                    "type": "hypothesis_refinement",
                }
                for event in events
                if event.get("event_type") == "experiment_bound"
                and event.get("source_hypothesis_id")
                and event.get("source_hypothesis_id") != event.get("hypothesis_id")
            ],
            "continuation_candidates": candidates,
            "deferred_hypotheses": [
                self._planner_hypothesis(node) for node in hypotheses.values()
                if node["status"] in {"deferred", "proposed", "pending_implementation", "pending_human_approval"}
            ],
            "failed_branches": [
                item for item in experiment_nodes if item["decision"] in {"failed", "rejected"}
            ],
        }
        self.store.write_root_json("research_tree.json", snapshot)
        self.store.write_root_text("research_tree.md", self._render_mermaid(snapshot))
        return snapshot

    def _backfill_historical_planner_slates(self) -> None:
        """Recover pre-tree LLM slates without inventing their historical incumbent."""
        for event in self.store.read_events():
            metadata: Mapping[str, Any] | None = None
            if event.get("action") == "llm_hypothesis_generated":
                details = event.get("details", {})
                metadata = details if isinstance(details, Mapping) else None
            elif event.get("action") == "llm_planning_blocked":
                details = event.get("details", {})
                candidate = details.get("planner_metadata", {}) if isinstance(details, Mapping) else {}
                metadata = candidate if isinstance(candidate, Mapping) else None
            if metadata:
                self.record_planner_batch(
                    metadata,
                    incumbent_experiment_id="unknown_historical_incumbent",
                )

    def planner_context(self, state: Any) -> dict[str, Any]:
        snapshot = self.refresh(state)
        return {
            "incumbent": snapshot["incumbent"],
            "continuation_candidates": snapshot["continuation_candidates"][:8],
            "deferred_hypotheses": snapshot["deferred_hypotheses"][-8:],
            "failed_branches": snapshot["failed_branches"][-8:],
            "experiment_count": len(snapshot["experiments"]),
            "hypothesis_count": len(snapshot["hypotheses"]),
        }

    @staticmethod
    def _slate(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        planner = metadata.get("planner", {})
        if not isinstance(planner, Mapping):
            planner = {}
        if not planner:
            planner = metadata.get("ideator", {})
        slate = planner.get("slate", {}) if isinstance(planner, Mapping) else {}
        return slate if isinstance(slate, Mapping) else {}

    @staticmethod
    def _advices(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        values: list[Mapping[str, Any]] = []
        implementer = metadata.get("implementer", {})
        if isinstance(implementer, Mapping) and isinstance(implementer.get("plan"), Mapping):
            values.append(implementer["plan"])
        for item in metadata.get("implementers", []):
            if isinstance(item, Mapping) and isinstance(item.get("plan"), Mapping):
                values.append(item["plan"])
        # Read older artifacts without keeping specialist routing in the active path.
        specialist = metadata.get("specialist", {})
        if isinstance(specialist, Mapping) and isinstance(specialist.get("advice"), Mapping):
            values.append(specialist["advice"])
        for item in metadata.get("specialists", []):
            if isinstance(item, Mapping) and isinstance(item.get("advice"), Mapping):
                values.append(item["advice"])
        return values

    @staticmethod
    def _new_hypothesis_node(value: Mapping[str, Any]) -> dict[str, Any]:
        hypothesis = str(value.get("hypothesis", ""))
        return {
            "hypothesis_id": str(value.get("hypothesis_id") or hypothesis_id(hypothesis)),
            "hypothesis": hypothesis,
            "domain": value.get("domain"),
            "expected_mechanism": value.get("expected_mechanism"),
            "required_capabilities": list(value.get("required_capabilities", [])),
            "lineage_parent_id": value.get("lineage_parent_id"),
            "lineage_action": value.get("lineage_action"),
            "evidence_reference": value.get("evidence_reference"),
            "status": value.get("status", "proposed"),
            "first_seen": value.get("timestamp"),
            "last_seen": value.get("timestamp"),
            "visits": 0,
            "deferred_count": 0,
            "experiment_ids": [],
            "capability_action_ids": [],
            "parent_experiment_ids": [],
            "best_primary": None,
        }

    @classmethod
    def _render_mermaid(cls, snapshot: Mapping[str, Any]) -> str:
        hypotheses = {
            str(item["hypothesis_id"]): item for item in snapshot.get("hypotheses", [])
        }
        incumbent_id = str(snapshot.get("incumbent", {}).get("experiment_id", "baseline"))
        lines = [
            "# Research Tree",
            "",
            "This diagram is generated from the append-only research evidence. Solid arrows are executed "
            "experiment branches; dashed arrows are unresolved hypotheses.",
            "",
            "```mermaid",
            "flowchart TD",
            '  baseline["baseline"]',
        ]
        for experiment in snapshot.get("experiments", []):
            experiment_id = str(experiment["experiment_id"])
            hypothesis = hypotheses.get(str(experiment.get("hypothesis_id")), {}).get("hypothesis", "")
            decision = str(experiment.get("decision") or "unknown")
            primary = experiment.get("primary")
            score = f"primary={primary:.4f}" if isinstance(primary, (int, float)) else "primary unavailable"
            label = cls._mermaid_label(f"{experiment_id} · {decision} · {score}\n{hypothesis}")
            lines.append(f'  {experiment_id}["{label}"]')
            parent = str(experiment.get("parent_experiment_id") or "baseline")
            lines.append(f"  {parent} --> {experiment_id}")

        unresolved_all = list(snapshot.get("deferred_hypotheses", []))
        unresolved = unresolved_all[-12:]
        known_experiments = {"baseline", *(str(item["experiment_id"]) for item in snapshot.get("experiments", []))}
        unknown_parent_nodes: list[str] = []
        for node in unresolved:
            node_id = str(node["hypothesis_id"])
            label = cls._mermaid_label(
                f"{node_id} · {node.get('status')}\n{node.get('hypothesis', '')}"
            )
            lines.append(f'  {node_id}{{"{label}"}}')
            parents = node.get("parent_experiment_ids", [])
            parent = next((str(item) for item in reversed(parents) if str(item) in known_experiments), None)
            if parent:
                lines.append(f"  {parent} -.-> {node_id}")
            else:
                unknown_parent_nodes.append(node_id)
        if unknown_parent_nodes:
            lines.append('  historical_context_unknown["historical planning context · exact incumbent unavailable"]')
            for node_id in unknown_parent_nodes:
                lines.append(f"  historical_context_unknown -.-> {node_id}")
        omitted = len(unresolved_all) - len(unresolved)
        if omitted:
            lines.append(f'  deferred_archive["{omitted} older deferred hypotheses · see research_tree.json"]')
            lines.append("  historical_context_unknown -.-> deferred_archive")

        lines.extend([
            "  classDef incumbent fill:#14532d,color:#ffffff,stroke:#22c55e,stroke-width:4px",
            "  classDef accepted fill:#dcfce7,color:#14532d,stroke:#22c55e",
            "  classDef rejected fill:#fef3c7,color:#78350f,stroke:#f59e0b",
            "  classDef failed fill:#fee2e2,color:#7f1d1d,stroke:#ef4444",
            "  classDef deferred fill:#e0e7ff,color:#312e81,stroke:#818cf8,stroke-dasharray: 5 5",
        ])
        for experiment in snapshot.get("experiments", []):
            experiment_id = str(experiment["experiment_id"])
            decision = str(experiment.get("decision") or "")
            style = "incumbent" if experiment_id == incumbent_id else decision
            if style in {"incumbent", "accepted", "rejected", "failed"}:
                lines.append(f"  class {experiment_id} {style}")
        if incumbent_id == "baseline":
            lines.append("  class baseline incumbent")
        for node in unresolved:
            lines.append(f"  class {node['hypothesis_id']} deferred")
        if omitted:
            lines.append("  class deferred_archive deferred")
        lines.extend(["```", "", "## Legend", "", "- Dark green: current incumbent", "- Green: previously accepted", "- Amber: rejected with evidence", "- Red: failed", "- Dashed blue: deferred or capability-blocked hypothesis", ""])
        return "\n".join(lines)

    @staticmethod
    def _mermaid_label(value: Any, limit: int = 110) -> str:
        compact = re.sub(r"\s+", " ", str(value)).strip().replace('"', "'")
        if len(compact) > limit:
            compact = compact[: limit - 1].rstrip() + "…"
        return compact

    @staticmethod
    def _ancestry(experiments: Sequence[Mapping[str, Any]], incumbent_id: str) -> list[str]:
        parents = {str(item["experiment_id"]): str(item["parent_experiment_id"]) for item in experiments}
        ancestry: list[str] = []
        current = incumbent_id
        seen: set[str] = set()
        while current and current not in seen:
            ancestry.append(current)
            seen.add(current)
            if current == "baseline":
                break
            current = parents.get(current, "baseline")
        return list(reversed(ancestry))

    @classmethod
    def _continuation_candidates(
        cls,
        hypotheses: Mapping[str, Mapping[str, Any]],
        experiments: Sequence[Mapping[str, Any]],
        incumbent_id: str,
        incumbent_primary: float,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        incumbent = next((item for item in experiments if item["experiment_id"] == incumbent_id), None)
        if incumbent:
            node = hypotheses[incumbent["hypothesis_id"]]
            candidates.append({
                **cls._planner_hypothesis(node),
                "reason": "continue_accepted_lineage",
                "source_experiment_id": incumbent_id,
                "priority": 0,
            })
        rejected = sorted(
            (
                item for item in experiments
                if item["decision"] == "rejected" and item["primary"] is not None
            ),
            key=lambda item: item["primary"],
            reverse=True,
        )
        for item in rejected[:4]:
            node = hypotheses[item["hypothesis_id"]]
            candidates.append({
                **cls._planner_hypothesis(node),
                "reason": "revisit_near_incumbent_branch",
                "source_experiment_id": item["experiment_id"],
                "gap_to_incumbent": incumbent_primary - float(item["primary"]),
                "priority": 1,
            })
        for node in hypotheses.values():
            if node["status"] in {"deferred", "proposed", "pending_implementation", "pending_human_approval"}:
                candidates.append({
                    **cls._planner_hypothesis(node),
                    "reason": "reconsider_unresolved_hypothesis",
                    "source_experiment_id": None,
                    "priority": 2,
                })
        return sorted(candidates, key=lambda item: (item["priority"], item.get("gap_to_incumbent", 0.0)))

    @staticmethod
    def _planner_hypothesis(node: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "hypothesis_id": node["hypothesis_id"],
            "hypothesis": node["hypothesis"],
            "domain": node.get("domain"),
            "status": node["status"],
            "visits": node["visits"],
            "deferred_count": node["deferred_count"],
            "experiment_ids": list(node["experiment_ids"]),
            "parent_experiment_ids": list(node.get("parent_experiment_ids", [])),
            "best_primary": node["best_primary"],
            "lineage_parent_id": node.get("lineage_parent_id"),
            "lineage_action": node.get("lineage_action"),
            "evidence_reference": node.get("evidence_reference"),
        }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
