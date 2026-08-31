"""Structured event and iteration logging for research-agent runs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .store import ArtifactStore


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchLogger:
    """Writes all material research actions to append-only storage."""

    def __init__(self, store: ArtifactStore, clock: Clock = utc_now) -> None:
        self.store = store
        self.clock = clock

    def log_action(
        self,
        action: str,
        *,
        experiment_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not action.strip():
            raise ValueError("action must not be empty")
        event = {
            "timestamp": self._timestamp(),
            "action": action,
            "experiment_id": experiment_id,
            "details": details or {},
        }
        self.store.append_event(event)
        return event

    def record_iteration(self, record: dict[str, Any]) -> dict[str, Any]:
        required = {"experiment_id", "hypothesis", "rationale", "config", "decision"}
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"iteration record missing required fields: {', '.join(missing)}")
        completed = {"timestamp": self._timestamp(), **record}
        self.store.append_iteration(completed)
        metrics = completed.get("metrics", {})
        self.store.append_metric_summary(
            {
                "experiment_id": completed["experiment_id"],
                "parent_experiment_id": completed.get("parent_experiment_id"),
                "decision": completed["decision"],
                "GAUC": metrics.get("GAUC"),
                "nDCG@5": metrics.get("nDCG@5"),
                "primary": metrics.get("primary"),
                "delta_primary": completed.get("delta_primary"),
                "runtime_seconds": completed.get("runtime_seconds"),
            }
        )
        self.log_action(
            "iteration_recorded",
            experiment_id=completed["experiment_id"],
            details={"decision": completed["decision"]},
        )
        return completed

    def record_stage(self, record: dict[str, Any]) -> dict[str, Any]:
        required = {"experiment_id", "stage_id", "config", "status"}
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"stage record missing required fields: {', '.join(missing)}")
        completed = {"timestamp": self._timestamp(), **record}
        self.store.append_stage(completed)
        self.log_action(
            "experiment_stage_recorded",
            experiment_id=completed["experiment_id"],
            details={"stage_id": completed["stage_id"], "status": completed["status"]},
        )
        return completed

    def record_manual_intervention(
        self,
        *,
        description: str,
        reason: str,
        effect: str,
        experiment_id: str | None = None,
        approval_id: str | None = None,
        authority_scope: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        intervention = {
            "timestamp": self._timestamp(),
            "experiment_id": experiment_id,
            "description": description,
            "reason": reason,
            "effect": effect,
            "approval_id": approval_id,
            "authority_scope": authority_scope,
            "status": status,
        }
        self.store.append_intervention(intervention)
        self.log_action(
            "manual_intervention_recorded",
            experiment_id=experiment_id,
            details={"reason": reason, "approval_id": approval_id},
        )
        return intervention

    def record_manual_approval(
        self,
        *,
        approval_id: str,
        authority_scope: str,
        description: str,
        reason: str,
        effect: str,
        experiment_id: str | None = None,
        reuse_action: str = "approval_reused",
    ) -> bool:
        """Persist versioned authority once and log every later reuse."""
        intervention = {
            "timestamp": self._timestamp(),
            "experiment_id": experiment_id,
            "description": description,
            "reason": reason,
            "effect": effect,
            "approval_id": approval_id,
            "authority_scope": authority_scope,
            "status": "active",
        }
        recorded = self.store.append_intervention_once(
            intervention,
            approval_id=approval_id,
        )
        self.log_action(
            "manual_intervention_recorded" if recorded else reuse_action,
            experiment_id=experiment_id,
            details={"reason": reason, "approval_id": approval_id},
        )
        return recorded

    def record_capability_action(
        self,
        action: dict[str, Any],
        *,
        status: str,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action_id = f"cap_{len(self.store.read_capability_actions()) + 1:03d}"
        record = {
            "timestamp": self._timestamp(),
            "action_id": action_id,
            **action,
            "status": status,
            "evidence": evidence or {},
        }
        self.store.append_capability_action(record)
        self.log_action(
            "capability_action_recorded",
            details={
                "action_id": action_id,
                "action": action.get("action"),
                "capability_gap_id": action.get("capability_gap_id"),
                "status": status,
            },
        )
        return record

    def record_diagnostic(self, diagnostic: dict[str, Any]) -> dict[str, Any]:
        record = {"timestamp": self._timestamp(), **diagnostic}
        self.store.append_diagnostic(record)
        self.log_action(
            "planner_diagnostic_completed",
            details={
                "action_id": diagnostic.get("action_id"),
                "capability_gap_id": diagnostic.get("capability_gap_id"),
            },
        )
        return record

    def record_research_tree_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if not event.get("event_type"):
            raise ValueError("research-tree event requires event_type")
        record = {"timestamp": self._timestamp(), **event}
        self.store.append_research_tree_event(record)
        return record

    def record_critic_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        if feedback.get("phase") not in {"pre_execution", "post_execution"}:
            raise ValueError("critic feedback requires a valid phase")
        record = {"timestamp": self._timestamp(), **feedback}
        self.store.append_critic_feedback(record)
        return record

    def record_research_strategy(self, strategy: dict[str, Any]) -> dict[str, Any]:
        required = {
            "strategy_id", "phase_label", "decision", "focus_domains",
            "metric_emphasis", "frozen_factors", "worker_assignments",
            "rationale", "evidence_reference", "transition_criteria",
        }
        missing = sorted(required - set(strategy))
        if missing:
            raise ValueError(
                "research strategy missing required fields: " + ", ".join(missing)
            )
        record = {"timestamp": self._timestamp(), **strategy}
        self.store.append_research_strategy(record)
        self.log_action(
            "llm_research_strategy_decided",
            details={
                "strategy_id": strategy["strategy_id"],
                "phase_label": strategy["phase_label"],
                "decision": strategy["decision"],
                "focus_domains": strategy["focus_domains"],
                "metric_emphasis": strategy["metric_emphasis"],
                "frozen_factors": strategy["frozen_factors"],
                "worker_assignments": strategy["worker_assignments"],
                "transition_criteria": strategy["transition_criteria"],
            },
        )
        return record

    def record_code_diff(self, experiment_id: str, diff_text: str) -> Path:
        path = self.store.write_patch(experiment_id, diff_text)
        self.log_action(
            "code_diff_recorded",
            experiment_id=experiment_id,
            details={"patch_path": str(path)},
        )
        return path

    def _timestamp(self) -> str:
        return self.clock().astimezone(timezone.utc).isoformat()
