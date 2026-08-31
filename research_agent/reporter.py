"""Renders human-readable reports only from structured run records."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any

from .store import ArtifactStore


class MarkdownReporter:
    """Generates the readable log from the append-only artifact store."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def write(self, destination: str | Path | None = None) -> Path:
        target = Path(destination) if destination else self.store.root / "research_log.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
        return target

    def render(self) -> str:
        lines = ["# Research Run Log", ""]
        iterations = self.store.read_iterations()
        if not iterations:
            lines.extend(["No completed iterations have been recorded.", ""])
        for record in iterations:
            lines.extend(self._render_iteration(record))
        lines.extend(self._render_run_statistics(iterations))
        lines.extend(self._render_research_strategies())
        lines.extend(self._render_research_tree())
        lines.extend(self._render_critic_memory())
        lines.extend(self._render_capability_backlog())
        lines.extend(self._render_interventions())
        lines.extend(self._render_final_summary())
        return "\n".join(lines).rstrip() + "\n"

    def _render_iteration(self, record: dict[str, Any]) -> list[str]:
        metrics = record.get("metrics", {})
        lines = [
            f"## {record['experiment_id']}",
            "",
            f"- Decision: {record['decision']}",
            f"- Hypothesis: {record['hypothesis']}",
            f"- Rationale: {record['rationale']}",
            f"- GAUC: {metrics.get('GAUC', 'unavailable')}",
            f"- nDCG@5: {metrics.get('nDCG@5', 'unavailable')}",
            f"- Primary: {metrics.get('primary', 'unavailable')}",
            f"- Delta primary: {record.get('delta_primary', 'unavailable')}",
            f"- Runtime seconds: {record.get('runtime_seconds', 'unavailable')}",
        ]
        if record.get("error"):
            lines.append(f"- Error: {record['error']}")
        if record.get("recovery"):
            lines.append(f"- Recovery: {record['recovery']}")
        stages = record.get("stages", [])
        if stages:
            lines.append(f"- Fidelity/confirmation stages: {len(stages)}")
            for stage in stages:
                stage_metrics = stage.get("metrics", {})
                lines.append(
                    f"  - {stage.get('stage_id')}: {stage.get('status')}"
                    f"; primary={stage_metrics.get('primary', 'unavailable')}"
                    f"; runtime={stage.get('runtime_seconds', 'unavailable')}s"
                )
        lines.append("")
        return lines

    def _render_research_strategies(self) -> list[str]:
        strategies = self.store.read_research_strategies()
        lines = [
            "## LLM Research Strategy History", "",
            f"Strategy decisions: {len(strategies)}", "",
        ]
        for item in strategies:
            lines.extend([
                f"- {item.get('strategy_id')} — {item.get('decision')} · {item.get('phase_label')}",
                f"  - Focus domains: {', '.join(item.get('focus_domains', []))}",
                f"  - Metric emphasis: {item.get('metric_emphasis')}",
                f"  - Frozen factors: {', '.join(item.get('frozen_factors', [])) or 'none'}",
                f"  - Worker assignments: {item.get('worker_assignments', [])}",
                f"  - Evidence: {item.get('evidence_reference')}",
                f"  - Transition: {item.get('transition_criteria')}",
                "",
            ])
        return lines

    def _render_capability_backlog(self) -> list[str]:
        actions = self.store.read_capability_actions()
        lines = ["## Capability Backlog", "", f"Capability actions: {len(actions)}", ""]
        for item in actions:
            lines.extend([
                f"- {item.get('action_id')} — {item.get('action')}",
                f"  - Gap: {item.get('capability_gap_id')}",
                f"  - Status: {item.get('status')}",
                f"  - Description: {item.get('capability_gap_description')}",
                f"  - Approval reason: {item.get('approval_reason') or 'not required'}",
                "",
            ])
        return lines

    def _render_research_tree(self) -> list[str]:
        tree = self.store.read_root_json("research_tree.json")
        if not tree:
            return ["## Research Tree", "", "No research-tree snapshot is available.", ""]
        incumbent = tree.get("incumbent", {})
        lines = [
            "## Research Tree",
            "",
            f"- Hypotheses retained: {len(tree.get('hypotheses', []))}",
            f"- Experiment branches: {len(tree.get('experiments', []))}",
            f"- Deferred hypotheses: {len(tree.get('deferred_hypotheses', []))}",
            f"- Failed/rejected branches: {len(tree.get('failed_branches', []))}",
            f"- Incumbent ancestry: {' → '.join(incumbent.get('ancestry', [])) or 'unavailable'}",
            "",
        ]
        for candidate in tree.get("continuation_candidates", [])[:5]:
            lines.append(
                f"- Continuation: {candidate.get('reason')} — "
                f"{candidate.get('hypothesis_id')} ({candidate.get('hypothesis')})"
            )
        lines.append("")
        return lines

    def _render_critic_memory(self) -> list[str]:
        memory = self.store.read_root_json("critic_memory.json")
        if not memory:
            return ["## Critic Feedback", "", "No critic feedback memory is available.", ""]
        lines = [
            "## Critic Feedback",
            "",
            f"- Feedback records: {memory.get('feedback_count', 0)}",
            f"- Dispositions: {memory.get('disposition_counts', {})}",
            "",
        ]
        for item in memory.get("recent_feedback", [])[-5:]:
            lines.append(
                f"- {item.get('experiment_id')} · {item.get('phase')} · "
                f"{item.get('disposition')}: {item.get('lesson')}"
            )
        lines.append("")
        return lines

    def _render_run_statistics(self, iterations: list[dict[str, Any]]) -> list[str]:
        events = self.store.read_events()
        decisions = Counter(str(item.get("decision", "unknown")) for item in iterations)
        total_runtime = sum(float(item.get("runtime_seconds") or 0.0) for item in iterations)
        cpu_seconds = sum(
            float(item.get("resource_usage", {}).get(key) or 0.0)
            for item in iterations
            for key in ("cpu_user_seconds", "cpu_system_seconds")
        )
        peak_rss = max(
            (float(item.get("resource_usage", {}).get("peak_rss_bytes") or 0.0) for item in iterations),
            default=0.0,
        )
        token_totals = Counter()
        for event in events:
            usage = event.get("details", {}).get("usage", {})
            for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    token_totals[key] += usage[key]
        failures = sum(event.get("action") in {"failed", "timed_out", "safety_rejected"} for event in events)
        recoveries = sum(event.get("action") in {"accepted_candidate_restored", "stale_active_experiment_recovered", "demo_failure_recovery_completed"} for event in events)
        return [
            "## Run Statistics",
            "",
            f"- Completed hypotheses: {len(iterations)}",
            f"- Decisions: {dict(sorted(decisions.items()))}",
            f"- Failures recorded: {failures}",
            f"- Recovery events: {recoveries}",
            f"- Total iteration runtime seconds: {total_runtime:.3f}",
            f"- Recorded CPU seconds: {cpu_seconds:.3f}",
            f"- Peak recorded RSS bytes: {int(peak_rss) if peak_rss else 'unavailable'}",
            f"- LLM input tokens: {token_totals.get('input_tokens', 'unavailable')}",
            f"- LLM output tokens: {token_totals.get('output_tokens', 'unavailable')}",
            f"- LLM total tokens: {token_totals.get('total_tokens', 'unavailable')}",
            f"- LLM cached input tokens: {token_totals.get('cached_input_tokens', 'unavailable')}",
            "",
        ]

    def _render_interventions(self) -> list[str]:
        raw_interventions = self.store.read_interventions()
        interventions: list[dict[str, Any]] = []
        duplicate_counts: Counter[str] = Counter()
        seen: set[str] = set()
        for item in raw_interventions:
            identity = str(item.get("approval_id") or "legacy:" + "\x1f".join(
                str(item.get(key) or "")
                for key in ("experiment_id", "description", "reason", "effect")
            ))
            duplicate_counts[identity] += 1
            if identity not in seen:
                seen.add(identity)
                interventions.append({**item, "_identity": identity})
        reuse_events = Counter(
            str(event.get("details", {}).get("approval_id"))
            for event in self.store.read_events()
            if event.get("action") in {"architecture_approval_reused", "approval_reused"}
            and event.get("details", {}).get("approval_id")
        )
        lines = [
            "## Manual Intervention Summary", "",
            f"Manual interventions: {len(interventions)}",
        ]
        if len(raw_interventions) != len(interventions):
            lines.append(f"- Historical append-only records: {len(raw_interventions)}")
        lines.append("")
        for index, item in enumerate(interventions, start=1):
            identity = item["_identity"]
            lines.extend(
                [
                    f"{index}. {item['timestamp']} — {item.get('experiment_id') or 'run-wide'}",
                    f"   - Action: {item['description']}",
                    f"   - Approval ID: {item.get('approval_id') or 'legacy-unversioned'}",
                    f"   - Authority scope: {item.get('authority_scope') or 'not specified'}",
                    f"   - Reason: {item['reason']}",
                    f"   - Effect: {item['effect']}",
                    f"   - Reuses: {reuse_events.get(str(item.get('approval_id')), 0) + duplicate_counts[identity] - 1}",
                    "",
                ]
            )
        return lines

    def _render_final_summary(self) -> list[str]:
        summary = self.store.read_root_json("final_summary.json")
        if not summary:
            return []
        lines = ["## Final Summary", ""]
        for key in (
            "selected_experiment_id", "selection_primary", "test_GAUC", "test_nDCG@5",
            "test_primary", "submission_path", "submission_checked", "readiness_passed",
            "readiness_issues",
        ):
            lines.append(f"- {key}: {summary.get(key, 'unavailable')}")
        lines.append("")
        return lines
