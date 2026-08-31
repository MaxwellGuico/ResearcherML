"""Final evidence-integrity audit for a completed autonomous research run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .state import ResearchState
from .store import ArtifactStore


@dataclass(frozen=True)
class ReadinessResult:
    passed: bool
    checks: Mapping[str, bool]
    issues: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": dict(self.checks), "issues": list(self.issues)}


def audit_readiness(store: ArtifactStore, submission_path: str | Path) -> ReadinessResult:
    """Check that the final claim is backed by required structured artifacts."""
    events = store.read_events()
    iterations = store.read_iterations()
    stages = store.read_stages()
    state_payload = store.read_root_json("state.json")
    summary = store.read_root_json("final_summary.json")
    submission = Path(submission_path)
    actions = {str(event.get("action")) for event in events}
    required_iteration_fields = {
        "experiment_id", "parent_experiment_id", "hypothesis", "rationale", "config",
        "changed_factors", "direction_id", "metrics", "runtime_seconds", "resource_usage",
        "decision", "error", "recovery", "safety", "code_diff_path",
    }
    normal_iterations = [item for item in iterations if item.get("direction_id")]
    iteration_ids = {str(item.get("experiment_id")) for item in iterations}
    pre_run_semantic_rejections = {
        str(event.get("experiment_id"))
        for event in events
        if event.get("action") == "semantic_rejection_recorded"
        and event.get("details", {}).get("semantic_trace")
    }
    patches_exist = all(
        bool(item.get("code_diff_path")) and Path(str(item["code_diff_path"])).is_file()
        for item in iterations
    )
    semantic_records_complete = all(
        (
            isinstance(item.get("semantic_review"), Mapping)
            and "approved" in item["semantic_review"]
            and isinstance(item.get("diagnostic_evidence"), Mapping)
        )
        or (
            item.get("decision") == "rejected"
            and str(item.get("experiment_id")) in pre_run_semantic_rejections
            and not item.get("metrics")
        )
        for item in normal_iterations
    )
    selected_consistent = False
    if state_payload:
        state = ResearchState.from_dict(state_payload)
        selected_consistent = state.current_best_experiment_id == "baseline" or any(
            item.get("experiment_id") == state.current_best_experiment_id
            and item.get("decision") == "accepted"
            for item in iterations
        )
    checks = {
        "benchmark_verified": "benchmark_verified" in actions,
        "run_lifecycle_logged": {"research_run_started", "research_run_finished"} <= actions,
        "state_persisted": state_payload is not None,
        "iterations_have_mandatory_fields": all(
            required_iteration_fields <= set(item) for item in iterations
        ),
        "normal_iterations_have_diagnostics_and_semantic_review": semantic_records_complete,
        "stage_records_link_to_completed_experiments": all(
            str(item.get("experiment_id")) in iteration_ids for item in stages
        ),
        "configuration_patches_exist": patches_exist,
        "selected_incumbent_is_consistent": selected_consistent,
        "final_summary_persisted": summary is not None,
        "submission_exists_and_is_nonempty": submission.is_file() and submission.stat().st_size > 0,
        "submission_schema_checked": bool(summary and summary.get("submission_checked")),
    }
    issues = tuple(name for name, passed in checks.items() if not passed)
    result = ReadinessResult(not issues, checks, issues)
    store.write_root_json("readiness.json", result.as_dict())
    return result
