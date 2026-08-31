"""Compact durable lessons that make critic judgments influence later planning."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any


def refresh_critic_memory(store: Any) -> dict[str, Any]:
    _backfill_completed_iterations(store)
    records = store.read_critic_feedback()
    dispositions = Counter(str(item.get("disposition", "unknown")) for item in records)
    blocking = [item for item in records if item.get("disposition") in {
        "change_configuration_or_hypothesis",
        "repair_lineage_reasoning",
        "repair_execution_alignment",
        "repair_safety_or_semantic_contract",
        "repair_evidence_chain",
        "diagnose_execution_failure",
    }]
    valid_negatives = [
        item for item in records if item.get("disposition") == "record_valid_negative_and_branch"
    ]
    supported = [
        item for item in records if item.get("disposition") == "continue_supported_lineage"
    ]
    memory = {
        "feedback_count": len(records),
        "disposition_counts": dict(sorted(dispositions.items())),
        "blocking_lessons": [_compact(item) for item in blocking[-8:]],
        "valid_negative_lessons": [_compact(item) for item in valid_negatives[-8:]],
        "supported_lineages": [_compact(item) for item in supported[-8:]],
        "recent_feedback": [_compact(item) for item in records[-12:]],
    }
    store.write_root_json("critic_memory.json", memory)
    return memory


def _backfill_completed_iterations(store: Any) -> None:
    existing = {
        (str(item.get("experiment_id")), str(item.get("phase")))
        for item in store.read_critic_feedback()
    }
    for item in store.read_iterations():
        experiment_id = str(item.get("experiment_id", ""))
        if not experiment_id or (experiment_id, "post_execution") in existing:
            continue
        trace = item.get("semantic_review", {}).get("trace", {}) or {}
        feedback = trace.get("planner_feedback")
        if not isinstance(feedback, dict):
            primary = item.get("metrics", {}).get("primary")
            delta = item.get("delta_primary")
            semantic_approved = item.get("semantic_review", {}).get("approved", True)
            if not semantic_approved:
                disposition = "repair_evidence_chain"
                lesson = "Historical semantic review failed; repair the evidence chain before drawing conclusions."
            elif isinstance(primary, (int, float)) and math.isfinite(float(primary)):
                disposition = "continue_supported_lineage" if isinstance(delta, (int, float)) and delta > 0 else "record_valid_negative_and_branch"
                lesson = (
                    "Historical controlled evidence improved on its incumbent; preserve and refine this lineage."
                    if disposition == "continue_supported_lineage"
                    else "Historical measured evidence did not improve the incumbent; do not repeat its exact configuration."
                )
            else:
                disposition = "diagnose_execution_failure"
                lesson = "Historical branch produced no finite ranking evidence; diagnose execution before scientific reuse."
            feedback = {
                "phase": "post_execution",
                "direction_id": item.get("direction_id"),
                "disposition": disposition,
                "lesson": lesson,
                "delta_primary": delta,
                "do_not_repeat_exact_configuration": isinstance(primary, (int, float)),
            }
        store.append_critic_feedback({
            "timestamp": item.get("timestamp"),
            "experiment_id": experiment_id,
            "source": "historical_iteration_backfill",
            **feedback,
        })
        existing.add((experiment_id, "post_execution"))


def _compact(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "timestamp", "experiment_id", "phase", "direction_id", "disposition", "lesson",
            "failed_checks", "delta_primary", "do_not_repeat_exact_configuration",
            "critic_recommended_action", "next_hypothesis_constraint",
        )
        if item.get(key) is not None
    }
