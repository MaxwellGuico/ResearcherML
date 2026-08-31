"""Compact, structured evidence supplied to research-planning agents."""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from .architecture import parse_architecture_id


CONFIRMATION_SEEDS = frozenset({0, 1, 2})


def robust_stage_evidence(
    records: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None]:
    """Aggregate one full-fidelity result per required confirmation seed."""
    by_seed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        config = record.get("config", {})
        metrics = record.get("metrics", {})
        if config.get("fidelity") != "full" or not isinstance(metrics, Mapping):
            continue
        try:
            seed = int(config.get("seed", 0))
            values = [float(metrics[name]) for name in metric_names]
        except (KeyError, TypeError, ValueError):
            continue
        if all(value == value and abs(value) != float("inf") for value in values):
            by_seed[seed] = record
    if set(by_seed) != CONFIRMATION_SEEDS:
        return None, None
    aggregate = {
        name: sum(float(by_seed[seed]["metrics"][name]) for seed in sorted(by_seed)) / len(by_seed)
        for name in metric_names
    }
    seed_zero_metrics = by_seed[0]["metrics"]
    for name in ("rows", "users", "evaluator_sha256"):
        if name in seed_zero_metrics:
            aggregate[name] = seed_zero_metrics[name]
    representative = min(
        by_seed.values(),
        key=lambda item: abs(float(item["metrics"]["primary"]) - float(aggregate["primary"])),
    )
    return aggregate, representative


def build_experiment_evidence(
    stages: Sequence[Mapping[str, Any]],
    *,
    baseline_primary: float,
    improvement_threshold: float,
) -> dict[str, Any]:
    metric_stages = [
        stage for stage in stages
        if isinstance(stage.get("metrics", {}).get("primary"), (int, float))
    ]
    best = max(metric_stages, key=lambda item: float(item["metrics"]["primary"]), default=None)
    seed_stages = [stage for stage in metric_stages if str(stage.get("stage_id", "")).startswith("seed_")]
    seed_primaries = [float(stage["metrics"]["primary"]) for stage in seed_stages]
    failures = [_failure_summary(stage) for stage in stages if stage.get("error") or not stage.get("metrics")]
    diagnostics = []
    for stage in stages:
        model_diagnostics = stage.get("runner_metadata", {}).get("diagnostics")
        if model_diagnostics:
            diagnostics.append({"stage_id": stage.get("stage_id"), **_compact_model_diagnostics(model_diagnostics)})
    best_stage_primary = float(best["metrics"]["primary"]) if best else None
    robust_metrics, _ = robust_stage_evidence(
        stages, ("GAUC", "nDCG@5", "primary")
    )
    decision_primary = (
        float(robust_metrics["primary"]) if robust_metrics is not None
        else best_stage_primary
    )
    gain = decision_primary - baseline_primary if decision_primary is not None else None
    return {
        "baseline_primary_before": baseline_primary,
        "best_primary": decision_primary,
        "best_stage_primary": best_stage_primary,
        "robust_primary": robust_metrics.get("primary") if robust_metrics else None,
        "gain_over_incumbent": gain,
        "exceeds_incumbent": bool(gain is not None and gain > 0),
        "exceeds_convergence_threshold": bool(gain is not None and gain > improvement_threshold),
        "best_stage_id": best.get("stage_id") if best else None,
        "fidelity_results": [
            {
                "stage_id": stage.get("stage_id"),
                "fidelity": stage.get("config", {}).get("fidelity"),
                "seed": stage.get("config", {}).get("seed"),
                "status": stage.get("status"),
                "metrics": stage.get("metrics", {}),
                "runtime_seconds": stage.get("runtime_seconds"),
            }
            for stage in stages
        ],
        "seed_confirmation": {
            "count": len(seed_primaries),
            "primaries": seed_primaries,
            "mean_primary": mean(seed_primaries) if seed_primaries else None,
            "std_primary": pstdev(seed_primaries) if seed_primaries else None,
            "required_seeds": sorted(CONFIRMATION_SEEDS),
            "complete": robust_metrics is not None,
            "robust_mean_primary": robust_metrics.get("primary") if robust_metrics else None,
        },
        "model_diagnostics": diagnostics,
        "failures": failures,
        "resource_usage": {
            "runtime_seconds": sum(float(stage.get("runtime_seconds") or 0.0) for stage in stages),
            "cpu_seconds": sum(
                float(stage.get("resource_usage", {}).get(key) or 0.0)
                for stage in stages for key in ("cpu_user_seconds", "cpu_system_seconds")
            ),
            "peak_rss_bytes": max(
                (float(stage.get("resource_usage", {}).get("peak_rss_bytes") or 0.0) for stage in stages),
                default=0.0,
            ),
        },
    }


def refresh_evidence_memory(store: Any, state: Any) -> dict[str, Any]:
    """Persist the planner-facing research memory from completed hypotheses."""
    hypotheses = [
        {
            "experiment_id": item.get("experiment_id"),
            "hypothesis": item.get("hypothesis"),
            "direction_id": item.get("direction_id"),
            "decision": item.get("decision"),
            "diagnostic_evidence": item.get("diagnostic_evidence", {}),
            "semantic_review": item.get("semantic_review", {}),
            "terminal_reason": item.get("terminal_reason"),
            "error": item.get("error"),
        }
        for item in store.read_iterations()
    ]
    memory = {
        "current_incumbent": {
            "experiment_id": state.current_best_experiment_id,
            "primary": state.current_best_primary,
        },
        "completed_hypotheses": state.completed_iterations,
        "consecutive_below_convergence_threshold": state.consecutive_non_improvements,
        "hypotheses": hypotheses,
    }
    store.write_root_json("evidence_memory.json", memory)
    return memory


def architecture_ablation_evidence(
    *,
    parent_config: Mapping[str, Any] | None,
    candidate_config: Mapping[str, Any],
    parent_primary: float,
    candidate_primary: float | None,
) -> dict[str, Any] | None:
    """Describe only the mechanism isolated by a valid one-path ablation."""
    if not parent_config:
        return None
    try:
        parent = parse_architecture_id(str(parent_config.get("architecture", "")))
        candidate = parse_architecture_id(str(candidate_config.get("architecture", "")))
    except ValueError:
        return None
    if parent is None or candidate is None or len(parent.interaction_paths) != 2:
        return None
    removed = set(parent.interaction_paths) - set(candidate.interaction_paths)
    retained = set(candidate.interaction_paths)
    if len(removed) != 1 or len(retained) != 1 or not retained.issubset(parent.interaction_paths):
        return None
    return {
        "design": "controlled_single_path_ablation",
        "parent_architecture": parent.architecture_id,
        "candidate_architecture": candidate.architecture_id,
        "removed_path": next(iter(removed)),
        "retained_path": next(iter(retained)),
        "frozen_parent_primary": parent_primary,
        "candidate_primary": candidate_primary,
        "delta_from_frozen_parent": (
            candidate_primary - parent_primary if candidate_primary is not None else None
        ),
        "claim_boundary": (
            "This comparison estimates the contribution of the removed path in the "
            "parent composition; it does not prove a broader causal mechanism."
        ),
    }


def _failure_summary(stage: Mapping[str, Any]) -> dict[str, Any]:
    error = str(stage.get("error") or "")
    safety = stage.get("safety", {})
    if safety and not safety.get("passed", True):
        category = "safety_rejection"
    elif "timed out" in error.lower() or "timeout" in error.lower():
        category = "infrastructure_timeout"
    elif error:
        category = "execution_or_model_failure"
    else:
        category = "no_metric_evidence"
    return {"stage_id": stage.get("stage_id"), "category": category, "error": error or None}


def _compact_model_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    curve = list(value.get("epoch_curve", []))
    selected_curve = curve if len(curve) <= 4 else [curve[0], curve[-1]]
    return {
        "training": value.get("training", {}),
        "score_distribution": value.get("score_distribution", {}),
        "feature_coverage": value.get("feature_coverage", {}),
        "user_segments": value.get("user_segments", {}),
        "stratified_validation": value.get("stratified_validation", {}),
        "model": value.get("model", {}),
        "epoch_curve_summary": selected_curve,
    }
