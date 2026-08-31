"""Compact cross-run coverage for mechanisms the hypothesis tree cannot infer."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .architecture import REVIEWED_OPERATORS, parse_architecture_id
from .store import ArtifactStore


_ARCHITECTURES = (
    ("fm", "executable"),
    ("deepfm", "executable"),
    ("nfm_residual", "executable"),
    *((operator, "executable") for operator in REVIEWED_OPERATORS),
)
_OBJECTIVES = (
    ("pointwise", "executable"),
    ("pairwise", "executable"),
    ("top_list_weighted", "missing_capability"),
    ("pointwise_top_list_mixture", "missing_capability"),
    ("multi_task_combined_loss", "executable"),
)
_FEATURES = (
    ("baseline", "executable"),
    ("author_affinity", "executable"),
    ("user_history", "executable"),
    ("weekday", "executable"),
)


def build_research_coverage(
    store: ArtifactStore,
    current_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a small auditable snapshot from valid local and sibling-run evidence."""
    records = [{
        "source": "official_verified_baseline",
        "experiment_id": "baseline",
        "decision": "accepted",
        "architecture": "fm",
        "loss": "pointwise",
        "training_objective": "pointwise",
        "feature_variant": "baseline",
        "primary": 0.6015,
        "GAUC": 0.6671,
        "nDCG@5": 0.5358,
        "config": {
            "architecture": "fm", "loss": "pointwise", "training_objective": "pointwise",
            "feature_variant": "baseline"
        },
    }]
    records.extend(_validated_records(current_history, source=store.root.name))
    source_dirs = _artifact_directories(store.root.parent)
    for directory in source_dirs:
        if directory.resolve() == store.root.resolve():
            continue
        records.extend(_validated_records(_read_jsonl(directory / "iterations.jsonl"), source=directory.name))

    records = _deduplicated_records(records)
    backlog = store.read_capability_actions()
    architecture = _coverage_group(_ARCHITECTURES, records, "architecture", backlog)
    objectives = _coverage_group(_OBJECTIVES, records, "objective", backlog)
    features = _coverage_group(_FEATURES, records, "feature", backlog)
    best = sorted(records, key=lambda item: float(item["primary"]), reverse=True)[:8]
    snapshot = {
        "version": 1,
        "purpose": "Coverage lists tested and untested mechanisms; research_tree records hypothesis lineage.",
        "architectures": architecture,
        "objectives": objectives,
        "features": features,
        "cross_run_evidence": [
            {
                "source": item["source"],
                "experiment_id": item.get("experiment_id"),
                "architecture": item["architecture"],
                "loss": item["loss"],
                "training_objective": item.get("training_objective", "pointwise"),
                "feature_variant": item["feature_variant"],
                "primary": item["primary"],
                "GAUC": item.get("GAUC"),
                "nDCG@5": item.get("nDCG@5"),
                "decision": item.get("decision"),
            }
            for item in best
        ],
        "source_artifact_count": len({item["source"] for item in records}),
        "validated_configuration_count": len(records),
    }
    store.write_root_json("research_coverage.json", snapshot)
    return snapshot


def _artifact_directories(workspace: Path) -> list[Path]:
    try:
        values = [
            path for path in workspace.iterdir()
            if path.is_dir() and (path / "iterations.jsonl").is_file()
            and not any(
                marker in path.name.lower()
                for marker in ("smoke", "probe", "demo")
            )
        ]
    except OSError:
        return []
    return sorted(values, key=lambda path: path.stat().st_mtime, reverse=True)[:50]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _validated_records(
    records: Iterable[Mapping[str, Any]], *, source: str
) -> Iterable[dict[str, Any]]:
    for record in records:
        metrics = record.get("metrics", {})
        config = record.get("config", {})
        semantic = record.get("semantic_review", {})
        if not isinstance(metrics, Mapping) or not isinstance(config, Mapping):
            continue
        if record.get("decision") not in {"accepted", "rejected"}:
            continue
        aggregate = _seed_aggregate(record)
        try:
            primary = float(aggregate.get("primary", metrics["primary"]))
            gauc = float(aggregate.get("GAUC", metrics["GAUC"]))
            ndcg = float(aggregate.get("nDCG@5", metrics["nDCG@5"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (primary, gauc, ndcg)):
            continue
        if abs(primary - ((gauc + ndcg) / 2.0)) > 1e-9:
            continue
        if isinstance(semantic, Mapping) and semantic and semantic.get("approved") is not True:
            continue
        yield {
            "source": source,
            "experiment_id": record.get("experiment_id"),
            "decision": record.get("decision"),
            "architecture": str(config.get("architecture", "fm")),
            "loss": str(config.get("loss", "pointwise")),
            "training_objective": str(config.get("training_objective", "pointwise")),
            "feature_variant": str(config.get("feature_variant", "baseline")),
            "primary": primary,
            "GAUC": gauc,
            "nDCG@5": ndcg,
            "config": dict(config),
        }


def _deduplicated_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best_by_config: dict[str, dict[str, Any]] = {}
    for record in records:
        config = record.get("config", {})
        scientific_config = {
            key: config.get(key)
            for key in (
                "architecture", "batch_size", "embedding_dim", "feature_variant",
                "l2", "learning_rate", "loss", "training_objective",
            )
            if key in config
        } if isinstance(config, Mapping) else {}
        signature = json.dumps(scientific_config, sort_keys=True, default=str)
        current = best_by_config.get(signature)
        if current is None or float(record["primary"]) > float(current["primary"]):
            best_by_config[signature] = dict(record)
    return list(best_by_config.values())


def _seed_aggregate(record: Mapping[str, Any]) -> dict[str, float]:
    evidence = record.get("diagnostic_evidence", {})
    if not isinstance(evidence, Mapping):
        return {}
    stages = evidence.get("fidelity_results", [])
    if not isinstance(stages, list):
        return {}
    by_seed: dict[int, Mapping[str, Any]] = {}
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("fidelity") != "full":
            continue
        metrics = stage.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        try:
            seed = int(stage.get("seed", 0))
            float(metrics["primary"])
        except (KeyError, TypeError, ValueError):
            continue
        by_seed[seed] = metrics
    if len(by_seed) < 2:
        return {}
    return {
        metric: sum(float(value[metric]) for value in by_seed.values()) / len(by_seed)
        for metric in ("GAUC", "nDCG@5", "primary")
    }


def _coverage_group(
    inventory: Sequence[tuple[str, str]],
    records: Sequence[Mapping[str, Any]],
    kind: str,
    backlog: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    backlog_text = " ".join(
        f"{item.get('hypothesis', '')} {item.get('capability_gap_id', '')}"
        for item in backlog
    ).lower()
    for mechanism, availability in inventory:
        matching = [item for item in records if _record_has_mechanism(item, kind, mechanism)]
        accepted = [item for item in matching if item.get("decision") == "accepted"]
        isolated_accepted = [
            item for item in accepted
            if _record_is_isolated_mechanism(item, kind, mechanism)
        ]
        isolated = [
            item for item in matching
            if _record_is_isolated_mechanism(item, kind, mechanism)
        ]
        pending = availability != "executable" and _backlog_mentions(mechanism, backlog_text)
        status = (
            "accepted" if isolated_accepted
            else "present_in_accepted" if accepted
            else "tested" if matching
            else "pending_implementation" if pending
            else "untested" if availability == "executable"
            else "unavailable"
        )
        best = max(matching, key=lambda item: float(item["primary"]), default=None)
        result.append({
            "mechanism": mechanism,
            "availability": availability,
            "status": status,
            "experiment_count": len(matching),
            "isolated_experiment_count": len(isolated),
            "evidence_scope": (
                "isolated" if isolated_accepted
                else "isolated_tested" if isolated
                else "combined" if accepted
                else "tested" if matching
                else "none"
            ),
            "best_primary": best.get("primary") if best else None,
            "best_source": best.get("source") if best else None,
        })
    return result


def _record_has_mechanism(record: Mapping[str, Any], kind: str, mechanism: str) -> bool:
    if kind == "objective":
        if mechanism == "multi_task_combined_loss":
            return str(record.get("training_objective", "pointwise")).startswith("multitask_")
        return record.get("loss") == mechanism
    if kind == "feature":
        return record.get("feature_variant") == mechanism
    architecture = str(record.get("architecture", "fm"))
    if mechanism in {"fm", "deepfm", "nfm_residual"}:
        return architecture == mechanism
    try:
        spec = parse_architecture_id(architecture)
    except ValueError:
        return False
    return spec is not None and mechanism in spec.interaction_paths


def _record_is_isolated_mechanism(
    record: Mapping[str, Any], kind: str, mechanism: str
) -> bool:
    if kind != "architecture" or mechanism in {"fm", "deepfm", "nfm_residual"}:
        return _record_has_mechanism(record, kind, mechanism)
    try:
        spec = parse_architecture_id(str(record.get("architecture", "fm")))
    except ValueError:
        return False
    return spec is not None and spec.interaction_paths == (mechanism,)


def _backlog_mentions(mechanism: str, backlog_text: str) -> bool:
    aliases = {
        "multi_task_shared_backbone": ("multi-task", "multi_task"),
        "multi_task_combined_loss": ("multi-task", "multi_task"),
        "top_list_weighted": ("top-of-list", "top_list", "toplist"),
        "pointwise_top_list_mixture": ("mixture", "top-of-list"),
    }
    return any(value in backlog_text for value in aliases.get(mechanism, (mechanism,)))
