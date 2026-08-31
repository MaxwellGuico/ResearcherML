"""Final, explicitly authorised test confirmation and submission creation."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from data import encode_candidate, load
from submit import read_submission, write_submission

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .metrics import evaluate_predictions
from .models.torch_fm import _predict, build_candidate_model
from .readiness import audit_readiness
from .reporter import MarkdownReporter
from .state import ResearchState
from .store import ArtifactStore


@dataclass(frozen=True)
class FinalizationResult:
    selected_experiment_id: str
    selection_primary: float
    submission_path: Path
    submission_checked: bool
    test_metrics: dict[str, Any] | None
    report_path: Path


def finalize_run(
    store: ArtifactStore,
    *,
    contract: BenchmarkContract = BENCHMARK_CONTRACT,
    submission_path: str | Path | None = None,
) -> FinalizationResult:
    """Write one valid submission and perform the permitted final test check.

    Test rows enter only here, after validation-based selection is complete.
    """
    persisted = store.read_root_json("state.json") or {}
    state = ResearchState.from_dict(persisted)
    target = Path(submission_path) if submission_path else store.root / "final_submission.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = _selected_iteration(store, state.current_best_experiment_id)
    if selected is None:
        _make_official_baseline_submission(target, contract)
        test_metrics = None
    else:
        test_metrics = _write_selected_torch_submission(target, selected, contract)
    splits = load(str(contract.data_dir))
    read_submission(target, splits[contract.test_split])
    summary: dict[str, Any] = {
        "selected_experiment_id": state.current_best_experiment_id,
        "selection_primary": state.current_best_primary,
        "submission_path": str(target),
        "submission_checked": True,
        "test_GAUC": test_metrics.get("GAUC") if test_metrics else "unavailable (official baseline submission; no agent-selected candidate)",
        "test_nDCG@5": test_metrics.get("nDCG@5") if test_metrics else "unavailable (official baseline submission; no agent-selected candidate)",
        "test_primary": test_metrics.get("primary") if test_metrics else "unavailable (official baseline submission; no agent-selected candidate)",
    }
    store.write_root_json("final_summary.json", summary)
    readiness = audit_readiness(store, target)
    summary["readiness_passed"] = readiness.passed
    summary["readiness_issues"] = list(readiness.issues)
    store.write_root_json("final_summary.json", summary)
    report_path = MarkdownReporter(store).write()
    return FinalizationResult(
        selected_experiment_id=state.current_best_experiment_id,
        selection_primary=state.current_best_primary,
        submission_path=target,
        submission_checked=True,
        test_metrics=test_metrics,
        report_path=report_path,
    )


def _selected_iteration(store: ArtifactStore, experiment_id: str) -> dict[str, Any] | None:
    if experiment_id == "baseline":
        return None
    for record in reversed(store.read_iterations()):
        if record.get("experiment_id") == experiment_id:
            return record
    raise ValueError(f"selected experiment is missing from iteration history: {experiment_id}")


def _write_selected_torch_submission(target: Path, record: dict[str, Any], contract: BenchmarkContract) -> dict[str, Any]:
    checkpoint_path = Path(record.get("runner_metadata", {}).get("checkpoint_path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"selected PyTorch checkpoint is unavailable: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    splits = load(str(contract.data_dir))
    encoded, feature_dim = encode_candidate(
        splits,
        feature_variant=str(checkpoint["config"].get("feature_variant", "baseline")),
    )
    test_x, test_y, test_users = encoded[contract.test_split]
    if int(checkpoint.get("feature_dim", feature_dim)) != feature_dim:
        raise ValueError("selected checkpoint feature dimension does not match canonical encoded data")
    model = build_candidate_model(
        checkpoint["config"],
        feature_dim=feature_dim,
        field_count=int(test_x.shape[1]),
        embedding_dim=int(checkpoint["config"].get("embedding_dim", 16)),
    )
    model.load_state_dict(checkpoint["model_state"])
    scores = _predict(model, torch.as_tensor(test_x, dtype=torch.long))
    write_submission(target, splits[contract.test_split], scores)
    return evaluate_predictions(test_users, test_y, scores, split=contract.test_split, allow_test=True).as_dict()


def _make_official_baseline_submission(target: Path, contract: BenchmarkContract) -> None:
    command = [
        sys.executable,
        "submit.py",
        str(target),
        "--make",
        "--split",
        contract.test_split,
        "--data_dir",
        str(contract.data_dir),
    ]
    environment = dict(os.environ)
    # submit.py prints Chinese status text; force a portable encoding on Windows.
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"official baseline submission failed: {completed.stderr.strip() or completed.stdout.strip()}")
