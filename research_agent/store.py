"""Append-only artifact storage for autonomous research runs."""
from __future__ import annotations

import csv
import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable


class ArtifactStore:
    """Stores run evidence without overwriting prior experiment history."""

    _SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    METRIC_COLUMNS = (
        "experiment_id",
        "parent_experiment_id",
        "decision",
        "GAUC",
        "nDCG@5",
        "primary",
        "delta_primary",
        "runtime_seconds",
    )

    def __init__(self, root: str | Path = "runs") -> None:
        self._lock = threading.RLock()
        self.root = Path(root)
        self.events_path = self.root / "experiments.jsonl"
        self.iterations_path = self.root / "iterations.jsonl"
        self.stages_path = self.root / "stages.jsonl"
        self.metrics_path = self.root / "metrics.csv"
        self.interventions_path = self.root / "manual_interventions.jsonl"
        self.capability_backlog_path = self.root / "capability_backlog.jsonl"
        self.diagnostics_path = self.root / "diagnostics.jsonl"
        self.research_tree_events_path = self.root / "research_tree.jsonl"
        self.critic_feedback_path = self.root / "critic_feedback.jsonl"
        self.research_strategies_path = self.root / "research_strategies.jsonl"
        self.patches_dir = self.root / "patches"
        self.runs_dir = self.root / "runs"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.patches_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def run_dir(self, experiment_id: str, stage_id: str | None = None) -> Path:
        self._validate_identifier(experiment_id)
        with self._lock:
            self.initialize()
            path = self.runs_dir / experiment_id
            if stage_id is None:
                path.mkdir(exist_ok=False)
                return path
            self._validate_identifier(stage_id)
            path.mkdir(exist_ok=True)
            path = path / stage_id
            path.mkdir(exist_ok=False)
            return path

    def append_event(self, event: dict[str, Any]) -> None:
        self._append_jsonl(self.events_path, event)

    def append_iteration(self, iteration: dict[str, Any]) -> None:
        self._append_jsonl(self.iterations_path, iteration)

    def append_stage(self, stage: dict[str, Any]) -> None:
        self._append_jsonl(self.stages_path, stage)

    def append_intervention(self, intervention: dict[str, Any]) -> None:
        self._append_jsonl(self.interventions_path, intervention)

    def append_intervention_once(
        self,
        intervention: dict[str, Any],
        *,
        approval_id: str,
    ) -> bool:
        """Atomically append one versioned approval within this process."""
        with self._lock:
            if any(
                item.get("approval_id") == approval_id
                for item in self._read_jsonl(self.interventions_path)
            ):
                return False
            self._append_jsonl(self.interventions_path, intervention)
            return True

    def append_capability_action(self, action: dict[str, Any]) -> None:
        self._append_jsonl(self.capability_backlog_path, action)

    def append_diagnostic(self, diagnostic: dict[str, Any]) -> None:
        self._append_jsonl(self.diagnostics_path, diagnostic)

    def append_research_tree_event(self, event: dict[str, Any]) -> None:
        self._append_jsonl(self.research_tree_events_path, event)

    def append_critic_feedback(self, feedback: dict[str, Any]) -> None:
        self._append_jsonl(self.critic_feedback_path, feedback)

    def append_research_strategy(self, strategy: dict[str, Any]) -> None:
        self._append_jsonl(self.research_strategies_path, strategy)

    def append_metric_summary(self, summary: dict[str, Any]) -> None:
        with self._lock:
            self.initialize()
            has_file = self.metrics_path.exists()
            with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.METRIC_COLUMNS)
                if not has_file:
                    writer.writeheader()
                writer.writerow({column: summary.get(column, "") for column in self.METRIC_COLUMNS})

    def write_patch(self, experiment_id: str, diff_text: str) -> Path:
        self._validate_identifier(experiment_id)
        with self._lock:
            self.initialize()
            destination = self.patches_dir / f"{experiment_id}.patch"
            destination.write_text(diff_text, encoding="utf-8")
            return destination

    def write_run_json(
        self,
        experiment_id: str,
        filename: str,
        payload: dict[str, Any],
        *,
        stage_id: str | None = None,
    ) -> Path:
        self._validate_identifier(experiment_id)
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        run_dir = self.runs_dir / experiment_id
        if stage_id is not None:
            self._validate_identifier(stage_id)
            run_dir = run_dir / stage_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run directory does not exist: {experiment_id}")
        with self._lock:
            destination = run_dir / filename
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return destination

    def write_root_json(self, filename: str, payload: dict[str, Any]) -> Path:
        """Write a small current-state document at the artifact root."""
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        with self._lock:
            self.initialize()
            destination = self.root / filename
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return destination

    def write_root_text(self, filename: str, content: str) -> Path:
        """Write a generated, human-readable current-state artifact."""
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        with self._lock:
            self.initialize()
            destination = self.root / filename
            destination.write_text(content, encoding="utf-8")
            return destination

    def read_events(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.events_path)

    def read_iterations(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.iterations_path)

    def read_stages(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.stages_path)

    def read_interventions(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.interventions_path)

    def read_capability_actions(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.capability_backlog_path)

    def read_diagnostics(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.diagnostics_path)

    def read_research_tree_events(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.research_tree_events_path)

    def read_critic_feedback(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.critic_feedback_path)

    def read_research_strategies(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.research_strategies_path)

    def read_root_json(self, filename: str) -> dict[str, Any] | None:
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        with self._lock:
            source = self.root / filename
            if not source.exists():
                return None
            return json.loads(source.read_text(encoding="utf-8"))

    def _append_jsonl(self, destination: Path, payload: dict[str, Any]) -> None:
        with self._lock:
            self.initialize()
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _read_jsonl(self, source: Path) -> list[dict[str, Any]]:
        with self._lock:
            if not source.exists():
                return []
            with source.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]

    def _validate_identifier(self, identifier: str) -> None:
        if not self._SAFE_IDENTIFIER.fullmatch(identifier):
            raise ValueError(f"unsafe experiment identifier: {identifier!r}")
