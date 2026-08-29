"""Isolated execution wrapper for PyTorch research candidates."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from data import load

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .logger import ResearchLogger


@dataclass(frozen=True)
class PreparedData:
    """Prepared data exposed to a candidate; test rows are intentionally absent."""

    train_rows: Sequence[Any]
    validation_rows: Sequence[Any]


@dataclass(frozen=True)
class CandidateOutput:
    """Prediction output returned by a candidate training callable."""

    user_ids: Sequence[Any]
    labels: Sequence[Any]
    scores: Sequence[Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerResult:
    experiment_id: str
    status: str
    run_dir: Path
    runtime_seconds: float
    output: CandidateOutput | None = None
    error: str | None = None


CandidateCallable = Callable[[PreparedData, Mapping[str, Any], Path], CandidateOutput]
DataLoader = Callable[[str], Mapping[str, Sequence[Any]]]
Clock = Callable[[], float]


class ExperimentRunner:
    """Runs one candidate in a dedicated directory with evidence capture.

    The candidate callable receives no test data. Future PyTorch models can use
    the supplied run directory for checkpoints and epoch-level artifacts.
    """

    def __init__(
        self,
        logger: ResearchLogger,
        *,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
        data_loader: DataLoader = load,
        clock: Clock = time.monotonic,
    ) -> None:
        self.logger = logger
        self.contract = contract
        self.data_loader = data_loader
        self.clock = clock

    def run(
        self,
        *,
        experiment_id: str,
        hypothesis: str,
        config: Mapping[str, Any],
        candidate: CandidateCallable,
        timeout_seconds: float | None = None,
    ) -> RunnerResult:
        run_dir = self.logger.store.run_dir(experiment_id)
        self.logger.store.write_run_json(
            experiment_id,
            "plan.json",
            {"experiment_id": experiment_id, "hypothesis": hypothesis},
        )
        self.logger.store.write_run_json(experiment_id, "config.json", dict(config))
        self.logger.log_action(
            "candidate_created",
            experiment_id=experiment_id,
            details={"run_dir": str(run_dir)},
        )
        started = self.clock()
        try:
            prepared = self._load_prepared_data()
            self.logger.log_action(
                "data_loaded",
                experiment_id=experiment_id,
                details={
                    "train_rows": len(prepared.train_rows),
                    "validation_rows": len(prepared.validation_rows),
                    "test_rows_exposed": 0,
                },
            )
            self.logger.log_action("training_started", experiment_id=experiment_id)
            output = candidate(prepared, config, run_dir)
            runtime_seconds = self.clock() - started
            if timeout_seconds is not None and runtime_seconds > timeout_seconds:
                message = f"runtime {runtime_seconds:.3f}s exceeded budget {timeout_seconds:.3f}s"
                self._write_failure(experiment_id, run_dir, "timed_out", message, runtime_seconds)
                return RunnerResult(experiment_id, "timed_out", run_dir, runtime_seconds, error=message)
            self.logger.store.write_run_json(
                experiment_id,
                "runner_result.json",
                {"status": "completed", "runtime_seconds": runtime_seconds, "metadata": dict(output.metadata)},
            )
            self.logger.log_action(
                "training_completed",
                experiment_id=experiment_id,
                details={"runtime_seconds": runtime_seconds},
            )
            return RunnerResult(experiment_id, "completed", run_dir, runtime_seconds, output=output)
        except Exception as exc:  # Candidate errors must be recorded, not lost.
            runtime_seconds = self.clock() - started
            message = f"{type(exc).__name__}: {exc}"
            self._write_failure(experiment_id, run_dir, "failed", message, runtime_seconds)
            return RunnerResult(experiment_id, "failed", run_dir, runtime_seconds, error=message)

    def _load_prepared_data(self) -> PreparedData:
        splits = self.data_loader(str(self.contract.data_dir))
        try:
            return PreparedData(
                train_rows=splits[self.contract.train_split],
                validation_rows=splits[self.contract.validation_split],
            )
        except KeyError as exc:
            raise ValueError(f"data.py did not return required split: {exc.args[0]}") from exc

    def _write_failure(
        self,
        experiment_id: str,
        run_dir: Path,
        status: str,
        message: str,
        runtime_seconds: float,
    ) -> None:
        self.logger.store.write_run_json(
            experiment_id,
            "error.json",
            {"status": status, "error": message, "runtime_seconds": runtime_seconds},
        )
        self.logger.log_action(
            status,
            experiment_id=experiment_id,
            details={"error": message, "runtime_seconds": runtime_seconds},
        )
