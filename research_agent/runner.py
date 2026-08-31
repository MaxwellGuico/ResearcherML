"""Isolated execution wrapper for PyTorch research candidates."""
from __future__ import annotations

import time
import contextlib
import hashlib
import multiprocessing as mp
import os
import pickle
import sys
import threading
import traceback
try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None
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
    row_ids: Sequence[Any] | None = None


@dataclass(frozen=True)
class RunnerResult:
    experiment_id: str
    status: str
    run_dir: Path
    runtime_seconds: float
    output: CandidateOutput | None = None
    error: str | None = None
    resource_usage: Mapping[str, Any] = field(default_factory=dict)
    stage_id: str | None = None


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
        self._data_lock = threading.Lock()
        self._prepared_cache: PreparedData | None = None

    def run(
        self,
        *,
        experiment_id: str,
        hypothesis: str,
        config: Mapping[str, Any],
        candidate: CandidateCallable,
        timeout_seconds: float | None = None,
        stage_id: str | None = None,
    ) -> RunnerResult:
        run_dir = self.logger.store.run_dir(experiment_id, stage_id).resolve()
        self.logger.store.write_run_json(
            experiment_id,
            "plan.json",
            {"experiment_id": experiment_id, "hypothesis": hypothesis},
            stage_id=stage_id,
        )
        self.logger.store.write_run_json(experiment_id, "config.json", dict(config), stage_id=stage_id)
        self.logger.log_action(
            "candidate_created",
            experiment_id=experiment_id,
            details={"run_dir": str(run_dir), "stage_id": stage_id},
        )
        started = self.clock()
        protected_before = self._protected_fingerprints()
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
            output, child_error, child_status, resource_usage = self._run_isolated(
                candidate, prepared, config, run_dir, timeout_seconds
            )
            runtime_seconds = self.clock() - started
            protected_after = self._protected_fingerprints()
            changed_protected = sorted(
                path for path, digest in protected_after.items()
                if protected_before.get(path) != digest
            )
            if changed_protected:
                message = "candidate modified protected files: " + ", ".join(changed_protected)
                self._write_failure(experiment_id, run_dir, "protected_file_modified", message, runtime_seconds, stage_id)
                return RunnerResult(experiment_id, "failed", run_dir, runtime_seconds, error=message, resource_usage=resource_usage, stage_id=stage_id)
            if child_status != "completed" or output is None:
                message = child_error or f"isolated candidate status: {child_status}"
                self._write_failure(experiment_id, run_dir, child_status, message, runtime_seconds, stage_id)
                return RunnerResult(experiment_id, child_status, run_dir, runtime_seconds, error=message, resource_usage=resource_usage, stage_id=stage_id)
            alignment_error = self._validate_output_alignment(output, prepared)
            if alignment_error:
                self._write_failure(experiment_id, run_dir, "invalid_output", alignment_error, runtime_seconds, stage_id)
                return RunnerResult(experiment_id, "failed", run_dir, runtime_seconds, error=alignment_error, resource_usage=resource_usage, stage_id=stage_id)
            if timeout_seconds is not None and runtime_seconds > timeout_seconds:
                message = f"runtime {runtime_seconds:.3f}s exceeded budget {timeout_seconds:.3f}s"
                self._write_failure(experiment_id, run_dir, "timed_out", message, runtime_seconds, stage_id)
                return RunnerResult(experiment_id, "timed_out", run_dir, runtime_seconds, error=message, resource_usage=resource_usage, stage_id=stage_id)
            self.logger.store.write_run_json(
                experiment_id,
                "runner_result.json",
                {"status": "completed", "runtime_seconds": runtime_seconds, "resource_usage": dict(resource_usage), "metadata": dict(output.metadata)},
                stage_id=stage_id,
            )
            self.logger.log_action(
                "training_completed",
                experiment_id=experiment_id,
                details={"runtime_seconds": runtime_seconds, "resource_usage": dict(resource_usage)},
            )
            return RunnerResult(experiment_id, "completed", run_dir, runtime_seconds, output=output, resource_usage=resource_usage, stage_id=stage_id)
        except Exception as exc:  # Candidate errors must be recorded, not lost.
            runtime_seconds = self.clock() - started
            message = f"{type(exc).__name__}: {exc}"
            self._write_failure(experiment_id, run_dir, "failed", message, runtime_seconds, stage_id)
            return RunnerResult(experiment_id, "failed", run_dir, runtime_seconds, error=message, stage_id=stage_id)

    def _run_isolated(
        self,
        candidate: CandidateCallable,
        prepared: PreparedData,
        config: Mapping[str, Any],
        run_dir: Path,
        timeout_seconds: float | None,
    ) -> tuple[CandidateOutput | None, str | None, str, Mapping[str, Any]]:
        """Run candidate code outside the controller and enforce its deadline."""
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        parent_conn, child_conn = context.Pipe(duplex=False)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        def invoke() -> None:
            resource_before = _resource_snapshot()
            try:
                worker_threads = max(1, int(config.get("worker_threads", 1)))
                os.environ["OMP_NUM_THREADS"] = str(worker_threads)
                os.environ["MKL_NUM_THREADS"] = str(worker_threads)
                # Environment limits apply when a candidate imports torch.
                # If it was already imported by the production entrypoint,
                # also update its runtime pool without importing a heavy ML
                # dependency for lightweight candidates and tests.
                torch_module = sys.modules.get("torch")
                if torch_module is not None:
                    torch_module.set_num_threads(worker_threads)
                os.chdir(run_dir)
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        result = candidate(prepared, config, run_dir)
                    with (run_dir / "candidate_output.pkl").open("wb") as handle:
                        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    child_conn.send(("completed", None, None, _resource_delta(resource_before)))
            except BaseException as exc:
                try:
                    child_conn.send(("failed", None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", _resource_delta(resource_before)))
                except (BrokenPipeError, EOFError):
                    pass
            finally:
                child_conn.close()

        process = context.Process(target=invoke, name=f"research-candidate-{run_dir.name}")
        process.start()
        child_conn.close()
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        message: tuple[str, CandidateOutput | None, str | None, Mapping[str, Any]] | None = None
        while process.is_alive():
            if parent_conn.poll(0.1):
                message = parent_conn.recv()
                break
            if deadline is not None and time.monotonic() >= deadline:
                process.terminate()
                process.join(2.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(2.0)
                return None, f"candidate exceeded runtime budget of {timeout_seconds:.3f}s", "timed_out", {}
        process.join(2.0)
        if message is None and parent_conn.poll():
            message = parent_conn.recv()
        parent_conn.close()
        if message is None:
            return None, f"candidate exited with code {process.exitcode}", "failed", {}
        if message[0] == "completed":
            try:
                with (run_dir / "candidate_output.pkl").open("rb") as handle:
                    output = pickle.load(handle)
            except Exception as exc:
                return None, f"could not load isolated candidate output: {type(exc).__name__}: {exc}", "failed", message[3]
            return output, message[2], message[0], message[3]
        return None, message[2], message[0], message[3]

    @staticmethod
    def _protected_fingerprints() -> dict[str, str]:
        root = Path(__file__).resolve().parent.parent
        result = {}
        for name in ("baseline.py", "evaluate.py"):
            path = root / name
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    @staticmethod
    def _validate_output_alignment(output: CandidateOutput, prepared: PreparedData) -> str | None:
        """Ensure candidate predictions correspond to canonical validation rows."""
        rows = prepared.validation_rows
        if not rows or not hasattr(rows[0], "__len__") or len(rows[0]) < 7:
            return None  # Lightweight test doubles may use opaque synthetic rows.
        expected_users = [row[1] for row in rows]
        expected_labels = [int(row[6]) for row in rows]
        actual_users = list(output.user_ids)
        actual_labels = [int(value) for value in output.labels]
        if actual_users != expected_users:
            return "candidate user_ids do not exactly match canonical validation row order"
        if actual_labels != expected_labels:
            return "candidate labels do not exactly match canonical validation labels"
        if output.row_ids is not None and list(output.row_ids) != list(range(len(rows))):
            return "candidate row_ids do not exactly match canonical validation row order"
        return None

    def _load_prepared_data(self) -> PreparedData:
        with self._data_lock:
            if self._prepared_cache is not None:
                return self._prepared_cache
            splits = self.data_loader(str(self.contract.data_dir))
            try:
                self._prepared_cache = PreparedData(
                    train_rows=splits[self.contract.train_split],
                    validation_rows=splits[self.contract.validation_split],
                )
            except KeyError as exc:
                raise ValueError(f"data.py did not return required split: {exc.args[0]}") from exc
            return self._prepared_cache

    def _write_failure(
        self,
        experiment_id: str,
        run_dir: Path,
        status: str,
        message: str,
        runtime_seconds: float,
        stage_id: str | None = None,
    ) -> None:
        self.logger.store.write_run_json(
            experiment_id,
            "error.json",
            {"status": status, "error": message, "runtime_seconds": runtime_seconds},
            stage_id=stage_id,
        )
        self.logger.log_action(
            status,
            experiment_id=experiment_id,
            details={"error": message, "runtime_seconds": runtime_seconds},
        )


def _resource_snapshot() -> dict[str, float | None]:
    if resource is None:
        return {"cpu_user_seconds": None, "cpu_system_seconds": None, "peak_rss_bytes": None}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB; macOS reports bytes. This environment is Linux, while
    # the fallback remains explicitly unavailable on platforms without resource.
    rss = float(usage.ru_maxrss) * (1024 if os.name == "posix" else 1)
    return {"cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime, "peak_rss_bytes": rss}


def _resource_delta(before: Mapping[str, float | None]) -> dict[str, float | None]:
    after = _resource_snapshot()
    result: dict[str, float | None] = {}
    for key in ("cpu_user_seconds", "cpu_system_seconds"):
        if before.get(key) is None or after.get(key) is None:
            result[key] = None
        else:
            result[key] = float(after[key]) - float(before[key])
    result["peak_rss_bytes"] = after.get("peak_rss_bytes")
    result["gpu_available"] = False
    result["gpu_peak_memory_bytes"] = None
    return result
