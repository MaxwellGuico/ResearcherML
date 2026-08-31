import tempfile
import threading
import time
import unittest
from pathlib import Path

from research_agent.logger import ResearchLogger
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.store import ArtifactStore


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)
        self.logger = ResearchLogger(self.store)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _loader(_data_dir):
        return {
            "train": [("train",)],
            "valid": [("valid",)],
            "test": [("test",)],
        }

    def test_runner_isolates_artifacts_and_hides_test_rows(self):
        def candidate(data, config, run_dir):
            assert data.train_rows == [("train",)]
            assert data.validation_rows == [("valid",)]
            assert not hasattr(data, "test_rows")
            (run_dir / "checkpoint.note").write_text("candidate artifact", encoding="utf-8")
            return CandidateOutput(["u"], [1], [0.9], {"framework": "pytorch"})

        result = ExperimentRunner(self.logger, data_loader=self._loader).run(
            experiment_id="exp_001",
            hypothesis="A safe candidate should run in isolation.",
            config={"seed": 0},
            candidate=candidate,
        )

        self.assertEqual(result.status, "completed")
        self.assertTrue((result.run_dir / "config.json").exists())
        self.assertTrue((result.run_dir / "checkpoint.note").exists())

    def test_runner_records_candidate_failure(self):
        def broken_candidate(_data, _config, _run_dir):
            raise RuntimeError("training exploded")

        result = ExperimentRunner(self.logger, data_loader=self._loader).run(
            experiment_id="exp_002",
            hypothesis="Failures must preserve evidence.",
            config={},
            candidate=broken_candidate,
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("training exploded", result.error)
        self.assertTrue((result.run_dir / "error.json").exists())
        self.assertEqual(self.store.read_events()[-1]["action"], "failed")

    def test_runner_marks_budget_overrun(self):
        times = iter((0.0, 3.0))

        def candidate(_data, _config, _run_dir):
            return CandidateOutput(["u"], [1], [0.9])

        result = ExperimentRunner(
            self.logger,
            data_loader=self._loader,
            clock=lambda: next(times),
        ).run(
            experiment_id="exp_003",
            hypothesis="Budget overruns are visible.",
            config={},
            candidate=candidate,
            timeout_seconds=2.0,
        )

        self.assertEqual(result.status, "timed_out")
        self.assertIn("exceeded budget", result.error)
        self.assertTrue((result.run_dir / "error.json").exists())

    def test_runner_terminates_isolated_candidate_on_hard_timeout(self):
        def hanging_candidate(_data, _config, _run_dir):
            time.sleep(2)

        result = ExperimentRunner(self.logger, data_loader=self._loader).run(
            experiment_id="exp_004",
            hypothesis="A hung candidate must be terminated.",
            config={},
            candidate=hanging_candidate,
            timeout_seconds=0.1,
        )

        self.assertEqual(result.status, "timed_out")
        self.assertIn("exceeded runtime budget", result.error)

    def test_runner_rejects_misaligned_canonical_validation_output(self):
        def loader(_data_dir):
            rows = [(20220422, "u1", "v1", "a", "t", 1.0, 1), (20220422, "u2", "v2", "a", "t", 1.0, 0)]
            return {"train": rows, "valid": rows, "test": rows}

        def misaligned_candidate(_data, _config, _run_dir):
            return CandidateOutput(["u2", "u1"], [0, 1], [0.9, 0.1])

        result = ExperimentRunner(self.logger, data_loader=loader).run(
            experiment_id="exp_005",
            hypothesis="Validation rows must remain aligned.",
            config={},
            candidate=misaligned_candidate,
            timeout_seconds=5,
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("canonical validation row order", result.error)

    def test_concurrent_runs_load_canonical_data_once(self):
        load_count = 0
        load_lock = threading.Lock()

        def counted_loader(_data_dir):
            nonlocal load_count
            with load_lock:
                load_count += 1
            return self._loader(_data_dir)

        def candidate(_data, _config, _run_dir):
            return CandidateOutput(["u"], [1], [0.9])

        runner = ExperimentRunner(self.logger, data_loader=counted_loader)
        results = []

        def invoke(experiment_id):
            results.append(runner.run(
                experiment_id=experiment_id,
                hypothesis="Concurrent workers should share immutable prepared data.",
                config={"worker_threads": 1},
                candidate=candidate,
            ))

        threads = [
            threading.Thread(target=invoke, args=("exp_006",)),
            threading.Thread(target=invoke, args=("exp_007",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(load_count, 1)
        self.assertEqual({result.status for result in results}, {"completed"})
