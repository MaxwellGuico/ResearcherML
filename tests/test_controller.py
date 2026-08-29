import tempfile
import unittest

from research_agent.controller import ExperimentController
from research_agent.logger import ResearchLogger
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.safety import ExperimentProposal, SafetyValidator
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


def loader(_data_dir):
    return {"train": [("train",)], "valid": [("valid",)], "test": [("test",)]}


def strong_candidate(_data, _config, _run_dir):
    return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])


def broken_candidate(_data, _config, _run_dir):
    raise RuntimeError("intentional runner failure")


def proposal(experiment_id, **changes):
    values = {
        "experiment_id": experiment_id,
        "hypothesis": "A single controlled change may improve ranking.",
        "rationale": "Use validation evidence only.",
        "config": {"candidate": experiment_id},
        "changed_factors": ("candidate",),
        "runtime_budget_seconds": 60,
    }
    values.update(changes)
    return ExperimentProposal(**values)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)
        self.logger = ResearchLogger(self.store)
        self.runner = ExperimentRunner(self.logger, data_loader=loader)
        self.validator = SafetyValidator(max_runtime_seconds=60)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_improving_candidate_is_accepted_and_updates_state(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
        )

        result = controller.run_iteration(proposal("exp_001"), strong_candidate)

        self.assertEqual(result.decision, "accepted")
        self.assertEqual(controller.state.current_best_experiment_id, "exp_001")
        self.assertEqual(controller.state.current_best_primary, 1.0)
        self.assertTrue((self.store.root / "state.json").exists())
        record = self.store.read_iterations()[0]
        self.assertEqual(record["decision"], "accepted")
        self.assertEqual(record["parent_experiment_id"], "baseline")

    def test_controller_recovers_state_and_history_after_restart(self):
        first_controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
        )
        first_controller.run_iteration(proposal("exp_001"), strong_candidate)

        resumed = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
        )
        duplicate = resumed.run_iteration(
            proposal("exp_002", config={"candidate": "exp_001"}),
            strong_candidate,
        )

        self.assertEqual(resumed.state.current_best_experiment_id, "exp_001")
        self.assertEqual(duplicate.decision, "rejected")
        self.assertIn("duplicates", duplicate.error)

    def test_non_improvement_is_rejected_and_preserves_best_pointer(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_experiment_id="baseline", current_best_primary=1.0),
        )

        result = controller.run_iteration(proposal("exp_002"), strong_candidate)

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        record = self.store.read_iterations()[0]
        self.assertIn("restored accepted candidate pointer", record["recovery"])

    def test_safety_rejection_never_invokes_runner(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
        )
        unsafe = proposal("exp_003", changed_factors=("a", "b"))

        result = controller.run_iteration(unsafe, strong_candidate)

        self.assertEqual(result.decision, "rejected")
        self.assertFalse((self.store.runs_dir / "exp_003").exists())
        self.assertIn("exactly one", result.error)

    def test_failure_recovers_and_stops_after_three_non_improvements(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=1.0),
        )

        first = controller.run_iteration(proposal("exp_004"), broken_candidate)
        controller.run_iteration(proposal("exp_005"), strong_candidate)
        third = controller.run_iteration(proposal("exp_006"), strong_candidate)

        self.assertEqual(first.decision, "failed")
        self.assertEqual(third.decision, "rejected")
        self.assertTrue(controller.state.stopped)
        self.assertIn("3 consecutive", controller.state.stop_reason)
        skipped = controller.run_iteration(proposal("exp_007"), strong_candidate)
        self.assertEqual(skipped.decision, "skipped")
