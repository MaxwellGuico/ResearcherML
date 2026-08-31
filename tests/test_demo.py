import tempfile
import unittest

from research_agent.controller import ExperimentController
from research_agent.demo import inject_failure_recovery_demo
from research_agent.logger import ResearchLogger
from research_agent.runner import ExperimentRunner
from research_agent.safety import SafetyValidator
from research_agent.store import ArtifactStore


def loader(_data_dir):
    return {"train": [("train",)], "valid": [("valid",)], "test": [("test",)]}


class DemoWorkflowTests(unittest.TestCase):
    def test_intentional_failure_is_contained_logged_and_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=SafetyValidator(max_runtime_seconds=60),
                max_iterations=2,
            )

            result = inject_failure_recovery_demo(controller, logger)

            self.assertEqual(result.decision, "failed")
            self.assertEqual(controller.state.current_best_experiment_id, "baseline")
            self.assertIsNone(controller.state.active_experiment_id)
            self.assertIn("intentional_demo_failure", result.error)
            actions = [event["action"] for event in store.read_events()]
            self.assertIn("demo_failure_injected", actions)
            self.assertIn("accepted_candidate_restored", actions)
