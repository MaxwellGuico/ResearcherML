import tempfile
import unittest

from research_agent.critic_memory import refresh_critic_memory
from research_agent.logger import ResearchLogger
from research_agent.store import ArtifactStore


class CriticMemoryTests(unittest.TestCase):
    def test_feedback_is_grouped_into_planner_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            logger.record_critic_feedback({
                "experiment_id": "exp_001", "phase": "post_execution",
                "direction_id": "fm_architecture", "disposition": "continue_supported_lineage",
                "lesson": "Preserve the accepted nonlinear path.", "delta_primary": 0.002,
            })
            logger.record_critic_feedback({
                "experiment_id": "exp_002", "phase": "post_execution",
                "direction_id": "pairwise_fm_ranking", "disposition": "record_valid_negative_and_branch",
                "lesson": "Do not repeat this pairwise configuration.",
                "do_not_repeat_exact_configuration": True,
            })

            memory = refresh_critic_memory(store)

            self.assertEqual(memory["feedback_count"], 2)
            self.assertEqual(memory["supported_lineages"][0]["experiment_id"], "exp_001")
            self.assertEqual(memory["valid_negative_lessons"][0]["experiment_id"], "exp_002")
            self.assertTrue((store.root / "critic_memory.json").exists())

    def test_historical_iterations_are_backfilled_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            ResearchLogger(store).record_iteration({
                "experiment_id": "exp_001", "hypothesis": "A measured historical hypothesis.",
                "rationale": "Historical evidence", "config": {"x": 1},
                "direction_id": "pointwise_fm_optimization", "decision": "rejected",
                "metrics": {"primary": 0.59}, "delta_primary": -0.01,
            })

            refresh_critic_memory(store)
            refresh_critic_memory(store)

            self.assertEqual(len(store.read_critic_feedback()), 1)
            self.assertEqual(
                store.read_critic_feedback()[0]["disposition"],
                "record_valid_negative_and_branch",
            )


if __name__ == "__main__":
    unittest.main()
