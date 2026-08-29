import unittest

from research_agent.planner import EvidencePlanner
from research_agent.search import SearchController
from research_agent.state import ResearchState


class SearchControllerTests(unittest.TestCase):
    def setUp(self):
        self.planner = EvidencePlanner(seed=0)
        self.search = SearchController(seed=0)

    def test_search_controller_selects_one_factor_not_a_fixed_template(self):
        direction = self.planner.propose([], ResearchState())
        proposal = self.search.propose_trial(direction, ResearchState(), [])

        self.assertEqual(len(proposal.changed_factors), 1)
        self.assertEqual(proposal.research_direction_id, direction.direction_id)
        self.assertEqual(proposal.config["fidelity"], "low")
        self.assertNotEqual(proposal.config[proposal.changed_factors[0]], self.search.BASELINE_CONFIG[proposal.changed_factors[0]])

    def test_pairwise_direction_changes_only_loss_conceptually(self):
        history = [{"direction_id": "pointwise_fm_optimization", "decision": "rejected"}]
        direction = self.planner.propose(history, ResearchState(completed_iterations=1))
        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), history)

        self.assertEqual(direction.direction_id, "pairwise_fm_ranking")
        self.assertEqual(proposal.changed_factors, ("loss",))
        self.assertEqual(proposal.config["loss"], "pairwise")

    def test_new_trial_ids_and_values_avoid_history_when_possible(self):
        direction = self.planner.propose([], ResearchState())
        history = [
            {
                "experiment_id": "exp_001",
                "direction_id": direction.direction_id,
                "changed_factors": ["learning_rate"],
                "config": {"learning_rate": 0.0005},
            }
        ]
        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), history)

        self.assertEqual(proposal.experiment_id, "exp_002")
