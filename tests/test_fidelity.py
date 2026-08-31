import unittest

from research_agent.fidelity import FidelityManager
from research_agent.planner import EvidencePlanner
from research_agent.search import SearchController
from research_agent.state import ResearchState


class FidelityTests(unittest.TestCase):
    def test_promotion_requires_validation_evidence_and_uses_full_budget(self):
        direction = EvidencePlanner(seed=0).propose([], ResearchState())
        proposal = SearchController(seed=0).propose_trial(direction, ResearchState(), [])
        manager = FidelityManager()

        self.assertTrue(manager.should_promote({"primary": 0.6}, 0.6015))
        self.assertFalse(manager.should_promote({"primary": 0.5}, 0.6015))
        promoted = manager.promote(proposal, direction, experiment_id="exp_002")
        self.assertEqual(promoted.experiment_id, proposal.experiment_id)
        self.assertEqual(promoted.parent_experiment_id, proposal.experiment_id)
        self.assertEqual(promoted.config["fidelity"], "medium")
        self.assertEqual(promoted.config["epochs"], direction.evaluation_budget["medium_epochs"])
        self.assertEqual(promoted.changed_factors, proposal.changed_factors)

        promoted_again = manager.promote(promoted, direction, experiment_id="exp_003")
        self.assertEqual(promoted_again.experiment_id, proposal.experiment_id)
        self.assertEqual(promoted_again.parent_experiment_id, proposal.experiment_id)
        self.assertEqual(promoted_again.config["fidelity"], "full")
        self.assertEqual(promoted_again.config["epochs"], direction.evaluation_budget["full_epochs"])

    def test_fidelity_budgets_are_explicit(self):
        direction = EvidencePlanner(seed=0).propose([], ResearchState())
        manager = FidelityManager()

        self.assertEqual(manager.budget(direction, "low"), 4)
        self.assertEqual(manager.budget(direction, "medium"), 8)
        self.assertEqual(manager.budget(direction, "full"), 12)
