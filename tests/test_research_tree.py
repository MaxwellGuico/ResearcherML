from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from research_agent.logger import ResearchLogger
from research_agent.planner import ResearchDirection
from research_agent.research_tree import ResearchTree, hypothesis_id
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


class ResearchTreeTests(unittest.TestCase):
    def test_tree_preserves_deferred_hypotheses_and_incumbent_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            tree = ResearchTree(logger)
            selected_hypothesis = "A lower learning rate may stabilize sparse FM optimization."
            deferred_hypothesis = "A cross network may improve explicit conditional interactions."
            metadata = {
                "planner": {
                    "response_id": "resp_1",
                    "slate": {
                        "candidates": [
                            {
                                "candidate_id": "h1", "domain": "training_optimization",
                                "hypothesis": selected_hypothesis,
                                "expected_mechanism": "Smaller updates reduce sparse optimizer overshoot.",
                                "required_capabilities": ["learning-rate control"],
                                "lineage_parent_id": "baseline", "lineage_action": "refine",
                                "evidence_reference": "baseline primary 0.6015",
                            },
                            {
                                "candidate_id": "h2", "domain": "model_architecture",
                                "hypothesis": deferred_hypothesis,
                                "expected_mechanism": "Explicit crosses model conditional field effects.",
                                "required_capabilities": ["cross network"],
                                "lineage_parent_id": "baseline", "lineage_action": "branch_new",
                                "evidence_reference": "baseline architecture diagnostics",
                            },
                        ],
                    },
                },
                "implementer": {
                    "plan": {
                        "selected_candidate_id": "h1",
                        "deferred_candidate_ids": ["h2"],
                    },
                },
            }
            tree.record_planner_batch(metadata, incumbent_experiment_id="baseline")
            direction = ResearchDirection(
                direction_id="pointwise_fm_optimization",
                hypothesis=selected_hypothesis,
                rationale="The controlled optimizer change tests convergence stability.",
                search_space={"learning_rate": [0.0005]},
                success_evidence="Validation primary exceeds the baseline.",
                evaluation_budget={"low_epochs": 4}, strategy="exploration",
                selected_candidate_id="h1",
            )
            tree.bind_experiment(direction, SimpleNamespace(
                experiment_id="exp_001", parent_experiment_id="baseline",
                search_region_id="region_01", search_strategy="exploration",
                portfolio_role="incumbent_exploit",
            ))
            logger.record_iteration({
                "experiment_id": "exp_001", "parent_experiment_id": "baseline",
                "hypothesis": selected_hypothesis, "rationale": direction.rationale,
                "config": {"learning_rate": 0.0005}, "changed_factors": ["learning_rate"],
                "direction_id": direction.direction_id, "search_region_id": "region_01",
                "search_strategy": "exploration", "portfolio_role": "incumbent_exploit",
                "metrics": {"primary": 0.603}, "delta_primary": 0.0015,
                "decision": "accepted",
            })
            state = ResearchState(current_best_experiment_id="exp_001", current_best_primary=0.603)
            tree.record_outcome("exp_001", incumbent_experiment_id="exp_001")

            snapshot = tree.refresh(state)

            self.assertEqual(snapshot["incumbent"]["ancestry"], ["baseline", "exp_001"])
            self.assertEqual(snapshot["experiments"][0]["hypothesis_id"], hypothesis_id(selected_hypothesis))
            self.assertEqual(snapshot["continuation_candidates"][0]["reason"], "continue_accepted_lineage")
            self.assertEqual(snapshot["deferred_hypotheses"][0]["hypothesis"], deferred_hypothesis)
            self.assertEqual(snapshot["deferred_hypotheses"][0]["lineage_action"], "branch_new")
            self.assertTrue(store.research_tree_events_path.exists())
            self.assertTrue((store.root / "research_tree.json").exists())
            diagram = (store.root / "research_tree.md").read_text(encoding="utf-8")
            self.assertIn("flowchart TD", diagram)
            self.assertIn("baseline --> exp_001", diagram)
            self.assertIn(hypothesis_id(deferred_hypothesis), diagram)
            self.assertIn("class exp_001 incumbent", diagram)

    def test_rejected_measured_branch_remains_a_revisit_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            logger.record_iteration({
                "experiment_id": "exp_001", "parent_experiment_id": "baseline",
                "hypothesis": "Pairwise loss may improve within-user ranking quality.",
                "rationale": "Ranking supervision directly compares exposed items.",
                "config": {"loss": "pairwise"}, "changed_factors": ["loss"],
                "direction_id": "pairwise_fm_ranking", "metrics": {"primary": 0.6009},
                "delta_primary": -0.0006, "decision": "rejected",
            })

            context = ResearchTree(logger).planner_context(
                ResearchState(current_best_experiment_id="baseline", current_best_primary=0.6015)
            )

            revisit = next(
                item for item in context["continuation_candidates"]
                if item["reason"] == "revisit_near_incumbent_branch"
            )
            self.assertEqual(revisit["source_experiment_id"], "exp_001")
            self.assertAlmostEqual(revisit["gap_to_incumbent"], 0.0006)

    def test_existing_llm_slate_is_backfilled_without_fabricated_parent_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            hypothesis = "A historical deferred cross feature may improve conditional interactions."
            logger.log_action("llm_hypothesis_generated", details={
                "ideator": {
                    "response_id": "historical_response",
                    "slate": {"candidates": [{
                        "candidate_id": "h3", "domain": "feature_data", "hypothesis": hypothesis,
                        "expected_mechanism": "A conditional cross exposes interactions hidden from additive fields.",
                        "required_capabilities": ["feature cross"],
                    }]},
                },
                "specialist": {"advice": {
                    "selected_candidate_id": "none", "deferred_candidate_ids": ["h3"],
                }},
            })

            snapshot = ResearchTree(logger).refresh(ResearchState())

            self.assertEqual(snapshot["deferred_hypotheses"][0]["hypothesis"], hypothesis)
            event = store.read_research_tree_events()[0]
            self.assertEqual(event["parent_experiment_id"], "unknown_historical_incumbent")


if __name__ == "__main__":
    unittest.main()
