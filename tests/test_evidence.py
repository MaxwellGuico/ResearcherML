import unittest

from research_agent.architecture import ReviewedArchitectureSpec
from research_agent.evidence import architecture_ablation_evidence, build_experiment_evidence


class EvidenceTests(unittest.TestCase):
    def test_architecture_ablation_evidence_limits_the_mechanism_claim(self):
        parent = ReviewedArchitectureSpec(
            interaction_paths=("embedding_mlp", "bi_interaction_mlp"), fusion="add",
            hidden_width=32, hidden_depth=2, dropout=0.1, cross_layers=2,
        )
        child = ReviewedArchitectureSpec(
            interaction_paths=("embedding_mlp",), fusion="add",
            hidden_width=32, hidden_depth=2, dropout=0.1, cross_layers=2,
        )

        evidence = architecture_ablation_evidence(
            parent_config={"architecture": parent.architecture_id},
            candidate_config={"architecture": child.architecture_id},
            parent_primary=0.603,
            candidate_primary=0.602,
        )

        self.assertEqual(evidence["removed_path"], "bi_interaction_mlp")
        self.assertEqual(evidence["retained_path"], "embedding_mlp")
        self.assertAlmostEqual(evidence["delta_from_frozen_parent"], -0.001)
        self.assertIn("does not prove", evidence["claim_boundary"])
    def test_evidence_groups_fidelity_diagnostics_and_seed_statistics(self):
        diagnostics = {
            "training": {"best_epoch": 2, "stop_reason": "early_stopping_patience"},
            "epoch_curve": [{"epoch": 1, "primary": 0.60}, {"epoch": 2, "primary": 0.61}],
            "score_distribution": {"finite_fraction": 1.0},
            "feature_coverage": {"feature_variant": "weekday"},
            "model": {"family": "fm", "trainable_parameters": 100},
        }
        stages = [
            {"stage_id": "low", "status": "accepted", "config": {"fidelity": "low", "seed": 0},
             "metrics": {"primary": 0.602}, "runner_metadata": {"diagnostics": diagnostics}, "runtime_seconds": 2.0},
            {"stage_id": "seed_1", "status": "completed", "config": {"fidelity": "full", "seed": 1},
             "metrics": {"primary": 0.603}, "runtime_seconds": 3.0},
            {"stage_id": "seed_2", "status": "completed", "config": {"fidelity": "full", "seed": 2},
             "metrics": {"primary": 0.601}, "runtime_seconds": 3.0},
        ]

        evidence = build_experiment_evidence(stages, baseline_primary=0.6, improvement_threshold=0.002)

        self.assertEqual(evidence["best_stage_id"], "seed_1")
        self.assertAlmostEqual(evidence["seed_confirmation"]["mean_primary"], 0.602)
        self.assertAlmostEqual(evidence["seed_confirmation"]["std_primary"], 0.001)
        self.assertEqual(evidence["model_diagnostics"][0]["training"]["best_epoch"], 2)

    def test_failures_are_classified_for_the_planner(self):
        evidence = build_experiment_evidence(
            [{"stage_id": "low", "status": "timed_out", "config": {}, "metrics": {},
              "error": "hard timeout after 60 seconds"}],
            baseline_primary=0.6,
            improvement_threshold=0.002,
        )

        self.assertEqual(evidence["failures"][0]["category"], "infrastructure_timeout")

    def test_complete_seed_confirmation_uses_robust_mean_as_decision_evidence(self):
        stages = [
            {
                "stage_id": "full" if seed == 0 else f"seed_{seed}",
                "status": "completed",
                "config": {"fidelity": "full", "seed": seed},
                "metrics": {"GAUC": value, "nDCG@5": value, "primary": value},
            }
            for seed, value in ((0, 0.70), (1, 0.55), (2, 0.55))
        ]

        evidence = build_experiment_evidence(
            stages, baseline_primary=0.61, improvement_threshold=0.002
        )

        self.assertAlmostEqual(evidence["best_stage_primary"], 0.70)
        self.assertAlmostEqual(evidence["best_primary"], 0.60)
        self.assertAlmostEqual(evidence["robust_primary"], 0.60)
        self.assertFalse(evidence["exceeds_incumbent"])
        self.assertTrue(evidence["seed_confirmation"]["complete"])
