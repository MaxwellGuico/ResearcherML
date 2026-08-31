from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.research_coverage import build_research_coverage
from research_agent.store import ArtifactStore


def iteration(experiment_id: str, architecture: str, primary: float, *, decision: str = "rejected"):
    gauc = primary + 0.05
    ndcg = primary - 0.05
    return {
        "experiment_id": experiment_id,
        "decision": decision,
        "config": {
            "architecture": architecture,
            "loss": "pointwise",
            "feature_variant": "baseline",
        },
        "metrics": {"GAUC": gauc, "nDCG@5": ndcg, "primary": primary},
        "semantic_review": {"approved": True},
    }


class ResearchCoverageTests(unittest.TestCase):
    def test_coverage_combines_current_tree_evidence_with_valid_sibling_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            current = ArtifactStore(workspace / "runs_current")
            prior = ArtifactStore(workspace / "runs_prior")
            prior.append_iteration(iteration("exp_prior", "deepfm", 0.604, decision="accepted"))
            history = [
                iteration(
                    "exp_current",
                    "composed:v1:embedding_mlp:add:w32:d2:p0.1:c2",
                    0.603,
                    decision="accepted",
                )
            ]

            coverage = build_research_coverage(current, history)

            architectures = {item["mechanism"]: item for item in coverage["architectures"]}
            self.assertEqual(architectures["deepfm"]["status"], "accepted")
            self.assertEqual(architectures["deepfm"]["best_source"], "runs_prior")
            self.assertEqual(architectures["embedding_mlp"]["status"], "accepted")
            self.assertEqual(architectures["cross_network"]["status"], "untested")
            self.assertEqual(architectures["cross_network"]["isolated_experiment_count"], 0)
            objectives = {item["mechanism"]: item for item in coverage["objectives"]}
            self.assertEqual(objectives["multi_task_combined_loss"]["status"], "untested")
            self.assertTrue((current.root / "research_coverage.json").is_file())

    def test_semantically_rejected_cross_run_record_is_not_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            current = ArtifactStore(workspace / "runs_current")
            prior = ArtifactStore(workspace / "runs_prior")
            invalid = iteration("exp_bad", "deepfm", 0.7, decision="accepted")
            invalid["semantic_review"] = {"approved": False}
            prior.append_iteration(invalid)

            coverage = build_research_coverage(current, [])

            architectures = {item["mechanism"]: item for item in coverage["architectures"]}
            self.assertEqual(architectures["deepfm"]["status"], "untested")
            self.assertEqual(coverage["validated_configuration_count"], 1)

    def test_combined_acceptance_does_not_count_as_isolated_architecture_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "runs_current")
            combined = iteration(
                "exp_combo",
                "composed:v1:embedding_mlp+bi_interaction_mlp:add:w32:d2:p0.1:c2",
                0.603,
                decision="accepted",
            )

            coverage = build_research_coverage(store, [combined])

            architectures = {item["mechanism"]: item for item in coverage["architectures"]}
            self.assertEqual(architectures["bi_interaction_mlp"]["status"], "present_in_accepted")
            self.assertEqual(architectures["bi_interaction_mlp"]["evidence_scope"], "combined")
            self.assertEqual(architectures["bi_interaction_mlp"]["isolated_experiment_count"], 0)


if __name__ == "__main__":
    unittest.main()
