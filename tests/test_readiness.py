from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.readiness import audit_readiness
from research_agent.store import ArtifactStore


class ReadinessTests(unittest.TestCase):
    def test_complete_evidence_bundle_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.write_root_json("state.json", {
                "current_best_experiment_id": "exp_001", "current_best_primary": 0.61,
                "completed_iterations": 1, "consecutive_non_improvements": 0,
                "plateau_restarts": 0, "active_experiment_id": None, "stop_reason": None,
            })
            submission = Path(directory) / "submission.csv"
            submission.write_text("row_id,user_id,video_id,score\n0,u,v,0.5\n", encoding="utf-8")
            patch_path = store.write_patch("exp_001", "learning_rate")
            store.append_event({"action": "benchmark_verified"})
            store.append_event({"action": "research_run_started"})
            store.append_event({"action": "research_run_finished"})
            store.append_iteration({
                "experiment_id": "exp_001", "parent_experiment_id": "baseline",
                "hypothesis": "A controlled optimization hypothesis.", "rationale": "Evidence based.",
                "config": {"learning_rate": 0.0005}, "changed_factors": ["learning_rate"],
                "direction_id": "pointwise_fm_optimization", "metrics": {"primary": 0.61},
                "runtime_seconds": 1.0, "resource_usage": {}, "decision": "accepted",
                "error": None, "recovery": None, "safety": {"passed": True},
                "code_diff_path": str(patch_path), "semantic_review": {"approved": True},
                "diagnostic_evidence": {},
            })
            store.append_stage({"experiment_id": "exp_001", "stage_id": "low"})
            store.write_root_json("final_summary.json", {"submission_checked": True})

            result = audit_readiness(store, submission)

            self.assertTrue(result.passed)
            self.assertEqual(result.issues, ())
            self.assertTrue(store.read_root_json("readiness.json")["passed"])

    def test_missing_submission_and_patch_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.write_root_json("state.json", {
                "current_best_experiment_id": "baseline", "current_best_primary": 0.6015,
                "completed_iterations": 0, "consecutive_non_improvements": 0,
                "plateau_restarts": 0, "active_experiment_id": None, "stop_reason": None,
            })
            store.write_root_json("final_summary.json", {"submission_checked": False})

            result = audit_readiness(store, Path(directory) / "missing.csv")

            self.assertFalse(result.passed)
            self.assertIn("submission_exists_and_is_nonempty", result.issues)
            self.assertIn("submission_schema_checked", result.issues)

    def test_logged_pre_run_semantic_rejection_needs_no_execution_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.write_root_json("state.json", {
                "current_best_experiment_id": "baseline", "current_best_primary": 0.6015,
                "completed_iterations": 1, "consecutive_non_improvements": 1,
                "plateau_restarts": 0, "active_experiment_id": None, "stop_reason": None,
            })
            submission = Path(directory) / "submission.csv"
            submission.write_text("row_id,user_id,video_id,score\n0,u,v,0.5\n", encoding="utf-8")
            patch_path = store.write_patch("exp_001", "learning_rate")
            for action in ("benchmark_verified", "research_run_started", "research_run_finished"):
                store.append_event({"action": action})
            store.append_event({
                "action": "semantic_rejection_recorded", "experiment_id": "exp_001",
                "details": {"semantic_trace": {"verdict": "rejected"}},
            })
            store.append_iteration({
                "experiment_id": "exp_001", "parent_experiment_id": "baseline",
                "hypothesis": "A duplicate controlled hypothesis.", "rationale": "Critic test.",
                "config": {"learning_rate": 0.0005}, "changed_factors": ["learning_rate"],
                "direction_id": "pointwise_fm_optimization", "metrics": {},
                "runtime_seconds": 0.0, "resource_usage": {}, "decision": "rejected",
                "error": "duplicate", "recovery": "baseline", "safety": {"passed": False},
                "code_diff_path": str(patch_path),
            })
            store.write_root_json("final_summary.json", {"submission_checked": True})

            result = audit_readiness(store, submission)

            self.assertTrue(result.passed)
