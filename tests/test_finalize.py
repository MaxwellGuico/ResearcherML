from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from research_agent.contracts import BenchmarkContract
from research_agent.finalize import _selected_iteration, _write_selected_torch_submission
from research_agent.models.torch_fm import build_candidate_model, build_model
from research_agent.store import ArtifactStore


class FinalizationTests(unittest.TestCase):
    def test_baseline_is_selected_when_no_agent_candidate_was_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            self.assertIsNone(_selected_iteration(store, "baseline"))

    def test_missing_selected_experiment_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                _selected_iteration(store, "exp_999")

    def test_selected_architecture_is_reconstructed_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            model = build_model("deepfm", feature_dim=10, field_count=2, embedding_dim=4)
            torch.save({
                "model_state": model.state_dict(),
                "feature_dim": 10,
                "config": {"architecture": "deepfm", "embedding_dim": 4, "feature_variant": "baseline"},
            }, checkpoint_path)
            encoded = {
                "test": (
                    np.asarray([[1, 2], [3, 4]], dtype=np.int64),
                    np.asarray([1, 0], dtype=np.int64),
                    np.asarray(["u", "u"]),
                )
            }
            metric_result = unittest.mock.Mock()
            metric_result.as_dict.return_value = {"GAUC": 1.0, "nDCG@5": 1.0, "primary": 1.0}
            record = {"runner_metadata": {"checkpoint_path": str(checkpoint_path)}}
            contract = BenchmarkContract(data_dir=Path(directory))

            with (
                patch("research_agent.finalize.load", return_value={"test": [("row",), ("row",)]}),
                patch("research_agent.finalize.encode_candidate", return_value=(encoded, 10)),
                patch("research_agent.finalize.write_submission"),
                patch("research_agent.finalize.evaluate_predictions", return_value=metric_result),
            ):
                metrics = _write_selected_torch_submission(
                    Path(directory) / "submission.csv", record, contract
                )

            self.assertEqual(metrics["primary"], 1.0)

    def test_multitask_checkpoint_reconstructs_training_time_auxiliary_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            config = {
                "architecture": "deepfm", "embedding_dim": 4,
                "feature_variant": "baseline", "training_objective": "multitask_click_w0.1",
            }
            model = build_candidate_model(config, feature_dim=10, field_count=2, embedding_dim=4)
            torch.save({
                "model_state": model.state_dict(), "feature_dim": 10, "config": config,
            }, checkpoint_path)
            encoded = {"test": (
                np.asarray([[1, 2], [3, 4]], dtype=np.int64),
                np.asarray([1, 0], dtype=np.int64), np.asarray(["u", "u"]),
            )}
            metric_result = unittest.mock.Mock()
            metric_result.as_dict.return_value = {"GAUC": 1.0, "nDCG@5": 1.0, "primary": 1.0}
            record = {"runner_metadata": {"checkpoint_path": str(checkpoint_path)}}
            contract = BenchmarkContract(data_dir=Path(directory))

            with (
                patch("research_agent.finalize.load", return_value={"test": [("row",), ("row",)]}),
                patch("research_agent.finalize.encode_candidate", return_value=(encoded, 10)),
                patch("research_agent.finalize.write_submission"),
                patch("research_agent.finalize.evaluate_predictions", return_value=metric_result),
            ):
                metrics = _write_selected_torch_submission(
                    Path(directory) / "submission.csv", record, contract
                )

            self.assertEqual(metrics["primary"], 1.0)
