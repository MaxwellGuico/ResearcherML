import tempfile
import unittest

import torch

from research_agent.models.torch_fm import TorchFM, run_torch_fm_candidate
from research_agent.runner import PreparedData


def row(user, video, label):
    return (20220408, str(user), str(video), "author", "1", 10_000.0, label)


class TorchFMTests(unittest.TestCase):
    def test_forward_shape(self):
        model = TorchFM(feature_dim=20, embedding_dim=4)
        self.assertEqual(model(torch.tensor([[1, 2], [3, 4]])).shape, (2,))

    def test_pointwise_and_pairwise_candidates_produce_validation_scores(self):
        prepared = PreparedData(
            train_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
            validation_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
        )
        with tempfile.TemporaryDirectory() as directory:
            for loss in ("pointwise", "pairwise"):
                output = run_torch_fm_candidate(
                    prepared,
                    {"loss": loss, "learning_rate": 0.01, "l2": 0.0, "epochs": 2, "batch_size": 2, "seed": 0},
                    __import__("pathlib").Path(directory),
                )
                self.assertEqual(len(output.scores), 4)
                self.assertTrue((__import__("pathlib").Path(directory) / "epoch_metrics.csv").exists())
                self.assertTrue((__import__("pathlib").Path(directory) / "checkpoint.pt").exists())
