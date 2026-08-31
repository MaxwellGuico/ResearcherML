import tempfile
import unittest

import torch

from data import auxiliary_labels, encode_candidate
from research_agent.architecture import ReviewedArchitectureSpec, parse_architecture_id
from research_agent.models.torch_fm import (
    MultiTaskSharedBackbone,
    TorchFM,
    build_candidate_model,
    build_model,
    run_torch_fm_candidate,
)
from research_agent.runner import PreparedData


def row(user, video, label):
    return (20220408, str(user), str(video), "author", "1", 10_000.0, label)


def multitask_row(user, video, label, click):
    return (*row(user, video, label), click)


class TorchFMTests(unittest.TestCase):
    def test_forward_shape(self):
        model = TorchFM(feature_dim=20, embedding_dim=4)
        self.assertEqual(model(torch.tensor([[1, 2], [3, 4]])).shape, (2,))

    def test_reviewed_architecture_specs_compile_to_distinct_models(self):
        features = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        baseline = build_model("fm", feature_dim=20, field_count=5, embedding_dim=4)
        for architecture in ("deepfm", "nfm_residual"):
            model = build_model(architecture, feature_dim=20, field_count=5, embedding_dim=4)
            self.assertEqual(model(features).shape, (2,))
            self.assertGreater(
                sum(parameter.numel() for parameter in model.parameters()),
                sum(parameter.numel() for parameter in baseline.parameters()),
            )

    def test_composed_architecture_round_trips_and_compiles_all_reviewed_operators(self):
        features = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        for paths in (
            ("embedding_mlp",),
            ("bi_interaction_mlp",),
            ("cross_network",),
            ("embedding_mlp", "cross_network"),
        ):
            spec = ReviewedArchitectureSpec(
                interaction_paths=paths,
                fusion="learned_gate" if len(paths) == 2 else "add",
                hidden_width=32,
                hidden_depth=2,
                dropout=0.1,
                cross_layers=2,
            )
            self.assertEqual(parse_architecture_id(spec.architecture_id), spec)
            model = build_model(spec.architecture_id, feature_dim=20, field_count=5, embedding_dim=4)
            self.assertEqual(model(features).shape, (2,))
            self.assertTrue(torch.isfinite(model(features)).all())

    def test_composed_architecture_rejects_unreviewed_or_unbounded_structures(self):
        with self.assertRaisesRegex(ValueError, "unreviewed architecture operators"):
            ReviewedArchitectureSpec(
                interaction_paths=("transformer",), fusion="add", hidden_width=32,
                hidden_depth=2, dropout=0.1, cross_layers=2,
            )
        with self.assertRaisesRegex(ValueError, "hidden_width"):
            ReviewedArchitectureSpec(
                interaction_paths=("embedding_mlp",), fusion="add", hidden_width=1024,
                hidden_depth=2, dropout=0.1, cross_layers=2,
            )

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
                diagnostics = output.metadata["diagnostics"]
                self.assertEqual(diagnostics["score_distribution"]["finite_fraction"], 1.0)
                self.assertEqual(diagnostics["training"]["epochs_budgeted"], 2)
                self.assertIn("validation_logloss", diagnostics["epoch_curve"][0])
                self.assertEqual(diagnostics["model"]["field_count"], 5)
                self.assertTrue(diagnostics["user_segments"])
                stratified = diagnostics["stratified_validation"]
                self.assertEqual(stratified["boundary_source"], "training_only")
                self.assertFalse(stratified["test_data_used"])
                self.assertEqual(len(stratified["training_vocabulary_sha256_by_field"]), 5)
                self.assertIn("all_fields_seen", stratified["feature_coverage"])
                self.assertTrue((__import__("pathlib").Path(directory) / "epoch_metrics.csv").exists())
                self.assertTrue((__import__("pathlib").Path(directory) / "checkpoint.pt").exists())

    def test_multitask_click_uses_shared_embeddings_and_records_auxiliary_diagnostics(self):
        rows = [
            multitask_row("u1", "v1", 1, 1), multitask_row("u1", "v2", 0, 0),
            multitask_row("u2", "v1", 0, 1), multitask_row("u2", "v2", 1, 0),
        ]
        prepared = PreparedData(train_rows=rows, validation_rows=rows)
        config = {
            "loss": "pointwise", "training_objective": "multitask_click_w0.1",
            "learning_rate": 0.01, "l2": 0.0, "epochs": 2, "batch_size": 2,
            "seed": 0, "architecture": "composed:v1:embedding_mlp:add:w32:d2:p0.1:c2",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = run_torch_fm_candidate(
                prepared, config, __import__("pathlib").Path(directory)
            )

        training = output.metadata["diagnostics"]["training"]
        self.assertEqual(training["auxiliary_task"], "is_click")
        self.assertEqual(training["auxiliary_loss_weight"], 0.1)
        self.assertEqual(training["auxiliary_train_positive_rate"], 0.5)
        self.assertIsNotNone(output.metadata["diagnostics"]["epoch_curve"][0]["auxiliary_train_loss"])
        self.assertTrue(output.metadata["diagnostics"]["model"]["multi_task_shared_backbone"])
        model = build_candidate_model(config, feature_dim=20, field_count=5, embedding_dim=4)
        self.assertIsInstance(model, MultiTaskSharedBackbone)
        self.assertEqual(auxiliary_labels(rows).tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_architecture_candidate_records_compiled_structure(self):
        prepared = PreparedData(
            train_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
            validation_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory)
            output = run_torch_fm_candidate(
                prepared,
                {
                    "loss": "pointwise", "learning_rate": 0.01, "l2": 0.0,
                    "epochs": 1, "batch_size": 2, "seed": 0, "architecture": "deepfm",
                },
                path,
            )

            model = output.metadata["diagnostics"]["model"]
            self.assertEqual(model["architecture"], "deepfm")
            self.assertEqual(model["family"], "fm_hybrid")
            self.assertEqual(model["architecture_spec"]["hidden_layers"], [64, 32])
            self.assertEqual(
                model["architecture_spec"]["structural_diff_from_fm"]["added_modules"],
                ["flattened_field_embedding_mlp"],
            )
            self.assertTrue((path / "architecture_spec.json").exists())
            self.assertEqual(output.metadata["architecture_spec_path"], str(path / "architecture_spec.json"))

    def test_composed_candidate_trains_and_records_exact_compiled_spec(self):
        prepared = PreparedData(
            train_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
            validation_rows=[row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0), row("u2", "v2", 1)],
        )
        spec = ReviewedArchitectureSpec(
            interaction_paths=("embedding_mlp", "cross_network"), fusion="learned_gate",
            hidden_width=16, hidden_depth=1, dropout=0.0, cross_layers=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = run_torch_fm_candidate(
                prepared,
                {
                    "loss": "pointwise", "learning_rate": 0.01, "l2": 0.0,
                    "epochs": 1, "batch_size": 2, "seed": 0,
                    "architecture": spec.architecture_id,
                },
                __import__("pathlib").Path(directory),
            )

        recorded = output.metadata["diagnostics"]["model"]["architecture_spec"]
        self.assertEqual(recorded["interaction_paths"], ["embedding_mlp", "cross_network"])
        self.assertEqual(recorded["fusion"], "learned_gate")
        self.assertEqual(recorded["hidden_width"], 16)

    def test_approved_feature_variants_encode_through_canonical_data_interface(self):
        splits = {
            "train": [row("u1", "v1", 1), row("u1", "v2", 0), row("u2", "v1", 0)],
            "valid": [row("u1", "v3", 1), row("u3", "v1", 0)],
        }
        for variant in ("weekday", "author_affinity", "user_history"):
            encoded, feature_dim = encode_candidate(splits, feature_variant=variant)
            self.assertEqual(encoded["train"][0].shape, (3, 6))
            self.assertEqual(encoded["valid"][0].shape, (2, 6))
            self.assertGreater(feature_dim, 6)

    def test_label_features_are_cold_for_unique_train_keys(self):
        splits = {
            "train": [
                (20220408, "u1", "v1", "a1", "1", 10_000.0, 1),
                (20220408, "u2", "v2", "a2", "1", 10_000.0, 0),
            ],
            "valid": [],
        }
        encoded, _ = encode_candidate(splits, feature_variant="author_affinity")
        derived_column = encoded["train"][0][:, -1]
        self.assertEqual(int(derived_column[0]), int(derived_column[1]))
