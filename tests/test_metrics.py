import math
import unittest

from research_agent.metrics import MetricsValidationError, evaluate_predictions


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.users = ["u1", "u1", "u2", "u2"]
        self.labels = [1, 0, 0, 1]
        self.scores = [0.9, 0.1, 0.1, 0.8]

    def test_validation_metrics_come_from_official_evaluator(self):
        result = evaluate_predictions(self.users, self.labels, self.scores)

        self.assertEqual(result.rows, 4)
        self.assertEqual(result.users, 2)
        self.assertEqual(result.gauc, 1.0)
        self.assertGreater(result.ndcg_at_5, 0.0)
        self.assertEqual(result.primary, (result.gauc + result.ndcg_at_5) / 2)
        self.assertEqual(len(result.evaluator_sha256), 64)

    def test_test_requires_explicit_final_confirmation(self):
        with self.assertRaisesRegex(MetricsValidationError, "allow_test=True"):
            evaluate_predictions(self.users, self.labels, self.scores, split="test")

    def test_invalid_scores_are_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                scores = list(self.scores)
                scores[0] = invalid
                with self.assertRaisesRegex(MetricsValidationError, "finite"):
                    evaluate_predictions(self.users, self.labels, scores)

    def test_invalid_shapes_labels_and_lengths_are_rejected(self):
        with self.assertRaisesRegex(MetricsValidationError, "one-dimensional"):
            evaluate_predictions(self.users, self.labels, [[0.1], [0.2], [0.3], [0.4]])
        with self.assertRaisesRegex(MetricsValidationError, "binary"):
            evaluate_predictions(self.users, [1, 0, 2, 1], self.scores)
        with self.assertRaisesRegex(MetricsValidationError, "equal lengths"):
            evaluate_predictions(self.users, self.labels[:-1], self.scores)
