import unittest

from research_agent.safety import ExperimentProposal, SafetyValidator


def safe_proposal(**changes):
    values = {
        "experiment_id": "exp_001",
        "hypothesis": "One controlled change may improve ranking.",
        "rationale": "The change is evaluated only on validation.",
        "config": {"seed": 0, "learning_rate": 0.001},
        "changed_factors": ("learning_rate",),
    }
    values.update(changes)
    return ExperimentProposal(**values)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.validator = SafetyValidator(max_runtime_seconds=60)

    def test_safe_validation_only_proposal_passes(self):
        report = self.validator.validate(safe_proposal(runtime_budget_seconds=60))
        self.assertTrue(report.passed)
        self.assertEqual(report.violations, ())

    def test_leakage_and_data_contract_violations_are_rejected(self):
        report = self.validator.validate(
            safe_proposal(
                training_split="valid",
                selection_split="test",
                external_data_sources=("outside.csv",),
                uses_test_labels=True,
                loads_raw_csv=True,
            )
        )
        self.assertFalse(report.passed)
        joined = " ".join(report.violations)
        self.assertIn("train split", joined)
        self.assertIn("valid split", joined)
        self.assertIn("external datasets", joined)
        self.assertIn("test labels", joined)
        self.assertIn("data.py", joined)

    def test_multiple_changes_and_protected_files_are_rejected(self):
        report = self.validator.validate(
            safe_proposal(
                changed_factors=("learning_rate", "l2"),
                modified_files=("evaluate.py", "baseline.py"),
            )
        )
        self.assertFalse(report.passed)
        self.assertIn("exactly one", " ".join(report.violations))
        self.assertIn("protected benchmark files", " ".join(report.violations))

    def test_duplicate_config_and_invalid_budget_are_rejected(self):
        proposal = safe_proposal(runtime_budget_seconds=61)
        report = self.validator.validate(proposal, historical_configs=[proposal.config])
        self.assertFalse(report.passed)
        self.assertIn("duplicates", " ".join(report.violations))
        self.assertIn("runtime budget", " ".join(report.violations))

    def test_unapproved_dependency_or_model_family_requires_review(self):
        report = self.validator.validate(
            safe_proposal(
                requested_dependencies=("optuna",),
                model_family="deepfm",
                human_reviewed=False,
            )
        )
        self.assertFalse(report.passed)
        self.assertIn("unapproved dependencies", " ".join(report.violations))
        self.assertIn("requires human review", " ".join(report.violations))
