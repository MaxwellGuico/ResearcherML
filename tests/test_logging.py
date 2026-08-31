import csv
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from research_agent.logger import ResearchLogger
from research_agent.reporter import MarkdownReporter
from research_agent.run_research import (
    _acquire_run_lock,
    _record_architecture_approval,
    _record_capability_approvals,
    _record_interrupted_worker_recoveries,
)
from research_agent.store import ArtifactStore


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)
        self.logger = ResearchLogger(
            self.store,
            clock=lambda: datetime(2026, 8, 29, tzinfo=timezone.utc),
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_actions_are_append_only_jsonl_records(self):
        self.logger.log_action("training_started", experiment_id="exp_001")
        self.logger.log_action("evaluation_started", experiment_id="exp_001")

        events = self.store.read_events()

        self.assertEqual([event["action"] for event in events], ["training_started", "evaluation_started"])
        self.assertEqual(events[0]["experiment_id"], "exp_001")

    def test_artifact_directory_rejects_a_second_live_process_lock(self):
        first = _acquire_run_lock(self.tempdir.name)
        try:
            with self.assertRaises(BlockingIOError):
                _acquire_run_lock(self.tempdir.name)
        finally:
            first.close()

    def test_iteration_writes_jsonl_and_metric_summary(self):
        self.logger.record_iteration(
            {
                "experiment_id": "exp_001",
                "parent_experiment_id": "baseline",
                "hypothesis": "A controlled change may improve ranking.",
                "rationale": "Tested with validation only.",
                "config": {"seed": 0},
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
                "delta_primary": 0.0035,
                "runtime_seconds": 12.5,
                "decision": "accepted",
            }
        )

        iterations = self.store.read_iterations()
        self.assertEqual(iterations[0]["decision"], "accepted")
        with self.store.metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["primary"], "0.605")

    def test_stage_is_separate_from_completed_hypothesis(self):
        self.logger.record_stage({
            "experiment_id": "exp_001",
            "stage_id": "medium",
            "config": {"fidelity": "medium"},
            "status": "completed",
            "metrics": {"primary": 0.602},
        })

        self.assertEqual(self.store.read_iterations(), [])
        self.assertEqual(self.store.read_stages()[0]["stage_id"], "medium")

    def test_llm_research_strategy_is_append_only_and_reported(self):
        self.logger.record_research_strategy({
            "strategy_id": "architecture_discovery_v1",
            "phase_label": "architecture discovery",
            "decision": "start",
            "focus_domains": ["model_architecture"],
            "metric_emphasis": "nDCG@5",
            "frozen_factors": ["feature_variant", "loss", "learning_rate", "l2"],
            "worker_assignments": [
                {"domain": "model_architecture", "portfolio_role": "incumbent_exploit"},
                {"domain": "model_architecture", "portfolio_role": "independent_explore"},
            ],
            "rationale": "Architecture evidence should be established before numerical tuning begins.",
            "evidence_reference": "baseline and prior architecture evidence",
            "transition_criteria": "Revise when controlled architecture experiments plateau or confirm an incumbent.",
        })

        self.assertEqual(len(self.store.read_research_strategies()), 1)
        self.assertEqual(
            self.store.read_events()[-1]["action"],
            "llm_research_strategy_decided",
        )
        report = MarkdownReporter(self.store).render()
        self.assertIn("LLM Research Strategy History", report)
        self.assertIn("architecture_discovery_v1", report)

    def test_patch_and_manual_summary_are_preserved(self):
        patch = self.logger.record_code_diff("exp_001", "diff --git a/a.py b/a.py\n")
        self.logger.record_manual_intervention(
            experiment_id="exp_001",
            description="Reduced the allowed runtime.",
            reason="Laptop battery limit.",
            effect="Future runs use the smaller budget.",
        )

        report = MarkdownReporter(self.store).write()

        self.assertTrue(patch.exists())
        self.assertIn("Manual interventions: 1", report.read_text(encoding="utf-8"))
        self.assertEqual(len(self.store.read_interventions()), 1)

    def test_report_explicitly_records_zero_interventions(self):
        report = MarkdownReporter(self.store).write()
        text = report.read_text(encoding="utf-8")
        self.assertIn("Manual interventions: 0", text)
        self.assertIn("## Run Statistics", text)

    def test_architecture_approval_is_persistent_and_not_duplicated_on_resume(self):
        self.assertTrue(_record_architecture_approval(self.logger))
        self.assertFalse(_record_architecture_approval(self.logger))

        self.assertEqual(len(self.store.read_interventions()), 1)
        self.assertEqual(
            self.store.read_interventions()[0]["approval_id"],
            "architecture_experiments:v1",
        )
        self.assertEqual(
            self.store.read_events()[-1]["action"],
            "architecture_approval_reused",
        )

        report = MarkdownReporter(self.store).render()
        self.assertIn("Approval ID: architecture_experiments:v1", report)
        self.assertIn("Reuses: 1", report)

    def test_versioned_approval_is_atomic_across_concurrent_callers(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            recorded = list(executor.map(
                lambda _index: _record_architecture_approval(self.logger),
                range(8),
            ))

        self.assertEqual(sum(recorded), 1)
        self.assertEqual(len(self.store.read_interventions()), 1)
        self.assertEqual(
            sum(
                event["action"] == "architecture_approval_reused"
                for event in self.store.read_events()
            ),
            7,
        )

    def test_interrupted_worker_recovery_is_append_only_and_idempotent(self):
        self.logger.log_action(
            "candidate_created", experiment_id="exp_009",
            details={"stage_id": "low", "run_dir": "runs/exp_009/low"},
        )

        self.assertEqual(_record_interrupted_worker_recoveries(self.logger), 1)
        self.assertEqual(_record_interrupted_worker_recoveries(self.logger), 0)
        recovery = self.store.read_events()[-1]
        self.assertEqual(recovery["action"], "interrupted_worker_recovered")
        self.assertIn("preserved incomplete artifacts", recovery["details"]["recovery"])

    def test_capability_backlog_is_append_only_and_reported(self):
        first = self.logger.record_capability_action({
            "action": "BUILD_CAPABILITY",
            "capability_gap_id": "listwise_loss",
            "capability_gap_description": "Implement a leakage-safe listwise ranking objective.",
            "hypothesis": "A listwise objective may improve top-ranked items.",
            "rationale": "The current objective does not optimize list position directly.",
        }, status="pending_implementation")
        second = self.logger.record_capability_action({
            "action": "REQUEST_HUMAN_APPROVAL",
            "capability_gap_id": "new_model_family",
            "capability_gap_description": "Review a substantially different recommendation model family.",
            "hypothesis": "A different model family may improve interaction capacity.",
            "rationale": "Existing reviewed FM hybrids may have reached their capacity limit.",
            "approval_reason": "A substantially different model family requires human review.",
        }, status="pending_human_approval")

        report = MarkdownReporter(self.store).render()

        self.assertEqual((first["action_id"], second["action_id"]), ("cap_001", "cap_002"))
        self.assertIn("Capability actions: 2", report)
        self.assertIn("pending_human_approval", report)

    def test_capability_approval_is_gap_scoped_and_records_intervention(self):
        self.logger.record_capability_action({
            "action": "REQUEST_HUMAN_APPROVAL",
            "capability_gap_id": "reviewed_gap",
            "capability_gap_description": "Review authority for one bounded architecture capability.",
            "hypothesis": "A bounded architecture operator may improve interaction capacity.",
            "rationale": "The operator exceeds the currently registered executable capability set.",
            "required_capabilities": ["bounded operator"],
            "specialist_id": "model_architecture_specialist",
            "approval_reason": "A structural capability expansion requires explicit human review.",
        }, status="pending_human_approval")

        _record_capability_approvals(["reviewed_gap"], self.logger)

        self.assertEqual(self.store.read_capability_actions()[-1]["status"], "human_approved")
        self.assertEqual(len(self.store.read_interventions()), 1)
        with self.assertRaisesRegex(ValueError, "no pending human-approval request"):
            _record_capability_approvals(["reviewed_gap"], self.logger)
