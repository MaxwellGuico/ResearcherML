import tempfile
import threading
import unittest

from research_agent.controller import ExperimentController
from research_agent.contracts import BenchmarkContract
from research_agent.logger import ResearchLogger
from research_agent.runner import CandidateOutput, ExperimentRunner, RunnerResult
from research_agent.safety import ExperimentProposal, SafetyValidator
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


def loader(_data_dir):
    return {"train": [("train",)], "valid": [("valid",)], "test": [("test",)]}


def strong_candidate(_data, _config, _run_dir):
    return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])


def broken_candidate(_data, _config, _run_dir):
    raise RuntimeError("intentional runner failure")


def weak_candidate(_data, _config, _run_dir):
    return CandidateOutput(["u", "u"], [1, 0], [0.1, 0.9])


def proposal(experiment_id, **changes):
    values = {
        "experiment_id": experiment_id,
        "hypothesis": "A single controlled change may improve ranking.",
        "rationale": "Use validation evidence only.",
        "config": {"candidate": experiment_id},
        "changed_factors": ("candidate",),
        "runtime_budget_seconds": 60,
    }
    values.update(changes)
    return ExperimentProposal(**values)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tempdir.name)
        self.logger = ResearchLogger(self.store)
        self.runner = ExperimentRunner(self.logger, data_loader=loader)
        self.validator = SafetyValidator(max_runtime_seconds=60)

    def tearDown(self):
        self.tempdir.cleanup()

    def _record_full_seed(self, experiment_id, seed, primary):
        self.logger.record_stage({
            "experiment_id": experiment_id,
            "stage_id": "full" if seed == 0 else f"seed_{seed}",
            "config": {"fidelity": "full", "seed": seed},
            "status": "completed",
            "metrics": {
                "GAUC": primary,
                "nDCG@5": primary,
                "primary": primary,
            },
            "runtime_seconds": 1.0,
            "resource_usage": {},
            "runner_metadata": {},
            "safety": {"passed": True, "violations": []},
        })

    def test_improving_candidate_is_accepted_and_updates_state(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
        )

        result = controller.run_iteration(proposal("exp_001"), strong_candidate)

        self.assertEqual(result.decision, "accepted")
        self.assertEqual(controller.state.current_best_experiment_id, "exp_001")
        self.assertEqual(controller.state.current_best_primary, 1.0)
        self.assertTrue((self.store.root / "state.json").exists())
        record = self.store.read_iterations()[0]
        self.assertEqual(record["decision"], "accepted")
        self.assertEqual(record["parent_experiment_id"], "baseline")

    def test_controller_recovers_state_and_history_after_restart(self):
        test_contract = BenchmarkContract(target_primary=1.1)
        first_controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            contract=test_contract,
        )
        first_controller.run_iteration(proposal("exp_001"), strong_candidate)

        resumed = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            contract=test_contract,
        )
        duplicate = resumed.run_iteration(
            proposal("exp_002", config={"candidate": "exp_001"}),
            strong_candidate,
        )

        self.assertEqual(resumed.state.current_best_experiment_id, "exp_001")
        self.assertEqual(duplicate.decision, "rejected")
        self.assertIn("duplicates", duplicate.error)

    def test_failed_unmeasured_config_is_retryable_in_same_process(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            contract=BenchmarkContract(target_primary=1.1),
        )
        config = {"candidate": "retryable"}

        failed = controller.run_iteration(
            proposal("exp_failed", config=config), broken_candidate
        )
        retried = controller.run_iteration(
            proposal("exp_retry", config=config), strong_candidate
        )

        self.assertEqual(failed.decision, "failed")
        self.assertEqual(retried.decision, "accepted")
        actions = [item["action"] for item in self.store.read_events()]
        self.assertIn("configuration_released_retryable", actions)

    def test_failed_unmeasured_config_is_retryable_after_resume(self):
        contract = BenchmarkContract(target_primary=1.1)
        first = ExperimentController(
            logger=self.logger, runner=self.runner, validator=self.validator,
            state=ResearchState(current_best_primary=0.5), contract=contract,
        )
        config = {"candidate": "resume_retry"}
        first.run_iteration(proposal("exp_failed", config=config), broken_candidate)

        resumed = ExperimentController(
            logger=self.logger, runner=self.runner, validator=self.validator,
            contract=contract,
        )
        result = resumed.run_iteration(
            proposal("exp_retry", config=config), strong_candidate
        )

        self.assertEqual(result.decision, "accepted")

    def test_inflight_duplicate_is_rejected_while_first_worker_runs(self):
        controller = ExperimentController(
            logger=self.logger, runner=self.runner, validator=self.validator,
            state=ResearchState(current_best_primary=0.5),
            contract=BenchmarkContract(target_primary=1.1),
        )
        entered = threading.Event()
        release = threading.Event()
        first_result = []

        def blocking_run(**kwargs):
            entered.set()
            release.wait(timeout=2)
            return RunnerResult(
                experiment_id=kwargs["experiment_id"],
                status="completed",
                run_dir=self.store.root,
                runtime_seconds=0.01,
                output=CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1]),
            )

        controller.runner.run = blocking_run

        config = {"candidate": "concurrent"}
        worker = threading.Thread(
            target=lambda: first_result.append(
                controller.run_iteration(
                    proposal("exp_first", config=config), strong_candidate
                )
            )
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=2))
        collision = controller.run_iteration(
            proposal("exp_collision", config=config), strong_candidate
        )
        release.set()
        worker.join(timeout=2)

        self.assertEqual(collision.decision, "rejected")
        self.assertIn("duplicates", collision.error)
        self.assertEqual(first_result[0].decision, "accepted")

    def test_old_invocation_budget_stop_is_resumable_under_hard_cap(self):
        self.store.write_root_json(
            "state.json",
            ResearchState(
                completed_iterations=1,
                consecutive_non_improvements=1,
                stop_reason="configured experiment budget reached: 1",
            ).as_dict(),
        )

        resumed = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            contract=BenchmarkContract(max_experiments=20),
            max_iterations=20,
        )

        self.assertFalse(resumed.state.stopped)
        actions = [item["action"] for item in self.store.read_events()]
        self.assertIn("configured_budget_extended", actions)

    def test_persistent_hard_cap_remains_terminal(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=1.0),
            contract=BenchmarkContract(target_primary=1.1, max_experiments=1),
            max_iterations=1,
        )

        controller.run_iteration(proposal("exp_cap"), strong_candidate)

        self.assertTrue(controller.state.stopped)
        self.assertEqual(controller.state.stop_reason, "configured experiment budget reached: 1")

    def test_non_improvement_is_rejected_and_preserves_best_pointer(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_experiment_id="baseline", current_best_primary=1.0),
        )

        result = controller.run_iteration(proposal("exp_002"), strong_candidate)

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        record = self.store.read_iterations()[0]
        self.assertIn("restored accepted candidate pointer", record["recovery"])

    def test_any_positive_gain_is_accepted_but_small_gain_counts_toward_convergence(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.999),
            contract=BenchmarkContract(target_primary=1.1, improvement_threshold=0.002),
        )

        result = controller.run_iteration(proposal("exp_small_gain"), strong_candidate)

        self.assertEqual(result.decision, "accepted")
        self.assertEqual(controller.state.current_best_primary, 1.0)
        self.assertEqual(controller.state.current_best_experiment_id, "exp_small_gain")
        self.assertEqual(controller.state.consecutive_non_improvements, 1)

    def test_semantically_invalid_evidence_cannot_replace_incumbent(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_experiment_id="baseline", current_best_primary=0.5),
            contract=BenchmarkContract(target_primary=1.1),
        )
        candidate_proposal = proposal("exp_semantic")
        stage = controller.run_iteration(
            candidate_proposal,
            strong_candidate,
            stage_id="low",
            complete_experiment=False,
        )

        result = controller.complete_staged_experiment(
            candidate_proposal,
            [stage],
            baseline_primary=0.5,
            terminal_reason="semantic_test",
            semantic_review={
                "approved": False,
                "reasons": ["configuration did not implement the claim"],
                "trace": {"verdict": "misaligned"},
            },
        )

        self.assertEqual(result.decision, "rejected")
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        self.assertEqual(controller.state.current_best_primary, 0.5)
        self.assertEqual(controller.state.consecutive_non_improvements, 1)
        self.assertFalse(self.store.read_iterations()[0]["semantic_review"]["approved"])

    def test_staged_acceptance_uses_three_seed_mean_not_best_seed(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(
                current_best_experiment_id="baseline", current_best_primary=0.61
            ),
            contract=BenchmarkContract(target_primary=1.1),
        )
        candidate = proposal("exp_noisy", config={"fidelity": "full", "seed": 0})
        for seed, primary in ((0, 0.70), (1, 0.55), (2, 0.55)):
            self._record_full_seed(candidate.experiment_id, seed, primary)

        result = controller.complete_staged_experiment(
            candidate,
            [],
            baseline_primary=0.61,
            acceptance_primary=0.61,
            terminal_reason="full_fidelity_completed",
            semantic_review={"approved": True},
        )

        self.assertEqual(result.decision, "rejected")
        self.assertAlmostEqual(result.metrics["primary"], 0.60)
        self.assertEqual(controller.state.current_best_experiment_id, "baseline")
        self.assertTrue(
            self.store.read_iterations()[0]["diagnostic_evidence"]["robust_acceptance"]["complete"]
        )

    def test_staged_candidate_cannot_be_accepted_without_all_confirmation_seeds(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.50),
            contract=BenchmarkContract(target_primary=1.1),
        )
        candidate = proposal("exp_incomplete", config={"fidelity": "full", "seed": 0})
        for seed in (0, 1):
            self._record_full_seed(candidate.experiment_id, seed, 0.65)

        result = controller.complete_staged_experiment(
            candidate,
            [],
            baseline_primary=0.50,
            terminal_reason="seed_confirmation_failed",
            semantic_review={"approved": True},
        )

        self.assertEqual(result.decision, "rejected")
        self.assertIn("seeds 0, 1, and 2", result.error)

    def test_resume_reconciles_incumbent_by_three_seed_mean(self):
        def accepted_record(experiment_id, primaries):
            fidelity_results = [
                {
                    "stage_id": "full" if seed == 0 else f"seed_{seed}",
                    "fidelity": "full",
                    "seed": seed,
                    "metrics": {"GAUC": value, "nDCG@5": value, "primary": value},
                }
                for seed, value in enumerate(primaries)
            ]
            self.logger.record_iteration({
                "experiment_id": experiment_id,
                "hypothesis": "Compare robust evidence.",
                "rationale": "Three seeds reduce selection noise.",
                "config": {"candidate": experiment_id},
                "decision": "accepted",
                "metrics": {"primary": primaries[0]},
                "diagnostic_evidence": {"fidelity_results": fidelity_results},
            })

        accepted_record("exp_robust", (0.61, 0.63, 0.62))
        accepted_record("exp_noisy", (0.70, 0.56, 0.57))
        self.store.write_root_json(
            "state.json",
            ResearchState(
                current_best_experiment_id="exp_noisy", current_best_primary=0.70
            ).as_dict(),
        )

        resumed = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            contract=BenchmarkContract(target_primary=1.1),
        )

        self.assertEqual(resumed.state.current_best_experiment_id, "exp_robust")
        self.assertAlmostEqual(resumed.state.current_best_primary, 0.62)
        self.assertIn(
            "robust_incumbent_reconciled",
            [event["action"] for event in self.store.read_events()],
        )

    def test_safety_rejection_never_invokes_runner(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
        )
        unsafe = proposal("exp_003", changed_factors=("a", "b"))

        result = controller.run_iteration(unsafe, strong_candidate)

        self.assertEqual(result.decision, "rejected")
        self.assertFalse((self.store.runs_dir / "exp_003").exists())
        self.assertIn("exactly one", result.error)

    def test_failure_recovers_and_requests_restart_after_three_non_improvements(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.64),
        )

        first = controller.run_iteration(proposal("exp_004"), broken_candidate)
        controller.run_iteration(proposal("exp_005"), weak_candidate)
        third = controller.run_iteration(proposal("exp_006"), weak_candidate)

        self.assertEqual(first.decision, "failed")
        self.assertEqual(third.decision, "rejected")
        self.assertFalse(controller.state.stopped)
        self.assertEqual(controller.state.consecutive_non_improvements, 3)

    def test_plateau_restart_resets_counter_without_stopping(self):
        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            state=ResearchState(current_best_primary=0.64),
        )
        for number in range(3):
            controller.run_iteration(proposal(f"exp_{number + 10:03d}"), broken_candidate)
        self.assertFalse(controller.state.stopped)
        self.assertEqual(controller.state.consecutive_non_improvements, 3)
        controller.begin_plateau_restart()
        self.assertEqual(controller.state.consecutive_non_improvements, 0)
        self.assertEqual(controller.state.plateau_restarts, 1)

    def test_restart_preserves_orphaned_run_as_interrupted_evidence(self):
        self.store.run_dir("exp_interrupted", "low")
        self.store.write_run_json(
            "exp_interrupted", "config.json", {"learning_rate": 0.0005}, stage_id="low"
        )
        self.store.write_run_json(
            "exp_interrupted", "plan.json",
            {"hypothesis": "An interrupted controlled experiment."}, stage_id="low",
        )
        self.store.write_patch("exp_interrupted", "learning_rate")
        self.store.write_root_json(
            "state.json", ResearchState(current_best_primary=0.6).as_dict()
        )

        controller = ExperimentController(
            logger=self.logger,
            runner=self.runner,
            validator=self.validator,
            contract=BenchmarkContract(target_primary=1.1),
        )

        record = self.store.read_iterations()[0]
        self.assertEqual(record["experiment_id"], "exp_interrupted")
        self.assertEqual(record["decision"], "failed")
        self.assertEqual(record["terminal_reason"], "interrupted")
        self.assertEqual(controller.state.completed_iterations, 1)
        self.assertIn(
            "interrupted_experiment_recovered",
            [event["action"] for event in self.store.read_events()],
        )
