import tempfile
import time
import unittest

from research_agent.controller import ExperimentController
from research_agent.contracts import BenchmarkContract
from research_agent.critic import ProposalCritic
from research_agent.fidelity import FidelityManager
from research_agent.logger import ResearchLogger
from research_agent.llm_planner import LLMPlanningError, ResearchCatalogueExhausted
from research_agent.loop import AutonomousResearchLoop
from research_agent.planner import CapabilityAction, EvidencePlanner, ResearchDirection
from research_agent.review import EvidenceReviewer
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.safety import SafetyValidator
from research_agent.search import SearchController
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


def loader(_data_dir):
    return {"train": [("train",)], "valid": [("valid",)], "test": [("test",)]}


def candidate(_data, _config, _run_dir):
    return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])


class AutonomousLoopTests(unittest.TestCase):
    def test_planner_context_includes_governance_interventions_and_recovery(self):
        class ContextPlanner:
            context = None

            def set_run_context(self, context):
                self.context = context

            def propose(self, history, state):
                return EvidencePlanner(seed=0).propose(history, state)

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            logger.record_manual_intervention(
                description="Approved reviewed FM hybrids",
                reason="Test governance propagation",
                effect="Architecture capability is allowed",
            )
            logger.log_action("timed_out", experiment_id="exp_old", details={"error": "bounded timeout"})
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            planner = ContextPlanner()
            loop = AutonomousResearchLoop(
                controller=controller, logger=logger, planner=planner,
                search=SearchController(), critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(), fidelity=FidelityManager(), candidate=candidate,
            )

            loop._prepare_planner_context([])

            self.assertEqual(planner.context["benchmark_contract"]["label"], "long_view")
            self.assertEqual(
                planner.context["manual_interventions"][0]["description"],
                "Approved reviewed FM hybrids",
            )
            self.assertEqual(planner.context["recent_errors_and_recoveries"][0]["action"], "timed_out")
            self.assertEqual(len(planner.context["architecture_guidance"]["sha256"]), 64)
            self.assertEqual(planner.context["research_tree"]["incumbent"]["ancestry"], ["baseline"])
            self.assertEqual(
                planner.context["research_coverage"]["architectures"][0]["mechanism"],
                "fm",
            )
            self.assertTrue((store.root / "research_coverage.json").is_file())

    def test_two_workers_execute_distinct_hypotheses_concurrently(self):
        directions = [
            ResearchDirection(
                direction_id="pointwise_fm_optimization",
                hypothesis="A lower learning rate may stabilize sparse pointwise FM optimization.",
                rationale="Controlled optimizer changes can expose convergence limitations.",
                search_space={"loss": ["pointwise"], "learning_rate": [0.0005, 0.001], "l2": [0.0, 1e-6]},
                success_evidence="Validation primary exceeds the incumbent.",
                evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
                strategy="exploration",
                portfolio_role="incumbent_exploit",
                specialist_id="training_specialist",
                preferred_factor="learning_rate",
                preferred_value="controller_select",
                claimed_behavior="Smaller optimizer steps stabilize sparse parameter updates.",
                required_capabilities=("learning-rate control",),
            ),
            ResearchDirection(
                direction_id="pairwise_fm_ranking",
                hypothesis="Pairwise score comparisons may improve within-user exposure ordering.",
                rationale="The validation metrics reward relative ordering within each user.",
                search_space={"loss": ["pairwise"], "learning_rate": [0.0005, 0.001], "l2": [0.0, 1e-6]},
                success_evidence="Validation primary exceeds the incumbent.",
                evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
                strategy="exploration",
                portfolio_role="independent_explore",
                specialist_id="ranking_specialist",
                preferred_factor="loss",
                preferred_value="controller_select",
                claimed_behavior="Positive-negative score differences receive direct supervision.",
                required_capabilities=("pairwise loss",),
            ),
        ]

        class BatchPlanner:
            last_metadata = {}

            def propose_batch(self, _history, _state, *, count):
                self.last_metadata = {"mode": "test_parallel", "planned_workers": count}
                return directions[:count]

            def propose(self, _history, _state):
                return directions[0]

        def slow_candidate(_data, _config, _run_dir):
            time.sleep(0.12)
            return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=BatchPlanner(),
                search=SearchController(seed=0, worker_threads=1),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=slow_candidate,
                max_workers=2,
            )

            started = time.monotonic()
            results = loop.run(max_cycles=2)
            elapsed = time.monotonic() - started

            self.assertEqual(len(results), 2)
            self.assertEqual({item.iteration.experiment_id for item in results}, {"exp_001", "exp_002"})
            self.assertEqual({item.iteration.decision for item in results}, {"accepted", "rejected"})
            self.assertLess(elapsed, 1.2)
            self.assertEqual(len(store.read_iterations()), 2)
            self.assertEqual(len(store.read_stages()), 10)
            self.assertTrue(all(item["config"]["worker_threads"] == 1 for item in store.read_stages()))
            actions = [event["action"] for event in store.read_events()]
            self.assertEqual(actions.count("parallel_worker_started"), 2)
            self.assertEqual(actions.count("parallel_worker_completed"), 2)
            self.assertIn("parallel_batch_completed", actions)
            started_events = [
                event for event in store.read_events()
                if event["action"] == "parallel_worker_started"
            ]
            self.assertEqual(
                {event["details"]["portfolio_role"] for event in started_events},
                {"incumbent_exploit", "independent_explore"},
            )
            self.assertEqual(
                {item["portfolio_role"] for item in store.read_iterations()},
                {"incumbent_exploit", "independent_explore"},
            )

    def test_capability_action_is_retained_while_empty_worker_slot_is_backfilled(self):
        executable = [
            ResearchDirection(
                direction_id="pointwise_fm_optimization",
                hypothesis="A lower learning rate may stabilize pointwise optimization updates.",
                rationale="This controlled optimizer change tests convergence without changing representations.",
                search_space={"loss": ["pointwise"], "learning_rate": [0.0005, 0.001]},
                success_evidence="Validation primary exceeds the frozen incumbent.",
                evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
                strategy="exploration", preferred_factor="learning_rate",
                portfolio_role="incumbent_exploit",
            ),
            ResearchDirection(
                direction_id="pairwise_fm_ranking",
                hypothesis="Pairwise supervision may improve within-user ordering quality.",
                rationale="This directly tests ranking supervision while preserving the model and features.",
                search_space={"loss": ["pairwise"], "learning_rate": [0.001]},
                success_evidence="Validation primary exceeds the frozen incumbent.",
                evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
                strategy="exploration", preferred_factor="loss",
                portfolio_role="independent_explore",
            ),
        ]
        capability = CapabilityAction(
            action="BUILD_CAPABILITY",
            hypothesis="Feature-coverage slices may localize the remaining ranking error.",
            rationale="Aggregate validation metrics cannot identify coverage-sensitive failures.",
            capability_gap_id="coverage_slices",
            capability_gap_description="Add training-derived validation coverage strata.",
            required_capabilities=("validation slicing",),
        )

        class BackfillPlanner:
            last_metadata = {}

            def propose_batch(self, _history, _state, *, count):
                self.last_metadata = {"mode": "initial", "requested": count}
                return [executable[0], capability]

            def propose(self, _history, _state):
                self.last_metadata = {"mode": "backfill"}
                return executable[1]

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger, runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator, state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller, logger=logger, planner=BackfillPlanner(),
                search=SearchController(worker_threads=1), critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(), fidelity=FidelityManager(), candidate=candidate,
                max_workers=2,
            )

            results = loop.run(max_cycles=2)

            self.assertEqual(len(results), 2)
            self.assertIsNone(loop.pause_reason)
            self.assertEqual(store.read_capability_actions()[0]["status"], "pending_implementation")
            actions = [item["action"] for item in store.read_events()]
            self.assertEqual(actions.count("parallel_worker_started"), 2)
            self.assertIn("execution_slots_refilled", actions)
            self.assertIn("capability_backlogged", actions)

    def test_loop_logs_direction_critic_and_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=candidate,
            )

            results = loop.run(max_cycles=1)

            self.assertEqual(len(results), 1)
            actions = [event["action"] for event in store.read_events()]
            self.assertIn("research_direction_proposed", actions)
            self.assertIn("proposal_critic_reviewed", actions)
            self.assertIn("seed_confirmation_summary", actions)
            self.assertGreaterEqual(actions.count("promotion_critic_reviewed"), 2)
            self.assertEqual(store.read_iterations()[0]["direction_id"], results[0].direction.direction_id)
            self.assertEqual(len(store.read_iterations()), 1)
            stages = store.read_stages()
            self.assertEqual({item["experiment_id"] for item in stages}, {"exp_001"})
            self.assertEqual(
                [item["stage_id"] for item in stages],
                ["low", "medium", "full", "seed_1", "seed_2"],
            )
            self.assertEqual(controller.state.completed_iterations, 1)
            completed = store.read_iterations()[0]
            self.assertEqual(completed["diagnostic_evidence"]["seed_confirmation"]["count"], 2)
            self.assertTrue(completed["semantic_review"]["approved"])
            self.assertEqual(completed["semantic_review"]["trace"]["verdict"], "verified")
            memory = store.read_root_json("evidence_memory.json")
            self.assertEqual(memory["completed_hypotheses"], 1)
            self.assertEqual(memory["hypotheses"][0]["experiment_id"], "exp_001")
            feedback = store.read_critic_feedback()
            self.assertEqual([item["phase"] for item in feedback], ["pre_execution", "post_execution"])
            critic_memory = store.read_root_json("critic_memory.json")
            self.assertEqual(critic_memory["feedback_count"], 2)
            self.assertEqual(
                critic_memory["supported_lineages"][0]["experiment_id"], "exp_001"
            )

    def test_blocked_llm_slate_is_logged_with_usage_before_pause(self):
        class BlockedPlanner:
            last_metadata = {"usage": {"total_tokens": 77}, "deferred_candidates": ["h1"]}

            def propose(self, _history, _state):
                raise LLMPlanningError("not executable without semantic drift")

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=BlockedPlanner(),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=candidate,
            )

            with self.assertRaises(LLMPlanningError):
                loop.run(max_cycles=1)

            blocked = next(event for event in store.read_events() if event["action"] == "llm_planning_blocked")
            self.assertEqual(blocked["details"]["usage"]["total_tokens"], 77)
            self.assertEqual(blocked["details"]["planner_metadata"]["deferred_candidates"], ["h1"])

    def test_capability_build_action_is_backlogged_and_pauses_non_terminally(self):
        action = CapabilityAction(
            action="BUILD_CAPABILITY",
            hypothesis="A listwise objective may improve top-ranked recommendation quality.",
            rationale="The hypothesis is valuable but has no exact executable implementation.",
            capability_gap_id="listwise_loss",
            capability_gap_description="Implement leakage-safe within-user lists and a listwise ranking loss.",
            required_capabilities=("within-user lists", "listwise loss"),
            specialist_id="ranking_specialist",
        )

        class ActionPlanner:
            last_metadata = {"mode": "test_action"}
            def propose(self, _history, _state):
                return action

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller, logger=logger, planner=ActionPlanner(),
                search=SearchController(), critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(), fidelity=FidelityManager(), candidate=candidate,
            )

            results = loop.run(max_cycles=1)

            self.assertEqual(results, [])
            self.assertFalse(controller.state.stopped)
            self.assertIn("BUILD_CAPABILITY", loop.pause_reason)
            self.assertEqual(store.read_iterations(), [])
            backlog = store.read_capability_actions()
            self.assertEqual(backlog[0]["status"], "pending_implementation")
            self.assertEqual(backlog[0]["capability_gap_id"], "listwise_loss")
            self.assertIn(
                "research_pause_requested",
                [item["action"] for item in store.read_events()],
            )

    def test_diagnostic_action_records_validation_only_evidence(self):
        action = CapabilityAction(
            action="RUN_DIAGNOSTIC",
            hypothesis="Segment diagnostics may identify where ranking errors concentrate.",
            rationale="Existing aggregate metrics may conceal exposure-segment failures.",
            capability_gap_id="segment_error_diagnostic",
            capability_gap_description="Summarize existing validation outcomes by research direction.",
            required_capabilities=("validation-only aggregation",),
            specialist_id="evaluation_specialist",
        )

        class DiagnosticPlanner:
            last_metadata = {}
            def propose(self, _history, _state):
                return action

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller, logger=logger, planner=DiagnosticPlanner(),
                search=SearchController(), critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(), fidelity=FidelityManager(), candidate=candidate,
            )

            results = loop.run(max_cycles=1)

            self.assertEqual(results, [])
            self.assertIn("diagnostic completed", loop.pause_reason)
            diagnostic = store.read_diagnostics()[0]
            self.assertEqual(diagnostic["evidence"]["selection_split"], "valid")
            self.assertFalse(diagnostic["evidence"]["test_data_used"])
            self.assertEqual(store.read_capability_actions()[0]["status"], "completed")

    def test_diagnostic_runs_after_backfilled_experiment_evidence_is_committed(self):
        action = CapabilityAction(
            action="RUN_DIAGNOSTIC",
            hypothesis="Stratified validation evidence may identify the weakest cohort.",
            rationale="The diagnostic should inspect evidence produced by the accompanying experiment.",
            capability_gap_id="stratified_validation_diagnostics",
            capability_gap_description="Inspect training-derived validation activity and coverage slices.",
            required_capabilities=("validation-only slicing",),
        )

        class DiagnosticThenExperimentPlanner:
            last_metadata = {}

            def __init__(self):
                self.calls = 0

            def propose(self, history, state):
                self.calls += 1
                if self.calls == 1:
                    return action
                return EvidencePlanner(seed=0).propose(history, state)

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger, runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator, state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller, logger=logger,
                planner=DiagnosticThenExperimentPlanner(), search=SearchController(),
                critic=ProposalCritic(validator), reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(), candidate=candidate,
            )

            results = loop.run(max_cycles=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(
                [item["status"] for item in store.read_capability_actions()],
                ["scheduled_after_batch", "completed"],
            )
            diagnostic = store.read_diagnostics()[0]
            self.assertEqual(diagnostic["evidence"]["measured_experiment_count"], 1)
            self.assertEqual(diagnostic["scheduled_action_id"], "cap_001")

    def test_catalogue_exhaustion_stops_and_returns_cleanly(self):
        class ExhaustedPlanner:
            last_metadata = {}

            def propose(self, _history, _state):
                raise ResearchCatalogueExhausted("catalogue exhausted")

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=60)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
                contract=BenchmarkContract(target_primary=1.1),
            )
            loop = AutonomousResearchLoop(
                controller=controller, logger=logger, planner=ExhaustedPlanner(),
                search=SearchController(), critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(), fidelity=FidelityManager(), candidate=candidate,
            )

            results = loop.run(max_cycles=20)

            self.assertEqual(results, [])
            self.assertEqual(controller.state.stop_reason, "catalogue exhausted")
            actions = [event["action"] for event in store.read_events()]
            self.assertIn("research_catalogue_exhausted", actions)
            self.assertIn("research_stopped", actions)
