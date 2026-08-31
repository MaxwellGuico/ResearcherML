"""Closed-loop orchestration across planner, search, safety, runner, and review."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from .controller import ExperimentController, IterationResult
from .critic import ProposalCritic
from .critic_memory import refresh_critic_memory
from .evidence import robust_stage_evidence
from .fidelity import FidelityManager
from .logger import ResearchLogger
from .llm_planner import LLMPlanningError, ResearchCatalogueExhausted
from .metrics import MetricsValidationError, evaluate_predictions
from .planner import CapabilityAction, ResearchDirection, ResearchPlanner
from .review import EvidenceReviewer
from .regions import SearchRegionManager
from .research_coverage import build_research_coverage
from .research_tree import ResearchTree
from .runner import CandidateCallable
from .search import SearchController, SearchState


IMPLEMENTED_DIAGNOSTIC_CAPABILITIES = {
    "stratified_validation_diagnostics": (
        "Every newly trained candidate records training-derived user-activity and "
        "feature-coverage slices with GAUC, nDCG@5, and primary."
    ),
}


@dataclass(frozen=True)
class CycleResult:
    direction: ResearchDirection
    iteration: IterationResult
    promoted_iteration: IterationResult | None


@dataclass(frozen=True)
class _WorkerOutcome:
    direction: ResearchDirection
    proposal: Any
    stage_results: Sequence[IterationResult]
    promoted: IterationResult | None
    terminal_reason: str
    semantic_review: Mapping[str, Any]


class AutonomousResearchLoop:
    """Runs evidence-based cycles; no hypothesis sequence is hard-coded."""

    def __init__(
        self,
        *,
        controller: ExperimentController,
        logger: ResearchLogger,
        planner: ResearchPlanner,
        search: SearchController,
        critic: ProposalCritic,
        reviewer: EvidenceReviewer,
        fidelity: FidelityManager,
        regions: SearchRegionManager | None = None,
        candidate: CandidateCallable,
        max_workers: int = 1,
        research_campaign: str | None = None,
    ) -> None:
        self.controller = controller
        self.logger = logger
        self.planner = planner
        self.search = search
        self.critic = critic
        self.reviewer = reviewer
        self.fidelity = fidelity
        self.regions = regions or SearchRegionManager()
        self.candidate = candidate
        self.research_tree = ResearchTree(logger)
        self.pause_reason: str | None = None
        self.research_campaign = research_campaign
        self.campaign_remaining: list[str] = []
        if max_workers not in {1, 2}:
            raise ValueError("max_workers must be 1 or 2")
        self.max_workers = max_workers

    def run(self, max_cycles: int) -> list[CycleResult]:
        results: list[CycleResult] = []
        while len(results) < max_cycles:
            if self.controller.state.stopped:
                break
            history = self.logger.store.read_iterations()
            self._prepare_planner_context(history)
            if self.research_campaign == "architecture_coverage" and not self.campaign_remaining:
                self.logger.log_action(
                    "research_campaign_completed",
                    details={"campaign": self.research_campaign, "reason": "all executable architecture mechanisms have evidence"},
                )
                break
            review = self.reviewer.review(history, self.controller.state)
            self.logger.log_action(
                "evidence_reviewed",
                details={
                    "action": review.action,
                    "rationale": review.rationale,
                    "regions": [snapshot.__dict__ for snapshot in self.regions.snapshots(history)],
                },
            )
            if review.action == "restart":
                self.controller.begin_plateau_restart()
            remaining_budget = min(
                max_cycles - len(results),
                self.controller.max_iterations - self.controller.state.completed_iterations,
            )
            batch_size = min(self.max_workers, remaining_budget)
            if self.research_campaign == "architecture_coverage":
                batch_size = min(batch_size, len(self.campaign_remaining))
            if batch_size <= 0:
                break
            try:
                if batch_size > 1 and hasattr(self.planner, "propose_batch"):
                    decisions = self.planner.propose_batch(history, self.controller.state, count=batch_size)
                else:
                    decisions = [self.planner.propose(history, self.controller.state)]
            except ResearchCatalogueExhausted as exc:
                self.logger.log_action(
                    "research_catalogue_exhausted",
                    details={"reason": str(exc), "completed_hypotheses": len(history)},
                )
                self.controller.stop(str(exc))
                break
            except LLMPlanningError as exc:
                self.logger.log_action(
                    "llm_planning_self_correction_requested",
                    details={
                        "error": str(exc),
                        "recovery": "retry once with the validator error in planner context",
                    },
                )
                correction_context = dict(getattr(self.planner, "run_context", {}) or {})
                correction_context["planner_self_correction"] = {
                    "validation_error": str(exc),
                    "instruction": (
                        "Return a new strategy and hypotheses that satisfy this invariant. "
                        "Do not freeze the factor required by any assigned worker domain."
                    ),
                }
                set_context = getattr(self.planner, "set_run_context", None)
                if callable(set_context):
                    set_context(correction_context)
                try:
                    if batch_size > 1 and hasattr(self.planner, "propose_batch"):
                        decisions = self.planner.propose_batch(
                            history, self.controller.state, count=batch_size
                        )
                    else:
                        decisions = [self.planner.propose(history, self.controller.state)]
                except LLMPlanningError as retry_exc:
                    planner_metadata = dict(getattr(self.planner, "last_metadata", {}) or {})
                    self.research_tree.record_planner_batch(
                        planner_metadata,
                        incumbent_experiment_id=self.controller.state.current_best_experiment_id,
                    )
                    self.research_tree.refresh(self.controller.state)
                    self.logger.log_action(
                        "llm_planning_blocked",
                        details={
                            "error": str(retry_exc),
                            "initial_error": str(exc),
                            "usage": planner_metadata.get("usage", {}),
                            "planner_metadata": planner_metadata,
                            "recovery": "pause after one bounded self-correction attempt",
                        },
                    )
                    raise
            llm_metadata = getattr(self.planner, "last_metadata", None)
            if llm_metadata:
                self.logger.log_action("llm_hypothesis_generated", details=dict(llm_metadata))
                if isinstance(llm_metadata.get("research_strategy"), Mapping):
                    self.logger.record_research_strategy(
                        dict(llm_metadata["research_strategy"])
                    )
                self.research_tree.record_planner_batch(
                    llm_metadata,
                    incumbent_experiment_id=self.controller.state.current_best_experiment_id,
                )
            decisions = list(decisions)
            decision_signatures = {
                self._planning_decision_signature(item) for item in decisions
            }
            refill_attempts = 0
            while (
                sum(signature[0] == "experiment" for signature in decision_signatures) < batch_size
                and refill_attempts < 2
            ):
                refill_attempts += 1
                missing = batch_size - sum(
                    signature[0] == "experiment" for signature in decision_signatures
                )
                try:
                    backfill_toggle = getattr(self.planner, "set_backfill_mode", None)
                    if callable(backfill_toggle):
                        backfill_toggle(True)
                    if missing > 1 and hasattr(self.planner, "propose_batch"):
                        refill = self.planner.propose_batch(
                            history, self.controller.state, count=missing
                        )
                    else:
                        refill = [self.planner.propose(history, self.controller.state)]
                except (ResearchCatalogueExhausted, LLMPlanningError) as exc:
                    self.logger.log_action(
                        "planner_backfill_exhausted",
                        details={
                            "attempt": refill_attempts,
                            "missing_slots": missing,
                            "reason": str(exc),
                        },
                    )
                    break
                finally:
                    if callable(locals().get("backfill_toggle")):
                        backfill_toggle(False)
                refill_metadata = dict(getattr(self.planner, "last_metadata", {}) or {})
                if refill_metadata:
                    self.logger.log_action(
                        "llm_hypothesis_backfill_generated",
                        details=refill_metadata,
                    )
                    if isinstance(refill_metadata.get("research_strategy"), Mapping):
                        self.logger.record_research_strategy(
                            dict(refill_metadata["research_strategy"])
                        )
                    self.research_tree.record_planner_batch(
                        refill_metadata,
                        incumbent_experiment_id=self.controller.state.current_best_experiment_id,
                    )
                added = 0
                for item in refill:
                    if isinstance(item, ResearchDirection) and item.portfolio_role is None:
                        occupied = sum(
                            signature[0] == "experiment"
                            for signature in decision_signatures
                        )
                        item = replace(
                            item,
                            portfolio_role=(
                                "incumbent_exploit" if occupied == 0
                                else "independent_explore"
                            ),
                        )
                    signature = self._planning_decision_signature(item)
                    if signature in decision_signatures:
                        continue
                    decision_signatures.add(signature)
                    decisions.append(item)
                    added += 1
                self.logger.log_action(
                    "execution_slots_refilled",
                    details={
                        "attempt": refill_attempts,
                        "missing_slots_before": missing,
                        "new_distinct_decisions": added,
                        "runnable_decisions_after": sum(
                            signature[0] == "experiment" for signature in decision_signatures
                        ),
                    },
                )
                if added == 0:
                    break
            baseline_primary = self.controller.state.current_best_primary
            incumbent_metrics = self._incumbent_metrics(baseline_primary)
            reserved_ids = self._reserved_experiment_ids()
            runnable: list[tuple[ResearchDirection, Any, Any]] = []
            seen_decisions: set[tuple[Any, ...]] = set()
            pause_after_batch = False
            pending_pause_reasons: list[str] = []
            scheduled_diagnostics: list[tuple[CapabilityAction, Mapping[str, Any]]] = []
            has_planned_experiment = any(
                isinstance(item, ResearchDirection) for item in decisions
            )
            for decision in decisions:
                if isinstance(decision, CapabilityAction):
                    action_signature = self._planning_decision_signature(decision)
                    if action_signature in seen_decisions:
                        continue
                    seen_decisions.add(action_signature)
                    status = {
                        "RUN_DIAGNOSTIC": (
                            "scheduled_after_batch" if has_planned_experiment else "completed"
                        ),
                        "BUILD_CAPABILITY": "pending_implementation",
                        "REQUEST_HUMAN_APPROVAL": "pending_human_approval",
                    }[decision.action]
                    capability_is_implemented = (
                        decision.capability_gap_id in IMPLEMENTED_DIAGNOSTIC_CAPABILITIES
                    )
                    if decision.action == "BUILD_CAPABILITY" and capability_is_implemented:
                        status = "implemented_pending_measurement"
                    evidence = (
                        self._diagnostic_snapshot(history, decision)
                        if (
                            decision.action == "RUN_DIAGNOSTIC" and not has_planned_experiment
                        ) or capability_is_implemented
                        else {}
                    )
                    recorded = self.logger.record_capability_action(
                        decision.as_dict(), status=status, evidence=evidence
                    )
                    self.research_tree.record_capability_action(
                        decision,
                        recorded,
                        incumbent_experiment_id=self.controller.state.current_best_experiment_id,
                    )
                    self.research_tree.refresh(self.controller.state)
                    if decision.action == "RUN_DIAGNOSTIC":
                        if has_planned_experiment:
                            scheduled_diagnostics.append((decision, recorded))
                            self.logger.log_action(
                                "diagnostic_scheduled_after_batch",
                                details={
                                    "action_id": recorded["action_id"],
                                    "capability_gap_id": decision.capability_gap_id,
                                },
                            )
                        else:
                            self.logger.record_diagnostic({
                                "action_id": recorded["action_id"],
                                "capability_gap_id": decision.capability_gap_id,
                                "hypothesis": decision.hypothesis,
                                "evidence": evidence,
                            })
                    elif capability_is_implemented:
                        self.logger.log_action(
                            "capability_already_implemented",
                            details={
                                "action_id": recorded["action_id"],
                                "capability_gap_id": decision.capability_gap_id,
                                "measurement": "attached automatically to the next candidate",
                            },
                        )
                    else:
                        pending_pause_reasons.append(
                            f"{decision.action} pending for {decision.capability_gap_id}"
                        )
                        if decision.action == "REQUEST_HUMAN_APPROVAL":
                            pause_after_batch = True
                        self.logger.log_action(
                            "capability_backlogged",
                            details={
                                "action_id": recorded["action_id"],
                                "action": decision.action,
                                "reason": decision.rationale,
                                "blocks_unrelated_experiments": False,
                            },
                        )
                    continue
                direction = decision
                direction_signature = self._planning_decision_signature(direction)
                if direction_signature in seen_decisions:
                    self.logger.log_action(
                        "parallel_hypothesis_deduplicated",
                        details={"direction_id": direction.direction_id, "hypothesis": direction.hypothesis},
                    )
                    continue
                seen_decisions.add(direction_signature)
                self.logger.log_action("research_direction_proposed", details=direction.as_dict())
                search_state = self.regions.choose_search_state(direction, history, self.controller.state, review)
                try:
                    proposal = self.search.propose_trial(
                        direction,
                        self.controller.state,
                        history,
                        search_state=search_state,
                        reserved_experiment_ids=reserved_ids,
                        reserved_configs=[item[1].config for item in runnable],
                    )
                except ValueError as exc:
                    self.logger.log_action(
                        "parallel_hypothesis_unavailable",
                        details={
                            "direction_id": direction.direction_id,
                            "hypothesis": direction.hypothesis,
                            "reason": str(exc),
                        },
                    )
                    continue
                self.research_tree.bind_experiment(direction, proposal)
                reserved_ids.add(proposal.experiment_id)
                critic_result = self.critic.review(proposal, history, direction)
                self.logger.log_action(
                    "proposal_critic_reviewed",
                    experiment_id=proposal.experiment_id,
                    details={
                        "approved": critic_result.approved,
                        "reasons": list(critic_result.reasons),
                        "semantic_trace": dict(critic_result.trace),
                    },
                )
                self._record_critic_feedback(proposal.experiment_id, critic_result)
                if not critic_result.approved:
                    iteration = self.controller.record_critic_rejection(
                        proposal,
                        reasons=critic_result.reasons,
                        semantic_trace=critic_result.trace,
                    )
                    self.research_tree.record_outcome(
                        proposal.experiment_id,
                        incumbent_experiment_id=self.controller.state.current_best_experiment_id,
                    )
                    self.research_tree.refresh(self.controller.state)
                    refresh_critic_memory(self.logger.store)
                    results.append(CycleResult(direction, iteration, None))
                    continue
                runnable.append((direction, proposal, critic_result))

            if not runnable:
                # An action-only batch must yield instead of repeatedly asking
                # the LLM for the same unsupported capability in one invocation.
                if self.pause_reason is None:
                    self.pause_reason = (
                        pending_pause_reasons[0]
                        if pending_pause_reasons
                        else "diagnostic completed; planner must review the new evidence"
                    )
                self.logger.log_action(
                    "research_pause_requested",
                    details={
                        "reason": self.pause_reason,
                        "persistent_stop": False,
                        "runnable_experiments": 0,
                    },
                )
                break

            self.logger.log_action(
                "parallel_batch_started",
                details={
                    "planned_workers": len(runnable),
                    "max_workers": self.max_workers,
                    "worker_threads": [item[1].config.get("worker_threads", 1) for item in runnable],
                    "baseline_primary": baseline_primary,
                },
            )
            outcomes: list[_WorkerOutcome] = []
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, max(1, len(runnable))),
                thread_name_prefix="experiment-worker",
            ) as executor:
                futures = {
                    executor.submit(
                        self._execute_worker,
                        direction,
                        proposal,
                        critic_result,
                        baseline_primary,
                        incumbent_metrics,
                        slot,
                    ): proposal.experiment_id
                    for slot, (direction, proposal, critic_result) in enumerate(runnable, start=1)
                }
                for future in as_completed(futures):
                    outcomes.append(future.result())

            outcomes.sort(key=self._robust_outcome_primary, reverse=True)
            for outcome in outcomes:
                decision_baseline = self.controller.state.current_best_primary
                iteration = self.controller.complete_staged_experiment(
                    outcome.proposal,
                    outcome.stage_results,
                    baseline_primary=baseline_primary,
                    acceptance_primary=decision_baseline,
                    terminal_reason=outcome.terminal_reason,
                    semantic_review=outcome.semantic_review,
                )
                self.research_tree.record_outcome(
                    outcome.proposal.experiment_id,
                    incumbent_experiment_id=self.controller.state.current_best_experiment_id,
                )
                self.research_tree.refresh(self.controller.state)
                results.append(CycleResult(outcome.direction, iteration, outcome.promoted))
            self.logger.log_action(
                "parallel_batch_completed",
                details={
                    "workers_completed": len(outcomes),
                    "reconciliation_order": [item.proposal.experiment_id for item in outcomes],
                    "incumbent_after": self.controller.state.current_best_experiment_id,
                },
            )
            for diagnostic_action, scheduled_record in scheduled_diagnostics:
                diagnostic_evidence = self._diagnostic_snapshot(
                    self.logger.store.read_iterations(), diagnostic_action
                )
                completed_record = self.logger.record_capability_action(
                    diagnostic_action.as_dict(),
                    status="completed",
                    evidence={
                        **diagnostic_evidence,
                        "scheduled_action_id": scheduled_record.get("action_id"),
                    },
                )
                self.logger.record_diagnostic({
                    "action_id": completed_record["action_id"],
                    "scheduled_action_id": scheduled_record.get("action_id"),
                    "capability_gap_id": diagnostic_action.capability_gap_id,
                    "hypothesis": diagnostic_action.hypothesis,
                    "evidence": diagnostic_evidence,
                })
                self.research_tree.record_capability_action(
                    diagnostic_action,
                    completed_record,
                    incumbent_experiment_id=self.controller.state.current_best_experiment_id,
                )
                self.logger.log_action(
                    "diagnostic_completed_after_batch",
                    details={
                        "action_id": completed_record["action_id"],
                        "scheduled_action_id": scheduled_record.get("action_id"),
                    },
                )
            if scheduled_diagnostics:
                self.research_tree.refresh(self.controller.state)
            refresh_critic_memory(self.logger.store)
            if pause_after_batch:
                self.pause_reason = next(
                    (
                        reason for reason in pending_pause_reasons
                        if reason.startswith("REQUEST_HUMAN_APPROVAL")
                    ),
                    "human approval required",
                )
                break
        return results

    def _robust_outcome_primary(self, outcome: _WorkerOutcome) -> float:
        """Order concurrent commits by the same evidence used for acceptance."""
        stage_records = [
            item for item in self.logger.store.read_stages()
            if item.get("experiment_id") == outcome.proposal.experiment_id
        ]
        aggregate, _ = robust_stage_evidence(
            stage_records, self.controller.contract.metric_names
        )
        if aggregate is None:
            return float("-inf")
        return float(aggregate[self.controller.contract.primary_metric])

    @staticmethod
    def _planning_decision_signature(decision: ResearchDirection | CapabilityAction) -> tuple[Any, ...]:
        if isinstance(decision, CapabilityAction):
            return ("capability", decision.action, decision.capability_gap_id)
        return (
            "experiment",
            decision.direction_id,
            decision.hypothesis,
            decision.preferred_factor,
            str(decision.preferred_value),
        )

    def _diagnostic_snapshot(
        self,
        history: Sequence[Mapping[str, Any]],
        action: CapabilityAction,
    ) -> dict[str, Any]:
        """Summarize existing validation evidence without training or test access."""
        measured = [
            item for item in history
            if isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]
        by_direction: dict[str, dict[str, Any]] = {}
        for item in measured:
            direction_id = str(item.get("direction_id") or "unassigned")
            primary = float(item["metrics"]["primary"])
            summary = by_direction.setdefault(direction_id, {"attempts": 0, "best_primary": None})
            summary["attempts"] += 1
            summary["best_primary"] = (
                primary if summary["best_primary"] is None
                else max(float(summary["best_primary"]), primary)
            )
        incumbent = next(
            (
                item for item in reversed(history)
                if item.get("experiment_id") == self.controller.state.current_best_experiment_id
            ),
            None,
        )
        return {
            "selection_split": self.controller.contract.selection_split,
            "test_data_used": False,
            "incumbent_experiment_id": self.controller.state.current_best_experiment_id,
            "incumbent_metrics": dict((incumbent or {}).get("metrics", {})),
            "incumbent_stratified_validation": next(
                (
                    diagnostic.get("stratified_validation", {})
                    for diagnostic in (incumbent or {}).get("diagnostic_evidence", {}).get(
                        "model_diagnostics", []
                    )
                    if diagnostic.get("stratified_validation")
                ),
                {},
            ),
            "measured_experiment_count": len(measured),
            "unmeasured_experiment_count": len(history) - len(measured),
            "direction_summary": by_direction,
            "requested_capabilities": list(action.required_capabilities),
            "implemented_capabilities": dict(IMPLEMENTED_DIAGNOSTIC_CAPABILITIES),
        }

    def _execute_worker(
        self,
        direction: ResearchDirection,
        proposal: Any,
        critic_result: Any,
        baseline_primary: float,
        incumbent_metrics: Mapping[str, Any],
        worker_slot: int,
    ) -> _WorkerOutcome:
        self.logger.log_action(
            "parallel_worker_started",
            experiment_id=proposal.experiment_id,
            details={
                "worker_slot": worker_slot,
                "worker_threads": proposal.config.get("worker_threads", 1),
                "portfolio_role": proposal.portfolio_role,
            },
        )
        first_stage = self.controller.run_iteration(
            proposal,
            self.candidate,
            stage_id=str(proposal.config.get("fidelity", "low")),
            complete_experiment=False,
        )
        stage_results, promoted, terminal_reason = self._maybe_promote(direction, proposal, first_stage)
        stage_records = [
            item for item in self.logger.store.read_stages()
            if item.get("experiment_id") == proposal.experiment_id
        ]
        evidence_critic = self.critic.review_evidence(
            proposal,
            stage_records,
            critic_result.trace,
            baseline_primary=baseline_primary,
            baseline_metrics=incumbent_metrics,
        )
        semantic_review = {
            "approved": evidence_critic.approved,
            "reasons": list(evidence_critic.reasons),
            "trace": dict(evidence_critic.trace),
        }
        agentic_metadata = evidence_critic.trace.get("agentic_review", {}).get("metadata", {})
        self.logger.log_action(
            "semantic_evidence_critic_reviewed",
            experiment_id=proposal.experiment_id,
            details={**semantic_review, "usage": dict(agentic_metadata.get("usage", {}))},
        )
        self._record_critic_feedback(proposal.experiment_id, evidence_critic)
        self.logger.log_action(
            "parallel_worker_completed",
            experiment_id=proposal.experiment_id,
            details={
                "worker_slot": worker_slot,
                "terminal_reason": terminal_reason,
                "portfolio_role": proposal.portfolio_role,
            },
        )
        return _WorkerOutcome(
            direction, proposal, stage_results, promoted, terminal_reason, semantic_review
        )

    def _confirm_seeds(self, proposal, iteration: IterationResult) -> None:
        """Repeat accepted candidates to distinguish signal from seed noise."""
        primaries: list[float] = []
        for seed in self.fidelity.DEFAULT_CONFIRMATION_SEEDS:
            stage_id = f"seed_{seed}"
            config = dict(proposal.config)
            config["seed"] = seed
            self.logger.log_action(
                "seed_confirmation_started",
                experiment_id=proposal.experiment_id,
                details={"stage_id": stage_id, "seed": seed},
            )
            result = self.controller.runner.run(
                experiment_id=proposal.experiment_id,
                hypothesis=f"Seed confirmation for {proposal.experiment_id}",
                config=config,
                candidate=self.candidate,
                timeout_seconds=proposal.runtime_budget_seconds,
                stage_id=stage_id,
            )
            if result.status != "completed" or result.output is None:
                self.logger.log_action(
                    "seed_confirmation_failed",
                    experiment_id=proposal.experiment_id,
                    details={"stage_id": stage_id, "seed": seed, "error": result.error or result.status},
                )
                self.logger.record_stage({
                    "experiment_id": proposal.experiment_id,
                    "stage_id": stage_id,
                    "status": result.status,
                    "config": config,
                    "metrics": {},
                    "runtime_seconds": result.runtime_seconds,
                    "error": result.error,
                })
                continue
            try:
                metrics = evaluate_predictions(
                    result.output.user_ids,
                    result.output.labels,
                    result.output.scores,
                    split=proposal.selection_split,
                ).as_dict()
            except MetricsValidationError as exc:
                self.logger.log_action(
                    "seed_confirmation_failed",
                    experiment_id=proposal.experiment_id,
                    details={"stage_id": stage_id, "seed": seed, "error": str(exc)},
                )
                continue
            primaries.append(float(metrics["primary"]))
            self.logger.record_stage({
                "experiment_id": proposal.experiment_id,
                "stage_id": stage_id,
                "status": "completed",
                "config": config,
                "metrics": metrics,
                "runtime_seconds": result.runtime_seconds,
                "runner_metadata": dict(result.output.metadata),
                "resource_usage": dict(result.resource_usage),
                "safety": {"passed": True, "violations": []},
            })
            self.logger.log_action(
                "seed_confirmation_completed",
                experiment_id=proposal.experiment_id,
                details={"stage_id": stage_id, "seed": seed, **metrics},
            )
        if primaries:
            self.logger.log_action(
                "seed_confirmation_summary",
                experiment_id=proposal.experiment_id,
                details={"seeds": list(self.fidelity.DEFAULT_CONFIRMATION_SEEDS), "mean_primary": mean(primaries), "std_primary": pstdev(primaries)},
            )

    def _maybe_promote(
        self,
        direction: ResearchDirection,
        proposal,
        iteration: IterationResult,
    ) -> tuple[list[IterationResult], IterationResult | None, str]:
        current_proposal = proposal
        current_iteration = iteration
        last_promotion: IterationResult | None = None
        stage_results = [iteration]
        terminal_reason = "stage_failed"
        while current_iteration.metrics:
            fidelity = str(current_proposal.config.get("fidelity", "low"))
            if self.fidelity.next_fidelity(fidelity) is None:
                terminal_reason = "full_fidelity_completed"
                if self.fidelity.is_promising(current_iteration.metrics, self.controller.state.current_best_primary):
                    self._confirm_seeds(current_proposal, current_iteration)
                break
            if not self.fidelity.should_promote(
                current_iteration.metrics,
                self.controller.state.current_best_primary,
                fidelity=fidelity,
            ):
                self.logger.log_action(
                    "fidelity_pruned",
                    experiment_id=current_proposal.experiment_id,
                    details={
                        "fidelity": fidelity,
                        "primary": current_iteration.metrics.get("primary"),
                        "incumbent_primary": self.controller.state.current_best_primary,
                    },
                )
                terminal_reason = f"pruned_at_{fidelity}"
                break
            history = self.logger.store.read_iterations()
            promoted = self.fidelity.promote(current_proposal, direction)
            critic_result = self.critic.review(promoted, history, direction)
            self.logger.log_action(
                "promotion_critic_reviewed",
                experiment_id=promoted.experiment_id,
                details={
                    "approved": critic_result.approved,
                    "reasons": list(critic_result.reasons),
                    "semantic_trace": dict(critic_result.trace),
                    "from_fidelity": fidelity,
                    "to_fidelity": promoted.config["fidelity"],
                },
            )
            if not critic_result.approved:
                self.logger.log_action(
                    "fidelity_pruned",
                    experiment_id=promoted.experiment_id,
                    details={"fidelity": promoted.config["fidelity"], "reason": list(critic_result.reasons)},
                )
                terminal_reason = f"promotion_rejected_after_{fidelity}"
                break
            current_iteration = self.controller.run_iteration(
                promoted,
                self.candidate,
                stage_id=str(promoted.config["fidelity"]),
                complete_experiment=False,
            )
            stage_results.append(current_iteration)
            current_proposal = promoted
            last_promotion = current_iteration
        return stage_results, last_promotion, terminal_reason

    def _reserved_experiment_ids(self) -> set[str]:
        """Reserve IDs from every event, including non-iteration seed runs."""
        return {
            str(event["experiment_id"])
            for event in self.logger.store.read_events()
            if event.get("experiment_id")
        }

    def _incumbent_metrics(self, baseline_primary: float) -> Mapping[str, Any]:
        incumbent_id = self.controller.state.current_best_experiment_id
        if incumbent_id == "baseline" and abs(baseline_primary - 0.6015) < 1e-9:
            return {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015}
        for record in reversed(self.logger.store.read_iterations()):
            if record.get("experiment_id") == incumbent_id:
                return dict(record.get("metrics", {}))
        return {"primary": baseline_primary}

    def _prepare_planner_context(self, history: Sequence[Mapping[str, Any]]) -> None:
        tree_context = self.research_tree.planner_context(self.controller.state)
        coverage_context = build_research_coverage(self.logger.store, history)
        self.campaign_remaining = [
            str(item.get("mechanism"))
            for item in coverage_context.get("architectures", [])
            if item.get("availability") == "executable"
            and item.get("evidence_scope") not in {"isolated", "isolated_tested"}
        ]
        critic_context_setter = getattr(self.critic, "set_research_context", None)
        if callable(critic_context_setter):
            critic_context_setter(tree_context)
        setter = getattr(self.planner, "set_run_context", None)
        if not callable(setter):
            return
        architecture_path = Path(__file__).resolve().parent.parent / "docs" / "agent-architecture.md"
        architecture_sha256 = hashlib.sha256(architecture_path.read_bytes()).hexdigest()
        interventions = self.logger.store.read_interventions()[-6:]
        recovery_actions = {
            "failed", "timed_out", "safety_rejected", "accepted_candidate_restored",
            "stale_active_experiment_recovered", "demo_failure_recovery_completed",
            "llm_planning_blocked", "llm_planning_failed",
            "interrupted_worker_recovered",
        }
        recovery_events = [
            {
                "timestamp": event.get("timestamp"),
                "action": event.get("action"),
                "experiment_id": event.get("experiment_id"),
                "details": {
                    key: event.get("details", {}).get(key)
                    for key in ("error", "reason", "recovery", "current_best_experiment_id")
                    if key in event.get("details", {})
                },
            }
            for event in self.logger.store.read_events()
            if event.get("action") in recovery_actions
        ][-6:]
        contract = self.controller.contract
        context = {
            "benchmark_contract": {
                "label": contract.label,
                "training_split": contract.train_split,
                "selection_split": contract.selection_split,
                "test_split": contract.test_split,
                "metrics": list(contract.metric_names),
                "acceptance_rule": "accept any positive primary improvement",
                "convergence_rule": (
                    f"restart after {contract.non_improvement_limit} hypotheses without "
                    f"an improvement greater than {contract.improvement_threshold}"
                ),
                "external_data_permitted": False,
            },
            "architecture_guidance": {
                "path": "docs/agent-architecture.md",
                "sha256": architecture_sha256,
                "operating_boundary": (
                    "LLM chooses a falsifiable scientific direction; deterministic search chooses exact "
                    "configurations; safety and semantic critics gate execution and acceptance."
                ),
            },
            "manual_interventions": interventions,
            "recent_errors_and_recoveries": recovery_events,
            "capability_backlog": self._compact_capability_backlog(),
            "recent_diagnostics": self._compact_diagnostics(),
            "implemented_diagnostic_capabilities": dict(
                IMPLEMENTED_DIAGNOSTIC_CAPABILITIES
            ),
            "research_tree": tree_context,
            "research_coverage": coverage_context,
            "research_campaign": (
                {
                    "type": "architecture_coverage",
                    "goal": "Obtain one controlled baseline for every executable architecture mechanism.",
                    "remaining_mechanisms": self.campaign_remaining,
                    "rules": [
                        "Select exactly one remaining untested architecture mechanism as a single-path baseline; do not combine paths.",
                        "Keep features, objective, optimizer, regularization, embedding dimension, and batch size fixed.",
                        "Do not spend experiments refining an already tested architecture until coverage is complete.",
                    ],
                    "mechanism_claim_constraints": {
                        "bi_interaction_mlp": (
                            "FM already models scalar pairwise interactions. Claim only that the bi-interaction path "
                            "retains the elementwise interaction vector and applies a nonlinear transformation."
                        ),
                    },
                }
                if self.research_campaign == "architecture_coverage"
                else {
                    "type": "multi_task_baseline",
                    "goal": "Run one controlled multi-task click-supervision experiment from the accepted incumbent.",
                    "rules": [
                        "Select multi_task_learning and change only training_objective.",
                        "Keep the accepted architecture, features, pointwise primary loss, optimizer, regularization, embedding dimension, and batch size fixed.",
                        "Claim only training-time shared-embedding supervision; click is not an inference feature or acceptance metric.",
                    ],
                }
                if self.research_campaign == "multi_task_baseline" else None
            ),
            "critic_memory": refresh_critic_memory(self.logger.store),
            "recent_llm_research_strategies": self.logger.store.read_research_strategies()[-4:],
            "metric_history_count": len(history),
        }
        setter(context)
        self.logger.log_action(
            "planning_context_prepared",
            details={
                "architecture_sha256": architecture_sha256,
                "manual_intervention_count": len(interventions),
                "recovery_event_count": len(recovery_events),
                "completed_hypothesis_count": len(history),
                "capability_backlog_count": len(self.logger.store.read_capability_actions()),
                "research_tree_hypothesis_count": context["research_tree"]["hypothesis_count"],
                "coverage_validated_configuration_count": coverage_context[
                    "validated_configuration_count"
                ],
                "critic_feedback_count": context["critic_memory"]["feedback_count"],
                "prior_research_strategy_count": len(
                    context["recent_llm_research_strategies"]
                ),
            },
        )

    def _compact_capability_backlog(self) -> list[dict[str, Any]]:
        return [
            {
                "action_id": item.get("action_id"),
                "action": item.get("action"),
                "status": item.get("status"),
                "capability_gap_id": item.get("capability_gap_id"),
                "capability_gap_description": item.get("capability_gap_description"),
                "approval_reason": item.get("approval_reason"),
                "incumbent_experiment_id": item.get("evidence", {}).get(
                    "incumbent_experiment_id"
                ),
                "weakest_stratum": item.get("evidence", {}).get(
                    "incumbent_stratified_validation", {}
                ).get("weakest_statistically_eligible_stratum"),
            }
            for item in self.logger.store.read_capability_actions()[-8:]
        ]

    def _compact_diagnostics(self) -> list[dict[str, Any]]:
        return [
            {
                "action_id": item.get("action_id"),
                "capability_gap_id": item.get("capability_gap_id"),
                "hypothesis": item.get("hypothesis"),
                "selection_split": item.get("evidence", {}).get("selection_split"),
                "test_data_used": item.get("evidence", {}).get("test_data_used"),
                "incumbent_experiment_id": item.get("evidence", {}).get(
                    "incumbent_experiment_id"
                ),
                "measured_experiment_count": item.get("evidence", {}).get(
                    "measured_experiment_count"
                ),
                "weakest_stratum": item.get("evidence", {}).get(
                    "incumbent_stratified_validation", {}
                ).get("weakest_statistically_eligible_stratum"),
            }
            for item in self.logger.store.read_diagnostics()[-4:]
        ]

    def _record_critic_feedback(self, experiment_id: str, result: Any) -> None:
        feedback = result.trace.get("planner_feedback", {}) if result.trace else {}
        if feedback:
            self.logger.record_critic_feedback({
                "experiment_id": experiment_id,
                **dict(feedback),
                "approved": result.approved,
            })
