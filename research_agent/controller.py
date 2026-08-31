"""Deterministic lifecycle controller for safe research iterations."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import threading
from typing import Any, Mapping, Sequence

from .contracts import BENCHMARK_CONTRACT, BenchmarkContract
from .evidence import (
    CONFIRMATION_SEEDS,
    architecture_ablation_evidence,
    build_experiment_evidence,
    refresh_evidence_memory,
    robust_stage_evidence,
)
from .logger import ResearchLogger
from .metrics import MetricsValidationError, evaluate_predictions
from .runner import CandidateCallable, ExperimentRunner, RunnerResult
from .safety import (
    ExperimentProposal,
    SafetyReport,
    SafetyValidator,
    measured_historical_configs,
)
from .state import ResearchState


_BASELINE_CONFIG = {
    "loss": "pointwise",
    "training_objective": "pointwise",
    "learning_rate": 0.001,
    "l2": 1e-6,
    "embedding_dim": 16,
    "batch_size": 8192,
    "seed": 0,
    "feature_variant": "baseline",
    "architecture": "fm",
    "worker_threads": 2,
}

@dataclass(frozen=True)
class IterationResult:
    experiment_id: str
    decision: str
    state: ResearchState
    runner_result: RunnerResult | None = None
    metrics: Mapping[str, Any] | None = None
    error: str | None = None
    stage_id: str | None = None


class ExperimentController:
    """Coordinates one controlled experiment without changing the baseline."""

    def __init__(
        self,
        *,
        logger: ResearchLogger,
        runner: ExperimentRunner,
        validator: SafetyValidator,
        contract: BenchmarkContract = BENCHMARK_CONTRACT,
        state: ResearchState | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.logger = logger
        self.runner = runner
        self.validator = validator
        self.contract = contract
        persisted_state = logger.store.read_root_json("state.json")
        self.state = state or (ResearchState.from_dict(persisted_state) if persisted_state else ResearchState())
        self.max_iterations = max_iterations or contract.max_experiments
        self._stage_lock = threading.Lock()
        if (
            self.state.stop_reason
            and self.state.stop_reason.startswith("configured experiment budget reached:")
            and self.state.completed_iterations < self.max_iterations
        ):
            previous_reason = self.state.stop_reason
            self.state.stop_reason = None
            self.logger.log_action(
                "configured_budget_extended",
                details={
                    "previous_stop_reason": previous_reason,
                    "persistent_experiment_cap": self.max_iterations,
                    "completed_iterations": self.state.completed_iterations,
                },
            )
        if self.state.stop_reason == "the approved executable research catalogue is exhausted":
            self.state.stop_reason = None
            self.logger.log_action(
                "research_catalogue_recheck_requested",
                details={"reason": "failed and interrupted attempts do not consume executable configurations"},
            )
        if self.state.active_experiment_id:
            stale_id = self.state.active_experiment_id
            self.state.active_experiment_id = None
            self.logger.log_action(
                "stale_active_experiment_recovered",
                experiment_id=stale_id,
                details={"recovery": "cleared stale active pointer; candidate will be reconsidered by the next loop"},
            )
        self._recover_interrupted_experiments()
        self._historical_configs = measured_historical_configs(
            (*logger.store.read_iterations(), *logger.store.read_stages())
        )
        self._inflight_configs: dict[str, Mapping[str, Any]] = {}
        self._reconcile_robust_incumbent()
        self._persist_state()

    def _reconcile_robust_incumbent(self) -> None:
        """Migrate older seed-0 incumbents to the strongest three-seed mean."""
        candidates: list[tuple[float, str]] = []
        for record in self.logger.store.read_iterations():
            if record.get("decision") != "accepted":
                continue
            stages = record.get("diagnostic_evidence", {}).get("fidelity_results", [])
            if not isinstance(stages, list):
                continue
            pseudo_records = [
                {"config": {"fidelity": item.get("fidelity"), "seed": item.get("seed", 0)},
                 "metrics": item.get("metrics", {})}
                for item in stages if isinstance(item, Mapping)
            ]
            aggregate, _ = robust_stage_evidence(pseudo_records, self.contract.metric_names)
            if aggregate is not None:
                candidates.append((float(aggregate[self.contract.primary_metric]), str(record["experiment_id"])))
        if not candidates:
            return
        robust_primary, robust_id = max(candidates)
        if (
            robust_id != self.state.current_best_experiment_id
            or abs(robust_primary - self.state.current_best_primary) > 1e-12
        ):
            previous = {
                "experiment_id": self.state.current_best_experiment_id,
                "primary": self.state.current_best_primary,
            }
            self.state.current_best_experiment_id = robust_id
            self.state.current_best_primary = robust_primary
            self.logger.log_action(
                "robust_incumbent_reconciled",
                experiment_id=robust_id,
                details={
                    "previous": previous,
                    "selection_rule": "mean validation primary across full-fidelity seeds 0, 1, and 2",
                    "robust_primary": robust_primary,
                },
            )

    def _recover_interrupted_experiments(self) -> None:
        """Convert orphaned run directories into explicit failed iterations."""
        completed_ids = {
            str(record.get("experiment_id")) for record in self.logger.store.read_iterations()
        }
        stages = self.logger.store.read_stages()
        events = self.logger.store.read_events()
        implementation_directions = {
            "torch_fm.pointwise_optimizer": "pointwise_fm_optimization",
            "torch_fm.within_user_pairwise_logistic": "pairwise_fm_ranking",
            "data.train_only_author_affinity": "leakage_safe_author_affinity",
            "data.train_only_user_history": "leakage_safe_user_history",
            "data.calendar_weekday_field": "weekday_features",
            "torch_fm.reviewed_architecture_compiler": "fm_architecture",
            "torch_fm.multitask_click_shared_embeddings": "multi_task_learning",
        }
        recovered = 0
        run_ids = {
            path.name for path in self.logger.store.runs_dir.iterdir() if path.is_dir()
        } if self.logger.store.runs_dir.exists() else set()
        orphan_ids = run_ids | {
            str(item.get("experiment_id")) for item in stages if item.get("experiment_id")
        }
        for experiment_id in sorted(orphan_ids - completed_ids):
            run_directory = self.logger.store.runs_dir / experiment_id
            experiment_stages = [
                item for item in stages if item.get("experiment_id") == experiment_id
            ]
            latest_stage = experiment_stages[-1] if experiment_stages else {}
            config: dict[str, Any] = dict(latest_stage.get("config", {}))
            hypothesis = str(
                latest_stage.get("hypothesis", "Interrupted experiment with unavailable hypothesis metadata.")
            )
            for config_path in sorted(run_directory.glob("*/config.json")):
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    break
                except (OSError, json.JSONDecodeError):
                    continue
            for plan_path in sorted(run_directory.glob("*/plan.json")):
                try:
                    hypothesis = str(
                        json.loads(plan_path.read_text(encoding="utf-8")).get("hypothesis", hypothesis)
                    )
                    break
                except (OSError, json.JSONDecodeError):
                    continue
            critic_event = next(
                (
                    event for event in reversed(events)
                    if event.get("experiment_id") == experiment_id
                    and event.get("action") == "proposal_critic_reviewed"
                ),
                {},
            )
            trace = critic_event.get("details", {}).get("semantic_trace", {}) or {}
            direction_id = latest_stage.get("direction_id") or implementation_directions.get(
                trace.get("implementation_id")
            )
            changed_factors = list(latest_stage.get("changed_factors", [])) or list(
                (trace.get("configuration_diff", {}) or {}).keys()
            )
            patch_path = self.logger.store.patches_dir / f"{experiment_id}.patch"
            error = "experiment interrupted before lifecycle reconciliation"
            self.state.completed_iterations += 1
            self.state.consecutive_non_improvements += 1
            record = {
                "experiment_id": experiment_id,
                "parent_experiment_id": self.state.current_best_experiment_id,
                "hypothesis": hypothesis,
                "rationale": "Recovered an interrupted isolated run without fabricating metrics.",
                "config": config,
                "changed_factors": changed_factors,
                "direction_id": direction_id,
                "search_strategy": "interrupted_recovery",
                "search_region_id": "recovered",
                "portfolio_role": latest_stage.get("portfolio_role"),
                "metrics": {},
                "delta_primary": None,
                "runtime_seconds": sum(float(item.get("runtime_seconds") or 0.0) for item in experiment_stages),
                "runner_metadata": {},
                "resource_usage": {},
                "decision": "failed",
                "error": error,
                "recovery": self._restore_message(),
                "safety": {"passed": False, "violations": [error]},
                "code_diff_path": str(patch_path) if patch_path.is_file() else "",
                "terminal_reason": "interrupted",
                "diagnostic_evidence": {"failures": [{"category": "interrupted", "error": error}]},
                "semantic_review": {
                    "approved": False,
                    "reasons": [error],
                    "trace": {"verdict": "incomplete"},
                },
                "stages": [
                    {
                        "stage_id": item.get("stage_id"),
                        "status": item.get("status"),
                        "metrics": item.get("metrics", {}),
                        "runtime_seconds": item.get("runtime_seconds"),
                    }
                    for item in experiment_stages
                ],
                "state_after": self.state.as_dict(),
            }
            self.logger.record_iteration(record)
            self.logger.log_action(
                "interrupted_experiment_recovered",
                experiment_id=experiment_id,
                details={"recovery": self._restore_message(), "stage_count": len(experiment_stages)},
            )
            recovered += 1
        if recovered:
            self._apply_stop_rule()
            self._persist_state()

    def run_iteration(
        self,
        proposal: ExperimentProposal,
        candidate: CandidateCallable,
        *,
        code_diff: str = "",
        stage_id: str | None = None,
        complete_experiment: bool = True,
    ) -> IterationResult:
        if proposal.runtime_budget_seconds > self.validator.max_runtime_seconds:
            proposal = replace(
                proposal,
                runtime_budget_seconds=self.validator.max_runtime_seconds,
            )
        if self.state.stopped:
            message = f"controller stopped: {self.state.stop_reason}"
            self.logger.log_action(
                "iteration_skipped_stopped",
                experiment_id=proposal.experiment_id,
                details={"reason": self.state.stop_reason},
            )
            return IterationResult(proposal.experiment_id, "skipped", self.state, error=message)

        self.logger.log_action(
            "hypothesis_selected",
            experiment_id=proposal.experiment_id,
            details={"hypothesis": proposal.hypothesis, "rationale": proposal.rationale},
        )
        parent_experiment_id = proposal.parent_experiment_id or self.state.current_best_experiment_id
        parent_config = self._parent_config(parent_experiment_id)
        if parent_experiment_id == "baseline" and not set(_BASELINE_CONFIG) <= set(proposal.config):
            parent_config = None
        existing_patch = self.logger.store.patches_dir / f"{proposal.experiment_id}.patch"
        if stage_id and stage_id != "low" and existing_patch.exists():
            patch_path = existing_patch
            self.logger.log_action(
                "code_diff_reused_for_stage",
                experiment_id=proposal.experiment_id,
                details={"stage_id": stage_id, "patch_path": str(patch_path)},
            )
        else:
            patch_path = self.logger.record_code_diff(
                proposal.experiment_id,
                code_diff or self._configuration_diff(proposal, parent_config),
            )
        with self._stage_lock:
            report = self.validator.validate(
                proposal,
                historical_configs=[
                    *self._historical_configs,
                    *self._inflight_configs.values(),
                ],
                parent_config=parent_config,
            )
            if report.passed:
                self._inflight_configs[proposal.config_fingerprint()] = proposal.config
        if not report.passed:
            self.logger.log_action(
                "safety_rejected",
                experiment_id=proposal.experiment_id,
                details={"violations": list(report.violations)},
            )
            return self._finish_without_run(
                proposal,
                decision="rejected",
                error="; ".join(report.violations),
                recovery=self._restore_message(),
                safety_report=report,
                patch_path=str(patch_path),
                parent_experiment_id=parent_experiment_id,
                stage_id=stage_id,
                complete_experiment=complete_experiment,
            )

        self.logger.log_action("safety_passed", experiment_id=proposal.experiment_id)
        self.logger.log_action(
            "configuration_reserved",
            experiment_id=proposal.experiment_id,
            details={"scope": "in_flight_only", "retryable_without_metrics": True},
        )
        if complete_experiment:
            self.state.active_experiment_id = proposal.experiment_id
            self._persist_state()
        runner_result = self.runner.run(
            experiment_id=proposal.experiment_id,
            hypothesis=proposal.hypothesis,
            config=proposal.config,
            candidate=candidate,
            timeout_seconds=proposal.runtime_budget_seconds,
            stage_id=stage_id,
        )
        if runner_result.status != "completed" or runner_result.output is None:
            self._finish_configuration_reservation(proposal, measured=False)
            return self._finish_without_run(
                proposal,
                decision="failed",
                error=runner_result.error or f"runner status: {runner_result.status}",
                recovery=self._restore_message(),
                safety_report=report,
                patch_path=str(patch_path),
                runtime_seconds=runner_result.runtime_seconds,
                runner_result=runner_result,
                parent_experiment_id=parent_experiment_id,
                stage_id=stage_id,
                complete_experiment=complete_experiment,
            )

        try:
            self.logger.log_action("evaluation_started", experiment_id=proposal.experiment_id)
            metric_result = evaluate_predictions(
                runner_result.output.user_ids,
                runner_result.output.labels,
                runner_result.output.scores,
                split=proposal.selection_split,
            )
            metrics = metric_result.as_dict()
            self.logger.log_action(
                "metrics_received",
                experiment_id=proposal.experiment_id,
                details={name: metrics[name] for name in self.contract.metric_names},
            )
        except MetricsValidationError as exc:
            self._finish_configuration_reservation(proposal, measured=False)
            return self._finish_without_run(
                proposal,
                decision="failed",
                error=f"metrics validation failed: {exc}",
                recovery=self._restore_message(),
                safety_report=report,
                patch_path=str(patch_path),
                runtime_seconds=runner_result.runtime_seconds,
                runner_result=runner_result,
                parent_experiment_id=parent_experiment_id,
                stage_id=stage_id,
                complete_experiment=complete_experiment,
            )

        delta = float(metrics[self.contract.primary_metric]) - self.state.current_best_primary
        self._finish_configuration_reservation(proposal, measured=True)
        if delta > 0:
            decision = "accepted"
            if complete_experiment:
                self.state.current_best_experiment_id = proposal.experiment_id
                self.state.current_best_primary = float(metrics[self.contract.primary_metric])
            recovery = None
            self.logger.log_action(
                "candidate_accepted",
                experiment_id=proposal.experiment_id,
                details={"delta_primary": delta},
            )
        else:
            decision = "inconclusive" if delta > 0 else "rejected"
            recovery = self._restore_message()
            self.logger.log_action(
                "candidate_not_promoted",
                experiment_id=proposal.experiment_id,
                details={"decision": decision, "delta_primary": delta},
            )
        return self._complete_iteration(
            proposal,
            decision=decision,
            metrics=metrics,
            delta_primary=delta,
            runtime_seconds=runner_result.runtime_seconds,
            error=None,
            recovery=recovery,
            safety_report=report,
            patch_path=str(patch_path),
            runner_result=runner_result,
            parent_experiment_id=parent_experiment_id,
            stage_id=stage_id,
            complete_experiment=complete_experiment,
        )

    def _finish_configuration_reservation(
        self,
        proposal: ExperimentProposal,
        *,
        measured: bool,
    ) -> None:
        """Release a live reservation and persist only measured configurations."""
        fingerprint = proposal.config_fingerprint()
        with self._stage_lock:
            self._inflight_configs.pop(fingerprint, None)
            if measured and all(
                SafetyValidator._fingerprint(config) != fingerprint
                for config in self._historical_configs
            ):
                self._historical_configs.append(proposal.config)
        self.logger.log_action(
            "configuration_committed_measured" if measured else "configuration_released_retryable",
            experiment_id=proposal.experiment_id,
            details={
                "measured_validation_evidence": measured,
                "retryable": not measured,
            },
        )

    def record_critic_rejection(
        self,
        proposal: ExperimentProposal,
        *,
        reasons: Sequence[str],
        semantic_trace: Mapping[str, Any],
    ) -> IterationResult:
        """Record a pre-execution semantic rejection without invoking training."""
        parent_experiment_id = proposal.parent_experiment_id or self.state.current_best_experiment_id
        parent_config = self._parent_config(parent_experiment_id)
        patch_path = self.logger.record_code_diff(
            proposal.experiment_id,
            self._configuration_diff(proposal, parent_config),
        )
        result = self._finish_without_run(
            proposal,
            decision="rejected",
            error="; ".join(reasons),
            recovery=self._restore_message(),
            safety_report=SafetyReport(False, tuple(reasons)),
            patch_path=str(patch_path),
            parent_experiment_id=parent_experiment_id,
        )
        records = self.logger.store.read_iterations()
        if records and records[-1].get("experiment_id") == proposal.experiment_id:
            self.logger.log_action(
                "semantic_rejection_recorded",
                experiment_id=proposal.experiment_id,
                details={"semantic_trace": dict(semantic_trace), "reasons": list(reasons)},
            )
        return result

    def record_manual_intervention(
        self,
        *,
        description: str,
        reason: str,
        effect: str,
        experiment_id: str | None = None,
    ) -> None:
        self.logger.record_manual_intervention(
            description=description,
            reason=reason,
            effect=effect,
            experiment_id=experiment_id,
        )

    def begin_plateau_restart(self) -> None:
        """Record a plateau and allow the LLM to investigate a new direction."""
        if self.state.consecutive_non_improvements < self.contract.non_improvement_limit:
            return
        self.state.plateau_restarts += 1
        self.state.consecutive_non_improvements = 0
        self.logger.log_action(
            "plateau_restart_requested",
            details={
                "restart_number": self.state.plateau_restarts,
                "trigger": f"{self.contract.non_improvement_limit} consecutive non-improvements",
                "next_step": "LLM must propose a new research direction",
            },
        )
        self._persist_state()

    def stop(self, reason: str) -> None:
        """Persist a clean terminal condition discovered by orchestration."""
        self.state.stop_reason = reason
        self.state.active_experiment_id = None
        self.logger.log_action("research_stopped", details={"reason": reason})
        self._persist_state()

    def _finish_without_run(
        self,
        proposal: ExperimentProposal,
        *,
        decision: str,
        error: str,
        recovery: str,
        safety_report: SafetyReport,
        patch_path: str,
        runtime_seconds: float = 0.0,
        runner_result: RunnerResult | None = None,
        parent_experiment_id: str,
        stage_id: str | None = None,
        complete_experiment: bool = True,
    ) -> IterationResult:
        self.logger.log_action(
            "accepted_candidate_restored",
            experiment_id=proposal.experiment_id,
            details={"current_best_experiment_id": self.state.current_best_experiment_id},
        )
        return self._complete_iteration(
            proposal,
            decision=decision,
            metrics={},
            delta_primary=None,
            runtime_seconds=runtime_seconds,
            error=error,
            recovery=recovery,
            safety_report=safety_report,
            patch_path=patch_path,
            runner_result=runner_result,
            parent_experiment_id=parent_experiment_id,
            stage_id=stage_id,
            complete_experiment=complete_experiment,
        )

    def _complete_iteration(
        self,
        proposal: ExperimentProposal,
        *,
        decision: str,
        metrics: Mapping[str, Any],
        delta_primary: float | None,
        runtime_seconds: float,
        error: str | None,
        recovery: str | None,
        safety_report: SafetyReport,
        patch_path: str,
        runner_result: RunnerResult | None,
        parent_experiment_id: str,
        stage_id: str | None = None,
        complete_experiment: bool = True,
    ) -> IterationResult:
        if complete_experiment:
            self.state.active_experiment_id = None
        record = {
            "experiment_id": proposal.experiment_id,
            "parent_experiment_id": parent_experiment_id,
            "hypothesis": proposal.hypothesis,
            "rationale": proposal.rationale,
            "config": dict(proposal.config),
            "changed_factors": list(proposal.changed_factors),
            "direction_id": proposal.research_direction_id,
            "search_strategy": proposal.search_strategy,
            "search_region_id": proposal.search_region_id,
            "portfolio_role": proposal.portfolio_role,
            "metrics": dict(metrics),
            "delta_primary": delta_primary,
            "runtime_seconds": runtime_seconds,
            "runner_metadata": dict(runner_result.output.metadata) if runner_result and runner_result.output else {},
            "resource_usage": dict(runner_result.resource_usage) if runner_result else {},
            "decision": decision,
            "error": error,
            "recovery": recovery,
            "safety": {"passed": safety_report.passed, "violations": list(safety_report.violations)},
            "code_diff_path": patch_path,
        }
        if not complete_experiment:
            record.update({"stage_id": stage_id or str(proposal.config.get("fidelity", "stage")), "status": decision})
            self.logger.record_stage(record)
            return IterationResult(
                proposal.experiment_id,
                decision,
                self.state,
                runner_result=runner_result,
                metrics=metrics,
                error=error,
                stage_id=str(record["stage_id"]),
            )

        self.state.completed_iterations += 1
        if delta_primary is not None and delta_primary > self.contract.improvement_threshold:
            self.state.consecutive_non_improvements = 0
        else:
            self.state.consecutive_non_improvements += 1
        self._apply_stop_rule()
        record["state_after"] = self.state.as_dict()
        self.logger.record_iteration(record)
        self._persist_state()
        refresh_evidence_memory(self.logger.store, self.state)
        return IterationResult(
            proposal.experiment_id,
            decision,
            self.state,
            runner_result=runner_result,
            metrics=metrics,
            error=error,
            stage_id=stage_id,
        )

    def complete_staged_experiment(
        self,
        proposal: ExperimentProposal,
        stage_results: Sequence[IterationResult],
        *,
        baseline_primary: float,
        terminal_reason: str,
        semantic_review: Mapping[str, Any] | None = None,
        acceptance_primary: float | None = None,
    ) -> IterationResult:
        """Commit one hypothesis after all of its fidelity/seed stages finish."""
        records = [
            item for item in self.logger.store.read_stages()
            if item.get("experiment_id") == proposal.experiment_id
        ]
        metric_records = [
            item for item in records
            if isinstance(item.get("metrics", {}).get(self.contract.primary_metric), (int, float))
        ]
        best_record = max(
            metric_records,
            key=lambda item: float(item["metrics"][self.contract.primary_metric]),
            default=None,
        )
        robust_metrics, representative_record = robust_stage_evidence(
            records, self.contract.metric_names
        )
        semantic_approved = bool((semantic_review or {}).get("approved", True))
        if best_record is not None:
            metrics = dict(robust_metrics or best_record["metrics"])
            best_primary = float(metrics[self.contract.primary_metric])
            delta = best_primary - baseline_primary
            reconciliation_primary = (
                baseline_primary if acceptance_primary is None else acceptance_primary
            )
            reconciliation_delta = best_primary - reconciliation_primary
            robust_confirmation_complete = robust_metrics is not None
            decision = (
                "accepted"
                if reconciliation_delta > 0 and semantic_approved and robust_confirmation_complete
                else "rejected"
            )
            error = None
            if reconciliation_delta > 0 and not semantic_approved:
                error = "post-execution semantic critic rejected the evidence chain"
            elif reconciliation_delta > 0 and not robust_confirmation_complete:
                error = "candidate lacks complete full-fidelity confirmation for seeds 0, 1, and 2"
            if decision == "accepted" and best_primary > self.state.current_best_primary:
                self.state.current_best_primary = best_primary
                self.state.current_best_experiment_id = proposal.experiment_id
            recovery = None if decision == "accepted" else self._restore_message()
        else:
            metrics = {}
            delta = None
            decision = "failed" if any(result.decision == "failed" for result in stage_results) else "rejected"
            errors = [result.error for result in stage_results if result.error]
            error = "; ".join(errors) or "experiment produced no valid metrics"
            recovery = self._restore_message()

        self.state.active_experiment_id = None
        self.state.completed_iterations += 1
        convergence_delta = (
            reconciliation_delta if best_record is not None else None
        )
        if (
            decision == "accepted"
            and convergence_delta is not None
            and convergence_delta > self.contract.improvement_threshold
        ):
            self.state.consecutive_non_improvements = 0
        else:
            self.state.consecutive_non_improvements += 1
        self._apply_stop_rule()

        runtime_seconds = sum(float(item.get("runtime_seconds") or 0.0) for item in records)
        diagnostic_evidence = build_experiment_evidence(
            records,
            baseline_primary=baseline_primary,
            improvement_threshold=self.contract.improvement_threshold,
        )
        if best_record is not None:
            diagnostic_evidence["robust_acceptance"] = {
                "required_seeds": sorted(CONFIRMATION_SEEDS),
                "complete": robust_metrics is not None,
                "aggregate_metrics": dict(robust_metrics or {}),
                "selection_rule": "three-seed full-fidelity mean exceeds incumbent",
            }
            diagnostic_evidence["reconciliation"] = {
                "frozen_parent_primary": baseline_primary,
                "incumbent_primary_at_commit": reconciliation_primary,
                "delta_from_frozen_parent": delta,
                "delta_from_incumbent_at_commit": reconciliation_delta,
            }
            ablation = architecture_ablation_evidence(
                parent_config=self._parent_config(proposal.parent_experiment_id),
                candidate_config=proposal.config,
                parent_primary=baseline_primary,
                candidate_primary=float(metrics[self.contract.primary_metric]),
            )
            if ablation:
                diagnostic_evidence["architecture_ablation"] = ablation
        record = {
            "experiment_id": proposal.experiment_id,
            "parent_experiment_id": proposal.parent_experiment_id or "baseline",
            "hypothesis": proposal.hypothesis,
            "rationale": proposal.rationale,
            # The experiment's conceptual configuration stays at its canonical
            # seed. A seed-confirmation checkpoint may supply the winning model
            # and metrics without becoming a new scientific configuration.
            "config": dict(proposal.config),
            "changed_factors": list(proposal.changed_factors),
            "direction_id": proposal.research_direction_id,
            "search_strategy": proposal.search_strategy,
            "search_region_id": proposal.search_region_id,
            "portfolio_role": proposal.portfolio_role,
            "metrics": metrics,
            "delta_primary": delta,
            "runtime_seconds": runtime_seconds,
            "runner_metadata": dict(
                (representative_record or best_record).get("runner_metadata", {})
                if best_record else {}
            ),
            "resource_usage": {
                "cpu_user_seconds": sum(float(item.get("resource_usage", {}).get("cpu_user_seconds") or 0.0) for item in records),
                "cpu_system_seconds": sum(float(item.get("resource_usage", {}).get("cpu_system_seconds") or 0.0) for item in records),
                "peak_rss_bytes": max((float(item.get("resource_usage", {}).get("peak_rss_bytes") or 0.0) for item in records), default=0.0),
            },
            "decision": decision,
            "error": error,
            "recovery": recovery,
            "safety": {"passed": all(item.get("safety", {}).get("passed", False) for item in records), "violations": []},
            "code_diff_path": records[0].get("code_diff_path") if records else "",
            "terminal_reason": terminal_reason,
            "diagnostic_evidence": diagnostic_evidence,
            "semantic_review": dict(semantic_review or {}),
            "stages": [
                {
                    "stage_id": item.get("stage_id"),
                    "status": item.get("status"),
                    "fidelity": item.get("config", {}).get("fidelity"),
                    "seed": item.get("config", {}).get("seed"),
                    "metrics": item.get("metrics", {}),
                    "runtime_seconds": item.get("runtime_seconds"),
                }
                for item in records
            ],
            "state_after": self.state.as_dict(),
        }
        self.logger.record_iteration(record)
        self.logger.log_action(
            "experiment_completed",
            experiment_id=proposal.experiment_id,
            details={
                "decision": decision,
                "terminal_reason": terminal_reason,
                "stage_count": len(records),
                "delta_primary": delta,
                "convergence_improvement_threshold": self.contract.improvement_threshold,
            },
        )
        self._persist_state()
        refresh_evidence_memory(self.logger.store, self.state)
        best_result = max(
            (result for result in stage_results if result.metrics),
            key=lambda result: float(result.metrics.get(self.contract.primary_metric, float("-inf"))),
            default=None,
        )
        return IterationResult(
            proposal.experiment_id,
            decision,
            self.state,
            runner_result=best_result.runner_result if best_result else None,
            metrics=metrics,
            error=error,
        )

    def _apply_stop_rule(self) -> None:
        if self.state.current_best_primary >= self.contract.target_primary:
            self.state.stop_reason = (
                f"target validation primary reached: {self.state.current_best_primary:.4f} "
                f">= {self.contract.target_primary:.4f}"
            )
        else:
            budget = self.max_iterations
            if self.state.completed_iterations >= budget:
                self.state.stop_reason = f"configured experiment budget reached: {budget}"

    def _persist_state(self) -> None:
        self.logger.store.write_root_json("state.json", self.state.as_dict())

    def _restore_message(self) -> str:
        return f"restored accepted candidate pointer: {self.state.current_best_experiment_id}"

    def _parent_config(self, experiment_id: str | None) -> Mapping[str, Any] | None:
        if experiment_id == "baseline":
            # Historical test doubles may intentionally provide partial
            # configs. Real search proposals are complete configs and are
            # therefore subject to the computed-diff check.
            return _BASELINE_CONFIG
        for record in reversed(self.logger.store.read_stages()):
            if record.get("experiment_id") == experiment_id:
                return {**_BASELINE_CONFIG, **dict(record.get("config", {}))}
        for record in reversed(self.logger.store.read_iterations()):
            if record.get("experiment_id") == experiment_id:
                return {**_BASELINE_CONFIG, **dict(record.get("config", {}))}
        return None

    @staticmethod
    def _configuration_diff(proposal: ExperimentProposal, parent_config: Mapping[str, Any] | None) -> str:
        import difflib
        import json

        before = json.dumps(dict(parent_config or {}), indent=2, sort_keys=True).splitlines(keepends=True)
        after = json.dumps(dict(proposal.config), indent=2, sort_keys=True).splitlines(keepends=True)
        return "".join(difflib.unified_diff(
            before,
            after,
            fromfile=f"parent/{proposal.parent_experiment_id or 'baseline'}.json",
            tofile=f"candidate/{proposal.experiment_id}.json",
        ))
