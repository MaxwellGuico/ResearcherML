"""Semantic integrity checks across hypotheses, implementations, configs, and evidence."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .architecture import parse_architecture_id
from .evidence import robust_stage_evidence
from .planner import ResearchDirection
from .research_tree import hypothesis_id
from .safety import (
    ExperimentProposal,
    SafetyReport,
    SafetyValidator,
    measured_historical_configs,
)


_IMPLEMENTATION_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "pointwise_fm_optimization": {
        "implementation_id": "torch_fm.pointwise_optimizer",
        "factors": ("learning_rate", "l2"),
        "required_config": {"loss": "pointwise"},
        "behavior": "Optimise the existing pointwise FM without changing its representation or objective.",
        "implementation_evidence": {
            "source_symbol": "research_agent.models.torch_fm._train_epoch",
            "activation": "config.loss == pointwise selects binary cross-entropy with logits",
        },
    },
    "pairwise_fm_ranking": {
        "implementation_id": "torch_fm.within_user_pairwise_logistic",
        "factors": ("loss", "learning_rate", "l2"),
        "required_config": {"loss": "pairwise"},
        "behavior": "Train score differences for positive-negative exposures sampled within each user.",
        "implementation_evidence": {
            "grouping_symbol": "research_agent.models.torch_fm._pair_groups",
            "training_symbol": "research_agent.models.torch_fm._train_epoch",
            "activation": "config.loss == pairwise builds per-user positive/negative groups and optimises -logsigmoid(score_positive-score_negative)",
        },
    },
    "leakage_safe_author_affinity": {
        "implementation_id": "data.train_only_author_affinity",
        "factors": ("feature_variant",),
        "required_config": {"feature_variant": "author_affinity"},
        "behavior": "Add train-only, leave-one-out user-author affinity buckets.",
        "implementation_evidence": {"source_symbol": "data.py author_affinity feature variant"},
    },
    "leakage_safe_user_history": {
        "implementation_id": "data.train_only_user_history",
        "factors": ("feature_variant",),
        "required_config": {"feature_variant": "user_history"},
        "behavior": "Add train-only, leave-one-out user response-history buckets.",
        "implementation_evidence": {"source_symbol": "data.py user_history feature variant"},
    },
    "weekday_features": {
        "implementation_id": "data.calendar_weekday_field",
        "factors": ("feature_variant",),
        "required_config": {"feature_variant": "weekday"},
        "behavior": "Add calendar weekday as one categorical FM field.",
        "implementation_evidence": {"source_symbol": "data.py weekday feature variant"},
    },
    "fm_architecture": {
        "implementation_id": "torch_fm.reviewed_architecture_compiler",
        "factors": ("architecture",),
        "required_config": {},
        "behavior": "Compile a reviewed FM-hybrid structure that adds a nonlinear interaction path.",
        "implementation_evidence": {
            "source_symbol": "research_agent.models.torch_fm.build_model",
            "spec_registry": "research_agent.models.torch_fm.ARCHITECTURE_SPECS",
            "spec_language": "research_agent.architecture.ReviewedArchitectureSpec",
            "activation": "config.architecture selects a legacy alias or canonical bounded reviewed composition",
        },
    },
    "multi_task_learning": {
        "implementation_id": "torch_fm.multitask_click_shared_embeddings",
        "factors": ("training_objective",),
        "required_config": {"loss": "pointwise"},
        "allowed_config": {
            "training_objective": (
                "multitask_click_w0.05", "multitask_click_w0.1", "multitask_click_w0.2",
            ),
        },
        "behavior": (
            "Train long_view as the primary task while a click head shares the model embeddings "
            "and contributes a bounded weighted auxiliary loss only during training."
        ),
        "implementation_evidence": {
            "data_symbol": "data.auxiliary_labels",
            "model_symbol": "research_agent.models.torch_fm.MultiTaskSharedBackbone",
            "training_symbol": "research_agent.models.torch_fm._train_epoch",
            "activation": "config.training_objective selects a reviewed click auxiliary-loss weight",
        },
    },
}


def _is_reviewed_architecture(value: Any) -> bool:
    if value in {"deepfm", "nfm_residual"}:
        return True
    if not isinstance(value, str):
        return False
    try:
        return parse_architecture_id(value) is not None
    except (TypeError, ValueError):
        return False


def _pre_execution_feedback(
    proposal: ExperimentProposal,
    checks: Mapping[str, bool],
    reasons: Sequence[str],
    config_diff: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [name for name, passed in checks.items() if not passed]
    if not reasons:
        disposition = "execute_controlled_test"
        lesson = "The hypothesis, executable behavior, lineage, and one-factor configuration diff agree."
    elif any("duplicates an existing experiment configuration" in reason for reason in reasons):
        disposition = "change_configuration_or_hypothesis"
        lesson = "Do not repeat this measured configuration; propose a materially different test."
    elif any(name.startswith("planner_lineage") for name in failed_checks):
        disposition = "repair_lineage_reasoning"
        lesson = "Relate the hypothesis to a known evidence node and label continuation versus novelty honestly."
    elif "configuration_activates_behavior" in failed_checks or "controlled_configuration_diff" in failed_checks:
        disposition = "repair_execution_alignment"
        lesson = "Revise the executable configuration so its sole scientific diff activates the claimed mechanism."
    else:
        disposition = "repair_safety_or_semantic_contract"
        lesson = "Resolve every failed deterministic contract before spending training compute."
    return {
        "phase": "pre_execution",
        "disposition": disposition,
        "lesson": lesson,
        "failed_checks": failed_checks,
        "do_not_repeat_exact_configuration": any(
            "duplicates an existing experiment configuration" in reason for reason in reasons
        ),
        "configuration_diff": dict(config_diff),
        "direction_id": proposal.research_direction_id,
    }


def _post_execution_feedback(
    proposal: ExperimentProposal,
    reasons: Sequence[str],
    *,
    best_primary: float | None,
    baseline_primary: float | None,
    agentic_review: Mapping[str, Any],
) -> dict[str, Any]:
    if reasons:
        disposition = "repair_evidence_chain"
        lesson = "Do not infer model quality from this run until the failed semantic or evidence checks are repaired."
    elif best_primary is None:
        disposition = "diagnose_execution_failure"
        lesson = "No ranking metric was measured; diagnose execution before revising the scientific hypothesis."
    elif baseline_primary is not None and best_primary > baseline_primary:
        disposition = "continue_supported_lineage"
        lesson = "The controlled change beat its incumbent; test a focused refinement while preserving the accepted mechanism."
    else:
        disposition = "record_valid_negative_and_branch"
        lesson = "The implementation validly tested the claim but did not beat the incumbent; do not repeat the exact configuration."
    return {
        "phase": "post_execution",
        "disposition": disposition,
        "lesson": lesson,
        "direction_id": proposal.research_direction_id,
        "best_primary": best_primary,
        "incumbent_primary_before": baseline_primary,
        "delta_primary": (
            best_primary - baseline_primary
            if best_primary is not None and baseline_primary is not None else None
        ),
        "do_not_repeat_exact_configuration": best_primary is not None,
        "critic_recommended_action": agentic_review.get("recommended_planner_action"),
        "next_hypothesis_constraint": agentic_review.get("next_hypothesis_constraint"),
        "failed_reasons": list(reasons),
    }
@dataclass(frozen=True)
class CriticResult:
    approved: bool
    reasons: tuple[str, ...]
    trace: Mapping[str, Any] = field(default_factory=dict)


class ProposalCritic:
    """Independently verifies semantic intent before and after execution."""

    def __init__(
        self,
        validator: SafetyValidator,
        *,
        semantic_client: Any | None = None,
        semantic_token_budget: int = 0,
    ) -> None:
        self.validator = validator
        self.semantic_client = semantic_client
        self.semantic_token_budget = semantic_token_budget
        self.known_lineage_ids: set[str] = {"baseline", "new"}

    def set_research_context(self, tree_context: Mapping[str, Any]) -> None:
        """Receive bounded graph identities; no mutable controller state is delegated."""
        known = {"baseline", "new"}
        incumbent = tree_context.get("incumbent", {})
        known.update(str(item) for item in incumbent.get("ancestry", []) if item)
        for section in ("continuation_candidates", "deferred_hypotheses", "failed_branches"):
            for item in tree_context.get(section, []):
                for key in ("hypothesis_id", "source_experiment_id", "experiment_id"):
                    if item.get(key):
                        known.add(str(item[key]))
        self.known_lineage_ids = known

    def review(
        self,
        proposal: ExperimentProposal,
        history: Sequence[Mapping[str, Any]],
        direction: ResearchDirection | None = None,
    ) -> CriticResult:
        if proposal.runtime_budget_seconds > self.validator.max_runtime_seconds:
            proposal = replace(
                proposal,
                runtime_budget_seconds=self.validator.max_runtime_seconds,
            )
        report: SafetyReport = self.validator.validate(
            proposal,
            historical_configs=measured_historical_configs(history),
        )
        reasons = list(report.violations)
        for value, message in (
            (proposal.research_direction_id, "proposal must record its research direction"),
            (proposal.search_strategy, "proposal must record its search strategy"),
            (proposal.search_region_id, "proposal must record its search region"),
        ):
            if not value:
                reasons.append(message)

        contract = _IMPLEMENTATION_CONTRACTS.get(str(proposal.research_direction_id))
        factor = proposal.changed_factors[0] if len(proposal.changed_factors) == 1 else None
        parent = self._parent_config(proposal, history)
        config_diff = {
            key: {"before": parent.get(key), "after": proposal.config.get(key)}
            for key in sorted(proposal.config)
            if key not in {"epochs", "fidelity", "worker_threads", "seed"}
            and parent.get(key) != proposal.config.get(key)
        }
        if proposal.search_strategy == "promotion":
            config_diff = {}
        # A promoted stage points to its own experiment so the safety layer can
        # compare it with the preceding fidelity. Its scientific lineage is
        # still the planner direction's original parent, however.
        expected_lineage_parent = (
            direction.lineage_parent_id
            if proposal.search_strategy == "promotion" and direction is not None
            else proposal.parent_experiment_id
        )
        lineage_checks = self._lineage_checks(
            direction, history, expected_parent=expected_lineage_parent
        )
        semantic_checks: dict[str, bool] = {
            "known_implementation": contract is not None,
            "direction_identity_preserved": bool(
                direction is None or direction.direction_id == proposal.research_direction_id
            ),
            "hypothesis_identity_preserved": bool(
                direction is None or direction.hypothesis.strip() == proposal.hypothesis.strip()
            ),
            "factor_supported_by_implementation": bool(contract and factor in contract["factors"]),
            "configuration_activates_behavior": bool(
                contract and all(
                    proposal.config.get(key) == value
                    for key, value in contract["required_config"].items()
                )
                and all(
                    proposal.config.get(key) in values
                    for key, values in contract.get("allowed_config", {}).items()
                )
                and (
                    proposal.research_direction_id != "fm_architecture"
                    or _is_reviewed_architecture(proposal.config.get("architecture"))
                )
            ),
            "factor_value_in_declared_search_space": bool(
                direction is None
                or (factor in direction.search_space and proposal.config.get(factor) in direction.search_space[factor])
            ),
            "expected_evidence_declared": bool(direction is None or direction.success_evidence.strip()),
            "controlled_configuration_diff": bool(
                proposal.search_strategy == "promotion" or (factor and set(config_diff) == {factor})
            ),
            **lineage_checks,
        }
        reasons.extend(
            f"semantic check failed: {name}"
            for name, passed in semantic_checks.items() if not passed
        )
        trace: dict[str, Any] = {
            "phase": "pre_execution",
            "hypothesis": proposal.hypothesis,
            "claimed_behavior": getattr(direction, "claimed_behavior", None) or (contract or {}).get("behavior"),
            "required_capabilities": list(getattr(direction, "required_capabilities", ()) or ()),
            "implementation_id": (contract or {}).get("implementation_id"),
            "implementation_behavior": (contract or {}).get("behavior"),
            "implementation_evidence": (contract or {}).get("implementation_evidence", {}),
            "configuration_diff": config_diff,
            "expected_evidence": getattr(direction, "success_evidence", None),
            "checks": semantic_checks,
        }
        trace["planner_feedback"] = _pre_execution_feedback(
            proposal, semantic_checks, reasons, config_diff
        )
        return CriticResult(approved=not reasons, reasons=tuple(reasons), trace=trace)

    def review_evidence(
        self,
        proposal: ExperimentProposal,
        stage_records: Sequence[Mapping[str, Any]],
        pre_execution_trace: Mapping[str, Any],
        *,
        baseline_primary: float | None = None,
        baseline_metrics: Mapping[str, Any] | None = None,
    ) -> CriticResult:
        """Bind the pre-run claim to artifacts and evidence from this experiment only."""
        records = [item for item in stage_records if item.get("experiment_id") == proposal.experiment_id]
        factor = proposal.changed_factors[0] if len(proposal.changed_factors) == 1 else None
        expected_value = proposal.config.get(factor) if factor else None
        metric_records = [
            item for item in records
            if isinstance(item.get("metrics", {}).get("primary"), (int, float))
        ]
        failure_records = [item for item in records if item.get("error")]
        best_stage_primary = max(
            (float(item["metrics"]["primary"]) for item in metric_records),
            default=None,
        )
        robust_metrics, _ = robust_stage_evidence(
            records, ("GAUC", "nDCG@5", "primary")
        )
        best_primary = (
            float(robust_metrics["primary"])
            if robust_metrics is not None else best_stage_primary
        )
        patch_paths = {str(item.get("code_diff_path")) for item in records if item.get("code_diff_path")}
        checks = {
            "pre_execution_semantics_passed": bool(pre_execution_trace.get("checks")) and all(
                pre_execution_trace.get("checks", {}).values()
            ),
            "evidence_belongs_to_experiment": bool(records),
            "configuration_preserved_across_fidelities": bool(records) and all(
                item.get("config", {}).get(factor) == expected_value for item in records
            ),
            "configuration_diff_artifact_present": bool(patch_paths) and any(
                self._patch_mentions(path, factor) for path in patch_paths
            ),
            "measured_metric_or_failure_evidence_present": bool(metric_records or failure_records),
            "validation_metrics_are_finite_numbers": all(
                isinstance(item.get("metrics", {}).get(metric), (int, float))
                and math.isfinite(float(item["metrics"][metric]))
                for item in metric_records for metric in ("GAUC", "nDCG@5", "primary")
            ),
            "primary_matches_metric_contract": all(
                abs(
                    float(item["metrics"]["primary"])
                    - (float(item["metrics"]["GAUC"]) + float(item["metrics"]["nDCG@5"])) / 2
                ) <= 1e-9
                for item in metric_records
                if all(metric in item.get("metrics", {}) for metric in ("GAUC", "nDCG@5", "primary"))
            ),
        }
        reasons = tuple(
            f"semantic evidence check failed: {name}"
            for name, passed in checks.items() if not passed
        )
        trace = {
            **dict(pre_execution_trace),
            "phase": "post_execution",
            "checks": checks,
            "evidence": {
                "experiment_id": proposal.experiment_id,
                "stage_ids": [item.get("stage_id") for item in records],
                "metric_stage_count": len(metric_records),
                "failure_stage_count": len(failure_records),
                "metrics": [dict(item.get("metrics", {})) for item in metric_records],
                "execution_diagnostics": [
                    {
                        "stage_id": item.get("stage_id"),
                        "training": item.get("runner_metadata", {}).get("diagnostics", {}).get("training", {}),
                        "model": item.get("runner_metadata", {}).get("diagnostics", {}).get("model", {}),
                        "feature_coverage": item.get("runner_metadata", {}).get("diagnostics", {}).get("feature_coverage", {}),
                        "stratified_validation": item.get("runner_metadata", {}).get("diagnostics", {}).get("stratified_validation", {}),
                    }
                    for item in metric_records
                ],
                "incumbent_metrics_before": dict(baseline_metrics or {}),
                "incumbent_primary_before": baseline_primary,
                "best_candidate_primary": best_primary,
                "best_stage_primary": best_stage_primary,
                "robust_confirmation_complete": robust_metrics is not None,
                "delta_primary": (
                    best_primary - baseline_primary
                    if best_primary is not None and baseline_primary is not None else None
                ),
                "patch_paths": sorted(patch_paths),
            },
        }
        if not reasons and self.semantic_client is not None:
            try:
                agentic, metadata = self.semantic_client.create_json(
                    _agentic_critic_instructions(),
                    _agentic_critic_prompt(trace),
                    schema=_agentic_critic_schema(),
                    schema_name="semantic_evidence_audit",
                    max_output_tokens=1400,
                    prompt_cache_key="researcher-ml-semantic-critic-v1",
                )
                total_tokens = metadata.get("usage", {}).get("total_tokens", 0)
                if (
                    self.semantic_token_budget > 0
                    and isinstance(total_tokens, int)
                    and total_tokens > self.semantic_token_budget
                ):
                    reasons = reasons + (
                        f"agentic semantic critic token budget exceeded: {total_tokens} > {self.semantic_token_budget}",
                    )
                failed_agent_checks = [
                    key for key in _AGENTIC_CHECK_KEYS if not agentic.get(key)
                ]
                reasons = reasons + tuple(
                    f"agentic semantic check failed: {key}" for key in failed_agent_checks
                )
                trace["agentic_review"] = {
                    "verdict": dict(agentic),
                    "metadata": dict(metadata),
                }
            except Exception as exc:
                reasons = reasons + (f"agentic semantic critic unavailable: {type(exc).__name__}: {exc}",)
                trace["agentic_review"] = {"error": f"{type(exc).__name__}: {exc}"}
        trace["verdict"] = (
            "verified" if not reasons and metric_records
            else "failure_evidence_verified" if not reasons
            else "misaligned"
        )
        trace["planner_feedback"] = _post_execution_feedback(
            proposal,
            reasons,
            best_primary=best_primary,
            baseline_primary=baseline_primary,
            agentic_review=trace.get("agentic_review", {}).get("verdict", {}),
        )
        return CriticResult(approved=not reasons, reasons=reasons, trace=trace)

    def _lineage_checks(
        self,
        direction: ResearchDirection | None,
        history: Sequence[Mapping[str, Any]],
        *,
        expected_parent: str | None = None,
    ) -> dict[str, bool]:
        online_direction = bool(direction and direction.selected_candidate_id)
        if not online_direction:
            return {
                "planner_lineage_declared": True,
                "planner_lineage_reference_known": True,
                "planner_lineage_evidence_cites_parent": True,
                "planner_lineage_claim_consistent": True,
            }
        parent = str(direction.lineage_parent_id or "")
        action = str(direction.lineage_action or "")
        evidence = str(direction.evidence_reference or "")
        known_ids = set(self.known_lineage_ids)
        known_ids.update(str(item.get("experiment_id")) for item in history if item.get("experiment_id"))
        known_ids.update(
            hypothesis_id(str(item.get("hypothesis", "")))
            for item in history if item.get("hypothesis")
        )
        prior_hypotheses = {
            " ".join(str(item.get("hypothesis", "")).lower().split()) for item in history
        }
        normalized = " ".join(direction.hypothesis.lower().split())
        return {
            "planner_lineage_declared": bool(
                parent and action in {"continue", "refine", "revisit", "branch_new"} and len(evidence) >= 4
            ),
            "planner_lineage_reference_known": parent in known_ids,
            "planner_lineage_evidence_cites_parent": bool(
                parent
                and (
                    parent.lower() in evidence.lower()
                    or parent == str(expected_parent or "")
                )
            ),
            "planner_lineage_parent_matches_execution": bool(
                not expected_parent or parent == str(expected_parent)
            ),
            "planner_lineage_claim_consistent": not (
                action == "branch_new" and normalized in prior_hypotheses
            ),
        }

    @staticmethod
    def _parent_config(
        proposal: ExperimentProposal,
        history: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        for item in reversed(history):
            if item.get("experiment_id") == proposal.parent_experiment_id:
                from .search import SearchController

                return {**SearchController.BASELINE_CONFIG, **dict(item.get("config", {}))}
        from .search import SearchController

        return SearchController.BASELINE_CONFIG

    @staticmethod
    def _patch_mentions(path: str, factor: str | None) -> bool:
        if not factor:
            return False
        source = Path(path)
        try:
            return source.is_file() and factor in source.read_text(encoding="utf-8")
        except OSError:
            return False


_AGENTIC_CHECK_KEYS = (
    "hypothesis_matches_implementation",
    "implementation_matches_configuration_diff",
    "configuration_matches_measured_evidence",
    "evidence_meaningfully_tests_hypothesis",
    "no_unimplemented_behavior_claimed",
)


def _agentic_critic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            *_AGENTIC_CHECK_KEYS, "rationale", "limitations",
            "recommended_planner_action", "next_hypothesis_constraint",
        ],
        "properties": {
            **{key: {"type": "boolean"} for key in _AGENTIC_CHECK_KEYS},
            "rationale": {"type": "string", "minLength": 20},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "recommended_planner_action": {
                "type": "string",
                "enum": ["continue", "refine", "revisit", "branch_new", "repair_alignment"],
            },
            "next_hypothesis_constraint": {"type": "string", "minLength": 20},
        },
    }


def _agentic_critic_instructions() -> str:
    return (
        "You are an independent semantic auditor for a recommender-system experiment. Verify correspondence across "
        "the hypothesis, claimed mechanism, named implementation, configuration diff, and measured validation "
        "evidence. Judge whether the experiment genuinely tested the claim, not whether its score improved. A valid "
        "negative result still meaningfully tests a hypothesis. Official aggregate validation ranking metrics are "
        "sufficient to test the main directional performance claim when deterministic checks establish the named "
        "implementation and configuration. Treat missing subgroup analyses, uncertainty estimates, or secondary "
        "mechanism diagnostics as limitations, not semantic failure, unless the experiment's primary acceptance "
        "criterion explicitly depends on them. Fail claims that promise behavior absent from the implementation or "
        "configuration, evidence from the wrong experiment, or a configuration that did not activate the mechanism. "
        "Return a concrete constraint for the next hypothesis and whether the planner should continue, refine, "
        "revisit, branch, or repair alignment. Use only the supplied trace."
    )


def _agentic_critic_prompt(trace: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "semantic_trace": _compact_agentic_trace(trace),
            "request": "Audit each correspondence independently and state evidence limitations.",
        },
        sort_keys=True,
        default=str,
    )


def _compact_agentic_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Keep semantic correspondence while leaving full diagnostics in artifacts."""
    evidence = trace.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    diagnostics = evidence.get("execution_diagnostics", [])
    if not isinstance(diagnostics, list):
        diagnostics = []
    best_primary = evidence.get("best_candidate_primary")
    metrics = evidence.get("metrics", [])
    best_index = 0
    if isinstance(metrics, list) and metrics:
        best_index = max(
            range(len(metrics)),
            key=lambda index: float(metrics[index].get("primary", float("-inf"))),
        )
    best_diagnostic = diagnostics[min(best_index, len(diagnostics) - 1)] if diagnostics else {}
    stratified = best_diagnostic.get("stratified_validation", {}) if isinstance(best_diagnostic, Mapping) else {}
    user_activity = stratified.get("user_activity", {}) if isinstance(stratified, Mapping) else {}
    compact = {
        key: trace.get(key)
        for key in (
            "phase", "hypothesis", "claimed_behavior", "implementation_id",
            "implementation_evidence", "expected_evidence", "checks",
        )
    }
    # The deterministic critic records this as ``configuration_diff``. Keep the
    # shorter external name explicit so the semantic auditor receives the actual
    # diff instead of a misleading null value.
    compact["config_diff"] = trace.get("configuration_diff", {})
    return compact | {
        "evidence": {
            "experiment_id": evidence.get("experiment_id"),
            "stage_ids": evidence.get("stage_ids", []),
            "metrics": metrics,
            "incumbent_metrics_before": evidence.get("incumbent_metrics_before", {}),
            "best_candidate_primary": best_primary,
            "delta_primary": evidence.get("delta_primary"),
            "patch_paths": evidence.get("patch_paths", []),
            "best_stage_diagnostics": {
                "stage_id": best_diagnostic.get("stage_id") if isinstance(best_diagnostic, Mapping) else None,
                "training": best_diagnostic.get("training", {}) if isinstance(best_diagnostic, Mapping) else {},
                "model": best_diagnostic.get("model", {}) if isinstance(best_diagnostic, Mapping) else {},
                "weakest_eligible_stratum": user_activity.get(
                    "weakest_statistically_eligible_stratum", {}
                ) if isinstance(user_activity, Mapping) else {},
            },
        }
    }
