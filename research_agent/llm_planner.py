"""OpenAI-powered high-level hypothesis planning; no model code or data access."""
from __future__ import annotations

import json
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .architecture import (
    REVIEWED_CROSS_LAYERS,
    REVIEWED_DEPTHS,
    REVIEWED_DROPOUTS,
    REVIEWED_FUSIONS,
    REVIEWED_OPERATORS,
    REVIEWED_WIDTHS,
    ReviewedArchitectureSpec,
    architecture_schema,
    controlled_single_path_ablations,
    parse_architecture_id,
)
from .planner import CapabilityAction, PLANNER_ACTIONS, ResearchDirection
from .safety import has_measured_validation_evidence
from .state import ResearchState


class LLMPlanningError(RuntimeError):
    """A planning failure must pause the run rather than fall back silently."""


class ResearchCatalogueExhausted(LLMPlanningError):
    """No distinct approved executable configuration remains."""


class OpenAITransportError(LLMPlanningError):
    """An API transport or HTTP failure with a stable error category."""


_GENERIC_IMPLEMENTER_ID = "generic_implementer"
_LEGACY_SPECIALIST_IDS = frozenset({
    "training_specialist", "ranking_specialist", "evaluation_specialist",
    "feature_data_specialist", "temporal_specialist", "model_architecture_specialist",
})


@dataclass(frozen=True)
class OpenAIResponsesClient:
    api_key: str
    model: str
    endpoint: str = "https://api.openai.com/v1/responses"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    backoff_seconds: float = 1.0
    sleeper: Callable[[float], None] = time.sleep
    opener: Callable[..., Any] = urlopen

    def create_json(
        self,
        instructions: str,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        schema_name: str = "research_direction",
        max_output_tokens: int | None = None,
        prompt_cache_key: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = dict(schema or _ideator_schema())
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
            "store": False,
            "text": {
                "verbosity": "low",
                "format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema},
            },
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        raw = None
        attempts = 0
        while attempts <= self.max_retries:
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                category = _http_error_category(exc.code)
                if not _retryable_http_status(exc.code) or attempts >= self.max_retries:
                    detail = _http_error_detail(exc)
                    suffix = f": {detail}" if detail else ""
                    raise OpenAITransportError(
                        f"OpenAI planning {category} ({exc.code}){suffix}"
                    ) from exc
            except (URLError, TimeoutError) as exc:
                category = "timeout" if isinstance(exc, TimeoutError) else "network_error"
                if attempts >= self.max_retries:
                    raise OpenAITransportError(f"OpenAI planning {category}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LLMPlanningError("OpenAI planning returned invalid JSON") from exc
            attempts += 1
            self.sleeper(self.backoff_seconds * (2 ** (attempts - 1)))
        if not isinstance(raw, dict):
            raise LLMPlanningError("OpenAI planning returned a non-object response")
        if raw.get("status") in {"failed", "cancelled", "incomplete"}:
            error = raw.get("error") or raw.get("incomplete_details") or {}
            raise LLMPlanningError(f"OpenAI planning response status={raw.get('status')}: {error}")
        # ``output_text`` is an SDK convenience property, not guaranteed to be
        # present in the raw HTTP JSON returned by the Responses API.  The raw
        # response carries the same text in output[].content[].
        text = raw.get("output_text")
        if not isinstance(text, str):
            text_parts = [
                part.get("text")
                for item in raw.get("output", [])
                if isinstance(item, dict)
                for part in item.get("content", [])
                if isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ]
            text = "".join(text_parts)
        if not text:
            raise LLMPlanningError("OpenAI response did not include output text")
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise LLMPlanningError("OpenAI planning JSON must be an object")
            return value, {"model": self.model, "usage": raw.get("usage", {}), "response_id": raw.get("id"), "retry_count": attempts}
        except json.JSONDecodeError as exc:
            raise LLMPlanningError("OpenAI response was not valid structured JSON") from exc


class OpenAIPlanner:
    """Open research planner followed by identical generic implementer calls."""

    def __init__(
        self,
        client: OpenAIResponsesClient,
        *,
        token_budget: int = 0,
        allow_architecture_experiments: bool = False,
    ) -> None:
        self.client = client
        self.token_budget = token_budget
        self.allow_architecture_experiments = allow_architecture_experiments
        self.require_executable_backfill = False
        self.last_metadata: dict[str, Any] = {}
        self.run_context: dict[str, Any] = {}

    def set_run_context(self, context: Mapping[str, Any]) -> None:
        """Receive bounded governance context without expanding planner authority."""
        self.run_context = dict(context)

    def set_backfill_mode(self, enabled: bool) -> None:
        """Require an explicit executable choice when filling an idle worker slot."""
        self.require_executable_backfill = bool(enabled)

    @classmethod
    def from_environment(cls, *, allow_architecture_experiments: bool = False) -> "OpenAIPlanner":
        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna").strip()
        if not key:
            raise LLMPlanningError("OPENAI_API_KEY is not set; add it to .env before starting the LLM agent")
        token_budget = int(os.environ.get("OPENAI_PLANNER_TOKEN_BUDGET", "0"))
        return cls(
            OpenAIResponsesClient(api_key=key, model=model),
            token_budget=token_budget,
            allow_architecture_experiments=allow_architecture_experiments,
        )

    def propose(
        self, history: Sequence[Mapping[str, Any]], state: ResearchState
    ) -> ResearchDirection | CapabilityAction:
        self.last_metadata = {}
        incumbent_config = _planner_incumbent_config(history, state)
        available_directions = _available_direction_ids(
            history,
            include_architecture=self.allow_architecture_experiments,
            incumbent_config=incumbent_config,
        )
        if not available_directions:
            raise ResearchCatalogueExhausted("the approved executable research catalogue is exhausted")
        slate, ideator_metadata = self.client.create_json(
            _ideator_instructions(),
            _ideator_prompt(history, state, self.run_context, worker_count=1),
            schema=_ideator_schema(),
            schema_name="research_hypothesis_slate",
            max_output_tokens=2_400,
            prompt_cache_key="researcher-ml-open-ideator-v1",
        )
        _enforce_token_budget(self.token_budget, ideator_metadata)
        candidates = slate.get("candidates", [])
        recommended_id = str(slate.get("recommended_candidate_id", ""))
        candidate_ids = [str(item.get("candidate_id", "")) for item in candidates if isinstance(item, dict)]
        if len(candidates) < 3 or recommended_id not in candidate_ids:
            raise LLMPlanningError("LLM ideator returned an invalid hypothesis slate")
        research_strategy = _validated_research_strategy(slate, worker_count=1)
        capability_manifest = _capability_manifest(
            history, available_directions, incumbent_config=incumbent_config
        )
        assignment = _portfolio_assignments(
            candidates, recommended_id, 1,
            research_strategy=research_strategy,
        )[0]
        assigned_candidate, portfolio_role = assignment
        assigned_domain = str(assigned_candidate.get("domain"))
        implementation, implementer_metadata = self.client.create_json(
            _implementer_instructions(),
            _implementer_prompt(
                slate, capability_manifest, history, state,
                assigned_candidate_id=str(assigned_candidate.get("candidate_id")),
                worker_slot=1,
                portfolio_role=portfolio_role,
                run_context=self.run_context,
                require_executable=self.require_executable_backfill,
            ),
            schema=_implementer_schema(candidate_ids, capability_manifest),
            schema_name="generic_implementer_execution_plan",
            max_output_tokens=1_400,
            prompt_cache_key="researcher-ml-generic-implementer-v1",
        )
        if implementation.get("selected_candidate_id") != assigned_candidate.get("candidate_id"):
            raise LLMPlanningError("implementer changed or substituted the assigned hypothesis")
        usage = _combined_usage(ideator_metadata, implementer_metadata)
        self.last_metadata = {
            "mode": "online_planner_generic_implementer",
            "model": ideator_metadata.get("model"),
            "token_budget": self.token_budget,
            "research_strategy": research_strategy,
            "planner": {**ideator_metadata, "slate": slate},
            "implementer": {**implementer_metadata, "implementer_id": _GENERIC_IMPLEMENTER_ID, "plan": implementation},
            "usage": usage,
            "deferred_candidates": implementation.get("deferred_candidate_ids", []),
        }
        _enforce_token_budget(self.token_budget, {"usage": usage})
        decision = _align_execution_lineage(
            _planner_decision_from_implementation(
                implementation, history, incumbent_config=incumbent_config
            ),
            state,
        )
        if isinstance(decision, ResearchDirection):
            decision = replace(decision, portfolio_role=portfolio_role)
        else:
            decision = replace(decision, portfolio_role=portfolio_role)
        _validate_strategy_decision(
            decision, research_strategy, assigned_domain=assigned_domain
        )
        _validate_research_campaign(decision, self.run_context)
        return decision

    def propose_batch(
        self,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        *,
        count: int = 2,
    ) -> list[ResearchDirection | CapabilityAction]:
        """Use one planning call and identical concurrent implementers for a diverse batch."""
        if count < 1 or count > 2:
            raise ValueError("online planner batch count must be 1 or 2")
        self.last_metadata = {}
        incumbent_config = _planner_incumbent_config(history, state)
        available_directions = _available_direction_ids(
            history,
            include_architecture=self.allow_architecture_experiments,
            incumbent_config=incumbent_config,
        )
        if not available_directions:
            raise ResearchCatalogueExhausted("the approved executable research catalogue is exhausted")
        slate, ideator_metadata = self.client.create_json(
            _ideator_instructions(),
            _ideator_prompt(history, state, self.run_context, worker_count=count),
            schema=_ideator_schema(),
            schema_name="parallel_research_hypothesis_slate",
            max_output_tokens=4_000,
            prompt_cache_key="researcher-ml-open-ideator-v1",
        )
        _enforce_token_budget(self.token_budget, ideator_metadata)
        candidates = [item for item in slate.get("candidates", []) if isinstance(item, dict)]
        candidate_ids = [str(item.get("candidate_id", "")) for item in candidates]
        recommended_id = str(slate.get("recommended_candidate_id", ""))
        if len(candidates) < 3 or recommended_id not in candidate_ids:
            raise LLMPlanningError("LLM ideator returned an invalid hypothesis slate")
        research_strategy = _validated_research_strategy(slate, worker_count=count)
        manifest = _capability_manifest(
            history, available_directions, incumbent_config=incumbent_config
        )
        assignments = _portfolio_assignments(
            candidates, recommended_id, count,
            research_strategy=research_strategy,
        )

        def consult(
            slot: int,
            candidate: Mapping[str, Any],
            portfolio_role: str,
        ) -> tuple[dict[str, Any], dict[str, Any], str]:
            plan, metadata = self.client.create_json(
                _implementer_instructions(),
                _implementer_prompt(
                    slate, manifest, history, state,
                    assigned_candidate_id=str(candidate.get("candidate_id")),
                    worker_slot=slot,
                    portfolio_role=portfolio_role,
                    run_context=self.run_context,
                    require_executable=self.require_executable_backfill,
                ),
                schema=_implementer_schema(candidate_ids, manifest),
                schema_name=f"parallel_generic_implementer_{slot}_plan",
                max_output_tokens=1_400,
                prompt_cache_key="researcher-ml-generic-implementer-v1",
            )
            return plan, metadata, portfolio_role

        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="llm-implementer") as executor:
            consultations = list(
                executor.map(lambda pair: consult(pair[0], *pair[1]), enumerate(assignments, start=1))
            )
        usage = _combined_usage(ideator_metadata, *(item[1] for item in consultations))
        self.last_metadata = {
            "mode": "online_planner_parallel_implementers",
            "model": ideator_metadata.get("model"),
            "token_budget": self.token_budget,
            "research_strategy": research_strategy,
            "planner": {**ideator_metadata, "slate": slate},
            "implementers": [
                {**metadata, "implementer_id": _GENERIC_IMPLEMENTER_ID, "portfolio_role": role, "plan": plan}
                for plan, metadata, role in consultations
            ],
            "usage": usage,
            "requested_workers": count,
            "executable_registry_ids": list(manifest),
        }
        _enforce_token_budget(self.token_budget, {"usage": usage})
        decisions: list[ResearchDirection | CapabilityAction] = []
        failures: list[str] = []
        signatures: set[tuple[Any, ...]] = set()
        for (implementation, _metadata, portfolio_role), assignment in zip(
            consultations, assignments
        ):
            try:
                assigned_domain = str(assignment[0].get("domain"))
                planner_action = str(implementation.get("planner_action") or (
                    "RUN_EXPERIMENT"
                    if implementation.get("implementation_alignment") == "exact"
                    else "BUILD_CAPABILITY"
                ))
                if implementation.get("selected_candidate_id") != assignment[0].get("candidate_id"):
                    raise LLMPlanningError("implementer changed or substituted the assigned hypothesis")
                if planner_action == "RUN_EXPERIMENT" and implementation.get("direction_id") not in manifest:
                    raise LLMPlanningError("implementer selected behavior absent from the executable registry")
                decision = _planner_decision_from_implementation(
                    implementation, history, incumbent_config=incumbent_config
                )
                if isinstance(decision, ResearchDirection):
                    decision = _align_execution_lineage(decision, state)
                    decision = replace(
                        decision,
                        portfolio_role=portfolio_role,
                        strategy=(
                            "local_refinement"
                            if portfolio_role == "incumbent_exploit"
                            else "exploration"
                        ),
                    )
                    signature = (
                        decision.direction_id,
                        decision.preferred_factor,
                        decision.preferred_value,
                    )
                else:
                    decision = replace(decision, portfolio_role=portfolio_role)
                    signature = (decision.action, decision.capability_gap_id)
                _validate_strategy_decision(
                    decision, research_strategy, assigned_domain=assigned_domain
                )
                _validate_research_campaign(decision, self.run_context)
            except LLMPlanningError as exc:
                failures.append(str(exc))
                continue
            if signature not in signatures:
                signatures.add(signature)
                decisions.append(decision)
        self.last_metadata["implementer_failures"] = failures
        self.last_metadata["planned_workers"] = sum(
            isinstance(item, ResearchDirection) for item in decisions
        )
        self.last_metadata["capability_actions"] = [
            item.as_dict() for item in decisions if isinstance(item, CapabilityAction)
        ]
        if not decisions:
            raise LLMPlanningError("parallel implementer portfolio produced no distinct decisions")
        return decisions


def _align_execution_lineage(
    decision: ResearchDirection | CapabilityAction,
    state: ResearchState,
) -> ResearchDirection | CapabilityAction:
    """Bind executable work to the incumbent config the Search Controller clones."""
    if not isinstance(decision, ResearchDirection):
        return decision
    incumbent_id = state.current_best_experiment_id
    evidence = str(decision.evidence_reference or "").strip()
    if incumbent_id not in evidence:
        evidence = f"{evidence}; execution parent {incumbent_id}".strip("; ")
    return replace(
        decision,
        lineage_parent_id=incumbent_id,
        evidence_reference=evidence,
    )


_RESEARCH_DOMAINS = frozenset({
    "model_architecture",
    "ranking_objective",
    "training_optimization",
    "feature_data",
    "temporal",
    "evaluation_diagnostics",
})

_DOMAIN_DIRECTIONS = {
    "model_architecture": ("fm_architecture", "multi_task_learning"),
    "ranking_objective": ("pairwise_fm_ranking", "multi_task_learning"),
    "training_optimization": (
        "pointwise_fm_optimization", "pairwise_fm_ranking", "multi_task_learning",
    ),
    "feature_data": ("leakage_safe_author_affinity", "leakage_safe_user_history"),
    "temporal": ("weekday_features",),
    "evaluation_diagnostics": (
        "pointwise_fm_optimization",
        "pairwise_fm_ranking",
        "leakage_safe_author_affinity",
        "leakage_safe_user_history",
        "weekday_features",
        "fm_architecture",
        "multi_task_learning",
    ),
}

_DOMAIN_FACTORS = {
    "model_architecture": frozenset({"architecture", "training_objective"}),
    "ranking_objective": frozenset({"loss", "training_objective"}),
    "training_optimization": frozenset({"learning_rate", "l2", "training_objective"}),
    "feature_data": frozenset({"feature_variant"}),
    "temporal": frozenset({"feature_variant"}),
    "evaluation_diagnostics": frozenset(),
}

_STRATEGY_FACTORS = (
    "architecture", "feature_variant", "loss", "training_objective", "learning_rate", "l2",
    "embedding_dim", "batch_size",
)


def _portfolio_assignments(
    candidates: Sequence[Mapping[str, Any]],
    recommended_id: str,
    count: int,
    *,
    research_strategy: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], str]]:
    """Apply the LLM-authored worker allocation without inventing priorities."""
    ordered = sorted(
        candidates,
        key=lambda item: str(item.get("candidate_id")) != recommended_id,
    )
    worker_assignments = list(research_strategy.get("worker_assignments", []))
    if len(worker_assignments) != count:
        raise LLMPlanningError(
            f"research strategy allocated {len(worker_assignments)} workers; expected {count}"
        )
    assignments: list[tuple[Mapping[str, Any], str]] = []
    excluded: set[str] = set()
    for worker in worker_assignments:
        domain = str(worker.get("domain", ""))
        role = str(worker.get("portfolio_role", ""))
        candidate = next(
            (
                item for item in ordered
                if str(item.get("candidate_id")) not in excluded
                and str(item.get("domain")) == domain
            ),
            None,
        )
        if candidate is None:
            raise LLMPlanningError(
                f"research strategy assigned domain {domain!r} without a distinct matching hypothesis"
            )
        excluded.add(str(candidate.get("candidate_id")))
        assignments.append((candidate, role))
    return assignments


def _validated_research_strategy(
    slate: Mapping[str, Any], *, worker_count: int
) -> dict[str, Any]:
    strategy = slate.get("research_strategy")
    if not isinstance(strategy, Mapping):
        raise LLMPlanningError("LLM ideator omitted the research strategy")
    focus_domains = [str(item) for item in strategy.get("focus_domains", [])]
    assignments = strategy.get("worker_assignments", [])
    frozen = {str(item) for item in strategy.get("frozen_factors", [])}
    required_text = (
        "strategy_id", "phase_label", "rationale", "evidence_reference",
        "transition_criteria",
    )
    if any(len(str(strategy.get(key, "")).strip()) < (20 if key in {"rationale", "transition_criteria"} else 3) for key in required_text):
        raise LLMPlanningError("LLM research strategy lacks specific rationale or transition evidence")
    if strategy.get("decision") not in {"start", "continue", "revise"}:
        raise LLMPlanningError("LLM research strategy has an invalid phase decision")
    if strategy.get("metric_emphasis") not in {"primary", "GAUC", "nDCG@5"}:
        raise LLMPlanningError("LLM research strategy has an invalid metric emphasis")
    if not focus_domains or any(item not in _RESEARCH_DOMAINS for item in focus_domains):
        raise LLMPlanningError("LLM research strategy has invalid focus domains")
    if not isinstance(assignments, list) or len(assignments) != worker_count:
        raise LLMPlanningError(
            f"LLM research strategy must allocate exactly {worker_count} workers"
        )
    for assignment in assignments:
        domain = str(assignment.get("domain", "")) if isinstance(assignment, Mapping) else ""
        role = str(assignment.get("portfolio_role", "")) if isinstance(assignment, Mapping) else ""
        if domain not in focus_domains:
            raise LLMPlanningError("worker assignment falls outside the LLM strategy focus")
        if role not in {"single_worker", "incumbent_exploit", "independent_explore"}:
            raise LLMPlanningError("LLM research strategy has an invalid worker role")
        if worker_count == 1 and role != "single_worker":
            raise LLMPlanningError("a one-worker strategy must use the single_worker role")
        if worker_count > 1 and role == "single_worker":
            raise LLMPlanningError("a multi-worker strategy cannot use the single_worker role")
        if _DOMAIN_FACTORS[domain] <= frozen:
            raise LLMPlanningError(
                f"LLM research strategy freezes every factor available to assigned domain {domain!r}"
            )
    if any(item not in _STRATEGY_FACTORS for item in frozen):
        raise LLMPlanningError("LLM research strategy freezes an unknown factor")
    return {
        "strategy_id": str(strategy["strategy_id"]),
        "phase_label": str(strategy["phase_label"]),
        "decision": str(strategy["decision"]),
        "focus_domains": focus_domains,
        "metric_emphasis": str(strategy["metric_emphasis"]),
        "frozen_factors": sorted(frozen),
        "worker_assignments": [dict(item) for item in assignments],
        "rationale": str(strategy["rationale"]),
        "evidence_reference": str(strategy["evidence_reference"]),
        "transition_criteria": str(strategy["transition_criteria"]),
    }


def _validate_strategy_decision(
    decision: ResearchDirection | CapabilityAction,
    strategy: Mapping[str, Any],
    *,
    assigned_domain: str,
) -> None:
    if isinstance(decision, CapabilityAction):
        return
    if decision.direction_id not in _DOMAIN_DIRECTIONS.get(assigned_domain, ()):
        raise LLMPlanningError(
            f"implementer direction {decision.direction_id!r} violates assigned strategy domain {assigned_domain!r}"
        )
    if decision.preferred_factor in set(strategy.get("frozen_factors", [])):
        raise LLMPlanningError(
            f"implementer attempted to change strategy-frozen factor {decision.preferred_factor!r}"
        )


def _validate_research_campaign(
    decision: ResearchDirection | CapabilityAction,
    run_context: Mapping[str, Any],
) -> None:
    campaign = run_context.get("research_campaign", {})
    if not isinstance(campaign, Mapping):
        return
    if campaign.get("type") == "multi_task_baseline":
        if not isinstance(decision, ResearchDirection):
            raise LLMPlanningError("multi-task baseline requires an executable experiment")
        if (
            decision.direction_id != "multi_task_learning"
            or decision.preferred_factor != "training_objective"
        ):
            raise LLMPlanningError("multi-task baseline campaign selected a non-multi-task experiment")
        return
    if campaign.get("type") != "architecture_coverage":
        return
    remaining = [str(item) for item in campaign.get("remaining_mechanisms", [])]
    if not remaining:
        raise LLMPlanningError("architecture coverage campaign has no remaining executable mechanism")
    if not isinstance(decision, ResearchDirection):
        raise LLMPlanningError(
            "architecture coverage requires an executable architecture baseline, not a capability action"
        )
    if decision.direction_id != "fm_architecture" or decision.preferred_factor != "architecture":
        raise LLMPlanningError("architecture coverage campaign selected a non-architecture experiment")
    selected = str(decision.preferred_value or "")
    selected_mechanisms: tuple[str, ...]
    try:
        spec = parse_architecture_id(selected)
    except ValueError as exc:
        raise LLMPlanningError(f"architecture coverage selected invalid architecture {selected!r}: {exc}") from exc
    if spec is not None:
        selected_mechanisms = spec.interaction_paths
    else:
        selected_mechanisms = (selected,)
    if len(selected_mechanisms) != 1 or selected_mechanisms[0] not in remaining:
        raise LLMPlanningError(
            f"architecture coverage selected {selected!r}; a baseline must isolate exactly one of "
            f"the remaining mechanisms {remaining}"
        )


def _direction_from_implementation(
    implementation: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    incumbent_config: Mapping[str, Any] | None = None,
) -> ResearchDirection:
    if implementation.get("implementation_alignment") != "exact":
        reason = str(implementation.get("alignment_reason", "no exact implementation exists"))
        raise LLMPlanningError(f"hypothesis slate is not executable without semantic drift: {reason}")
    direction_id = implementation.get("direction_id")
    template = _APPROVED_DIRECTIONS.get(direction_id)
    if template is None:
        raise LLMPlanningError("LLM proposed an unsupported direction")
    execution_implementer_id = str(
        implementation.get("implementer_id")
        or implementation.get("execution_specialist_id", "")
    ).strip()
    if execution_implementer_id not in {_GENERIC_IMPLEMENTER_ID, *_LEGACY_SPECIALIST_IDS}:
        raise LLMPlanningError(
            f"LLM routed {direction_id} to an unknown implementer {execution_implementer_id!r}"
        )
    available_factors = _available_factors(
        str(direction_id), template, history, incumbent_config=incumbent_config
    )
    hypothesis = str(implementation.get("hypothesis", "")).strip()
    rationale = str(implementation.get("rationale", "")).strip()
    preferred_factor = str(implementation.get("preferred_factor", "")).strip()
    preferred_value = implementation.get("preferred_value")
    if len(hypothesis) < 20 or len(rationale) < 20:
        raise LLMPlanningError("LLM proposal lacks a sufficiently specific hypothesis or rationale")
    if preferred_factor not in available_factors:
        raise LLMPlanningError("implementer proposed a factor outside the approved search space")
    if direction_id == "fm_architecture":
        controlled_ablations = controlled_single_path_ablations(
            str((incumbent_config or {}).get("architecture", "fm"))
        )
        if preferred_value == "composed_spec":
            raw_spec = implementation.get("architecture_spec")
            if not isinstance(raw_spec, Mapping):
                raise LLMPlanningError("implementer omitted the composed architecture structure")
            try:
                preferred_value = ReviewedArchitectureSpec.from_mapping(raw_spec).architecture_id
            except ValueError as exc:
                raise LLMPlanningError(str(exc)) from exc
            if controlled_ablations and preferred_value not in controlled_ablations:
                raise LLMPlanningError(
                    "a two-path incumbent requires a controlled one-path architecture ablation"
                )
            search_space = {
                "architecture": list(controlled_ablations) or [preferred_value]
            }
        elif preferred_value in template["search_space"]["architecture"]:
            if controlled_ablations:
                raise LLMPlanningError(
                    "a two-path incumbent requires a controlled one-path architecture ablation"
                )
            search_space = template["search_space"]
        else:
            raise LLMPlanningError("implementer proposed an unavailable architecture structure")
    elif preferred_value != "controller_select":
        raise LLMPlanningError("numeric and non-architecture values must be selected by the search controller")
    else:
        search_space = template["search_space"]
    return ResearchDirection(
        direction_id=str(direction_id),
        hypothesis=hypothesis,
        rationale=rationale,
        search_space=search_space,
        success_evidence=str(implementation.get("success_evidence", "")).strip(),
        evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
        strategy=str(implementation.get("strategy", "")).strip(),
        specialist_id=_GENERIC_IMPLEMENTER_ID,
        preferred_factor=preferred_factor,
        preferred_value=preferred_value,
        specialist_rationale=rationale,
        selected_candidate_id=str(implementation.get("selected_candidate_id", "")).strip(),
        claimed_behavior=str(implementation.get("claimed_behavior", "")).strip(),
        required_capabilities=tuple(
            str(item) for item in implementation.get("implementation_requirements", [])
        ),
        lineage_parent_id=str(implementation.get("lineage_parent_id", "")).strip() or None,
        lineage_action=str(implementation.get("lineage_action", "")).strip() or None,
        evidence_reference=str(implementation.get("evidence_reference", "")).strip() or None,
    )


def _planner_decision_from_implementation(
    implementation: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    incumbent_config: Mapping[str, Any] | None = None,
) -> ResearchDirection | CapabilityAction:
    action = str(implementation.get("planner_action") or (
        "RUN_EXPERIMENT"
        if implementation.get("implementation_alignment") == "exact"
        else "BUILD_CAPABILITY"
    ))
    if action not in PLANNER_ACTIONS:
        raise LLMPlanningError(f"implementer proposed unsupported planner action {action!r}")
    if action == "RUN_EXPERIMENT":
        return _direction_from_implementation(
            implementation, history, incumbent_config=incumbent_config
        )
    gap_id = str(implementation.get("capability_gap_id", "")).strip()
    description = str(implementation.get("capability_gap_description", "")).strip()
    if action == "RUN_DIAGNOSTIC" and (not gap_id or gap_id == "none"):
        diagnostic_text = " ".join([
            str(implementation.get("hypothesis", "")),
            " ".join(str(item) for item in implementation.get("implementation_requirements", [])),
        ]).lower()
        if (
            "validation" in diagnostic_text
            and (
                "stratif" in diagnostic_text
                or ("user-activity" in diagnostic_text and "feature-coverage" in diagnostic_text)
            )
        ):
            gap_id = "stratified_validation_diagnostics"
        else:
            digest = hashlib.sha256(
                str(implementation.get("hypothesis", "")).strip().lower().encode("utf-8")
            ).hexdigest()[:12]
            gap_id = f"validation_diagnostic_{digest}"
        if not description or description == "none":
            description = (
                "Run validation-only diagnostic evidence for the selected hypothesis: "
                + str(implementation.get("hypothesis", "")).strip()
            )
    if not gap_id or gap_id == "none" or len(description) < 20:
        raise LLMPlanningError("non-experiment planner action lacks a specific capability gap")
    hypothesis = str(implementation.get("hypothesis", "")).strip()
    rationale = str(implementation.get("rationale", "")).strip()
    if len(hypothesis) < 20 or len(rationale) < 20:
        raise LLMPlanningError("capability action lacks a sufficiently specific hypothesis or rationale")
    approval_reason = str(implementation.get("approval_reason", "")).strip()
    if action == "REQUEST_HUMAN_APPROVAL" and (
        approval_reason == "none" or len(approval_reason) < 20
    ):
        raise LLMPlanningError("human approval action lacks a specific approval reason")
    return CapabilityAction(
        action=action,
        hypothesis=hypothesis,
        rationale=rationale,
        capability_gap_id=gap_id,
        capability_gap_description=description,
        required_capabilities=tuple(
            str(item) for item in implementation.get("implementation_requirements", [])
        ),
        specialist_id=_GENERIC_IMPLEMENTER_ID,
        approval_reason=approval_reason or None,
    )


# Compatibility alias for existing integrations while artifacts migrate to
# the generic implementer vocabulary.
_planner_decision_from_specialist = _planner_decision_from_implementation


_APPROVED_DIRECTIONS: Mapping[str, Mapping[str, Any]] = {
    "pointwise_fm_optimization": {
        "search_space": {"loss": ["pointwise"], "learning_rate": [0.0005, 0.001, 0.002], "l2": [0.0, 1e-6, 1e-5]},
        "scope": "Improve the existing pointwise FM training and regularisation without changing data or model family.",
    },
    "pairwise_fm_ranking": {
        "search_space": {"loss": ["pairwise"], "learning_rate": [0.0005, 0.001, 0.002], "l2": [0.0, 1e-6, 1e-5]},
        "scope": "Investigate pairwise objective alignment, then tune its learning dynamics using its own evidence lineage.",
    },
    "leakage_safe_author_affinity": {
        "search_space": {"feature_variant": ["author_affinity"]},
        "scope": "Add train-only, leave-one-out user-author affinity buckets to the FM.",
    },
    "leakage_safe_user_history": {
        "search_space": {"feature_variant": ["user_history"]},
        "scope": "Add train-only, leave-one-out user response-history buckets to the FM.",
    },
    "weekday_features": {
        "search_space": {"feature_variant": ["weekday"]},
        "scope": "Add calendar weekday as a categorical FM field without using future labels.",
    },
    "fm_architecture": {
        "search_space": {"architecture": ["deepfm", "nfm_residual"]},
        "scope": (
            "Change the FM computation graph using a reviewed declarative architecture. Preserve the FM path and "
            "compose one or two bounded embedding-MLP, bi-interaction-MLP, or cross-network residual paths with "
            "additive or learned-gate fusion. Legacy DeepFM and residual-NFM aliases remain executable."
        ),
    },
    "multi_task_learning": {
        "search_space": {
            "training_objective": [
                "multitask_click_w0.05", "multitask_click_w0.1", "multitask_click_w0.2",
            ],
        },
        "scope": (
            "Keep the accepted ranking architecture and pointwise long_view head, add a training-only "
            "click auxiliary head sharing embeddings, and select one bounded auxiliary-loss weight."
        ),
    },
}


def load_dotenv(path: str = ".env") -> None:
    """Load simple local secrets without adding a dependency."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def _ideator_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "research_strategy", "candidates", "recommended_candidate_id",
            "portfolio_rationale",
        ],
        "properties": {
            "research_strategy": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "strategy_id", "phase_label", "decision", "focus_domains",
                    "metric_emphasis", "frozen_factors", "worker_assignments",
                    "rationale", "evidence_reference", "transition_criteria",
                ],
                "properties": {
                    "strategy_id": {"type": "string", "minLength": 3, "maxLength": 80},
                    "phase_label": {"type": "string", "minLength": 3, "maxLength": 80},
                    "decision": {"type": "string", "enum": ["start", "continue", "revise"]},
                    "focus_domains": {
                        "type": "array", "minItems": 1, "maxItems": 3,
                        "items": {"type": "string", "enum": sorted(_RESEARCH_DOMAINS)},
                    },
                    "metric_emphasis": {
                        "type": "string", "enum": ["primary", "GAUC", "nDCG@5"],
                    },
                    "frozen_factors": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(_STRATEGY_FACTORS)},
                    },
                    "worker_assignments": {
                        "type": "array", "minItems": 1, "maxItems": 2,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["domain", "portfolio_role"],
                            "properties": {
                                "domain": {"type": "string", "enum": sorted(_RESEARCH_DOMAINS)},
                                "portfolio_role": {
                                    "type": "string",
                                    "enum": ["single_worker", "incumbent_exploit", "independent_explore"],
                                },
                            },
                        },
                    },
                    "rationale": {"type": "string", "minLength": 20},
                    "evidence_reference": {"type": "string", "minLength": 4},
                    "transition_criteria": {"type": "string", "minLength": 20},
                },
            },
            "candidates": {
                "type": "array", "minItems": 3, "maxItems": 4,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": [
                        "candidate_id", "domain", "hypothesis", "rationale",
                        "expected_mechanism", "required_capabilities", "proposed_action",
                        "lineage_parent_id", "lineage_action", "evidence_reference",
                    ],
                    "properties": {
                        "candidate_id": {"type": "string", "minLength": 2},
                        "domain": {"type": "string", "enum": [
                            "model_architecture", "ranking_objective", "training_optimization",
                            "feature_data", "temporal", "evaluation_diagnostics",
                        ]},
                        "hypothesis": {"type": "string", "minLength": 20},
                        "rationale": {"type": "string", "minLength": 20},
                        "expected_mechanism": {"type": "string", "minLength": 20},
                        "required_capabilities": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "proposed_action": {
                            "type": "string",
                            "enum": ["RUN_EXPERIMENT", "RUN_DIAGNOSTIC"],
                        },
                        "lineage_parent_id": {"type": "string", "minLength": 3},
                        "lineage_action": {
                            "type": "string",
                            "enum": ["continue", "refine", "revisit", "branch_new"],
                        },
                        "evidence_reference": {"type": "string", "minLength": 4},
                    },
                },
            },
            "recommended_candidate_id": {"type": "string", "minLength": 2},
            "portfolio_rationale": {"type": "string", "minLength": 20},
        },
    }


def _implementer_schema(
    candidate_ids: Sequence[str],
    capability_manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    direction_ids = list(capability_manifest) + ["none"]
    factors = sorted({
        factor
        for item in capability_manifest.values()
        for factor in item.get("available_factors", [])
    }) + ["none"]
    architecture_only = set(capability_manifest) == {"fm_architecture"}
    preferred_values = (
        ["deepfm", "nfm_residual", "composed_spec", "unavailable"]
        if architecture_only
        else ["controller_select", "deepfm", "nfm_residual", "composed_spec", "unavailable"]
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "selected_candidate_id", "direction_id", "implementer_id",
            "implementation_alignment", "alignment_reason",
            "hypothesis", "claimed_behavior", "implementation_requirements", "rationale",
            "preferred_factor", "preferred_value", "success_evidence", "strategy",
            "deferred_candidate_ids", "planner_action", "capability_gap_id",
            "capability_gap_description", "approval_reason", "architecture_spec",
            "lineage_parent_id", "lineage_action", "evidence_reference",
        ],
        "properties": {
            "selected_candidate_id": {"type": "string", "enum": list(candidate_ids)},
            "direction_id": {"type": "string", "enum": direction_ids},
            "implementer_id": {"type": "string", "enum": [_GENERIC_IMPLEMENTER_ID]},
            "implementation_alignment": {"type": "string", "enum": ["exact", "unavailable"]},
            "alignment_reason": {"type": "string", "minLength": 20},
            "hypothesis": {"type": "string", "minLength": 20},
            "claimed_behavior": {"type": "string", "minLength": 20},
            "implementation_requirements": {
                "type": "array", "items": {"type": "string"}, "minItems": 1,
            },
            "rationale": {"type": "string", "minLength": 20},
            "preferred_factor": {"type": "string", "enum": factors},
            "preferred_value": {
                "type": "string",
                "enum": preferred_values,
            },
            "architecture_spec": {
                "anyOf": [architecture_schema(), {"type": "null"}],
            },
            "success_evidence": {"type": "string", "minLength": 20},
            "strategy": {"type": "string", "enum": ["exploration", "local_refinement", "diverse_restart"]},
            "deferred_candidate_ids": {"type": "array", "items": {"type": "string", "enum": list(candidate_ids)}},
            "planner_action": {"type": "string", "enum": list(PLANNER_ACTIONS)},
            "capability_gap_id": {"type": "string", "minLength": 3, "maxLength": 80},
            "capability_gap_description": {"type": "string", "minLength": 4},
            "approval_reason": {"type": "string", "minLength": 4},
            "lineage_parent_id": {"type": "string", "minLength": 3},
            "lineage_action": {
                "type": "string",
                "enum": ["continue", "refine", "revisit", "branch_new"],
            },
            "evidence_reference": {"type": "string", "minLength": 4},
        },
    }


def _ideator_instructions() -> str:
    return (
        "You are an open-ended recommender-systems research lead. First decide the current research strategy from "
        "the supplied evidence: whether to start, continue, or revise a phase; which scientific domains deserve "
        "focus; which metric needs diagnostic emphasis; which unrelated factors must remain frozen; how workers "
        "should be allocated; and what evidence should trigger a phase transition. This strategy is your scientific "
        "decision, not a fixed controller policy. The official acceptance objective always remains primary. Then "
        "generate a portfolio of falsifiable next hypotheses consistent with that strategy. Diagnose the evidence and generate a diverse "
        "portfolio of falsifiable next hypotheses before considering implementation availability. Explore model "
        "architecture, interaction structure, objectives, optimisation, representations, and diagnostics when the "
        "evidence supports them. Describe required scientific behavior, not repository identifiers or assumed code. "
        "Use the research_tree evidence to label every candidate as continuing, refining, revisiting, or branching "
        "from prior work, and cite the supporting hypothesis or experiment ID. Do not repeat a failed branch unless "
        "new evidence or a changed mechanism makes its prediction materially different. "
        "Treat critic_memory as binding feedback: preserve supported mechanisms, do not repeat measured valid-negative "
        "configurations, and repair lineage or implementation alignment before proposing the affected branch again. "
        "When stratified validation evidence is available, name the weakest statistically eligible stratum before "
        "proposing a targeted intervention and do not generalize beyond that slice evidence. "
        "Do not use test labels, external data, or pretrained benchmark weights. A new dependency or substantially "
        "different model family requires later human review. Avoid repeating a failed mechanism unless new evidence "
        "specifically changes its prediction. Mark evidence-only investigations as RUN_DIAGNOSTIC and trainable "
        "hypotheses as RUN_EXPERIMENT; implementation availability is assessed later by one generic implementer."
    )


def _ideator_prompt(
    history: Sequence[Mapping[str, Any]],
    state: ResearchState,
    run_context: Mapping[str, Any] | None = None,
    *,
    worker_count: int = 1,
) -> str:
    return json.dumps(
        {
            "benchmark": {
                "selection_split": "valid",
                "objective": "primary=(GAUC+nDCG@5)/2",
                "acceptance_rule": "any positive primary gain over the incumbent",
                "convergence_threshold": 0.002,
            },
            "state": state.as_dict(),
            "available_worker_count": worker_count,
            "governance_context": _compact_planner_context(run_context or {}),
            "recent_evidence": _compact_history(history, include_direction_ids=False),
            "request": (
                "Author a research strategy for this batch and generate 3-4 materially different hypotheses under "
                "it. Assign one worker domain per available worker described in governance context, freeze unrelated "
                "factors; use single_worker only when one worker is available, and use incumbent_exploit or "
                "independent_explore for every slot in a multi-worker batch; "
                "factors, state evidence-based transition criteria, identify required capabilities and scientific "
                "actions, explicitly relate each hypothesis to the research tree, and recommend the strongest "
                "candidate. Metric emphasis guides diagnosis and hypotheses but cannot replace the "
                "fixed primary acceptance rule. Do not assume what is implemented."
            ),
        },
        sort_keys=True, default=str,
    )


def _implementer_instructions() -> str:
    return (
        "You are the single generic ML research implementer. Adopt whatever architecture, feature, objective, training, "
        "or evaluation expertise the assigned hypothesis requires. Compare the exact scientific hypothesis with the "
        "full executable capability registry. Use RUN_EXPERIMENT only when the assigned candidate's "
        "claimed behavior has an exact implementation path. Never relabel, weaken, or replace an assigned hypothesis "
        "merely to make it executable. Use RUN_DIAGNOSTIC for an evidence-only question; BUILD_CAPABILITY when a safe "
        "missing implementation can be added within current authority; or REQUEST_HUMAN_APPROVAL when it requires a "
        "new dependency, substantially different model family, external authority, or expanded scope. For every "
        "non-experiment action, provide a stable gap id and concrete description. Numeric values are selected later. "
        "Set implementer_id=generic_implementer. Only architecture work may select a categorical reviewed "
        "architecture value; all other decisions must return preferred_value=controller_select. For fm_architecture, choose a legacy alias or "
        "preferred_value=composed_spec with an exact architecture_spec assembled from the manifest's reviewed "
        "operators and bounds. Return architecture_spec=null for non-architecture actions. Never request arbitrary "
        "model code through this field."
        " Preserve the selected ideator candidate's lineage_parent_id, lineage_action, and evidence_reference so "
        "the critic can verify why this branch follows from prior evidence. Apply every relevant critic_memory "
        "constraint explicitly; do not send a known alignment defect to training again."
    )


def _implementer_prompt(
    slate: Mapping[str, Any],
    capability_manifest: Mapping[str, Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    state: ResearchState,
    *,
    assigned_candidate_id: str | None = None,
    worker_slot: int | None = None,
    portfolio_role: str | None = None,
    run_context: Mapping[str, Any] | None = None,
    require_executable: bool = False,
) -> str:
    candidates = [
        item for item in slate.get("candidates", [])
        if isinstance(item, Mapping)
    ]
    assigned_candidate = next(
        (
            dict(item) for item in candidates
            if str(item.get("candidate_id")) == str(assigned_candidate_id)
        ),
        {},
    )
    return json.dumps({
        "assigned_hypothesis": assigned_candidate,
        "research_strategy": dict(slate.get("research_strategy", {})),
        "executable_capability_manifest": capability_manifest,
        "incumbent": {
            "experiment_id": state.current_best_experiment_id,
            "primary": state.current_best_primary,
        },
        "implementation_context": _compact_implementer_context(run_context or {}),
        "recent_evidence": _compact_history(history)[-2:],
        "implementation_assignment": {
            "worker_slot": worker_slot,
            "candidate_id": assigned_candidate_id,
            "portfolio_role": portfolio_role,
            "role_requirement": (
                "Develop a controlled descendant of the accepted incumbent and preserve all unrelated settings."
                if portfolio_role == "incumbent_exploit"
                else (
                    "Test a materially different mechanism from the exploitation worker while still branching from the incumbent."
                    if portfolio_role == "independent_explore"
                    else "Execute the single worker domain chosen by the LLM research strategy."
                )
            ),
            "diversity_requirement": "Do not replace this assignment with the other worker's mechanism.",
        } if assigned_candidate_id else None,
        "scheduler_backfill": {
            "required": require_executable,
            "instruction": (
                "Fill an idle experiment-worker slot using the assigned hypothesis. Return RUN_EXPERIMENT only when "
                "it is exactly executable; otherwise return a diagnostic, capability, or approval action without substitution."
                if require_executable else "Normal planning action semantics apply."
            ),
        },
        "request": (
            "When implementation_assignment is present, evaluate only that assigned candidate. If it is exactly executable, "
            "return RUN_EXPERIMENT and name the matching direction and factor. If it is diagnostic, return "
            "RUN_DIAGNOSTIC. If implementation is missing, return BUILD_CAPABILITY or REQUEST_HUMAN_APPROVAL; do not "
            "substitute another manifest experiment. For RUN_EXPERIMENT use capability_gap_id=none, "
            "capability_gap_description=none, and approval_reason=none. Mark other slate ideas deferred."
            + (
                " This is a scheduler backfill consultation, but exact hypothesis identity remains mandatory."
                if require_executable else ""
            )
        ),
    }, sort_keys=True, default=str)


def _capability_manifest(
    history: Sequence[Mapping[str, Any]],
    direction_ids: Sequence[str],
    *,
    incumbent_config: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    manifest = {
        direction_id: {
            "behavior": _APPROVED_DIRECTIONS[direction_id]["scope"],
            "available_factors": _available_factors(
                direction_id, _APPROVED_DIRECTIONS[direction_id], history,
                incumbent_config=incumbent_config,
            ),
            "implementer": _GENERIC_IMPLEMENTER_ID,
        }
        for direction_id in direction_ids
    }
    if "fm_architecture" in manifest:
        manifest["fm_architecture"]["reviewed_architecture_language"] = {
            "operators": list(REVIEWED_OPERATORS),
            "fusions": list(REVIEWED_FUSIONS),
            "hidden_widths": list(REVIEWED_WIDTHS),
            "hidden_depths": list(REVIEWED_DEPTHS),
            "dropouts": list(REVIEWED_DROPOUTS),
            "cross_layers": list(REVIEWED_CROSS_LAYERS),
            "note": "The response JSON schema independently enforces these values.",
        }
        manifest["fm_architecture"]["legacy_aliases"] = ["deepfm", "nfm_residual"]
        manifest["fm_architecture"]["resource_bounds"] = {
            "max_interaction_paths": 2,
            "max_hidden_width": 64,
            "max_hidden_depth": 3,
            "max_cross_layers": 3,
        }
        manifest["fm_architecture"]["parameter_relevance"] = {
            "hidden_width_hidden_depth_dropout": "used only by embedding_mlp or bi_interaction_mlp; otherwise use 32, 2, 0.1",
            "cross_layers": "used only by cross_network; otherwise use 2",
            "interaction_path_order": list(REVIEWED_OPERATORS),
        }
        controlled_ablations = controlled_single_path_ablations(
            str((incumbent_config or {}).get("architecture", "fm"))
        )
        manifest["fm_architecture"]["controlled_single_path_ablations"] = list(
            controlled_ablations
        )
        manifest["fm_architecture"]["attribution_rule"] = (
            "When this list is non-empty, select exactly one listed ablation and "
            "claim only the contribution isolated by removing the other path."
        )
    return manifest


def _compact_planner_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Retain strategic evidence without copying full run artifacts into prompts."""
    tree = context.get("research_tree", {})
    if not isinstance(tree, Mapping):
        tree = {}
    critic = context.get("critic_memory", {})
    if not isinstance(critic, Mapping):
        critic = {}
    compact_critic: dict[str, Any] = {
        "feedback_count": critic.get("feedback_count", 0),
    }
    for key in (
        "supported_lineages", "valid_negative_results", "alignment_repairs",
        "execution_failures",
    ):
        values = critic.get(key, [])
        compact_critic[key] = list(values[-3:]) if isinstance(values, list) else []
    strategies = context.get("recent_llm_research_strategies", [])
    return {
        "benchmark_contract": context.get("benchmark_contract", {}),
        "architecture_guidance": context.get("architecture_guidance", {}),
        "manual_interventions": list(context.get("manual_interventions", []))[-3:],
        "recent_errors_and_recoveries": list(context.get("recent_errors_and_recoveries", []))[-3:],
        "capability_backlog": list(context.get("capability_backlog", []))[-4:],
        "recent_diagnostics": list(context.get("recent_diagnostics", []))[-2:],
        "implemented_diagnostic_capabilities": context.get(
            "implemented_diagnostic_capabilities", {}
        ),
        "research_tree": {
            "incumbent": tree.get("incumbent", {}),
            "continuation_candidates": list(tree.get("continuation_candidates", []))[:6],
            "experiment_count": tree.get("experiment_count", 0),
            "hypothesis_count": tree.get("hypothesis_count", 0),
        },
        "research_coverage": context.get("research_coverage", {}),
        "research_campaign": context.get("research_campaign"),
        "critic_memory": compact_critic,
        "recent_llm_research_strategies": list(strategies[-2:]) if isinstance(strategies, list) else [],
        "metric_history_count": context.get("metric_history_count", 0),
        "planner_self_correction": context.get("planner_self_correction"),
    }


def _compact_implementer_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Give implementers authority and repair constraints, not the planner's full memory."""
    critic = context.get("critic_memory", {})
    if not isinstance(critic, Mapping):
        critic = {}
    return {
        "benchmark_contract": context.get("benchmark_contract", {}),
        "manual_interventions": [
            {
                "approval_id": item.get("approval_id"),
                "authority_scope": item.get("authority_scope"),
                "description": item.get("description"),
            }
            for item in list(context.get("manual_interventions", []))[-3:]
            if isinstance(item, Mapping)
        ],
        "alignment_repairs": list(critic.get("alignment_repairs", []))[-3:]
        if isinstance(critic.get("alignment_repairs", []), list) else [],
        "valid_negative_results": list(critic.get("valid_negative_results", []))[-3:]
        if isinstance(critic.get("valid_negative_results", []), list) else [],
        "research_campaign": context.get("research_campaign"),
    }


def _compact_history(
    history: Sequence[Mapping[str, Any]],
    *,
    include_direction_ids: bool = True,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in history[-6:]:
        evidence = item.get("diagnostic_evidence", {}) or {}
        semantic_review = item.get("semantic_review", {}) or {}
        semantic_trace = semantic_review.get("trace", {}) or {}
        diagnostics = evidence.get("model_diagnostics", []) or []
        best_diagnostic = next(
            (value for value in diagnostics if value.get("stage_id") == evidence.get("best_stage_id")),
            diagnostics[-1] if diagnostics else {},
        )
        coverage = best_diagnostic.get("feature_coverage", {}) or {}
        fields = coverage.get("fields", []) if isinstance(coverage, dict) else []
        record = {
            "experiment_id": item.get("experiment_id"),
            "hypothesis": item.get("hypothesis"),
            "decision": item.get("decision"),
            "metrics": item.get("metrics"),
            "terminal_reason": item.get("terminal_reason"),
            "error": item.get("error"),
            "evidence": {
                "gain_over_incumbent": evidence.get("gain_over_incumbent"),
                "robust_primary": evidence.get("robust_primary"),
                "best_stage_id": evidence.get("best_stage_id"),
                "seed_confirmation": evidence.get("seed_confirmation", {}),
                "reconciliation": evidence.get("reconciliation", {}),
                "architecture_ablation": evidence.get("architecture_ablation", {}),
                "training": best_diagnostic.get("training", {}),
                "score_distribution": best_diagnostic.get("score_distribution", {}),
                "user_segments": best_diagnostic.get("user_segments", {}),
                "stratified_validation": _compact_stratified_validation(
                    best_diagnostic.get("stratified_validation", {})
                ),
                "feature_coverage_summary": {
                    "field_count": len(fields),
                    "max_unseen_validation_fraction": max(
                        (float(field.get("validation_unseen_fraction", 0.0)) for field in fields), default=0.0
                    ),
                },
                "failures": evidence.get("failures", []),
                "semantic_integrity": {
                    "approved": semantic_review.get("approved"),
                    "verdict": semantic_trace.get("verdict"),
                    "implementation_id": semantic_trace.get("implementation_id"),
                    "failed_checks": [
                        name for name, passed in semantic_trace.get("checks", {}).items() if not passed
                    ],
                    "planner_feedback": semantic_trace.get("planner_feedback", {}),
                },
            },
        }
        if include_direction_ids:
            record["direction_id"] = item.get("direction_id")
        compact.append(record)
    return compact


def _compact_stratified_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep actionable slice evidence without repeating hashes and full artifacts."""
    if not isinstance(value, Mapping):
        return {}

    def compact_group(group: Any) -> dict[str, Any]:
        if not isinstance(group, Mapping):
            return {}
        return {
            str(name): {
                key: metrics.get(key)
                for key in ("rows", "users", "primary", "GAUC", "nDCG@5", "metrics_available")
                if key in metrics
            }
            for name, metrics in group.items()
            if isinstance(metrics, Mapping)
        }

    return {
        "selection_split": value.get("selection_split"),
        "test_data_used": value.get("test_data_used"),
        "boundary_source": value.get("boundary_source"),
        "minimum_rows_for_weakest_stratum": value.get("minimum_rows_for_weakest_stratum"),
        "weakest_statistically_eligible_stratum": value.get(
            "weakest_statistically_eligible_stratum"
        ),
        "user_activity": compact_group(value.get("user_activity")),
        "feature_coverage": compact_group(value.get("feature_coverage")),
    }


def _combined_usage(*metadata: Mapping[str, Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in metadata:
        usage = item.get("usage", {})
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
        input_details = usage.get("input_tokens_details", {})
        cached_tokens = input_details.get("cached_tokens") if isinstance(input_details, Mapping) else None
        if isinstance(cached_tokens, int):
            totals["cached_input_tokens"] = totals.get("cached_input_tokens", 0) + cached_tokens
    return totals


def _enforce_token_budget(token_budget: int, metadata: Mapping[str, Any]) -> None:
    if token_budget <= 0:
        return
    usage = metadata.get("usage", {})
    total = usage.get("total_tokens")
    if not isinstance(total, int):
        total = sum(
            value for key, value in usage.items()
            if key in {"input_tokens", "output_tokens"} and isinstance(value, int)
        )
    if total > token_budget:
        raise LLMPlanningError(f"planner token budget exceeded: {total} > {token_budget}")


def _available_direction_ids(
    history: Sequence[Mapping[str, Any]],
    *,
    include_architecture: bool = True,
    incumbent_config: Mapping[str, Any] | None = None,
) -> list[str]:
    return [
        direction_id
        for direction_id, template in _APPROVED_DIRECTIONS.items()
        if include_architecture or direction_id != "fm_architecture"
        if _available_factors(
            direction_id, template, history, incumbent_config=incumbent_config
        )
    ]


def _available_factors(
    direction_id: str,
    template: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    incumbent_config: Mapping[str, Any] | None = None,
) -> list[str]:
    current = dict(incumbent_config or {
        "loss": "pointwise", "learning_rate": 0.001, "l2": 1e-6,
        "feature_variant": "baseline", "architecture": "fm", "training_objective": "pointwise",
    })
    relevant = [
        item for item in history
        if item.get("direction_id") == direction_id and _has_measured_evidence(item)
    ]
    if direction_id == "fm_architecture":
        # This language contains many bounded compositions. Exact duplicate
        # structures are still excluded by measured-config checks downstream.
        return ["architecture"]
    if direction_id == "pairwise_fm_ranking" and current.get("loss") != "pairwise":
        pairwise_tested = any(
            item.get("config", {}).get("loss") == "pairwise"
            and _same_planner_context(
                item.get("config", {}), current, ignored_factor="loss"
            )
            for item in relevant
        )
        return [] if pairwise_tested else ["loss"]
    factors: list[str] = []
    for factor, values in template["search_space"].items():
        if factor == "loss":
            continue
        candidates = [value for value in values if value != current.get(factor)]
        seen = {
            item.get("config", {}).get(factor) for item in relevant
            if _same_planner_context(item.get("config", {}), current, ignored_factor=factor)
        }
        if any(value not in seen for value in candidates):
            factors.append(factor)
    return factors


def _same_planner_context(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    ignored_factor: str | None = None,
) -> bool:
    ignored = {"epochs", "fidelity", "worker_threads", "seed"}
    if ignored_factor:
        ignored.add(ignored_factor)
    # Historical test fixtures and migrated records may be partial; omitted
    # keys mean "unchanged from incumbent". Real records contain full configs.
    keys = set(candidate) - ignored
    return all(candidate.get(key) == incumbent.get(key) for key in keys)


def _planner_incumbent_config(
    history: Sequence[Mapping[str, Any]], state: ResearchState
) -> Mapping[str, Any]:
    baseline = {
        "loss": "pointwise", "learning_rate": 0.001, "l2": 1e-6,
        "embedding_dim": 16, "batch_size": 8192, "feature_variant": "baseline",
        "architecture": "fm", "training_objective": "pointwise",
    }
    if state.current_best_experiment_id == "baseline":
        return baseline
    record = next(
        (item for item in reversed(history) if item.get("experiment_id") == state.current_best_experiment_id),
        None,
    )
    if record is None:
        raise LLMPlanningError(
            "accepted incumbent configuration is unavailable for planner context: "
            f"{state.current_best_experiment_id}"
        )
    return {**baseline, **dict(record.get("config", {}))}


def _planner_incumbent_direction(
    history: Sequence[Mapping[str, Any]], state: ResearchState
) -> str | None:
    return next(
        (
            str(item.get("direction_id")) for item in reversed(history)
            if item.get("experiment_id") == state.current_best_experiment_id
            and item.get("direction_id")
        ),
        None,
    )


def _has_measured_evidence(record: Mapping[str, Any]) -> bool:
    return has_measured_validation_evidence(record)


class OfflinePlanner:
    """Deterministic planner for tests, demos, and runs without an API key."""

    def __init__(self, *, seed: int = 0) -> None:
        from .planner import EvidencePlanner

        self._planner = EvidencePlanner(seed=seed)
        self.last_metadata = {"mode": "offline", "model": "deterministic-evidence-planner"}
        self.run_context: dict[str, Any] = {}

    def set_run_context(self, context: Mapping[str, Any]) -> None:
        self.run_context = dict(context)

    def propose(self, history: Sequence[Mapping[str, Any]], state: ResearchState) -> ResearchDirection:
        return self._planner.propose(history, state)

    def propose_batch(
        self,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        *,
        count: int = 2,
    ) -> list[ResearchDirection]:
        available = list(self._planner._available_directions(history))
        incumbent_direction = _planner_incumbent_direction(history, state)
        objective_direction = (
            "pairwise_fm_ranking"
            if _planner_incumbent_config(history, state).get("loss") == "pairwise"
            else "pointwise_fm_optimization"
        )
        exploit = next(
            (
                item for direction_id in (incumbent_direction, objective_direction)
                for item in available if direction_id and item.direction_id == direction_id
            ),
            available[0] if available else None,
        )
        selected: list[ResearchDirection] = []
        if exploit is not None:
            selected.append(replace(
                exploit,
                portfolio_role="incumbent_exploit" if count > 1 else "single_worker",
                strategy="local_refinement" if count > 1 else exploit.strategy,
            ))
        if count > 1:
            explore = next((item for item in available if item.direction_id != exploit.direction_id), None)
            if explore is not None:
                selected.append(replace(
                    explore, portfolio_role="independent_explore", strategy="exploration"
                ))
        if not selected:
            raise ResearchCatalogueExhausted("the approved executable research catalogue is exhausted")
        self.last_metadata = {
            "mode": "offline_parallel_portfolio",
            "model": "deterministic-evidence-planner",
            "planned_workers": len(selected),
            "portfolio_roles": [item.portfolio_role for item in selected],
        }
        return selected


def _retryable_http_status(status: int) -> bool:
    return status in {408, 409, 429} or status >= 500


def _http_error_category(status: int) -> str:
    if status in {401, 403}:
        return "authentication_error"
    if status == 429:
        return "rate_limited"
    if 400 <= status < 500:
        return "client_error"
    return "server_error"


def _http_error_detail(error: HTTPError) -> str:
    """Extract a bounded provider message without logging request headers or secrets."""
    try:
        payload = json.loads(error.read().decode("utf-8"))
        detail = payload.get("error", {}).get("message")
        if not isinstance(detail, str):
            return ""
        return " ".join(detail.split())[:600]
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
