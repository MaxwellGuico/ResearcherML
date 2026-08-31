"""Dependency-free search controller for selecting exact trial configurations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .planner import ResearchDirection
from .safety import ExperimentProposal, has_measured_validation_evidence
from .state import ResearchState


@dataclass(frozen=True)
class SearchState:
    status: str = "BOOTSTRAP"
    region_id: str = "region_01"
    strategy: str = "exploration"
    fidelity: str = "low"


class SearchController:
    """Selects exact one-factor trials after the planner selects a direction."""

    BASELINE_CONFIG: Mapping[str, Any] = {
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

    def __init__(
        self,
        *,
        seed: int = 0,
        architecture_human_reviewed: bool = False,
        worker_threads: int = 1,
    ) -> None:
        if worker_threads not in {1, 2}:
            raise ValueError("worker_threads must be 1 or 2")
        self.seed = seed
        self.architecture_human_reviewed = architecture_human_reviewed
        self.worker_threads = worker_threads

    def propose_trial(
        self,
        direction: ResearchDirection,
        state: ResearchState,
        history: Sequence[Mapping[str, Any]],
        *,
        search_state: SearchState | None = None,
        reserved_experiment_ids: Sequence[str] = (),
        reserved_configs: Sequence[Mapping[str, Any]] = (),
    ) -> ExperimentProposal:
        search_state = search_state or SearchState(strategy=direction.strategy)
        parent_experiment_id = state.current_best_experiment_id
        config = self._incumbent_config(state, history)
        # Operational parallelism is invocation policy, not inherited research
        # state and not one of the scientific factors under test.
        config["worker_threads"] = self.worker_threads
        config["seed"] = self.seed
        active_factor = self._choose_active_factor(
            direction, history, state, search_state.strategy, current_config=config
        )
        value = self._choose_value(
            direction, active_factor, history, state, search_state.strategy,
            current_value=config.get(active_factor), incumbent_config=config,
            reserved_configs=reserved_configs,
        )
        config[active_factor] = value
        config["epochs"] = int(direction.evaluation_budget["low_epochs"])
        config["fidelity"] = search_state.fidelity
        experiment_id = self._next_experiment_id(history, reserved_experiment_ids)
        hypothesis = direction.hypothesis
        rationale = (
            f"Search Controller selected {active_factor} from direction "
            f"{direction.direction_id} using {search_state.strategy}, as a controlled "
            f"change from accepted incumbent {parent_experiment_id}."
        )
        model_family = "fm" if config.get("architecture", "fm") == "fm" else "fm_hybrid"
        inherited_architecture_review = (
            model_family == "fm_hybrid"
            and parent_experiment_id != "baseline"
            and active_factor != "architecture"
        )
        return ExperimentProposal(
            experiment_id=experiment_id,
            parent_experiment_id=parent_experiment_id,
            hypothesis=hypothesis,
            rationale=rationale,
            config=config,
            changed_factors=(active_factor,),
            runtime_budget_seconds=600.0,
            research_direction_id=direction.direction_id,
            search_strategy=search_state.strategy,
            search_region_id=search_state.region_id,
            portfolio_role=direction.portfolio_role,
            model_family=model_family,
            human_reviewed=(
                self.architecture_human_reviewed or inherited_architecture_review
                if model_family == "fm_hybrid" else False
            ),
        )

    def _choose_active_factor(
        self,
        direction: ResearchDirection,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        strategy: str = "exploration",
        current_config: Mapping[str, Any] | None = None,
    ) -> str:
        direction_history = [
            item for item in history
            if item.get("direction_id") == direction.direction_id
            and self._has_search_evidence(item)
        ]
        if (
            direction.direction_id == "pairwise_fm_ranking"
            and (current_config or {}).get("loss") != "pairwise"
        ):
            foundation_tested = any(
                item.get("config", {}).get("loss") == "pairwise"
                and self._same_factor_context(
                    item.get("config", {}), current_config or {}, "loss"
                )
                for item in direction_history
            )
            if not foundation_tested:
                return "loss"
        candidates = [key for key in direction.search_space if key != "loss"]
        if not candidates:
            raise ValueError(f"direction has no searchable factor: {direction.direction_id}")
        counts = {
            factor: sum(factor in item.get("changed_factors", []) for item in direction_history)
            for factor in candidates
        }
        if direction.preferred_factor in candidates:
            preferred = str(direction.preferred_factor)
            if counts[preferred] == min(counts.values()):
                return preferred
        if strategy == "local_refinement":
            scored = {
                factor: max(
                    (float(item.get("metrics", {}).get("primary", float("-inf")))
                     for item in direction_history
                     if factor in item.get("changed_factors", [])),
                    default=float("-inf"),
                )
                for factor in candidates
            }
            return max(candidates, key=lambda factor: (scored[factor], -counts[factor], factor))
        minimum = min(counts.values())
        return sorted(factor for factor, count in counts.items() if count == minimum)[0]

    def _choose_value(
        self,
        direction: ResearchDirection,
        factor: str,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
        strategy: str = "exploration",
        current_value: Any = None,
        incumbent_config: Mapping[str, Any] | None = None,
        reserved_configs: Sequence[Mapping[str, Any]] = (),
    ) -> Any:
        values = list(direction.search_space[factor])
        candidates = [value for value in values if value != current_value]
        if not candidates:
            raise ValueError(
                f"direction {direction.direction_id} has no {factor} value different from "
                f"accepted incumbent value {current_value!r}"
            )
        seen = {
            item.get("config", {}).get(factor)
            for item in history
            if item.get("direction_id") == direction.direction_id
            and self._has_search_evidence(item)
            and self._same_factor_context(item.get("config", {}), incumbent_config or {}, factor)
        }
        unseen = [value for value in candidates if value not in seen]
        reserved_values = {
            item.get(factor)
            for item in reserved_configs
            if self._same_factor_context(item, incumbent_config or {}, factor)
        }
        unreserved = [value for value in unseen if value not in reserved_values]
        if unreserved:
            unseen = unreserved
        elif unseen and all(value in reserved_values for value in unseen):
            raise ValueError(
                f"all remaining {factor} values for {direction.direction_id} are already "
                "reserved by sibling experiments"
            )
        if direction.preferred_value in unseen:
            return direction.preferred_value
        if unseen:
            # Values are ordered by distance from the best observed value when
            # refining; exploration uses a stable low-to-high traversal.
            if strategy == "local_refinement":
                observations = [
                    (float(item.get("metrics", {}).get("primary", float("-inf"))), item.get("config", {}).get(factor))
                    for item in history
                    if item.get("config", {}).get(factor) in candidates
                ]
                if observations:
                    best_value = max(observations)[1]
                    return min(unseen, key=lambda value: abs(float(value) - float(best_value)))
            return sorted(unseen, key=lambda value: str(value))[0]
        return sorted(candidates, key=lambda value: str(value))[0]

    def _incumbent_config(
        self,
        state: ResearchState,
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return a complete canonical config for the accepted parent only."""
        if state.current_best_experiment_id == "baseline":
            return dict(self.BASELINE_CONFIG)
        parent = next(
            (
                item for item in reversed(history)
                if str(item.get("experiment_id")) == state.current_best_experiment_id
                and isinstance(item.get("config"), Mapping)
            ),
            None,
        )
        if parent is None:
            raise ValueError(
                "accepted incumbent configuration is unavailable for "
                f"{state.current_best_experiment_id}; refusing to branch from baseline"
            )
        config = dict(self.BASELINE_CONFIG)
        config.update(parent["config"])
        return config

    @staticmethod
    def _same_factor_context(
        candidate: Mapping[str, Any],
        incumbent: Mapping[str, Any],
        factor: str,
    ) -> bool:
        """Only consume values tested on the same model/feature/objective context."""
        ignored = {factor, "epochs", "fidelity", "worker_threads", "seed"}
        keys = (set(candidate) | set(incumbent)) - ignored
        return all(candidate.get(key) == incumbent.get(key) for key in keys)

    @staticmethod
    def _has_search_evidence(record: Mapping[str, Any]) -> bool:
        return has_measured_validation_evidence(record)

    @staticmethod
    def _next_experiment_id(
        history: Sequence[Mapping[str, Any]],
        reserved_experiment_ids: Sequence[str] = (),
    ) -> str:
        existing = {str(item.get("experiment_id", "")) for item in history}
        existing.update(str(item) for item in reserved_experiment_ids)
        number = 1
        while f"exp_{number:03d}" in existing:
            number += 1
        return f"exp_{number:03d}"
