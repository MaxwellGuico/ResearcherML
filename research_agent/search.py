"""Dependency-free search controller for selecting exact trial configurations."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .planner import ResearchDirection
from .safety import ExperimentProposal
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
        "learning_rate": 0.001,
        "l2": 1e-6,
        "embedding_dim": 16,
        "batch_size": 8192,
        "seed": 0,
    }

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def propose_trial(
        self,
        direction: ResearchDirection,
        state: ResearchState,
        history: Sequence[Mapping[str, Any]],
        *,
        search_state: SearchState | None = None,
    ) -> ExperimentProposal:
        search_state = search_state or SearchState(strategy=direction.strategy)
        active_factor = self._choose_active_factor(direction, history, state)
        value = self._choose_value(direction, active_factor, history, state)
        config = dict(self.BASELINE_CONFIG)
        config[active_factor] = value
        config["epochs"] = int(direction.evaluation_budget["low_epochs"])
        config["fidelity"] = search_state.fidelity
        experiment_id = self._next_experiment_id(history)
        hypothesis = direction.hypothesis
        rationale = (
            f"Search Controller selected {active_factor} from direction "
            f"{direction.direction_id} using {search_state.strategy}."
        )
        return ExperimentProposal(
            experiment_id=experiment_id,
            parent_experiment_id=state.current_best_experiment_id,
            hypothesis=hypothesis,
            rationale=rationale,
            config=config,
            changed_factors=(active_factor,),
            runtime_budget_seconds=600.0,
            research_direction_id=direction.direction_id,
            search_strategy=search_state.strategy,
            search_region_id=search_state.region_id,
        )

    def _choose_active_factor(
        self,
        direction: ResearchDirection,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> str:
        if direction.direction_id == "pairwise_fm_ranking":
            return "loss"
        candidates = [key for key in direction.search_space if key not in {"loss"}]
        if not candidates:
            raise ValueError(f"direction has no searchable factor: {direction.direction_id}")
        counts = {
            factor: sum(factor in item.get("changed_factors", []) for item in history)
            for factor in candidates
        }
        minimum = min(counts.values())
        tied = sorted(factor for factor, count in counts.items() if count == minimum)
        return random.Random(self.seed + state.completed_iterations).choice(tied)

    def _choose_value(
        self,
        direction: ResearchDirection,
        factor: str,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> Any:
        values = list(direction.search_space[factor])
        baseline = self.BASELINE_CONFIG[factor]
        candidates = [value for value in values if value != baseline] or values
        seen = {
            item.get("config", {}).get(factor)
            for item in history
            if item.get("direction_id") == direction.direction_id
        }
        unseen = [value for value in candidates if value not in seen] or candidates
        return random.Random(self.seed + state.completed_iterations + len(history)).choice(unseen)

    @staticmethod
    def _next_experiment_id(history: Sequence[Mapping[str, Any]]) -> str:
        existing = {str(item.get("experiment_id", "")) for item in history}
        number = 1
        while f"exp_{number:03d}" in existing:
            number += 1
        return f"exp_{number:03d}"
