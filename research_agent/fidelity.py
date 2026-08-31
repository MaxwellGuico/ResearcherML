"""Multi-fidelity promotion policy for evidence-based compute allocation."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .planner import ResearchDirection
from .safety import ExperimentProposal


class FidelityManager:
    """Promotes only candidates with recorded validation evidence."""

    DEFAULT_CONFIRMATION_SEEDS = (1, 2)

    def next_fidelity(self, fidelity: str) -> str | None:
        return {"low": "medium", "medium": "full"}.get(fidelity)

    def should_promote(
        self,
        metrics: Mapping[str, Any],
        incumbent_primary: float,
        *,
        fidelity: str = "low",
    ) -> bool:
        if self.next_fidelity(fidelity) is None:
            return False
        return self.is_promising(metrics, incumbent_primary)

    @staticmethod
    def is_promising(metrics: Mapping[str, Any], incumbent_primary: float) -> bool:
        primary = metrics.get("primary")
        return isinstance(primary, (int, float)) and primary >= incumbent_primary - 0.002

    @staticmethod
    def budget(direction: ResearchDirection, fidelity: str) -> int:
        budgets = direction.evaluation_budget
        if fidelity == "low":
            return int(budgets["low_epochs"])
        if fidelity == "medium":
            return int(budgets.get("medium_epochs", budgets["full_epochs"]))
        if fidelity == "full":
            return int(budgets["full_epochs"])
        raise ValueError(f"unsupported fidelity: {fidelity!r}")

    def promote(
        self,
        proposal: ExperimentProposal,
        direction: ResearchDirection,
        *,
        experiment_id: str | None = None,
    ) -> ExperimentProposal:
        current_fidelity = str(proposal.config.get("fidelity", "low"))
        next_fidelity = self.next_fidelity(current_fidelity)
        if next_fidelity is None:
            raise ValueError(f"cannot promote candidate at terminal fidelity: {current_fidelity!r}")
        config = dict(proposal.config)
        config["epochs"] = self.budget(direction, next_fidelity)
        config["fidelity"] = next_fidelity
        return replace(
            proposal,
            experiment_id=proposal.experiment_id,
            # Point to the preceding stage for configuration-diff validation;
            # the experiment ID remains unchanged, so this is not a new trial.
            parent_experiment_id=proposal.experiment_id,
            config=config,
            search_strategy="promotion",
        )
