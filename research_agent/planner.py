"""Evidence-driven research-orchestrator interface.

Both the online LLM planner and offline fallback emit a research direction,
never an executable numeric trial.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .state import ResearchState


PLANNER_ACTIONS = (
    "RUN_EXPERIMENT",
    "RUN_DIAGNOSTIC",
    "BUILD_CAPABILITY",
    "REQUEST_HUMAN_APPROVAL",
)


@dataclass(frozen=True)
class CapabilityAction:
    """A first-class non-training decision emitted when experimentation is premature."""

    action: str
    hypothesis: str
    rationale: str
    capability_gap_id: str
    capability_gap_description: str
    required_capabilities: tuple[str, ...] = ()
    specialist_id: str | None = None
    portfolio_role: str | None = None
    approval_reason: str | None = None

    @property
    def implementer_id(self) -> str | None:
        """Canonical role name; ``specialist_id`` remains an artifact compatibility field."""
        return self.specialist_id

    def __post_init__(self) -> None:
        if self.action not in PLANNER_ACTIONS[1:]:
            raise ValueError(f"unsupported capability action: {self.action}")
        if not self.capability_gap_id or self.capability_gap_id == "none":
            raise ValueError("capability action requires a stable capability_gap_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "capability_gap_id": self.capability_gap_id,
            "capability_gap_description": self.capability_gap_description,
            "required_capabilities": list(self.required_capabilities),
            "specialist_id": self.specialist_id,
            "implementer_id": self.implementer_id,
            "portfolio_role": self.portfolio_role,
            "approval_reason": self.approval_reason,
        }


@dataclass(frozen=True)
class ResearchDirection:
    direction_id: str
    hypothesis: str
    rationale: str
    search_space: Mapping[str, Any]
    success_evidence: str
    evaluation_budget: Mapping[str, Any]
    strategy: str
    specialist_id: str | None = None
    preferred_factor: str | None = None
    preferred_value: Any | None = None
    specialist_rationale: str | None = None
    selected_candidate_id: str | None = None
    claimed_behavior: str | None = None
    required_capabilities: tuple[str, ...] = ()
    portfolio_role: str | None = None
    lineage_parent_id: str | None = None
    lineage_action: str | None = None
    evidence_reference: str | None = None

    @property
    def implementer_id(self) -> str | None:
        """Canonical role name; ``specialist_id`` remains an artifact compatibility field."""
        return self.specialist_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction_id": self.direction_id,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "search_space": dict(self.search_space),
            "success_evidence": self.success_evidence,
            "evaluation_budget": dict(self.evaluation_budget),
            "strategy": self.strategy,
            "specialist_id": self.specialist_id,
            "implementer_id": self.implementer_id,
            "preferred_factor": self.preferred_factor,
            "preferred_value": self.preferred_value,
            "specialist_rationale": self.specialist_rationale,
            "selected_candidate_id": self.selected_candidate_id,
            "claimed_behavior": self.claimed_behavior,
            "required_capabilities": list(self.required_capabilities),
            "portfolio_role": self.portfolio_role,
            "lineage_parent_id": self.lineage_parent_id,
            "lineage_action": self.lineage_action,
            "evidence_reference": self.evidence_reference,
        }


class ResearchPlanner(Protocol):
    """Boundary implemented by the OpenAI planner and test doubles."""

    def propose(
        self, history: Sequence[Mapping[str, Any]], state: ResearchState
    ) -> ResearchDirection | CapabilityAction: ...


class EvidencePlanner:
    """Proposes a research direction from logged outcomes, not fixed trials."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def propose(
        self,
        history: Sequence[Mapping[str, Any]],
        state: ResearchState,
    ) -> ResearchDirection:
        directions = self._available_directions(history)
        attempted = {item.get("direction_id") for item in history}
        unexplored = [direction for direction in directions if direction.direction_id not in attempted]
        pool = unexplored or directions
        rng = random.Random(self.seed + state.completed_iterations)
        chosen = rng.choice(pool)

        if not history:
            rationale = "No prior agent-run evidence exists, so begin with broad bootstrap exploration."
        else:
            recent = history[-3:]
            recent_decisions = [item.get("decision") for item in recent]
            rationale = (
                f"Recent decisions were {recent_decisions}; choose a {'new' if unexplored else 'diverse'} "
                "research direction rather than repeat an identical configuration."
            )
        return ResearchDirection(
            direction_id=chosen.direction_id,
            hypothesis=chosen.hypothesis,
            rationale=rationale,
            search_space=chosen.search_space,
            success_evidence=chosen.success_evidence,
            evaluation_budget=chosen.evaluation_budget,
            strategy="exploration" if unexplored else "diverse_restart",
        )

    @staticmethod
    def _available_directions(history: Sequence[Mapping[str, Any]]) -> tuple[ResearchDirection, ...]:
        # These are approved research domains, not a scheduled hypothesis list.
        # Exact values are selected later by SearchController.
        return (
            ResearchDirection(
                direction_id="pointwise_fm_optimization",
                hypothesis="The current FM training settings may leave ranking quality unrealized.",
                rationale="Optimize the existing PyTorch FM without changing its feature set or model family.",
                search_space={"loss": ["pointwise"], "learning_rate": [0.0005, 0.001, 0.002], "l2": [0.0, 1e-6, 1e-5]},
                success_evidence="Validation primary improves by more than 0.002 over the accepted parent.",
                evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
                strategy="bootstrap",
            ),
            ResearchDirection(
                direction_id="pairwise_fm_ranking",
                hypothesis="A pairwise within-user ranking objective may align FM training with GAUC and nDCG@5.",
                rationale="The benchmark is a within-user ranking task while the reference model uses pointwise loss.",
                search_space={"loss": ["pairwise"], "learning_rate": [0.0005, 0.001], "l2": [0.0, 1e-6]},
                success_evidence="Validation primary improves while both GAUC and nDCG@5 remain valid.",
                evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
                strategy="bootstrap",
            ),
        )
