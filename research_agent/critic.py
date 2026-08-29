"""Narrow deterministic critic for planner and search proposals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .safety import ExperimentProposal, SafetyReport, SafetyValidator


@dataclass(frozen=True)
class CriticResult:
    approved: bool
    reasons: tuple[str, ...]


class ProposalCritic:
    """Adds proposal-shape and logging checks around the safety validator."""

    def __init__(self, validator: SafetyValidator) -> None:
        self.validator = validator

    def review(
        self,
        proposal: ExperimentProposal,
        history: Sequence[Mapping[str, Any]],
    ) -> CriticResult:
        report: SafetyReport = self.validator.validate(
            proposal,
            historical_configs=[item.get("config", {}) for item in history],
        )
        reasons = list(report.violations)
        if not proposal.research_direction_id:
            reasons.append("proposal must record its research direction")
        if not proposal.search_strategy:
            reasons.append("proposal must record its search strategy")
        if not proposal.search_region_id:
            reasons.append("proposal must record its search region")
        return CriticResult(approved=not reasons, reasons=tuple(reasons))
