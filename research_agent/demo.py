"""Explicit demo-only failure injection for recovery evidence."""
from __future__ import annotations

from .controller import ExperimentController, IterationResult
from .logger import ResearchLogger
from .runner import PreparedData
from .safety import ExperimentProposal
from .search import SearchController


def inject_failure_recovery_demo(
    controller: ExperimentController,
    logger: ResearchLogger,
) -> IterationResult:
    """Run one safe candidate that intentionally fails after canonical data load."""
    config = dict(SearchController.BASELINE_CONFIG)
    config.update({"learning_rate": 0.002, "epochs": 4, "fidelity": "low"})
    proposal = ExperimentProposal(
        experiment_id=SearchController._next_experiment_id(logger.store.read_iterations()),
        parent_experiment_id=controller.state.current_best_experiment_id,
        hypothesis="An intentional candidate crash should be contained and restore the accepted model.",
        rationale="The final demonstration requires concrete failure and automatic recovery evidence.",
        config=config,
        changed_factors=("learning_rate",),
        runtime_budget_seconds=60.0,
        research_direction_id="pointwise_fm_optimization",
        search_strategy="demo_failure",
        search_region_id="region_demo_recovery",
    )
    logger.log_action(
        "demo_failure_injected",
        experiment_id=proposal.experiment_id,
        details={"expected_error": "intentional_demo_failure", "scope": "explicit --demo-failure mode only"},
    )
    return controller.run_iteration(proposal, _intentional_failure_candidate)


def _intentional_failure_candidate(
    _prepared: PreparedData,
    _config,
    _run_dir,
):
    raise RuntimeError("intentional_demo_failure: verifying subprocess containment and recovery")
