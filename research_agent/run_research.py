"""Command-line entrypoint for a complete autonomous research run."""
from __future__ import annotations

import argparse
import fcntl
import os
from dataclasses import replace
from pathlib import Path

from .controller import ExperimentController
from .critic import ProposalCritic
from .fidelity import FidelityManager
from .finalize import finalize_run
from .logger import ResearchLogger
from .loop import AutonomousResearchLoop
from .models.torch_fm import run_torch_fm_candidate
from .llm_planner import LLMPlanningError, OfflinePlanner, OpenAIPlanner
from .regions import SearchRegionManager
from .reporter import MarkdownReporter
from .review import EvidenceReviewer
from .runner import ExperimentRunner
from .safety import SafetyValidator
from .search import SearchController
from .store import ArtifactStore
from .contracts import BENCHMARK_CONTRACT, verify_benchmark_inputs
from .provenance import collect_provenance
from .demo import inject_failure_recovery_demo
from .research_coverage import build_research_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validation-only KuaiRand autonomous research loop.")
    parser.add_argument(
        "--cycles", "--round-budget", dest="round_budget", type=int, default=20,
        help="complete hypotheses available to each research round",
    )
    parser.add_argument("--max-rounds", type=int, default=1, help="research rounds pre-approved for this invocation")
    parser.add_argument("--max-total-experiments", type=int, default=20, help="persistent lifetime experiment safety cap")
    parser.add_argument("--target-primary", type=float, default=BENCHMARK_CONTRACT.target_primary)
    parser.add_argument(
        "--finalize", action="store_true",
        help="explicitly generate a test confirmation and submission at the end of this invocation",
    )
    parser.add_argument("--artifact-dir", default="runs", help="append-only artifact directory")
    parser.add_argument("--data-dir", default=str(BENCHMARK_CONTRACT.data_dir))
    parser.add_argument("--planner", choices=("openai", "offline"), default="openai")
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2),
        default=2,
        help="concurrent experiment workers (bounded to 1 or 2; default: 2)",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        choices=(1, 2),
        default=1,
        help="CPU threads available to each experiment worker (default: 1)",
    )
    parser.add_argument(
        "--approve-architecture-experiments",
        action="store_true",
        help="record human approval for reviewed compositional FM-hybrid architecture experiments",
    )
    parser.add_argument(
        "--approve-capability-gap",
        action="append",
        default=[],
        metavar="GAP_ID",
        help="approve one previously backlogged human-approval request; may be repeated",
    )
    parser.add_argument(
        "--architecture-coverage",
        action="store_true",
        help="test only executable architecture mechanisms still untested in validated cross-run coverage",
    )
    parser.add_argument(
        "--multi-task-baseline",
        action="store_true",
        help="scope this invocation to one LLM-authored controlled multi-task objective experiment",
    )
    parser.add_argument("--demo-failure", action="store_true", help="explicitly inject one contained failure before normal research")
    args = parser.parse_args()
    if args.round_budget <= 0:
        parser.error("--round-budget must be positive")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.max_total_experiments <= 0:
        parser.error("--max-total-experiments must be positive")
    if not 0.0 < args.target_primary <= 1.0:
        parser.error("--target-primary must be in (0, 1]")
    if args.demo_failure and args.round_budget < 2:
        parser.error("--demo-failure requires --cycles 2 or greater")
    if args.architecture_coverage and args.multi_task_baseline:
        parser.error("architecture coverage and multi-task baseline campaigns are mutually exclusive")
    research_campaign = (
        "architecture_coverage" if args.architecture_coverage
        else "multi_task_baseline" if args.multi_task_baseline else None
    )
    try:
        _run_lock_handle = _acquire_run_lock(args.artifact_dir)
    except BlockingIOError:
        parser.error(
            f"another research process is already using artifact directory {args.artifact_dir!r}"
        )

    contract = replace(
        BENCHMARK_CONTRACT,
        data_dir=Path(args.data_dir),
        target_primary=args.target_primary,
        max_experiments=args.max_total_experiments,
    )
    store = ArtifactStore(args.artifact_dir)
    logger = ResearchLogger(store)
    provenance = collect_provenance()
    store.write_root_json("provenance.json", provenance)
    logger.log_action("provenance_recorded", details={"git_commit": provenance.get("git_commit"), "package_count": len(provenance["packages"])})
    validator = SafetyValidator(max_runtime_seconds=600.0)
    controller = ExperimentController(
        logger=logger,
        runner=ExperimentRunner(logger, contract=contract),
        validator=validator,
        contract=contract, max_iterations=contract.max_experiments,
    )
    try:
        benchmark_identity = verify_benchmark_inputs(contract)
    except Exception as exc:
        logger.log_action(
            "benchmark_verification_failed",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise SystemExit(str(exc))
    logger.log_action("benchmark_verified", details=benchmark_identity)
    logger.log_action(
        "research_run_started",
        details={
            "round_budget": args.round_budget,
            "max_rounds": args.max_rounds,
            "max_total_experiments": args.max_total_experiments,
            "target_primary": args.target_primary,
            "data_interface": "data.py",
            "selection_split": contract.validation_split,
            "experiment_workers": args.workers,
            "threads_per_worker": args.worker_threads,
            "research_campaign": research_campaign,
        },
    )
    _record_interrupted_worker_recoveries(logger)
    if args.approve_architecture_experiments:
        _record_architecture_approval(logger)
    try:
        _record_capability_approvals(args.approve_capability_gap, logger)
    except ValueError as exc:
        logger.log_action("capability_approval_rejected", details={"error": str(exc)})
        raise SystemExit(str(exc))
    if args.demo_failure:
        demo_result = inject_failure_recovery_demo(controller, logger)
        logger.log_action(
            "demo_failure_recovery_completed",
            experiment_id=demo_result.experiment_id,
            details={"decision": demo_result.decision, "recovered_to": controller.state.current_best_experiment_id},
        )
    if args.planner == "offline":
        planner = OfflinePlanner(seed=0)
        logger.log_action(
            "planner_selected",
            details={"mode": "offline", "experiment_workers": args.workers},
        )
    else:
        try:
            planner = OpenAIPlanner.from_environment(
                allow_architecture_experiments=args.approve_architecture_experiments,
            )
            logger.log_action(
                "planner_selected",
                details={
                    "mode": "online_open_ideation",
                    "model": planner.client.model,
                    "planner_token_budget": planner.token_budget or "disabled",
                    "base_planning_api_calls_per_batch": 1 + args.workers,
                    "bounded_backfill_attempts_per_batch": 2,
                    "planning_batch_size": args.workers,
                    "semantic_audit_api_calls_per_experiment": 1,
                },
            )
        except LLMPlanningError as exc:
            logger.log_action("llm_configuration_failed", details={"error": str(exc), "recovery": "add OPENAI_API_KEY to .env or rerun with --planner offline"})
            raise SystemExit(str(exc))
    loop = AutonomousResearchLoop(
        controller=controller,
        logger=logger,
        planner=planner,
        search=SearchController(
            seed=0,
            architecture_human_reviewed=args.approve_architecture_experiments,
            worker_threads=args.worker_threads,
        ),
        critic=ProposalCritic(
            validator,
            semantic_client=getattr(planner, "client", None),
            semantic_token_budget=int(os.environ.get("OPENAI_SEMANTIC_CRITIC_TOKEN_BUDGET", "0")),
        ),
        reviewer=EvidenceReviewer(),
        fidelity=FidelityManager(),
        regions=SearchRegionManager(),
        candidate=run_torch_fm_candidate,
        max_workers=args.workers,
        research_campaign=research_campaign,
    )
    results = []
    rounds_completed = 0
    try:
        for round_number in range(1, args.max_rounds + 1):
            if controller.state.stopped:
                break
            available = contract.max_experiments - controller.state.completed_iterations
            if available <= 0:
                controller.stop(f"configured experiment budget reached: {contract.max_experiments}")
                break
            round_limit = min(args.round_budget, available)
            logger.log_action(
                "research_round_started",
                details={
                    "round_number": round_number,
                    "round_budget": round_limit,
                    "target_primary": contract.target_primary,
                    "incumbent_primary": controller.state.current_best_primary,
                },
            )
            round_results = loop.run(round_limit)
            results.extend(round_results)
            rounds_completed += 1
            checkpoint = {
                "round_number": round_number,
                "experiments_completed_this_round": len(round_results),
                "completed_experiments_total": controller.state.completed_iterations,
                "lifetime_experiment_cap": contract.max_experiments,
                "target_primary": contract.target_primary,
                "incumbent_experiment_id": controller.state.current_best_experiment_id,
                "incumbent_primary": controller.state.current_best_primary,
                "gap_to_target": max(0.0, contract.target_primary - controller.state.current_best_primary),
                "plateau_restarts": controller.state.plateau_restarts,
                "test_data_used": False,
                "status": (
                    "target_reached" if controller.state.current_best_primary >= contract.target_primary
                    else "terminal" if controller.state.stopped
                    else "continue_preapproved" if round_number < args.max_rounds
                    else "awaiting_next_round_approval"
                ),
            }
            store.write_root_json("research_checkpoint.json", checkpoint)
            logger.log_action("research_round_checkpoint", details=checkpoint)
            if loop.pause_reason or controller.state.stopped:
                break
    except LLMPlanningError as exc:
        logger.log_action("llm_planning_failed", details={"error": str(exc), "recovery": "run paused; correct the LLM configuration and resume"})
        raise SystemExit(str(exc))
    # The last completed hypothesis must appear in coverage even when the
    # invocation ends before another planner-context refresh.
    build_research_coverage(store, store.read_iterations())
    if loop.pause_reason:
        logger.log_action(
            "research_run_paused",
            details={
                "reason": loop.pause_reason,
                "cycles_completed": len(results),
                "rounds_completed": rounds_completed,
                "finalization_skipped": True,
                "test_data_used": False,
            },
        )
        report_path = MarkdownReporter(store).write()
        print(f"Research paused: {loop.pause_reason}")
        print(f"Capability backlog: {store.capability_backlog_path}")
        print(f"Log: {report_path}")
        return
    logger.log_action(
        "research_run_finished",
        details={
            "cycles_completed": len(results), "rounds_completed": rounds_completed,
            "stop_reason": controller.state.stop_reason,
            "checkpoint_status": (
                store.read_root_json("research_checkpoint.json").get("status")
                if (store.root / "research_checkpoint.json").exists()
                else "no_round_executed"
            ),
        },
    )
    should_finalize = args.finalize or controller.state.current_best_primary >= contract.target_primary
    if not should_finalize:
        report_path = MarkdownReporter(store).write()
        print(
            f"Research checkpoint: {controller.state.current_best_primary:.4f} / "
            f"target {contract.target_primary:.4f}"
        )
        print("Test data was not used. Approve another round by rerunning, or pass --finalize.")
        print(f"Checkpoint: {store.root / 'research_checkpoint.json'}")
        print(f"Log: {report_path}")
        return
    try:
        final = finalize_run(store, contract=contract)
    except Exception as exc:
        logger.log_action("finalization_failed", details={"error": f"{type(exc).__name__}: {exc}", "recovery": "no automatic retry available"})
        raise
    print(f"Selected: {final.selected_experiment_id} (validation primary {final.selection_primary:.4f})")
    if final.test_metrics:
        print(f"Final test confirmation: {final.test_metrics['primary']:.4f}")
    else:
        print("No candidate cleared the validation improvement threshold; wrote the official baseline submission.")
    print(f"Log: {final.report_path}")
    print(f"Submission: {final.submission_path}")


def _acquire_run_lock(artifact_dir: str | Path):
    """Hold an exclusive process lock for the lifetime of one CLI invocation."""
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".research.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise
    return handle


def _record_capability_approvals(gap_ids: list[str], logger: ResearchLogger) -> None:
    """Record explicit, gap-scoped authority without enabling unrelated work."""
    for gap_id in gap_ids:
        prior = next(
            (
                item for item in reversed(logger.store.read_capability_actions())
                if item.get("capability_gap_id") == gap_id
            ),
            None,
        )
        if prior is None or prior.get("status") != "pending_human_approval":
            raise ValueError(
                f"capability gap {gap_id!r} has no pending human-approval request"
            )
        logger.record_capability_action(
            {
                "action": prior.get("action"),
                "hypothesis": prior.get("hypothesis"),
                "rationale": prior.get("rationale"),
                "capability_gap_id": gap_id,
                "capability_gap_description": prior.get("capability_gap_description"),
                "required_capabilities": prior.get("required_capabilities", []),
                "specialist_id": prior.get("specialist_id"),
                "implementer_id": prior.get("implementer_id") or prior.get("specialist_id"),
                "approval_reason": prior.get("approval_reason"),
            },
            status="human_approved",
            evidence={"source_action_id": prior.get("action_id"), "scope": gap_id},
        )
        logger.record_manual_approval(
            description=f"Approved capability gap {gap_id}",
            reason=str(prior.get("approval_reason") or "Planner requested expanded authority"),
            effect="The approval is recorded for this gap only; implementation remains critic-gated.",
            approval_id=f"capability_gap:{gap_id}:v1",
            authority_scope=gap_id,
        )


def _record_interrupted_worker_recoveries(logger: ResearchLogger) -> int:
    """Make incomplete stage directories visible without deleting or reusing them."""
    events = logger.store.read_events()
    completed = {
        (str(item.get("experiment_id")), str(item.get("stage_id")))
        for item in logger.store.read_stages()
    }
    already_recorded = {
        (
            str(item.get("experiment_id")),
            str(item.get("details", {}).get("stage_id")),
        )
        for item in events
        if item.get("action") == "interrupted_worker_recovered"
    }
    recovered = 0
    for event in events:
        if event.get("action") != "candidate_created" or not event.get("experiment_id"):
            continue
        stage_id = str(event.get("details", {}).get("stage_id") or "")
        key = (str(event["experiment_id"]), stage_id)
        if not stage_id or key in completed or key in already_recorded:
            continue
        logger.log_action(
            "interrupted_worker_recovered",
            experiment_id=key[0],
            details={
                "stage_id": stage_id,
                "run_dir": event.get("details", {}).get("run_dir"),
                "recovery": (
                    "preserved incomplete artifacts and reserved the experiment ID; "
                    "future work receives a fresh experiment ID"
                ),
            },
        )
        already_recorded.add(key)
        recovered += 1
    return recovered


def _record_architecture_approval(logger: ResearchLogger) -> bool:
    """Record persistent architecture authority once per artifact directory."""
    approval_id = "architecture_experiments:v1"
    description = "Approved reviewed FM-hybrid architecture experiments"
    if any(
        item.get("approval_id") is None
        and item.get("description") == description
        for item in logger.store.read_interventions()
    ):
        logger.log_action(
            "architecture_approval_reused",
            details={"approval_id": approval_id, "description": description},
        )
        return False
    recorded = logger.record_manual_approval(
        approval_id=approval_id,
        authority_scope="reviewed_fm_hybrid_operator_grammar",
        reuse_action="architecture_approval_reused",
        description=description,
        reason=(
            "Enable the generic implementer to select controlled "
            "compositional FM-hybrid candidates"
        ),
        effect=(
            "The planner may select fm_architecture from the reviewed operator "
            "grammar; no new dependency or external data is allowed"
        ),
    )
    return recorded


if __name__ == "__main__":
    main()
