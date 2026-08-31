from __future__ import annotations

import unittest
import json
import threading
from urllib.error import URLError

from research_agent.llm_planner import (
    LLMPlanningError,
    OfflinePlanner,
    OpenAIPlanner,
    OpenAIResponsesClient,
    _available_direction_ids,
    _ideator_prompt,
    _planner_decision_from_specialist,
    _validate_research_campaign,
)
from research_agent.planner import CapabilityAction, ResearchDirection
from research_agent.architecture import ReviewedArchitectureSpec
from research_agent.state import ResearchState


def architecture_strategy():
    return {
        "strategy_id": "architecture_discovery_v1",
        "phase_label": "architecture discovery",
        "decision": "start",
        "focus_domains": ["model_architecture"],
        "metric_emphasis": "nDCG@5",
        "frozen_factors": ["feature_variant", "loss", "learning_rate", "l2"],
        "worker_assignments": [{
            "domain": "model_architecture", "portfolio_role": "single_worker",
        }],
        "rationale": "Architecture capacity is the evidence-backed bottleneck while unrelated factors remain controlled.",
        "evidence_reference": "baseline architecture and ranking metrics",
        "transition_criteria": "Revise after architecture candidates reach a measured plateau or establish a confirmed incumbent.",
    }


class FakeClient:
    def __init__(self):
        self.calls = []

    def create_json(self, instructions, prompt, **kwargs):
        self.calls.append((instructions, prompt, kwargs))
        if len(self.calls) == 1:
            value = {
                "research_strategy": {
                    "strategy_id": "ranking_bootstrap_v1",
                    "phase_label": "ranking objective diagnosis",
                    "decision": "start",
                    "focus_domains": ["ranking_objective"],
                    "metric_emphasis": "nDCG@5",
                    "frozen_factors": ["architecture", "feature_variant", "learning_rate", "l2"],
                    "worker_assignments": [{
                        "domain": "ranking_objective", "portfolio_role": "single_worker",
                    }],
                    "rationale": "The current evidence warrants testing objective alignment before numerical tuning.",
                    "evidence_reference": "baseline ranking metrics",
                    "transition_criteria": "Revise after a measured ranking-objective result establishes whether top-five ordering improves.",
                },
                "candidates": [
                    {
                        "candidate_id": "h1", "domain": "ranking_objective",
                        "hypothesis": "Pairwise training may improve ordering among each user's exposed items.",
                        "rationale": "The benchmark rewards within-user ordering through both reported ranking metrics.",
                        "expected_mechanism": "Relative score comparisons should align updates with ordering errors.",
                        "required_capabilities": ["within-user pair construction", "pairwise loss"],
                        "lineage_parent_id": "baseline", "lineage_action": "branch_new",
                        "evidence_reference": "baseline ranking metrics",
                    },
                    {
                        "candidate_id": "h2", "domain": "model_architecture",
                        "hypothesis": "A nonlinear interaction tower may capture conditional preference structure.",
                        "rationale": "A fixed second-order interaction may underfit context-dependent preferences.",
                        "expected_mechanism": "Learned nonlinear crosses could represent higher-order interactions.",
                        "required_capabilities": ["model graph editing", "architecture comparison"],
                        "lineage_parent_id": "baseline", "lineage_action": "branch_new",
                        "evidence_reference": "baseline architecture capacity",
                    },
                    {
                        "candidate_id": "h3", "domain": "training_optimization",
                        "hypothesis": "Regularisation adjustment may improve validation generalisation stability.",
                        "rationale": "The observed training curve can distinguish underfit from overfit behavior.",
                        "expected_mechanism": "Adjusted shrinkage should improve generalisation of sparse embeddings.",
                        "required_capabilities": ["regularisation search"],
                        "lineage_parent_id": "baseline", "lineage_action": "refine",
                        "evidence_reference": "baseline training behavior",
                    },
                ],
                "recommended_candidate_id": "h1",
                "recommended_specialist_id": "ranking_specialist",
                "portfolio_rationale": "The slate covers objective alignment, architecture capacity, and optimisation stability.",
            }
        else:
            value = {
                "selected_candidate_id": "h1",
                "direction_id": "pairwise_fm_ranking",
                "execution_specialist_id": "ranking_specialist",
                "implementation_alignment": "exact",
                "alignment_reason": "The executable pairwise loss implements the relative within-user comparison claimed.",
                "hypothesis": "Pairwise training may better optimise within-user ranking than the baseline loss.",
                "claimed_behavior": "Within-user score differences directly train positive exposures above negatives.",
                "implementation_requirements": ["within-user pair sampling", "pairwise logistic loss"],
                "rationale": "Ranking loss directly optimises score differences among exposures for each user.",
                "preferred_factor": "loss",
                "preferred_value": "controller_select",
                "success_evidence": "Validation primary improves by more than 0.002 without invalid component metrics.",
                "strategy": "exploration",
                "deferred_candidate_ids": ["h2", "h3"],
                "lineage_parent_id": "baseline",
                "lineage_action": "branch_new",
                "evidence_reference": "baseline ranking metrics",
            }
        return value, {"model": "fake", "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}, "response_id": f"test-{len(self.calls)}"}


class LLMPlannerTests(unittest.TestCase):
    def test_multitask_campaign_rejects_other_directions(self):
        campaign = {"research_campaign": {"type": "multi_task_baseline"}}
        multi = ResearchDirection(
            direction_id="multi_task_learning",
            hypothesis="Click supervision may improve shared representations for long-view ranking.",
            rationale="This is a controlled training-objective change from the accepted incumbent.",
            search_space={"training_objective": ["multitask_click_w0.1"]},
            success_evidence="Validation primary and both components are finite.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="exploration", preferred_factor="training_objective",
        )
        _validate_research_campaign(multi, campaign)
        with self.assertRaisesRegex(LLMPlanningError, "non-multi-task"):
            _validate_research_campaign(
                ResearchDirection(**{**multi.__dict__, "direction_id": "fm_architecture", "preferred_factor": "architecture"}),
                campaign,
            )

    def test_architecture_coverage_rejects_refinement_before_untested_mechanism(self):
        campaign = {
            "research_campaign": {
                "type": "architecture_coverage",
                "remaining_mechanisms": ["cross_network"],
            }
        }
        direction = ResearchDirection(
            direction_id="fm_architecture",
            hypothesis="A cross network may improve explicit conditional interactions.",
            rationale="This fills the remaining controlled architecture coverage gap.",
            search_space={"architecture": ["composed:v1:cross_network:add:w32:d2:p0.1:c2"]},
            success_evidence="Report finite validation primary and component metrics.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="exploration", preferred_factor="architecture",
            preferred_value="composed:v1:cross_network:add:w32:d2:p0.1:c2",
        )

        _validate_research_campaign(direction, campaign)
        with self.assertRaisesRegex(LLMPlanningError, "isolate exactly one"):
            _validate_research_campaign(
                ResearchDirection(
                    **{
                        **direction.__dict__,
                        "preferred_value": "composed:v1:embedding_mlp+cross_network:add:w32:d2:p0.1:c2",
                    }
                ),
                campaign,
            )
        with self.assertRaisesRegex(LLMPlanningError, "remaining mechanisms"):
            _validate_research_campaign(
                ResearchDirection(
                    **{
                        **direction.__dict__,
                        "preferred_value": "composed:v1:embedding_mlp:add:w32:d2:p0.1:c2",
                    }
                ),
                campaign,
            )

    def test_llm_strategy_can_allocate_both_workers_to_architecture(self):
        class ArchitectureBatchClient:
            def __init__(self):
                self.calls = []
                self.lock = threading.Lock()

            def create_json(self, _instructions, prompt, **kwargs):
                with self.lock:
                    self.calls.append(kwargs["schema_name"])
                metadata = {
                    "model": "fake", "usage": {"total_tokens": 10},
                    "response_id": kwargs["schema_name"],
                }
                if kwargs["schema_name"] == "parallel_research_hypothesis_slate":
                    candidates = [
                        {
                            "candidate_id": candidate_id,
                            "domain": "model_architecture",
                            "hypothesis": hypothesis,
                            "rationale": "The controlled structural change tests interaction capacity with all other factors frozen.",
                            "expected_mechanism": "A reviewed residual pathway can model interactions absent from scalar second-order FM terms.",
                            "required_capabilities": ["reviewed architecture compiler"],
                            "proposed_action": "RUN_EXPERIMENT",
                            "lineage_parent_id": "baseline",
                            "lineage_action": "branch_new",
                            "evidence_reference": "baseline architecture evidence",
                        }
                        for candidate_id, hypothesis in (
                            ("arch_deep", "A DeepFM residual pathway may improve top-five recommendation ordering."),
                            ("arch_nfm", "An NFM residual pathway may improve top-five recommendation ordering."),
                            ("arch_cross", "A bounded cross-network pathway may improve conditional interactions."),
                        )
                    ]
                    return {
                        "research_strategy": {
                            **architecture_strategy(),
                            "worker_assignments": [
                                {"domain": "model_architecture", "portfolio_role": "incumbent_exploit"},
                                {"domain": "model_architecture", "portfolio_role": "independent_explore"},
                            ],
                        },
                        "candidates": candidates,
                        "recommended_candidate_id": "arch_deep",
                        "recommended_specialist_id": "model_architecture_specialist",
                        "portfolio_rationale": "Two architecture workers compare distinct controlled structural mechanisms.",
                    }, metadata
                assignment = json.loads(prompt)["implementation_assignment"]
                deepfm = assignment["candidate_id"] == "arch_deep"
                return {
                    "selected_candidate_id": assignment["candidate_id"],
                    "direction_id": "fm_architecture",
                    "execution_specialist_id": "generic_implementer",
                    "implementation_alignment": "exact",
                    "alignment_reason": "The selected legacy architecture exactly implements the claimed residual mechanism.",
                    "hypothesis": (
                        "A DeepFM residual pathway may improve top-five recommendation ordering."
                        if deepfm else
                        "An NFM residual pathway may improve top-five recommendation ordering."
                    ),
                    "claimed_behavior": "Retain the FM path and add one reviewed nonlinear interaction pathway.",
                    "implementation_requirements": ["reviewed architecture compiler"],
                    "rationale": "This is a controlled architecture comparison with non-architecture factors frozen.",
                    "preferred_factor": "architecture",
                    "preferred_value": "deepfm" if deepfm else "nfm_residual",
                    "architecture_spec": None,
                    "success_evidence": "Validation primary improves and nDCG@5 movement is reported separately.",
                    "strategy": "exploration",
                    "deferred_candidate_ids": ["arch_cross"],
                    "planner_action": "RUN_EXPERIMENT",
                    "capability_gap_id": "none",
                    "capability_gap_description": "none",
                    "approval_reason": "none",
                    "lineage_parent_id": "baseline",
                    "lineage_action": "branch_new",
                    "evidence_reference": "baseline architecture evidence",
                }, metadata

        planner = OpenAIPlanner(
            ArchitectureBatchClient(), allow_architecture_experiments=True
        )

        directions = planner.propose_batch([], ResearchState(), count=2)

        self.assertEqual(
            {item.preferred_value for item in directions}, {"deepfm", "nfm_residual"}
        )
        self.assertEqual(
            planner.last_metadata["research_strategy"]["focus_domains"],
            ["model_architecture"],
        )

    def test_strategy_cannot_freeze_the_factor_its_worker_must_change(self):
        class ContradictoryClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                if len(self.calls) == 1:
                    value["research_strategy"]["frozen_factors"].append("loss")
                return value, metadata

        with self.assertRaisesRegex(LLMPlanningError, "strategy-frozen factor"):
            OpenAIPlanner(ContradictoryClient()).propose([], ResearchState())

    def test_online_batch_shares_ideation_and_uses_two_generic_implementers(self):
        class BatchClient:
            def __init__(self):
                self.calls = []
                self.lock = threading.Lock()

            def create_json(self, instructions, prompt, **kwargs):
                schema_name = kwargs["schema_name"]
                with self.lock:
                    self.calls.append((instructions, prompt, kwargs))
                metadata = {
                    "model": "fake",
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    "response_id": schema_name,
                }
                if schema_name == "parallel_research_hypothesis_slate":
                    return {
                        "research_strategy": {
                            "strategy_id": "mixed_bootstrap_v1",
                            "phase_label": "objective and optimization bootstrap",
                            "decision": "start",
                            "focus_domains": ["ranking_objective", "training_optimization"],
                            "metric_emphasis": "primary",
                            "frozen_factors": ["architecture", "feature_variant"],
                            "worker_assignments": [
                                {"domain": "ranking_objective", "portfolio_role": "incumbent_exploit"},
                                {"domain": "training_optimization", "portfolio_role": "independent_explore"},
                            ],
                            "rationale": "The bootstrap should compare objective alignment with optimization stability using two workers.",
                            "evidence_reference": "baseline primary and component metrics",
                            "transition_criteria": "Revise after both controlled hypotheses have measured validation evidence.",
                        },
                        "candidates": [
                            {
                                "candidate_id": "h1", "domain": "ranking_objective",
                                "hypothesis": "Pairwise comparisons may improve within-user exposure ordering.",
                                "rationale": "The ranking metrics reward relative ordering within each user.",
                                "expected_mechanism": "Positive-negative score differences receive direct supervision.",
                                "required_capabilities": ["pairwise loss"],
                            },
                            {
                                "candidate_id": "h2", "domain": "training_optimization",
                                "hypothesis": "A lower learning rate may stabilize sparse FM optimization.",
                                "rationale": "Noisy sparse embedding updates can overshoot useful minima.",
                                "expected_mechanism": "Smaller parameter updates improve convergence stability.",
                                "required_capabilities": ["learning-rate control"],
                            },
                            {
                                "candidate_id": "h3", "domain": "feature_data",
                                "hypothesis": "Leakage-safe affinity may expose persistent user preferences.",
                                "rationale": "User-author repetition can carry useful preference information.",
                                "expected_mechanism": "Train-only aggregation adds personalized interaction context.",
                                "required_capabilities": ["train-only aggregation"],
                            },
                        ],
                        "recommended_candidate_id": "h1",
                        "recommended_specialist_id": "ranking_specialist",
                        "portfolio_rationale": "The portfolio spans objectives, optimization, and representation evidence.",
                    }, metadata
                assignment = json.loads(prompt)["implementation_assignment"]
                pairwise = assignment["candidate_id"] == "h1"
                return {
                    "selected_candidate_id": assignment["candidate_id"],
                    "direction_id": "pairwise_fm_ranking" if pairwise else "pointwise_fm_optimization",
                    "execution_specialist_id": "generic_implementer",
                    "implementation_alignment": "exact",
                    "alignment_reason": "The declared implementation and controlled factor exactly provide the claimed behavior.",
                    "hypothesis": (
                        "Pairwise score comparisons may improve within-user exposure ordering."
                        if pairwise else
                        "A lower pointwise learning rate may stabilize sparse FM optimization."
                    ),
                    "claimed_behavior": (
                        "Positive-negative score differences receive direct supervision."
                        if pairwise else
                        "Smaller optimizer steps stabilize sparse embedding updates."
                    ),
                    "implementation_requirements": ["pairwise loss" if pairwise else "learning-rate control"],
                    "rationale": "The selected controlled mechanism directly addresses the diagnosed optimization behavior.",
                    "preferred_factor": "loss" if pairwise else "learning_rate",
                    "preferred_value": "controller_select",
                    "success_evidence": "Finite validation primary exceeds the incumbent score.",
                    "strategy": "exploration",
                    "deferred_candidate_ids": ["h3"],
                }, metadata

        client = BatchClient()
        planner = OpenAIPlanner(client)
        directions = planner.propose_batch([], ResearchState(), count=2)

        self.assertEqual({item.direction_id for item in directions}, {
            "pairwise_fm_ranking", "pointwise_fm_optimization",
        })
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(planner.last_metadata["mode"], "online_planner_parallel_implementers")
        self.assertEqual(planner.last_metadata["usage"]["total_tokens"], 45)
        self.assertEqual(planner.last_metadata["planned_workers"], 2)
        self.assertEqual(
            {item.portfolio_role for item in directions},
            {"incumbent_exploit", "independent_explore"},
        )
        self.assertNotIn("capability_partitions", planner.last_metadata)
        self.assertEqual(len(planner.last_metadata["implementers"]), 2)
        implementer_prompts = [json.loads(call[1]) for call in client.calls[1:]]
        self.assertEqual(
            {item["implementation_assignment"]["portfolio_role"] for item in implementer_prompts},
            {"incumbent_exploit", "independent_explore"},
        )

    def test_offline_batch_returns_distinct_directions(self):
        planner = OfflinePlanner(seed=0)
        directions = planner.propose_batch([], ResearchState(), count=2)

        self.assertEqual(len(directions), 2)
        self.assertEqual(len({item.direction_id for item in directions}), 2)
        self.assertEqual(planner.last_metadata["mode"], "offline_parallel_portfolio")
        self.assertEqual(directions[0].portfolio_role, "incumbent_exploit")
        self.assertEqual(directions[1].portfolio_role, "independent_explore")

    def test_deepfm_incumbent_reopens_optimizer_capability_in_its_context(self):
        history = [{
            "experiment_id": "exp_fm_lr",
            "direction_id": "pointwise_fm_optimization",
            "decision": "rejected",
            "metrics": {"primary": 0.59},
            "config": {
                "architecture": "fm", "loss": "pointwise",
                "learning_rate": 0.0005, "l2": 1e-6,
                "feature_variant": "baseline",
            },
        }, {
            "experiment_id": "exp_deepfm",
            "direction_id": "fm_architecture",
            "decision": "accepted",
            "metrics": {"primary": 0.61},
            "config": {
                "architecture": "deepfm", "loss": "pointwise",
                "learning_rate": 0.001, "l2": 1e-6,
                "feature_variant": "baseline",
            },
        }]
        incumbent = history[-1]["config"]

        available = _available_direction_ids(history, incumbent_config=incumbent)

        self.assertIn("pointwise_fm_optimization", available)

    def test_llm_response_supplies_the_hypothesis_and_is_logged_as_metadata(self):
        client = FakeClient()
        planner = OpenAIPlanner(client)
        direction = planner.propose([], ResearchState())
        self.assertEqual(direction.direction_id, "pairwise_fm_ranking")
        self.assertIn("Pairwise training", direction.hypothesis)
        self.assertEqual(planner.last_metadata["model"], "fake")
        self.assertEqual(direction.specialist_id, "generic_implementer")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(planner.last_metadata["mode"], "online_planner_generic_implementer")
        self.assertEqual(planner.last_metadata["deferred_candidates"], ["h2", "h3"])

    def test_open_ideator_does_not_receive_implementation_catalogue(self):
        client = FakeClient()
        OpenAIPlanner(client).propose([], ResearchState())

        ideator_prompt = client.calls[0][1]
        implementer_prompt = client.calls[1][1]
        self.assertNotIn("approved_directions", ideator_prompt)
        self.assertNotIn("pairwise_fm_ranking", ideator_prompt)
        self.assertNotIn("user_history", ideator_prompt)
        self.assertIn("executable_capability_manifest", implementer_prompt)
        self.assertIn("pairwise_fm_ranking", implementer_prompt)
        self.assertNotIn("conditional preference structure", implementer_prompt)
        self.assertNotIn("Regularisation adjustment", implementer_prompt)
        self.assertEqual(client.calls[0][2]["max_output_tokens"], 2400)
        self.assertIn("prompt_cache_key", client.calls[0][2])
        planner_schema = client.calls[0][2]["schema"]
        implementer_schema = client.calls[1][2]["schema"]
        self.assertNotIn("recommended_specialist_id", planner_schema["properties"])
        self.assertEqual(
            implementer_schema["properties"]["implementer_id"]["enum"],
            ["generic_implementer"],
        )
        self.assertNotIn(
            "specialist_new",
            implementer_schema["properties"]["selected_candidate_id"]["enum"],
        )

    def test_scheduler_backfill_preserves_hypothesis_identity(self):
        client = FakeClient()
        planner = OpenAIPlanner(client)
        planner.set_backfill_mode(True)

        planner.propose([], ResearchState())

        implementer_prompt = client.calls[1][1]
        self.assertIn('"required": true', implementer_prompt)
        self.assertIn("Fill an idle experiment-worker slot", implementer_prompt)
        self.assertIn("without substitution", implementer_prompt)

    def test_executable_lineage_is_bound_to_the_incumbent_that_search_will_clone(self):
        client = FakeClient()
        state = ResearchState(
            current_best_experiment_id="exp_005",
            current_best_primary=0.602,
        )

        direction = OpenAIPlanner(client).propose([{
            "experiment_id": "exp_005",
            "direction_id": "leakage_safe_author_affinity",
            "decision": "accepted",
            "metrics": {"primary": 0.602},
            "config": {
                "loss": "pointwise", "learning_rate": 0.001, "l2": 1e-6,
                "embedding_dim": 16, "batch_size": 8192,
                "feature_variant": "author_affinity", "architecture": "fm",
            },
        }], state)

        self.assertEqual(direction.lineage_parent_id, "exp_005")
        self.assertIn("execution parent exp_005", direction.evidence_reference)

    def test_unsupported_llm_direction_is_rejected(self):
        class UnsafeClient:
            def create_json(self, _instructions, _prompt, **_kwargs):
                return ({"candidates": [], "recommended_candidate_id": "x", "recommended_specialist_id": "training_specialist", "portfolio_rationale": "x" * 30}, {})
        with self.assertRaises(LLMPlanningError):
            OpenAIPlanner(UnsafeClient()).propose([], ResearchState())

    def test_raw_responses_payload_is_parsed_after_transient_retry(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self):
                return json.dumps({
                    "id": "resp-test",
                    "output": [{"content": [{"type": "output_text", "text": json.dumps({
                        "direction_id": "pairwise_fm_ranking",
                        "hypothesis": "Pairwise ranking should improve the validation ordering objective.",
                        "rationale": "The benchmark measures within-user ordering rather than calibration.",
                        "strategy": "exploration",
                    })}]}],
                    "usage": {"total_tokens": 12},
                }).encode()

        calls = []
        payloads = []
        def opener(request, timeout):
            calls.append(timeout)
            payloads.append(json.loads(request.data.decode("utf-8")))
            if len(calls) == 1:
                raise URLError("temporary network failure")
            return Response()

        client = OpenAIResponsesClient(
            api_key="test-key",
            model="gpt-5",
            opener=opener,
            sleeper=lambda _seconds: None,
        )
        value, metadata = client.create_json(
            "instructions", "prompt", max_output_tokens=321, prompt_cache_key="stable-test-key"
        )

        self.assertEqual(value["direction_id"], "pairwise_fm_ranking")
        self.assertEqual(metadata["retry_count"], 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(payloads[-1]["max_output_tokens"], 321)
        self.assertEqual(payloads[-1]["prompt_cache_key"], "stable-test-key")
        self.assertEqual(payloads[-1]["text"]["verbosity"], "low")
        self.assertFalse(payloads[-1]["store"])

    def test_offline_planner_is_deterministic_and_requires_no_key(self):
        planner = OfflinePlanner(seed=0)
        direction = planner.propose([], ResearchState())

        self.assertIn(direction.direction_id, {"pointwise_fm_optimization", "pairwise_fm_ranking"})
        self.assertEqual(planner.last_metadata["mode"], "offline")

    def test_exhausted_single_variant_direction_is_not_advertised(self):
        available = _available_direction_ids([{
            "direction_id": "weekday_features",
            "config": {"feature_variant": "weekday"},
            "changed_factors": ["feature_variant"],
            "metrics": {"primary": 0.59},
        }])

        self.assertNotIn("weekday_features", available)
        self.assertIn("pointwise_fm_optimization", available)

    def test_failed_unmeasured_attempt_does_not_exhaust_direction(self):
        available = _available_direction_ids([{
            "direction_id": "weekday_features",
            "config": {"feature_variant": "weekday"},
            "decision": "failed",
            "metrics": {},
        }])

        self.assertIn("weekday_features", available)

    def test_architecture_capability_requires_explicit_planner_approval(self):
        self.assertNotIn("fm_architecture", _available_direction_ids([], include_architecture=False))
        self.assertIn("fm_architecture", _available_direction_ids([], include_architecture=True))

        class ArchitecturePlanningClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                if len(self.calls) == 1:
                    value["recommended_candidate_id"] = "h2"
                    value["recommended_specialist_id"] = "model_architecture_specialist"
                    value["research_strategy"] = architecture_strategy()
                else:
                    value.update({
                        "selected_candidate_id": "h2",
                        "direction_id": "fm_architecture",
                        "execution_specialist_id": "model_architecture_specialist",
                        "hypothesis": "A residual DeepFM path may improve nonlinear interaction capacity.",
                        "claimed_behavior": "The model retains FM and adds a nonlinear embedding interaction path.",
                        "implementation_requirements": ["reviewed DeepFM compiler"],
                        "preferred_factor": "architecture",
                        "preferred_value": "deepfm",
                        "alignment_reason": "The reviewed DeepFM implementation exactly provides the claimed residual path.",
                    })
                return value, metadata

        client = ArchitecturePlanningClient()
        OpenAIPlanner(client, allow_architecture_experiments=True).propose([], ResearchState())
        self.assertIn("fm_architecture", client.calls[1][1])
        self.assertIn("generic_implementer", client.calls[1][1])

    def test_architecture_specialist_selects_executable_model_structure(self):
        class ArchitectureClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                if len(self.calls) == 1:
                    value["recommended_candidate_id"] = "h2"
                    value["recommended_specialist_id"] = "model_architecture_specialist"
                    value["research_strategy"] = architecture_strategy()
                else:
                    value.update({
                        "selected_candidate_id": "h2",
                        "direction_id": "fm_architecture",
                        "execution_specialist_id": "model_architecture_specialist",
                        "implementation_alignment": "exact",
                        "alignment_reason": "The reviewed DeepFM structure supplies the nonlinear field interaction path claimed.",
                        "hypothesis": "A residual DeepFM interaction path may capture conditional preferences beyond scalar FM terms.",
                        "rationale": "The architecture adds nonlinear crosses while retaining the controlled FM prediction path.",
                        "preferred_factor": "architecture",
                        "preferred_value": "deepfm",
                        "success_evidence": "Validation primary exceeds the incumbent with finite ranking metrics.",
                        "deferred_candidate_ids": ["h1", "h3"],
                    })
                return value, metadata

        direction = OpenAIPlanner(
            ArchitectureClient(), allow_architecture_experiments=True
        ).propose([], ResearchState())

        self.assertEqual(direction.direction_id, "fm_architecture")
        self.assertEqual(direction.specialist_id, "generic_implementer")
        self.assertEqual(direction.preferred_value, "deepfm")

    def test_architecture_specialist_compiles_a_bounded_composed_spec(self):
        class ComposedArchitectureClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                if len(self.calls) == 1:
                    value["recommended_candidate_id"] = "h2"
                    value["recommended_specialist_id"] = "model_architecture_specialist"
                    value["research_strategy"] = architecture_strategy()
                else:
                    value.update({
                        "selected_candidate_id": "h2",
                        "direction_id": "fm_architecture",
                        "execution_specialist_id": "model_architecture_specialist",
                        "implementation_alignment": "exact",
                        "alignment_reason": "Every requested path and fusion operator is present in the reviewed compiler.",
                        "hypothesis": "Gated explicit and nonlinear embedding crosses may improve conditional ranking capacity.",
                        "rationale": "The two paths expose complementary explicit and implicit higher-order interactions.",
                        "preferred_factor": "architecture",
                        "preferred_value": "composed_spec",
                        "architecture_spec": {
                            "interaction_paths": ["embedding_mlp", "cross_network"],
                            "fusion": "learned_gate", "hidden_width": 32, "hidden_depth": 2,
                            "dropout": 0.1, "cross_layers": 2,
                        },
                        "success_evidence": "Finite validation primary exceeds the accepted incumbent primary.",
                        "deferred_candidate_ids": ["h1", "h3"],
                    })
                return value, metadata

        direction = OpenAIPlanner(
            ComposedArchitectureClient(), allow_architecture_experiments=True
        ).propose([], ResearchState())

        self.assertTrue(direction.preferred_value.startswith("composed:v1:"))
        self.assertEqual(direction.search_space["architecture"], [direction.preferred_value])

    def test_two_path_incumbent_restricts_architecture_to_one_path_ablations(self):
        parent = ReviewedArchitectureSpec(
            interaction_paths=("embedding_mlp", "bi_interaction_mlp"), fusion="add",
            hidden_width=32, hidden_depth=2, dropout=0.1, cross_layers=2,
        )
        child = ReviewedArchitectureSpec(
            interaction_paths=("embedding_mlp",), fusion="add", hidden_width=32,
            hidden_depth=2, dropout=0.1, cross_layers=2,
        )
        advice = {
            "planner_action": "RUN_EXPERIMENT",
            "implementation_alignment": "exact",
            "direction_id": "fm_architecture",
            "execution_specialist_id": "model_architecture_specialist",
            "hypothesis": "Removing one interaction path will isolate its contribution to validation ranking.",
            "rationale": "The sibling comparison preserves the FM path and all non-architecture settings.",
            "preferred_factor": "architecture",
            "preferred_value": "composed_spec",
            "architecture_spec": child.as_dict(),
            "success_evidence": "Validation primary and component metrics quantify the path contribution.",
            "strategy": "local_refinement",
            "implementation_requirements": ["reviewed architecture compiler"],
        }

        direction = _planner_decision_from_specialist(
            advice, [], incumbent_config={"architecture": parent.architecture_id}
        )

        self.assertEqual(len(direction.search_space["architecture"]), 2)
        self.assertIn(child.architecture_id, direction.search_space["architecture"])
        with self.assertRaisesRegex(LLMPlanningError, "controlled one-path"):
            _planner_decision_from_specialist(
                {**advice, "preferred_value": "deepfm", "architecture_spec": None},
                [], incumbent_config={"architecture": parent.architecture_id},
            )

    def test_implementer_cannot_replace_assigned_hypothesis(self):
        class RecoveryClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                if len(self.calls) == 1:
                    value["recommended_candidate_id"] = "h3"
                    value["recommended_specialist_id"] = "training_specialist"
                    value["research_strategy"] = {
                        "strategy_id": "optimization_refinement_v1",
                        "phase_label": "optimization refinement",
                        "decision": "revise",
                        "focus_domains": ["training_optimization"],
                        "metric_emphasis": "primary",
                        "frozen_factors": ["architecture", "feature_variant", "loss"],
                        "worker_assignments": [{
                            "domain": "training_optimization", "portfolio_role": "single_worker",
                        }],
                        "rationale": "Optimization refinement is warranted while architecture and representation remain controlled.",
                        "evidence_reference": "baseline optimization evidence",
                        "transition_criteria": "Revise after the controlled learning-rate experiment produces validation evidence.",
                    }
                else:
                    value.update({
                        "selected_candidate_id": "specialist_new",
                        "hypothesis": "Pointwise learning-rate refinement may improve sparse embedding convergence.",
                        "claimed_behavior": "A smaller optimizer step may stabilize sparse FM parameter updates.",
                        "implementation_requirements": ["pointwise FM", "learning-rate control"],
                        "direction_id": "pointwise_fm_optimization",
                        "execution_specialist_id": "training_specialist",
                        "preferred_factor": "learning_rate",
                        "preferred_value": "controller_select",
                        "alignment_reason": "This new hypothesis exactly matches an untried manifest capability.",
                        "deferred_candidate_ids": ["h1", "h2", "h3"],
                    })
                return value, metadata

        with self.assertRaisesRegex(LLMPlanningError, "substituted"):
            OpenAIPlanner(RecoveryClient()).propose([], ResearchState())

    def test_architecture_only_manifest_routes_to_architecture_owner(self):
        class ArchitectureOnlyClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                if len(self.calls) == 1:
                    value["recommended_specialist_id"] = "evaluation_specialist"
                    value["recommended_candidate_id"] = "h2"
                    value["research_strategy"] = architecture_strategy()
                else:
                    value.update({
                        "selected_candidate_id": "h2",
                        "direction_id": "fm_architecture",
                        "execution_specialist_id": "generic_implementer",
                        "hypothesis": "A residual NFM path may model nonlinear combinations beyond scalar FM interactions.",
                        "claimed_behavior": "A nonlinear network transforms vector-valued bi-interactions before scoring.",
                        "implementation_requirements": ["bi-interaction vector", "nonlinear residual network"],
                        "preferred_factor": "architecture",
                        "preferred_value": "nfm_residual",
                        "alignment_reason": "The reviewed residual NFM compiler exactly implements this structure.",
                    })
                return value, metadata

        exhausted = []
        specifications = {
            "pointwise_fm_optimization": {"learning_rate": [0.0005, 0.002], "l2": [0.0, 1e-5]},
            "pairwise_fm_ranking": {"loss": ["pairwise"], "learning_rate": [0.0005, 0.002], "l2": [0.0, 1e-5]},
            "leakage_safe_author_affinity": {"feature_variant": ["author_affinity"]},
            "leakage_safe_user_history": {"feature_variant": ["user_history"]},
            "weekday_features": {"feature_variant": ["weekday"]},
        }
        for direction_id, factors in specifications.items():
            for factor, values in factors.items():
                for value in values:
                    exhausted.append({
                        "experiment_id": f"exp_{len(exhausted) + 1:03d}",
                        "direction_id": direction_id,
                        "changed_factors": [factor],
                        "config": {factor: value},
                        "metrics": {"primary": 0.59},
                    })
        client = ArchitectureOnlyClient()
        result = OpenAIPlanner(
            client, allow_architecture_experiments=True
        ).propose(exhausted, ResearchState())

        self.assertIn("single generic ML research implementer", client.calls[1][0])
        self.assertEqual(result.specialist_id, "generic_implementer")
        self.assertEqual(result.preferred_value, "nfm_residual")

    def test_planner_prompt_receives_seed_and_diagnostic_evidence(self):
        prompt = _ideator_prompt(
            [{
                "experiment_id": "exp_001",
                "direction_id": "pointwise_fm_optimization",
                "diagnostic_evidence": {
                    "seed_confirmation": {"mean_primary": 0.602, "std_primary": 0.0004},
                    "model_diagnostics": [{"training": {"stop_reason": "early_stopping_patience"}}],
                },
            }],
            ResearchState(),
        )

        self.assertIn("seed_confirmation", prompt)
        self.assertIn("early_stopping_patience", prompt)
        self.assertNotIn("pointwise_fm_optimization", prompt)

    def test_planner_prompt_receives_governance_and_intervention_context(self):
        context = {
            "architecture_guidance": {"sha256": "abc123"},
            "manual_interventions": [{"description": "Approved architecture experiments"}],
            "recent_errors_and_recoveries": [{"action": "timed_out"}],
        }

        prompt = _ideator_prompt([], ResearchState(), context)

        self.assertIn("abc123", prompt)
        self.assertIn("Approved architecture experiments", prompt)
        self.assertIn("timed_out", prompt)

    def test_token_budget_stops_before_specialist_call(self):
        class ExpensiveClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                metadata["usage"] = {"total_tokens": 101}
                return value, metadata

        client = ExpensiveClient()
        with self.assertRaisesRegex(LLMPlanningError, "token budget exceeded"):
            OpenAIPlanner(client, token_budget=100).propose([], ResearchState())
        self.assertEqual(len(client.calls), 1)

    def test_zero_token_budget_disables_aggregate_guard(self):
        client = FakeClient()
        direction = OpenAIPlanner(client, token_budget=0).propose([], ResearchState())

        self.assertEqual(direction.direction_id, "pairwise_fm_ranking")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(direction.lineage_parent_id, "baseline")
        self.assertEqual(direction.lineage_action, "branch_new")

    def test_unavailable_hypothesis_is_not_silently_mapped(self):
        class UnavailableClient(FakeClient):
            def create_json(self, *args, **kwargs):
                value, metadata = super().create_json(*args, **kwargs)
                if len(self.calls) == 2:
                    value.update({
                        "direction_id": "none",
                        "preferred_factor": "none",
                        "implementation_alignment": "unavailable",
                        "alignment_reason": "No current implementation provides the nonlinear architecture behavior claimed.",
                    })
                return value, metadata

        planner = OpenAIPlanner(UnavailableClient())
        with self.assertRaisesRegex(LLMPlanningError, "specific capability gap"):
            planner.propose([], ResearchState())
        self.assertEqual(planner.last_metadata["deferred_candidates"], ["h2", "h3"])

    def test_unsupported_hypothesis_becomes_capability_build_action(self):
        specialist = {
            "planner_action": "BUILD_CAPABILITY",
            "implementation_alignment": "unavailable",
            "execution_specialist_id": "ranking_specialist",
            "hypothesis": "A listwise top-weighted objective may improve nDCG at five.",
            "rationale": "The existing pairwise objective does not directly weight top-ranked positions.",
            "implementation_requirements": ["within-user list construction", "listwise loss"],
            "capability_gap_id": "listwise_top_weighted_loss",
            "capability_gap_description": "No executable listwise objective or within-user list builder currently exists.",
            "approval_reason": "none",
        }

        decision = _planner_decision_from_specialist(specialist, [])

        self.assertIsInstance(decision, CapabilityAction)
        self.assertEqual(decision.action, "BUILD_CAPABILITY")
        self.assertEqual(decision.capability_gap_id, "listwise_top_weighted_loss")

    def test_implemented_diagnostic_gets_stable_identity_when_specialist_uses_none(self):
        specialist = {
            "planner_action": "RUN_DIAGNOSTIC",
            "implementation_alignment": "exact",
            "execution_specialist_id": "evaluation_specialist",
            "hypothesis": "Training-derived user-activity and feature-coverage slices may reveal the weakest cohort.",
            "rationale": "Aggregate validation metrics conceal which sufficiently supported cohort is weakest.",
            "implementation_requirements": [
                "Compute user-activity validation slices",
                "Compute feature-coverage validation slices",
            ],
            "capability_gap_id": "none",
            "capability_gap_description": "none",
            "approval_reason": "none",
        }

        decision = _planner_decision_from_specialist(specialist, [])

        self.assertEqual(decision.action, "RUN_DIAGNOSTIC")
        self.assertEqual(decision.capability_gap_id, "stratified_validation_diagnostics")
