import json
import tempfile
import unittest
from pathlib import Path

from research_agent.critic import ProposalCritic
from research_agent.planner import ResearchDirection
from research_agent.safety import ExperimentProposal, SafetyValidator


def direction(**changes):
    values = {
        "direction_id": "pointwise_fm_optimization",
        "hypothesis": "Reducing the learning rate may improve sparse FM embedding convergence.",
        "rationale": "The learning curve suggests that optimization stability is limiting validation ranking.",
        "search_space": {"loss": ["pointwise"], "learning_rate": [0.0005, 0.001], "l2": [0.0, 1e-6]},
        "success_evidence": "Validation primary exceeds the incumbent with finite GAUC and nDCG metrics.",
        "evaluation_budget": {"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
        "strategy": "exploration",
        "preferred_factor": "learning_rate",
        "claimed_behavior": "Smaller updates should stabilize sparse embedding optimization.",
        "required_capabilities": ("learning-rate control",),
    }
    values.update(changes)
    return ResearchDirection(**values)


def proposal(**changes):
    values = {
        "experiment_id": "exp_001",
        "hypothesis": direction().hypothesis,
        "rationale": "Search selected one controlled factor from the research direction.",
        "config": {
            "loss": "pointwise", "learning_rate": 0.0005, "l2": 1e-6,
            "embedding_dim": 16, "batch_size": 8192, "seed": 0,
            "feature_variant": "baseline", "epochs": 4, "fidelity": "low",
        },
        "changed_factors": ("learning_rate",),
        "runtime_budget_seconds": 60,
        "research_direction_id": "pointwise_fm_optimization",
        "search_strategy": "exploration",
        "search_region_id": "region_01",
    }
    values.update(changes)
    return ExperimentProposal(**values)


class SemanticCriticTests(unittest.TestCase):
    def setUp(self):
        self.critic = ProposalCritic(SafetyValidator(max_runtime_seconds=60))

    def test_pre_execution_trace_binds_hypothesis_to_implementation_and_diff(self):
        result = self.critic.review(proposal(), [], direction())

        self.assertTrue(result.approved)
        self.assertEqual(result.trace["implementation_id"], "torch_fm.pointwise_optimizer")
        self.assertEqual(
            result.trace["configuration_diff"]["learning_rate"],
            {"before": 0.001, "after": 0.0005},
        )
        self.assertTrue(all(result.trace["checks"].values()))

    def test_unmeasured_duplicate_is_retryable_but_measured_duplicate_is_not(self):
        failed_history = [{
            "experiment_id": "exp_failed",
            "decision": "failed",
            "config": dict(proposal().config),
            "metrics": {},
        }]
        measured_history = [{
            "experiment_id": "exp_measured",
            "decision": "rejected",
            "config": dict(proposal().config),
            "metrics": {"primary": 0.59},
        }]

        retryable = self.critic.review(proposal(), failed_history, direction())
        consumed = self.critic.review(proposal(), measured_history, direction())

        self.assertTrue(retryable.approved)
        self.assertFalse(consumed.approved)
        self.assertIn("proposal duplicates an existing experiment configuration", consumed.reasons)

    def test_claim_is_rejected_when_configuration_does_not_activate_behavior(self):
        pairwise_direction = direction(
            direction_id="pairwise_fm_ranking",
            search_space={"loss": ["pairwise"]},
            preferred_factor="loss",
        )
        mismatched = proposal(
            hypothesis=pairwise_direction.hypothesis,
            research_direction_id="pairwise_fm_ranking",
            changed_factors=("loss",),
        )

        result = self.critic.review(mismatched, [], pairwise_direction)

        self.assertFalse(result.approved)
        self.assertIn("semantic check failed: configuration_activates_behavior", result.reasons)

    def test_critic_rejects_hidden_second_scientific_change(self):
        changed_twice = proposal(
            config={**proposal().config, "l2": 0.0},
            changed_factors=("learning_rate",),
        )

        result = self.critic.review(changed_twice, [], direction())

        self.assertFalse(result.approved)
        self.assertIn("semantic check failed: controlled_configuration_diff", result.reasons)
        self.assertEqual(result.trace["planner_feedback"]["disposition"], "repair_execution_alignment")

    def test_online_lineage_must_reference_known_evidence(self):
        online_direction = direction(
            selected_candidate_id="h1",
            lineage_parent_id="hyp_missing",
            lineage_action="refine",
            evidence_reference="hyp_missing showed unstable convergence",
        )

        result = self.critic.review(proposal(), [], online_direction)

        self.assertFalse(result.approved)
        self.assertIn("semantic check failed: planner_lineage_reference_known", result.reasons)
        self.assertEqual(result.trace["planner_feedback"]["disposition"], "repair_lineage_reasoning")

    def test_online_lineage_can_revisit_a_known_deferred_tree_node(self):
        self.critic.set_research_context({
            "incumbent": {"ancestry": ["baseline"]},
            "continuation_candidates": [],
            "deferred_hypotheses": [{"hypothesis_id": "hyp_deferred"}],
            "failed_branches": [],
        })
        online_direction = direction(
            selected_candidate_id="h2",
            lineage_parent_id="hyp_deferred",
            lineage_action="revisit",
            evidence_reference="hyp_deferred was previously capability blocked",
        )

        result = self.critic.review(proposal(), [], online_direction)

        self.assertTrue(result.approved)

    def test_promotion_preserves_scientific_lineage_while_parenting_previous_stage(self):
        self.critic.set_research_context({
            "incumbent": {"ancestry": ["exp_parent"]},
            "continuation_candidates": [],
            "deferred_hypotheses": [],
            "failed_branches": [],
        })
        online_direction = direction(
            selected_candidate_id="h3",
            lineage_parent_id="exp_parent",
            lineage_action="refine",
            evidence_reference="exp_parent established the incumbent",
        )
        promoted = proposal(
            parent_experiment_id="exp_001",
            search_strategy="promotion",
            config={**proposal().config, "epochs": 8, "fidelity": "medium"},
        )

        result = self.critic.review(promoted, [], online_direction)

        self.assertTrue(result.approved)
        self.assertTrue(result.trace["checks"]["planner_lineage_parent_matches_execution"])

    def test_post_execution_trace_verifies_patch_configuration_and_metrics(self):
        pre = self.critic.review(proposal(), [], direction())
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "exp_001.patch"
            patch.write_text('-  "learning_rate": 0.001\n+  "learning_rate": 0.0005\n', encoding="utf-8")
            stages = [{
                "experiment_id": "exp_001",
                "stage_id": "low",
                "config": dict(proposal().config),
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
                "code_diff_path": str(patch),
            }]

            result = self.critic.review_evidence(proposal(), stages, pre.trace)

        self.assertTrue(result.approved)
        self.assertEqual(result.trace["verdict"], "verified")
        self.assertEqual(result.trace["evidence"]["stage_ids"], ["low"])

    def test_post_execution_trace_detects_configuration_drift(self):
        pre = self.critic.review(proposal(), [], direction())
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "exp_001.patch"
            patch.write_text("learning_rate", encoding="utf-8")
            drifted_config = dict(proposal().config)
            drifted_config["learning_rate"] = 0.002
            result = self.critic.review_evidence(proposal(), [{
                "experiment_id": "exp_001", "stage_id": "full", "config": drifted_config,
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
                "code_diff_path": str(patch),
            }], pre.trace)

        self.assertFalse(result.approved)
        self.assertIn("semantic evidence check failed: configuration_preserved_across_fidelities", result.reasons)

    def test_valid_negative_result_becomes_an_actionable_planner_lesson(self):
        pre = self.critic.review(proposal(), [], direction())
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "exp_001.patch"
            patch.write_text("learning_rate", encoding="utf-8")
            result = self.critic.review_evidence(
                proposal(), [{
                    "experiment_id": "exp_001", "stage_id": "low", "config": dict(proposal().config),
                    "metrics": {"GAUC": 0.65, "nDCG@5": 0.53, "primary": 0.59},
                    "code_diff_path": str(patch),
                }], pre.trace, baseline_primary=0.60,
            )

        feedback = result.trace["planner_feedback"]
        self.assertTrue(result.approved)
        self.assertEqual(feedback["disposition"], "record_valid_negative_and_branch")
        self.assertTrue(feedback["do_not_repeat_exact_configuration"])

    def test_post_execution_rejects_primary_inconsistent_with_official_components(self):
        pre = self.critic.review(proposal(), [], direction())
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "exp_001.patch"
            patch.write_text("learning_rate", encoding="utf-8")
            result = self.critic.review_evidence(proposal(), [{
                "experiment_id": "exp_001", "stage_id": "low", "config": dict(proposal().config),
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.606},
                "code_diff_path": str(patch),
            }], pre.trace)

        self.assertFalse(result.approved)
        self.assertIn("semantic evidence check failed: primary_matches_metric_contract", result.reasons)

    def test_online_agentic_auditor_independently_verifies_complete_chain(self):
        class SemanticClient:
            def __init__(self):
                self.calls = []

            def create_json(self, instructions, prompt, **kwargs):
                self.calls.append((instructions, prompt, kwargs))
                return ({
                    "hypothesis_matches_implementation": True,
                    "implementation_matches_configuration_diff": True,
                    "configuration_matches_measured_evidence": True,
                    "evidence_meaningfully_tests_hypothesis": True,
                    "no_unimplemented_behavior_claimed": True,
                    "rationale": "The named pointwise implementation and learning-rate diff exactly test the claim.",
                    "limitations": ["Only low-fidelity evidence is currently available."],
                    "recommended_planner_action": "refine",
                    "next_hypothesis_constraint": "Preserve pointwise loss and test one nearby learning rate.",
                }, {"model": "fake", "usage": {"total_tokens": 9449}, "response_id": "audit-1"})

        client = SemanticClient()
        critic = ProposalCritic(
            SafetyValidator(max_runtime_seconds=60),
            semantic_client=client,
            semantic_token_budget=0,
        )
        pre = critic.review(proposal(), [], direction())
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "exp_001.patch"
            patch.write_text("learning_rate", encoding="utf-8")
            result = critic.review_evidence(proposal(), [{
                "experiment_id": "exp_001", "stage_id": "low", "config": dict(proposal().config),
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
                "code_diff_path": str(patch),
            }], pre.trace)

        self.assertTrue(result.approved)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][2]["max_output_tokens"], 1400)
        self.assertEqual(client.calls[0][2]["prompt_cache_key"], "researcher-ml-semantic-critic-v1")
        self.assertEqual(result.trace["agentic_review"]["metadata"]["usage"]["total_tokens"], 9449)
        semantic_prompt = json.loads(client.calls[0][1])["semantic_trace"]
        self.assertEqual(
            semantic_prompt["config_diff"]["learning_rate"],
            {"before": 0.001, "after": 0.0005},
        )

    def test_agentic_auditor_can_block_incumbent_eligibility(self):
        class RejectingClient:
            def create_json(self, *_args, **_kwargs):
                return ({
                    "hypothesis_matches_implementation": False,
                    "implementation_matches_configuration_diff": True,
                    "configuration_matches_measured_evidence": True,
                    "evidence_meaningfully_tests_hypothesis": False,
                    "no_unimplemented_behavior_claimed": False,
                    "rationale": "The claimed behavior exceeds what the named implementation actually provides.",
                    "limitations": [],
                }, {"usage": {"total_tokens": 30}})

        critic = ProposalCritic(SafetyValidator(max_runtime_seconds=60), semantic_client=RejectingClient())
        pre = critic.review(proposal(), [], direction())
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "exp_001.patch"
            patch.write_text("learning_rate", encoding="utf-8")
            result = critic.review_evidence(proposal(), [{
                "experiment_id": "exp_001", "stage_id": "low", "config": dict(proposal().config),
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
                "code_diff_path": str(patch),
            }], pre.trace)

        self.assertFalse(result.approved)
        self.assertEqual(result.trace["verdict"], "misaligned")
        self.assertIn("agentic semantic check failed: hypothesis_matches_implementation", result.reasons)

    def test_architecture_claim_maps_to_compiler_and_reviewed_structure(self):
        architecture_direction = direction(
            direction_id="fm_architecture",
            hypothesis="A nonlinear field-embedding path may improve context-dependent interaction capacity.",
            search_space={"architecture": ["deepfm", "nfm_residual"]},
            preferred_factor="architecture",
            preferred_value="deepfm",
            claimed_behavior="An MLP over concatenated field embeddings adds higher-order interactions.",
        )
        architecture_proposal = proposal(
            hypothesis=architecture_direction.hypothesis,
            research_direction_id="fm_architecture",
            config={**proposal().config, "learning_rate": 0.001, "architecture": "deepfm"},
            changed_factors=("architecture",),
            model_family="fm_hybrid",
            human_reviewed=True,
        )

        result = self.critic.review(architecture_proposal, [], architecture_direction)

        self.assertTrue(result.approved)
        self.assertEqual(result.trace["implementation_id"], "torch_fm.reviewed_architecture_compiler")
        self.assertEqual(result.trace["configuration_diff"]["architecture"]["after"], "deepfm")

    def test_multitask_claim_maps_to_shared_embedding_auxiliary_head(self):
        multi_direction = direction(
            direction_id="multi_task_learning",
            hypothesis="Click supervision may regularize shared embeddings for long-view ranking.",
            search_space={"training_objective": ["multitask_click_w0.1"]},
            preferred_factor="training_objective",
            preferred_value="multitask_click_w0.1",
            claimed_behavior="A bounded click loss updates embeddings shared with the primary long-view head.",
        )
        multi_proposal = proposal(
            hypothesis=multi_direction.hypothesis,
            research_direction_id="multi_task_learning",
            config={
                **proposal().config,
                "learning_rate": 0.001,
                "training_objective": "multitask_click_w0.1",
            },
            changed_factors=("training_objective",),
        )

        result = self.critic.review(multi_proposal, [], multi_direction)

        self.assertTrue(result.approved, result.reasons)
        self.assertEqual(
            result.trace["implementation_id"],
            "torch_fm.multitask_click_shared_embeddings",
        )

    def test_critic_accepts_only_canonical_reviewed_architecture_compositions(self):
        architecture = "composed:v1:embedding_mlp+cross_network:learned_gate:w32:d2:p0.1:c2"
        architecture_direction = direction(
            direction_id="fm_architecture",
            hypothesis="A gated explicit and nonlinear interaction residual may improve conditional ranking.",
            search_space={"architecture": [architecture]},
            preferred_factor="architecture",
            preferred_value=architecture,
            claimed_behavior="Reviewed embedding and cross paths are fused before addition to the FM score.",
        )
        architecture_proposal = proposal(
            hypothesis=architecture_direction.hypothesis,
            research_direction_id="fm_architecture",
            config={**proposal().config, "learning_rate": 0.001, "architecture": architecture},
            changed_factors=("architecture",),
            model_family="fm_hybrid",
            human_reviewed=True,
        )

        accepted = self.critic.review(architecture_proposal, [], architecture_direction)
        invalid = proposal(
            hypothesis=architecture_direction.hypothesis,
            research_direction_id="fm_architecture",
            config={**proposal().config, "learning_rate": 0.001, "architecture": "python:arbitrary_model"},
            changed_factors=("architecture",), model_family="fm_hybrid", human_reviewed=True,
        )
        rejected = self.critic.review(invalid, [], architecture_direction)

        self.assertTrue(accepted.approved)
        self.assertFalse(rejected.approved)
        self.assertIn("semantic check failed: configuration_activates_behavior", rejected.reasons)
