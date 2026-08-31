import unittest

from research_agent.planner import EvidencePlanner, ResearchDirection
from research_agent.search import SearchController, SearchState
from research_agent.architecture import ReviewedArchitectureSpec
from research_agent.safety import SafetyValidator
from research_agent.state import ResearchState


class SearchControllerTests(unittest.TestCase):
    def setUp(self):
        self.planner = EvidencePlanner(seed=0)
        self.search = SearchController(seed=0)

    def test_search_controller_selects_one_factor_not_a_fixed_template(self):
        direction = self.planner.propose([], ResearchState())
        proposal = self.search.propose_trial(direction, ResearchState(), [])

        self.assertEqual(len(proposal.changed_factors), 1)
        self.assertEqual(proposal.research_direction_id, direction.direction_id)
        self.assertEqual(proposal.config["fidelity"], "low")
        self.assertNotEqual(proposal.config[proposal.changed_factors[0]], self.search.BASELINE_CONFIG[proposal.changed_factors[0]])

    def test_pairwise_direction_changes_only_loss_conceptually(self):
        history = [{"direction_id": "pointwise_fm_optimization", "decision": "rejected"}]
        direction = self.planner.propose(history, ResearchState(completed_iterations=1))
        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), history)

        self.assertEqual(direction.direction_id, "pairwise_fm_ranking")
        self.assertEqual(proposal.changed_factors, ("loss",))
        self.assertEqual(proposal.config["loss"], "pairwise")

    def test_pairwise_direction_can_evolve_after_loss_foundation(self):
        direction = ResearchDirection(
            direction_id="pairwise_fm_ranking",
            hypothesis="Tune pairwise optimisation after establishing the ranking-loss foundation.",
            rationale="Pairwise convergence may require a different learning rate from pointwise training.",
            search_space={"loss": ["pairwise"], "learning_rate": [0.0005, 0.001, 0.002], "l2": [0.0, 1e-6, 1e-5]},
            success_evidence="Validation primary improves by more than the threshold.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="exploration",
            specialist_id="training_specialist",
            preferred_factor="learning_rate",
        )
        history = [{
            "experiment_id": "exp_001",
            "direction_id": "pairwise_fm_ranking",
            "changed_factors": ["loss"],
            "metrics": {"primary": 0.59},
            "config": {**self.search.BASELINE_CONFIG, "loss": "pairwise", "epochs": 4, "fidelity": "low"},
        }, {
            "experiment_id": "exp_unrelated",
            "direction_id": "pointwise_fm_optimization",
            "changed_factors": ["learning_rate"],
            "config": {**self.search.BASELINE_CONFIG, "learning_rate": 0.0005},
        }]

        proposal = self.search.propose_trial(
            direction,
            ResearchState(
                current_best_experiment_id="exp_001",
                current_best_primary=0.61,
                completed_iterations=1,
            ),
            history,
        )

        self.assertEqual(proposal.parent_experiment_id, "exp_001")
        self.assertEqual(proposal.changed_factors, ("learning_rate",))
        self.assertEqual(proposal.config["loss"], "pairwise")
        self.assertNotEqual(proposal.config["learning_rate"], 0.001)

    def test_new_trial_ids_and_values_avoid_history_when_possible(self):
        direction = self.planner.propose([], ResearchState())
        history = [
            {
                "experiment_id": "exp_001",
                "direction_id": direction.direction_id,
                "changed_factors": ["learning_rate"],
                "config": {"learning_rate": 0.0005},
            }
        ]
        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), history)

        self.assertEqual(proposal.experiment_id, "exp_002")

    def test_failed_unmeasured_value_can_be_retried(self):
        direction = ResearchDirection(
            direction_id="pointwise_fm_optimization",
            hypothesis="Retry an interrupted learning-rate experiment with measured evidence.",
            rationale="An interrupted run did not test the configuration.",
            search_space={"learning_rate": [0.0005, 0.001]},
            success_evidence="Validation primary is measured.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="exploration",
            preferred_factor="learning_rate",
        )
        failed = [{
            "experiment_id": "exp_001", "direction_id": direction.direction_id,
            "decision": "failed", "metrics": {}, "changed_factors": ["learning_rate"],
            "config": {"learning_rate": 0.0005},
        }]

        proposal = self.search.propose_trial(direction, ResearchState(completed_iterations=1), failed)

        self.assertEqual(proposal.config[proposal.changed_factors[0]], 0.0005)

    def test_trial_ids_reserve_non_iteration_runs(self):
        direction = self.planner.propose([], ResearchState())
        proposal = self.search.propose_trial(
            direction,
            ResearchState(),
            [],
            reserved_experiment_ids=("exp_001", "exp_002"),
        )

        self.assertEqual(proposal.experiment_id, "exp_003")

    def test_worker_threads_are_bounded_invocation_policy(self):
        direction = self.planner.propose([], ResearchState())
        proposal = SearchController(worker_threads=1).propose_trial(
            direction, ResearchState(), []
        )

        self.assertEqual(proposal.config["worker_threads"], 1)
        with self.assertRaisesRegex(ValueError, "worker_threads"):
            SearchController(worker_threads=3)

    def test_unknown_portfolio_role_is_rejected(self):
        direction = self.planner.propose([], ResearchState())
        proposal = self.search.propose_trial(direction, ResearchState(), [])
        invalid = proposal.__class__(
            **{**proposal.__dict__, "portfolio_role": "unbounded_role"}
        )

        report = SafetyValidator().validate(invalid)

        self.assertIn("proposal has an unsupported portfolio role", report.violations)

    def test_new_direction_branches_from_accepted_incumbent(self):
        direction = self.planner.propose([], ResearchState())
        proposal = self.search.propose_trial(
            direction,
            ResearchState(current_best_experiment_id="exp_deepfm", current_best_primary=0.61),
            [{
                "experiment_id": "exp_deepfm", "direction_id": "fm_architecture",
                "decision": "accepted", "metrics": {"primary": 0.61},
                "config": {**self.search.BASELINE_CONFIG, "architecture": "deepfm", "seed": 2},
            }],
        )

        self.assertEqual(proposal.parent_experiment_id, "exp_deepfm")
        self.assertEqual(proposal.config["architecture"], "deepfm")
        self.assertEqual(proposal.config["seed"], 0)
        self.assertEqual(proposal.model_family, "fm_hybrid")

    def test_incumbent_descendant_is_exactly_one_conceptual_change(self):
        direction = ResearchDirection(
            direction_id="pointwise_fm_optimization",
            hypothesis="Tune the accepted DeepFM learning rate.",
            rationale="The accepted nonlinear architecture may need local optimisation.",
            search_space={"loss": ["pointwise"], "learning_rate": [0.0005, 0.001, 0.002]},
            success_evidence="Validation primary exceeds the accepted parent.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="local_refinement",
            preferred_factor="learning_rate",
            preferred_value=0.0005,
        )
        incumbent = {
            "experiment_id": "exp_deepfm",
            "direction_id": "fm_architecture",
            "decision": "accepted",
            "metrics": {"primary": 0.61},
            "config": {**self.search.BASELINE_CONFIG, "architecture": "deepfm", "seed": 2},
        }
        state = ResearchState(current_best_experiment_id="exp_deepfm", current_best_primary=0.61)

        proposal = SearchController(worker_threads=1).propose_trial(direction, state, [incumbent])
        report = SafetyValidator(
            approved_model_families=frozenset({"fm"})
        ).validate(proposal, parent_config=incumbent["config"])

        self.assertTrue(report.passed, report.violations)
        self.assertEqual(proposal.parent_experiment_id, "exp_deepfm")
        self.assertEqual(proposal.changed_factors, ("learning_rate",))
        self.assertEqual(proposal.config["architecture"], "deepfm")
        self.assertEqual(proposal.config["feature_variant"], "baseline")
        self.assertEqual(proposal.config["learning_rate"], 0.0005)
        self.assertTrue(proposal.human_reviewed)

    def test_multitask_trial_inherits_embedding_mlp_incumbent(self):
        architecture = "composed:v1:embedding_mlp:add:w32:d2:p0.1:c2"
        incumbent = {
            "experiment_id": "exp_embedding", "direction_id": "fm_architecture",
            "decision": "accepted", "metrics": {"primary": 0.6034},
            "config": {**self.search.BASELINE_CONFIG, "architecture": architecture},
        }
        direction = ResearchDirection(
            direction_id="multi_task_learning",
            hypothesis="Click supervision may improve the accepted embedding representation.",
            rationale="The auxiliary label is dense and shares user-item representation signal.",
            search_space={"training_objective": ["multitask_click_w0.1"]},
            success_evidence="Validation primary improves with finite component metrics.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="exploration", preferred_factor="training_objective",
        )

        proposal = self.search.propose_trial(
            direction,
            ResearchState(current_best_experiment_id="exp_embedding", current_best_primary=0.6034),
            [incumbent],
        )

        self.assertEqual(proposal.parent_experiment_id, "exp_embedding")
        self.assertEqual(proposal.config["architecture"], architecture)
        self.assertEqual(proposal.changed_factors, ("training_objective",))
        self.assertEqual(proposal.config["training_objective"], "multitask_click_w0.1")

    def test_missing_incumbent_config_fails_closed(self):
        direction = self.planner.propose([], ResearchState())

        with self.assertRaisesRegex(ValueError, "accepted incumbent configuration is unavailable"):
            self.search.propose_trial(
                direction,
                ResearchState(current_best_experiment_id="exp_missing", current_best_primary=0.61),
                [],
            )

    def test_values_from_a_different_architecture_do_not_exhaust_incumbent_context(self):
        direction = ResearchDirection(
            direction_id="pointwise_fm_optimization",
            hypothesis="Tune DeepFM without treating plain-FM trials as equivalent.",
            rationale="Hyperparameter evidence is conditional on model architecture.",
            search_space={"learning_rate": [0.0005, 0.001]},
            success_evidence="Validation primary is measured.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="local_refinement",
            preferred_factor="learning_rate",
            preferred_value=0.0005,
        )
        history = [{
            "experiment_id": "exp_fm_lr", "direction_id": direction.direction_id,
            "decision": "rejected", "metrics": {"primary": 0.59},
            "changed_factors": ["learning_rate"],
            "config": {**self.search.BASELINE_CONFIG, "learning_rate": 0.0005},
        }, {
            "experiment_id": "exp_deepfm", "direction_id": "fm_architecture",
            "decision": "accepted", "metrics": {"primary": 0.61},
            "config": {**self.search.BASELINE_CONFIG, "architecture": "deepfm"},
        }]

        proposal = SearchController(architecture_human_reviewed=True).propose_trial(
            direction,
            ResearchState(current_best_experiment_id="exp_deepfm", current_best_primary=0.61),
            history,
        )

        self.assertEqual(proposal.config["learning_rate"], 0.0005)
        self.assertEqual(proposal.config["architecture"], "deepfm")

    def test_local_refinement_uses_best_factor_evidence(self):
        direction = ResearchDirection(
            direction_id="pointwise_fm_optimization",
            hypothesis="A focused optimization hypothesis is being tested.",
            rationale="Use prior validation evidence to refine one factor.",
            search_space={"loss": ["pointwise"], "learning_rate": [0.0005, 0.001, 0.002], "l2": [0.0, 1e-6, 1e-5]},
            success_evidence="Primary improves by more than the threshold.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="local_refinement",
        )
        history = [
            {"direction_id": direction.direction_id, "changed_factors": ["learning_rate"],
             "config": {"learning_rate": 0.0005}, "metrics": {"primary": 0.61}},
            {"direction_id": direction.direction_id, "changed_factors": ["l2"],
             "config": {"l2": 1e-5}, "metrics": {"primary": 0.60}},
        ]
        proposal = self.search.propose_trial(
            direction, ResearchState(completed_iterations=2), history,
            search_state=SearchState(strategy="local_refinement"),
        )

        self.assertEqual(proposal.changed_factors, ("learning_rate",))

    def test_architecture_trial_requires_and_records_human_review(self):
        architecture_direction = ResearchDirection(
            direction_id="fm_architecture",
            hypothesis="A nonlinear interaction path may capture preferences beyond second-order FM terms.",
            rationale="The FM structure may be the current representational bottleneck.",
            search_space={"architecture": ["deepfm", "nfm_residual"]},
            success_evidence="Validation primary exceeds the current incumbent.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="exploration",
            specialist_id="model_architecture_specialist",
            preferred_factor="architecture",
            preferred_value="nfm_residual",
        )

        unreviewed = SearchController().propose_trial(architecture_direction, ResearchState(), [])
        reviewed = SearchController(architecture_human_reviewed=True).propose_trial(
            architecture_direction, ResearchState(), []
        )

        self.assertEqual(reviewed.changed_factors, ("architecture",))
        self.assertEqual(reviewed.config["architecture"], "nfm_residual")
        self.assertEqual(reviewed.model_family, "fm_hybrid")
        self.assertFalse(unreviewed.human_reviewed)
        self.assertTrue(reviewed.human_reviewed)
        self.assertFalse(SafetyValidator().validate(unreviewed).passed)
        self.assertTrue(SafetyValidator().validate(reviewed).passed)

    def test_parallel_architecture_siblings_reserve_distinct_one_path_ablations(self):
        parent_spec = ReviewedArchitectureSpec(
            interaction_paths=("embedding_mlp", "bi_interaction_mlp"), fusion="add",
            hidden_width=32, hidden_depth=2, dropout=0.1, cross_layers=2,
        )
        children = [
            ReviewedArchitectureSpec(
                interaction_paths=(path,), fusion="add", hidden_width=32,
                hidden_depth=2, dropout=0.1, cross_layers=2,
            ).architecture_id
            for path in ("embedding_mlp", "bi_interaction_mlp")
        ]
        direction = ResearchDirection(
            direction_id="fm_architecture",
            hypothesis="Removing one nonlinear path will isolate its contribution to ranking quality.",
            rationale="Sibling ablations preserve all non-architecture settings and share a frozen parent.",
            search_space={"architecture": children},
            success_evidence="Validation metrics quantify the removed path contribution.",
            evaluation_budget={"low_epochs": 4, "medium_epochs": 8, "full_epochs": 12},
            strategy="local_refinement", specialist_id="model_architecture_specialist",
            preferred_factor="architecture", preferred_value=children[0],
        )
        parent = {
            "experiment_id": "exp_parent", "direction_id": "fm_architecture",
            "decision": "accepted", "metrics": {"primary": 0.603},
            "config": {**self.search.BASELINE_CONFIG, "architecture": parent_spec.architecture_id},
        }
        state = ResearchState(current_best_experiment_id="exp_parent", current_best_primary=0.603)
        search = SearchController(architecture_human_reviewed=True)

        first = search.propose_trial(direction, state, [parent])
        second = search.propose_trial(
            direction, state, [parent], reserved_experiment_ids=(first.experiment_id,),
            reserved_configs=(first.config,),
        )

        self.assertEqual(first.parent_experiment_id, "exp_parent")
        self.assertEqual(second.parent_experiment_id, "exp_parent")
        self.assertEqual({first.config["architecture"], second.config["architecture"]}, set(children))
