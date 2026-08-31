# Autonomous MLE Agent Architecture


## Document role and precedence

This is the project's architecture reference for the autonomous research system. It describes the target design and longer-term direction; it is not a second benchmark contract.

- The repository-root `AGENTS.md`, the Starter Kit README, and `evaluate.py` remain authoritative.
- If this document conflicts with those sources, the root rules and fixed benchmark contract win.
- Before designing, implementing, or modifying the research agent, consult this document.
- Implement the architecture in phases. The initial version should establish the smallest reliable closed loop before advanced orchestration or search is added.
- Advanced BO/TPE islands and additional model families remain future extensions requiring evidence and human review. A bounded two-worker hypothesis portfolio, one generic implementer, an independent verifier, and a reviewed compositional FM-hybrid architecture language are now implemented.
- Concrete experiment templates are implementation configuration, not part of this architecture document. Define, approve, and version them separately.

## Project implementation constraints

### Canonical data interface

- Treat the datasets returned by `data.py` as the project's canonical cleaned and prepared data.
- All controllers, runners, PyTorch datasets, and models must obtain benchmark data through `data.py`; they must not read the raw KuaiRand CSV files directly.
- Keep split construction, label preparation, metadata joining, feature encoding, and missing/unknown handling centralized in `data.py`.
- If preprocessing must change, change it deliberately in `data.py`, log the exact diff, run leakage and data-contract checks, and keep the change isolated as part of the experiment.
- Do not perform additional silent cleaning, deduplication, row removal, label rewriting, or split reconstruction downstream.

### Model implementation framework

- Implement candidate model architectures and training loops in PyTorch.
- Represent models as `torch.nn.Module` components and use explicit, reproducible PyTorch training and checkpoint logic.
- Preserve the existing NumPy FM in `baseline.py` as the immutable official reference; do not convert or overwrite that reference implementation.
- Before comparing new architectures, verify that the PyTorch baseline path consumes the same `data.py` outputs and uses the unchanged official evaluator.
- PyTorch is approved for this project. Any additional modelling or search dependency still requires human review under the root project rules.

## Current implementation snapshot — 30 August 2026

The repository currently implements Phases 1–8 of the improvement plan. The
sections later in this document continue to describe longer-term design where
they are explicitly marked as recommendations or future work.

```text
Experiment history + diagnostics + incumbent
                         │
                         ▼
                    LLM planner
       strategy + 3–4 falsifiable hypotheses
                         │
              one assigned hypothesis per slot
                         ▼
             Generic LLM implementer
      adopt the expertise required by the hypothesis
       exact registry mapping or capability request
                         │
                         ▼
              Independent verifier
 hypothesis → behavior → config/diff → evidence
                         │
                         ▼
              Search controller + runner
       up to two workers; low → medium → full
                         │
                         ▼
        append-only evidence + planner memory
```

Current properties:

- The default planner is online, uses `gpt-5.6-luna` unless overridden, and
  reads its OpenAI API key from `.env`.
- Each ideation response first authors a structured research strategy. The LLM
  decides whether to start, continue, or revise a phase; selects focus domains
  and metric emphasis; declares frozen factors; allocates worker domains and
  exploit/explore roles; cites its evidence; and states transition criteria.
  The deterministic orchestrator validates contradictions and enforces the
  decision but contains no architecture-first, feature-first, or
  hyperparameter-first priority list. Both workers may therefore investigate
  distinct architectures when that is the LLM's evidence-backed strategy.
  Decisions are append-only in `research_strategies.jsonl`, included in the
  next planner context, and rendered in `research_log.md`.
- A two-worker planning batch uses one shared planner call followed by two
  identical generic-implementer calls. There is no domain-specialist router and
  no capability partition. Every implementer sees the same executable registry
  and adopts the expertise required by its assigned hypothesis.
- An implementer may execute only its assigned hypothesis. Unsupported ideas
  become a diagnostic, capability-build, or human-approval action; they are
  retained in the research tree and are never replaced by `specialist_new` or
  silently mapped onto different behavior.
- The semantic critic verifies the hypothesis, claimed mechanism, named
  implementation, configuration diff, fidelity consistency, patch artifact,
  diagnostics, and measured evidence. Online runs add one independent LLM
  audit after deterministic checks pass.
- The semantic audit payload includes the measured feature-lineage and
  stratified-validation evidence when a hypothesis makes subgroup claims.
  Planner history receives only slice metrics and the weakest eligible stratum;
  training-vocabulary hashes remain in artifacts and are not repeated to the
  LLM.
- Critic output is persisted in `critic_feedback.jsonl` and condensed into
  `critic_memory.json` for the next planning cycle. Feedback distinguishes a
  supported lineage, a valid negative result, execution failure, duplicate
  configuration, lineage defect, and implementation/evidence misalignment.
  The planner must apply these lessons rather than merely observe that a run
  was rejected.
- Before compute, the critic verifies that the declared conceptual factor is
  the only scientific configuration diff and that online lineage references
  identify a known experiment or hypothesis node. After compute, it verifies
  that `primary == (GAUC + nDCG@5) / 2` in addition to experiment ownership,
  fidelity consistency, finite metrics, and patch evidence.
- `evidence_memory.json` gives future planning calls compact training curves,
  score distributions, feature coverage, segment metrics, seed statistics,
  failure classes, resource use, and semantic verdicts.
- `research_tree.jsonl` records append-only hypothesis consideration,
  experiment binding, capability branches, and outcomes. The derived
  `research_tree.json` snapshot links every experiment to its accepted parent,
  preserves deferred and failed branches, reconstructs incumbent ancestry, and
  supplies bounded continuation candidates to the next planning batch.
- `research_tree.md` is regenerated from that snapshot as a colour-coded
  Mermaid flowchart for GitHub and editor Markdown previews. It is a read-only
  projection and never becomes a source of controller state.
- `research_coverage.json` complements the tree with tested, accepted,
  present-in-an-accepted-combination, untested, pending, and unavailable architecture mechanisms, objectives, and
  feature families. It imports only metric-consistent, semantically approved
  non-smoke sibling-run evidence, deduplicates scientific configurations, and
  prefers multi-seed means when available. The tree answers lineage; coverage
  answers what has not yet been tried.
- Multi-task supervision is now an executable training-objective capability,
  not a separate inference architecture. The primary `long_view` head retains
  the accepted FM-hybrid graph; a bounded `is_click` auxiliary head shares its
  embeddings during training with reviewed weights 0.05, 0.1, and 0.2. The
  auxiliary outcome is carried by the canonical `data.py` rows, never becomes
  an input feature, and does not participate in candidate acceptance.
- `--architecture-coverage` turns that inventory into a bounded campaign. The
  LLM authors each hypothesis and ordering, while a deterministic invariant
  permits only mechanisms currently marked executable and untested. The
  campaign ends when no such mechanism remains, preventing local refinement
  from consuming an exhaustive-baseline budget.
- Online planner candidates explicitly declare whether they continue, refine,
  revisit, or start a new branch and identify the evidence node they depend on.
  Accepted lineages are offered first to the exploit worker; near-incumbent
  rejected branches and unresolved hypotheses remain available for deliberate
  revisiting rather than disappearing from a rolling text window.
- Experiment identity belongs to the hypothesis. `low`, `medium`, `full`, and
  seed confirmations are stages of that same experiment, not independent
  experiments. Promotion safety uses the preceding stage as its configuration
  parent, while semantic lineage continues to reference the original accepted
  experiment or hypothesis; these two parent concepts must not be conflated.
- Every new controlled experiment clones the complete accepted-incumbent
  configuration and applies exactly one conceptual change. Search directions
  never silently reset architecture, features, objective, or optimisation
  settings to baseline. Hyperparameter values tested under a different model,
  feature, or objective context do not exhaust the incumbent's search context.
- The search loop runs at most two experiments concurrently. Each worker has
  an explicit one- or two-thread CPU limit, an isolated run directory, and its
  own low → medium → full lifecycle. Completed workers are reconciled serially
  against the latest incumbent, highest three-seed full-fidelity mean first.
- Bounded autonomy separates a round budget from the persistent lifetime cap.
  `--round-budget` (or `--cycles`) limits one approved round, `--max-rounds`
  pre-approves repeated rounds in one invocation, and
  `--max-total-experiments` bounds the artifact directory across resumptions.
  The validation target is explicit through `--target-primary` and defaults to
  0.70. Every round emits `research_checkpoint.json` with progress, target gap,
  stop status, and `test_data_used: false`. A normal checkpoint returns control
  for approval without finalization; test confirmation occurs only with
  `--finalize` or after reaching the declared target.
- Planning overbooks execution slots when an implementer returns a diagnostic or
  capability action. The action remains in the append-only backlog while a
  bounded refill request selects another distinct executable hypothesis. Safe
  capability work does not block unrelated experiments; an invocation pauses
  only when no executable work remains or explicit human authority is required.
- Two-worker batches use the explicit portfolio roles and domains selected by
  the LLM strategy. Workers may occupy the same domain with distinct hypotheses
  or different domains. Both still clone the incumbent so results remain
  compositional. The role is
  included in implementer prompts and persisted in worker, stage, and iteration
  evidence. Capability exhaustion is evaluated in the incumbent's model,
  feature, and objective context rather than globally across incompatible
  configurations.
- Duplicate semantics use one evidence rule across planning, criticism, and
  execution: only a configuration with a finite validation `primary` consumes
  that configuration. Failed, interrupted, pre-run safety/semantic rejected,
  and otherwise unmeasured attempts may be retried. A thread-safe in-flight
  reservation still blocks simultaneous duplicate workers; it is released on
  unmeasured failure and committed to durable duplicate history only after
  metrics validation succeeds. Reservation, release, and commit events are
  append-only evidence.
- Planner output is a decision, not necessarily a training run. The implemented
  action vocabulary is `RUN_EXPERIMENT`, `RUN_DIAGNOSTIC`, `BUILD_CAPABILITY`,
  and `REQUEST_HUMAN_APPROVAL`. The generic implementer must use the latter three when an
  assigned hypothesis is diagnostic, lacks an exact safe implementation, or
  exceeds current authority; they may not translate it into an unrelated
  catalogue experiment. Diagnostics use validation evidence only. Capability
  and approval requests enter `capability_backlog.jsonl` and are fed back to
  subsequent planning calls. `BUILD_CAPABILITY` is non-blocking when independent
  executable work exists. `REQUEST_HUMAN_APPROVAL` pauses after already-started
  safe workers complete; an action-only batch pauses non-terminally rather than
  repeatedly requesting the same action.
- Every fidelity stage records training-derived user-activity and categorical
  feature-coverage strata with GAUC, nDCG@5, primary, row/user counts, positive
  rate, fixed boundaries, and training-vocabulary hashes. The weakest stratum is
  identified only when it has at least 100 rows. This modest diagnostic cost is
  paid at low fidelity as well so pruning cannot create an evidence deadlock.
- When the incumbent is a reviewed two-path composition, the architecture
  capability is narrowed to its two controlled one-path children. Parallel
  siblings share the same frozen parent, reserve different architecture values,
  and record which path was removed. Evidence explicitly limits attribution to
  the removed path rather than claiming a broader causal mechanism.
- Human approvals use stable versioned IDs and authority scopes. Repeated or
  concurrent reuse writes an event referencing the original approval instead of
  another intervention record. Reports collapse identical legacy records while
  preserving their raw append-only count.
- Before every planning batch, the orchestrator receives the immutable benchmark
  contract, a digest and operating boundary from this architecture document,
  recent manual interventions, compact errors/recoveries, state, metrics, and
  diagnostic evidence. Exact numeric trial selection remains outside the LLM.
- Finalization reconstructs the selected checkpoint's recorded FM, DeepFM, or
  residual-NFM architecture, validates the submission schema, and writes
  `readiness.json`. Readiness fails visibly when lifecycle logs, patches,
  diagnostics, semantic reviews, state, selection lineage, or submission
  evidence are missing.

## Autonomous MLE Experimentation System

This document defines the architecture, responsibilities, operating rules, and implementation guidance for an autonomous machine-learning experimentation system designed to improve recommender-system performance metrics such as **GAUC** and **nDCG@5**.

The central design principle is:

> **Agents reason about what is worth investigating; specialized search algorithms efficiently determine which exact configurations to test.**

The system must explicitly balance **exploration** and **exploitation**, use **multi-fidelity evaluation** to conserve compute, and maintain multiple promising search regions so that it does not become trapped in a local optimum.

---

## 1. System Goal

Given a recommender-system training pipeline and an evaluation function, the system should autonomously discover pipeline changes that improve a target metric while minimizing:

- GPU/CPU compute
- wall-clock time
- unnecessary model training
- LLM token usage
- repeated experiments
- overfitting to a single validation split

For the KuaiRand-Pure benchmark, the optimization target is fixed:

```text
primary = (GAUC + nDCG@5) / 2
```

GAUC and nDCG@5 must also be reported separately. The system must not substitute a different primary objective.

For the fixed objective:

\[
x^* = \arg\max_x f(x)
\]

where:

- `x` is a complete experiment configuration
- `f(x)` is the resulting evaluation score

The true function `f(x)` is expensive and unknown before experimentation. The system therefore combines reasoning agents with efficient black-box optimization.

---

## 2. Target High-Level Architecture

This diagram is the longer-term system shape. Refer to the implementation
snapshot above for components that are operational today.

```text
                         ┌───────────────────────────┐
                         │       LLM AGENT LAYER      │
                         │                           │
                         │ Diagnose                  │
                         │ Generate hypotheses       │
                         │ Decide what to investigate│
                         │ Detect plateaus           │
                         │ Review evidence           │
                         └─────────────┬─────────────┘
                                       │
                              Hypothesis / Search Task
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
             Feature / Data                         Model / Training
                 Agent                                  Agent
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │    SEARCH CONTROLLER      │
                         │                           │
                         │ Exploration/exploitation  │
                         │ Strategy selection        │
                         │ Compute allocation        │
                         │ Region management         │
                         └─────────────┬─────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
          EXPLORATION              EXPLOITATION          MULTI-FIDELITY
                │                      │                      │
        Random / Diverse         BO / TPE / Local        ASHA / Hyperband
        Search / Restarts          Search / UCB             / Pruning
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │    EXPERIMENT RUNNER      │
                         │                           │
                         │ Build config              │
                         │ Train model               │
                         │ Evaluate                  │
                         │ Record artifacts          │
                         └─────────────┬─────────────┘
                                       ▼
                             GAUC / nDCG@5 / Loss
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      EXPERIMENT DB        │
                         │                           │
                         │ Config                    │
                         │ Hypothesis                │
                         │ Metrics                   │
                         │ Search region             │
                         │ Budget                    │
                         │ Parent/lineage            │
                         │ Uncertainty               │
                         │ Compute cost              │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                                  LLM REVIEW
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                         Improving           Stuck
                              │                 │
                              ▼                 ▼
                         Exploit            Explore / Restart
                              │                 │
                              └────────┬────────┘
                                       ▼
                                     LOOP
```

---

## 3. Core Design Principle: Separate Strategy from Search

The LLM should **not** be the low-level hyperparameter optimizer.

Bad pattern:

```text
LLM → suggest LR = 0.000743
LLM → suggest LR = 0.000691
LLM → suggest LR = 0.000812
LLM → repeat
```

This consumes tokens while using a language model for a task that classical optimization algorithms handle better.

Preferred pattern:

```text
LLM:
    "Negative sampling appears to be a bottleneck."

Search Controller:
    "Search negative-sampling parameters in this defined space."

TPE/BO/ASHA:
    Select exact configurations and allocate evaluation budget.

Experiment Runner:
    Train and evaluate.

LLM:
    Review the accumulated evidence and decide what concept to investigate next.
```

### Responsibilities by level

| Layer | Main question |
|---|---|
| LLM agent | **What should we investigate? Why?** |
| Search controller | **Which search strategy and how much compute?** |
| Search algorithm | **Which exact configuration should be evaluated next?** |
| ASHA/Hyperband | **Which trials deserve more compute?** |
| Experiment runner | **Run the experiment correctly.** |
| Metrics layer | **How good was the experiment?** |
| Experiment DB | **What have we learned so far?** |

---

## 4. Agent Layer

### 4.1 Research/Orchestrator Agent

The main LLM agent is responsible for high-level experimental reasoning.

It should:

1. Read current experiment history.
2. Identify trends and plateaus.
3. Compare GAUC and nDCG@5 behavior.
4. Generate hypotheses.
5. Select a research direction.
6. Define a reasonable search space.
7. Decide whether the current direction is still worth pursuing.
8. Request additional exploration when evidence is weak.
9. Request a restart or change in research direction when the search plateaus.
10. Record a concise rationale for each major search decision.

It should **not** manually micromanage every individual trial.

### 4.2 Generic Implementer

The active system uses one generic LLM implementer contract rather than a
network of routed specialists. For each assigned hypothesis it adopts the
needed domain expertise, compares the claim with the complete executable
registry, and returns either an exact experiment plan or one explicit
non-training action. It cannot replace the assigned hypothesis with an easier
catalogue experiment. The verifier remains independent of the implementer.

Arbitrary generated source patches are not automatically executed. Existing
reviewed declarative structures and registered feature/objective implementations
are executable; missing behavior becomes `BUILD_CAPABILITY`, and substantially
different model families or dependencies become `REQUEST_HUMAN_APPROVAL`.

The role descriptions below are expertise the generic implementer may adopt;
they are not separate running agents.

#### Feature/Data Agent

Investigates:

- user features
- item features
- historical behavior windows
- recency features
- interaction aggregation
- missing-value handling
- data leakage risks
- feature normalization
- feature crossing

Example hypothesis:

> Recent user interactions may contain more predictive signal than long-term aggregate behavior.

#### Model Architecture Agent

The generic implementer owns structural model experiments when assigned an
architecture hypothesis. It receives the evidence-backed hypothesis and the
reviewed architecture registry. It may select a categorical model structure,
while the search controller still owns low-level numeric search.

The compiler accepts the legacy structures:

- `deepfm`: the immutable FM prediction path plus a residual MLP over
  concatenated field embeddings; fixed hidden layers `[64, 32]` and dropout
  `0.1`.
- `nfm_residual`: the immutable FM prediction path plus a residual nonlinear
  network over the vector-valued FM bi-interaction; fixed hidden layer `[32]`
  and dropout `0.1`.

It also accepts a canonical declarative specification that preserves the
immutable FM score and composes one or two reviewed residual paths:

- `embedding_mlp`: an MLP over concatenated field embeddings.
- `bi_interaction_mlp`: an MLP over the vector-valued FM bi-interaction.
- `cross_network`: one to three explicit DCN-style cross layers.

The implementer may select widths `16`, `32`, or `64`, depths `1` to `3`,
dropout `0`, `0.1`, or `0.2`, and additive fusion. Two-path structures may
instead use a learned gate. The compiler rejects duplicate or unknown paths,
out-of-range values, non-canonical identifiers, architecture overhead above
500,000 parameters, and total models above 10,000,000 parameters. It never
accepts source code, import names, arbitrary classes, or unconstrained numeric
dimensions.

All variants are compiled by `research_agent.models.torch_fm.build_model` from
the reviewed aliases or `ReviewedArchitectureSpec`. Each run records
`architecture_spec.json`, a structural diff from FM, parameter count,
checkpoint configuration, and diagnostics. They are FM hybrids and do not use
external data, pretrained weights, or a new dependency.

Architecture experiments are omitted from the executable registry
unless the run is started with `--approve-architecture-experiments`. That flag
records the human intervention required by the project rules. The independent
verifier audits architecture evidence; the generic implementer authors the
reviewed structural choice for its assigned architecture hypothesis.

The reviewed operator set is the safety and reproducibility boundary, not the
number of complete model recipes. Future expansion should add operators to the
declarative compiler rather than allow the LLM to execute arbitrary source
code. Every new operator still requires tests, resource bounds,
semantic-contract metadata, and human review before it becomes executable.

For architecture hypotheses, the implementer investigates:

- embedding architecture
- number of layers
- hidden dimensions
- attention mechanisms
- residual connections
- ranking heads
- model family changes

Example hypothesis:

> The current model may be under-expressive for the user/item interaction structure.

#### Training Agent

Investigates:

- learning rate
- optimizer
- scheduler
- regularization
- dropout
- batch size
- epochs
- warmup

#### Sampling Agent

Investigates:

- negative-sample ratio
- random negatives
- popularity-based negatives
- hard negatives
- sampling temperature
- candidate filtering

#### Evaluation Agent

Checks:

- metric correctness
- train/validation/test leakage
- user-level aggregation
- sample weighting
- reproducibility
- metric variance across runs

A system should not optimize a metric it cannot trust.

---

## 5. Search Controller

The Search Controller is the central decision layer between high-level hypotheses and low-level optimization.

It decides:

- exploration vs exploitation allocation
- search algorithm
- search region
- evaluation fidelity
- whether to continue, pause, or restart a search
- which candidates should receive additional compute

### 5.1 Search states

The controller should maintain an explicit search state.

Recommended states:

```text
BOOTSTRAP
EXPLORING
PROMISING
EXPLOITING
PLATEAU
RESTARTING
VALIDATING
FINISHED
```

### 5.2 Example state transitions

```text
BOOTSTRAP
   ↓
EXPLORING
   ↓
PROMISING
   ↓
EXPLOITING
   ↓
┌───────────────┐
│               │
│ improvement   │ plateau
│               │
▼               ▼
EXPLOITING     PLATEAU
                 │
          ┌──────┴──────┐
          ▼             ▼
      increase       NEW REGION /
      exploration       RESTART
          │             │
          └──────┬──────┘
                 ▼
             EXPLORING
```

---

## 6. Exploration Strategies

Exploration exists to discover useful regions the current model of the search space does not yet understand.

### 6.1 Random Search

Sample configurations from the allowed search space without trying to favor the current best region.

Use when:

- little information is available
- the search space is large
- a baseline is needed
- avoiding early bias is important

Random search is a deliberate hedge against premature commitment.

### 6.2 Diverse Search

Prefer configurations that are substantially different from prior experiments.

Conceptually:

\[
\text{exploration score}(x)=\text{distance}(x,\text{previous experiments})
\]

Useful distance functions may be defined separately for numeric and categorical variables.

Purpose:

> Explore regions that have not already been repeatedly sampled.

### 6.3 Random Restarts

When the search is stuck, create a new seed configuration far from the current incumbent and begin another local/search trajectory.

Restarts should not be treated as failure. They are a mechanism for escaping local optima.

### 6.4 Bayesian Uncertainty Exploration

If using a Bayesian surrogate, select some experiments because uncertainty is high.

For Upper Confidence Bound:

\[
UCB(x)=\mu(x)+\kappa\sigma(x)
\]

where:

- `μ(x)` = predicted performance
- `σ(x)` = uncertainty
- `κ` = exploration strength

A larger `κ` encourages more exploration.

### 6.5 Evolutionary Exploration

Maintain multiple candidate configurations and mutate them.

Exploration is preserved by:

- mutation
- population diversity
- occasional random immigrants
- maintaining multiple candidate lineages

---

## 7. Exploitation Strategies

Exploitation uses known information to improve promising candidates.

### 7.1 Bayesian Optimization

Use a surrogate model to estimate promising configurations and prioritize expensive evaluations.

Best suited when:

- experiments are expensive
- the search space is reasonably structured
- the number of evaluations is limited

The optimizer should update after observed experiment results.

### 7.2 TPE / Tree-structured Parzen Estimator

TPE is useful when the search space includes:

- categorical variables
- conditional parameters
- discrete values
- irregular parameter spaces

The practical mechanism is to learn which parameter configurations are associated with stronger outcomes and sample preferentially from promising regions.

### 7.3 Local Search

Perturb a strong configuration by small amounts.

Example:

```text
Current best:
embedding = 128
lr = 0.0007

Neighborhood:
embedding = 96
embedding = 112
embedding = 144
embedding = 160
```

This is useful for final refinement.

---

## 8. Multi-Fidelity Search

Never assume every configuration deserves a full training run.

### 8.0 Implemented fidelity lifecycle

The current controller uses an ASHA-like three-rung lifecycle:

```text
experiment/hypothesis ID
        │
        ▼
low (default 4 epochs)
        │ promising: primary >= incumbent - 0.002
        ▼
medium (default 8 epochs)
        │ promising: primary >= incumbent - 0.002
        ▼
full (default 12 epochs)
        │ promising
        ▼
seed_1 + seed_2 confirmation
```

All rungs retain the same experiment ID, hypothesis, conceptual factor, and
factor value. Only fidelity metadata, epoch budget, and confirmation seed may
change. A semantic post-run check verifies this invariance. Weak candidates are
pruned after low or medium fidelity. This is successive promotion within each
experiment. Up to two independent experiment lifecycles can now advance
concurrently; rungs within one experiment remain sequential because later
fidelity depends on earlier evidence.

Seed 0 is the full rung. Acceptance and concurrent commit ordering require the
mean of full-fidelity seeds 0, 1, and 2 to exceed the current incumbent. Missing
confirmation seeds make an experiment ineligible for acceptance. This same
robust rule is used when resuming older artifacts, so a noisy seed-0 incumbent
is automatically reconciled to the strongest available three-seed mean.

The system should evaluate candidates at progressively higher fidelity.

Example:

```text
1000 candidates
      ↓
cheap training / few epochs
      ↓
100 candidates
      ↓
medium training
      ↓
20 candidates
      ↓
long training
      ↓
5 candidates
      ↓
full training
      ↓
best candidates
```

### 8.1 ASHA / Hyperband

Successive-halving style methods should be used to stop weak candidates early.

Example:

```text
100 trials × 1 epoch
      ↓
keep 30
      ↓
30 trials × 3 epochs
      ↓
keep 10
      ↓
10 trials × 10 epochs
      ↓
keep 3
      ↓
3 trials × full budget
```

This is one of the strongest compute-saving mechanisms in the architecture.

### 8.2 Important constraint

Early performance must be reasonably predictive of final performance.

If a model often starts poorly but eventually wins, overly aggressive early stopping can eliminate the true optimum.

Therefore the system should periodically test whether early-fidelity ranking correlates with final-fidelity ranking.

---

## 9. Exploration vs Exploitation Controller

The controller should dynamically allocate search budget.

A simple initial policy may be:

```text
Early stage:
80% exploration / 20% exploitation

Promising stage:
30% exploration / 70% exploitation

Strong improvement:
20% exploration / 80% exploitation

Plateau:
60% exploration / 40% exploitation
```

These numbers are starting points, not hard-coded truths.

### 9.1 Signals for exploitation

Increase exploitation when:

- recent experiments consistently improve the target metric
- multiple nearby configurations are strong
- Bayesian uncertainty is low around the promising region
- improvement per compute is high
- independent runs reproduce the improvement

### 9.2 Signals for exploration

Increase exploration when:

- improvement has plateaued
- recent candidates are too similar
- surrogate uncertainty is high outside the current region
- all experiments are concentrated around one local region
- changes to hyperparameters no longer produce meaningful gains
- GAUC and nDCG@5 disagree persistently
- the current hypothesis has repeatedly failed

---

## 10. Preventing Local Optima

The system cannot mathematically guarantee that it will never reach a local optimum. It should instead use several independent safeguards.

### 10.1 Maintain multiple search regions

Do not maintain only a single "best configuration".

Maintain an archive of promising regions or lineages:

```text
Region A: embedding ≈ 64
Region B: embedding ≈ 128
Region C: different sampling strategy
Region D: different architecture
```

A weaker current region should not automatically be deleted if it remains scientifically plausible.

### 10.2 Keep an exploration quota

Reserve an explicit portion of compute for unexplored regions.

Example:

```text
70% → exploit known-good regions
20% → investigate uncertain regions
10% → completely new/random directions
```

### 10.3 Plateau detection

Monitor recent improvement.

For a rolling window:

\[
\Delta f = f_{best,t}-f_{best,t-N}
\]

If `Δf` remains below a practical threshold while compute continues to rise, mark the search as plateauing.

Do not endlessly optimize a saturated region.

### 10.4 Random restart

When a search region plateaus, initialize a new region using a randomized or deliberately distant configuration.

### 10.5 Conceptual restart

A hyperparameter plateau does not necessarily mean that the numerical values are wrong.

It may mean the **hypothesis is wrong**.

Example:

```text
Many learning-rate experiments
        ↓
very small metric improvement
        ↓
AGENT:
"Learning rate probably isn't the main bottleneck."
        ↓
Investigate negative sampling / features / architecture
```

This is a key reason to retain the LLM layer.

### 10.6 Diversity constraints

The controller should measure configuration diversity.

Do not allow 50 consecutive experiments to be near-identical unless there is strong evidence that local refinement is still productive.

---

## 11. Multiple Search Islands

A robust implementation should be able to run multiple search islands in parallel.

**Implementation status:** a bounded two-worker portfolio is implemented. One
planner creates a shared slate, two identical implementers receive distinct hypotheses,
and the workers execute in isolated experiment/stage directories. Artifact
writes and safety-history updates are synchronized. Worker results are sorted
by best validation primary and committed serially, so only a candidate that
beats the incumbent at reconciliation time is accepted. `--workers` selects one
or two workers and `--worker-threads` limits each worker to one or two CPU
threads.

This is deliberately smaller than the long-term island architecture below. It
does not yet provide independent BO/TPE/evolutionary populations or migration;
those remain future extensions.

```text
                 SEARCH SPACE

      ┌───────────────────────────────────┐
      │                                   │
      │  Island A      Island B            │
      │  BO/TPE        BO/TPE              │
      │                                   │
      │          Island C                 │
      │          Evolutionary             │
      │                                   │
      │  Island D                         │
      │  Random exploration               │
      │                                   │
      └───────────────────────────────────┘
```

Each island can explore a different:

- model family
- feature hypothesis
- sampling strategy
- hyperparameter region
- optimizer

### Island migration

Periodically, strong discoveries can be shared across islands.

Example:

```text
Island C discovers:
new negative sampling strategy → nDCG +0.04

        ↓
share discovery

Island A/B/D may test that strategy
```

Do not force every island to converge immediately. Diversity is valuable.

---

## 12. Experiment Database

Every experiment must be recorded in a structured form.

Recommended record:

```yaml
experiment_id: exp_00142
parent_id: exp_00131
hypothesis_id: hyp_00027
region_id: region_03
search_strategy: tpe

config:
  learning_rate: 0.0007
  embedding_dim: 128
  dropout: 0.10
  negative_samples: 20

budget:
  epochs: 10
  gpu_hours: 1.8

metrics:
  gauc: 0.721
  ndcg_at_5: 0.337
  train_loss: 0.841

status: completed
seed: 42

lineage:
  parent_experiment: exp_00131
  mutation_type: learning_rate

notes:
  hypothesis: "More aggressive negative sampling may improve top-k ranking."
```

The database is the system's shared memory.

---

## 13. Experiment Lineage

Experiments should form a graph rather than an unstructured list.

```text
exp_001
   │
   ├── exp_002
   │     └── exp_006
   │
   ├── exp_003
   │
   └── exp_004
         └── exp_007
```

This allows the agent to understand:

- which changes caused improvement
- which branches failed
- which search regions originated from which hypotheses
- whether gains depend on a particular parent configuration

Every experiment should ideally have:

- parent experiment
- hypothesis
- search strategy
- resource budget
- final metrics

---

## 14. Metric Handling

### 14.1 Primary objective

The benchmark primary objective is fixed as the arithmetic mean of GAUC and nDCG@5. Neither component may replace it for candidate acceptance. The agent should still inspect the two component metrics separately when interpreting results.

The implemented acceptance and convergence rules are intentionally separate:

- If a semantically valid completed candidate has `primary > incumbent_primary`,
  it becomes the new incumbent. No additional `+0.002` acceptance margin is
  required.
- The `0.002` value is a convergence threshold only. A completed hypothesis
  whose gain is not greater than `0.002` increments the consecutive
  low-improvement counter even when a smaller positive gain was accepted.
- Three consecutive completed hypotheses without a gain greater than `0.002`
  trigger a diverse research restart. Fidelity stages do not increment this
  counter independently.
- A post-run semantic-audit failure blocks incumbent promotion regardless of
  metric value.

### 14.2 Secondary metrics

Track GAUC and additional diagnostics simultaneously.

Example:

```text
Primary:    (GAUC + nDCG@5) / 2
Components: GAUC and nDCG@5
Diagnostic: log loss / calibration / coverage / diversity
```

### 14.3 Do not blindly maximize one metric

Possible outcome:

```text
GAUC ↑
nDCG@5 ↓
```

This is not necessarily contradictory. The model may improve overall ranking discrimination while becoming worse at placing the most useful items at the very top.

The agent should investigate metric disagreement rather than averaging everything blindly.

### 14.4 Statistical reliability

Small metric improvements may be noise.

Important candidates should be re-run across multiple seeds or evaluation samples where feasible.

The system should distinguish:

```text
apparent improvement
        vs.
reproducible improvement
```

---

## 15. Token-Efficient Agent Operation

LLM tokens are a scarce resource relative to deterministic search operations.

### 15.0 Implemented API-call and token budgets

For the default two-worker batch, the online loop makes:

1. one planner call producing a strategy and 3–4 hypothesis slate;
2. two concurrent generic-implementer calls, each preserving one assigned
   hypothesis and returning either an exact mapping or a capability action; and
3. after execution, one independent semantic-audit call per experiment when
   deterministic evidence checks pass.

Thus planning costs three API calls per two experiments (1.5 planning calls per
experiment), instead of independently repeating planning for both workers. A
`--workers 1` invocation retains the original two-call planning path.

`OPENAI_PLANNER_TOKEN_BUDGET` currently defaults to `0`, disabling the aggregate
planning stop during system development. It can be set to a positive per-batch
accounting guard; this is **not** the model's context-window or output-token
maximum. Observed usage is always recorded so a production cap can be selected
from evidence rather than guesswork.

The single-hypothesis planner has `max_output_tokens=2400`; the shared
two-worker strategy/slate call has `max_output_tokens=4000`; each implementer
has `max_output_tokens=1400`. These ceilings include generated reasoning and visible
structured output, so they constrain each response independently of the combined
optional aggregate accounting guard. When configured to a positive value, the
planner checks usage after the planner call and again after all implementer calls; it
pauses rather than silently exceeding the batch budget. Because exact input
usage is known only after a request, this is an accounting boundary, not a
guaranteed pre-request billing cap. Per-response output ceilings and usage
logging remain enabled when the aggregate guard is `0`.

`OPENAI_SEMANTIC_CRITIC_TOKEN_BUDGET` separately defaults to `0` during system
development, disabling its aggregate accounting rejection while preserving
usage logging. When configured to a positive value, it bounds the audit. The audit
call has `max_output_tokens=1400`; earlier uncompressed audits used about 1,800–3,100 total
tokens depending on evidence depth. Keeping the critic separate prevents its
cost from obscuring planner cost.

Cost controls currently implemented:

- at most six compact evidence records are sent to planning agents;
- implementers receive only their assigned hypothesis, not the other slate
  candidates or complete governance context;
- reviewed architecture values are not duplicated as a full prompt schema
  because the response schema already enforces them;
- semantic LLM audits receive one compact best-stage trace while full stage
  diagnostics remain local;
- full checkpoints, raw datasets, and full logs are never sent to the LLM;
- stable prompt-cache keys are used per role and cached-token usage is logged;
- low text verbosity and strict JSON schemas bound visible output;
- numeric search, fidelity promotion, metric calculation, and artifact checks
  remain deterministic and consume no LLM tokens;
- an unsupported assigned hypothesis becomes an explicit capability or approval
  action; substitution is forbidden.

A live two-worker Phase 6 batch used 5,142 planning tokens for two distinct
hypotheses under the retired specialist design. A later evidence-rich Phase 7 batch used
15,742 tokens, demonstrating that the original 12,000 guard was too low. During
system development the aggregate planning guard is now disabled (`0`), while
usage remains metered and semantic audits remain separately budgeted.

### Good use of tokens

- analyze experiment trends
- propose high-level hypotheses
- detect saturation
- identify promising research directions
- interpret metric conflicts
- select or modify search spaces
- decide whether to change the problem formulation

### Bad use of tokens

- manually enumerate hundreds of numeric configurations
- rewrite identical experiment instructions
- reason independently about every low-level trial
- perform calculations that a search library can execute
- repeatedly inspect unchanged experiment history

### Batch operation

Prefer:

```text
LLM reasoning pass
      ↓
define search task
      ↓
50–500 automated trials
      ↓
aggregate results
      ↓
LLM review pass
```

rather than:

```text
LLM → one trial
LLM → one trial
LLM → one trial
LLM → one trial
...
```

This can dramatically reduce token consumption.

---

## 16. Coarse-to-Fine Optimization

The system should usually move from cheap/broad search to expensive/narrow search.

```text
                    LARGE SEARCH SPACE
                           │
                           ▼
                  Random / TPE / cheap
                           │
                    many candidates
                           ▼
                       ASHA
                           │
                    remove weak trials
                           ▼
                       BO / TPE
                           │
                   promising regions
                           ▼
                    Local refinement
                           │
                           ▼
                    Full training
                           │
                           ▼
                   independent validation
```

The fundamental rule is:

> **Spend the minimum amount of computation needed to make the next decision.**

---

## 17. Suggested End-to-End Loop

```text
1. Establish baseline.

2. Validate evaluation code.
   - GAUC
   - nDCG@5
   - train/validation/test separation

3. Run a broad bootstrap search.
   - random search
   - diverse sampling
   - cheap fidelity

4. Identify several promising regions.

5. Start multiple search islands.
   - TPE/BO
   - local refinement
   - evolutionary branch

6. Use ASHA/Hyperband to eliminate weak trials.

7. Increase compute for promising candidates.

8. Track improvement per compute.

9. Maintain explicit exploration.

10. Detect plateaus.

11. If plateaued:
    - increase exploration
    - start a new region
    - perform a random restart
    - ask the agent whether the hypothesis itself is wrong

12. Promote strong candidates to full-budget training.

13. Re-run finalists with independent seeds.

14. Compare against the baseline.

15. Record the winning configuration and evidence.

16. Begin the next research cycle.
```

---

## 18. Example Search Cycle

Assume baseline:

```text
GAUC    = 0.710
nDCG@5  = 0.310
```

### Cycle 1: broad exploration

Random/TPE evaluates many cheap candidates.

Best region:

```text
embedding ≈ 128
learning rate ≈ 5e-4
```

### Cycle 2: exploitation

BO refines around that region.

```text
nDCG@5:
0.310 → 0.319 → 0.325 → 0.327
```

### Cycle 3: plateau

Additional local tuning gives:

```text
0.327 → 0.327 → 0.326 → 0.327
```

Controller marks the region as saturated.

### Cycle 4: exploration

A separate island investigates negative sampling.

```text
random negatives → 0.322
hard negatives   → 0.334
```

Now the agent has evidence that sampling, not another tiny learning-rate adjustment, may be the next major direction.

### Cycle 5: combined refinement

Search:

```text
embedding
learning rate
negative sampling
```

using BO/TPE with ASHA.

Potential result:

```text
nDCG@5 = 0.341
```

The improvement is then independently validated.

---

## 19. Controller Decision Policy

A practical controller can use a simple scoring framework.

For each candidate/search region estimate:

```text
Expected value
+ uncertainty value
+ diversity value
+ historical success
--------------------------------
compute cost
```

The controller then chooses experiments that have strong expected information or performance per unit compute.

A conceptual utility function is:

\[
U(x)=\frac{E[\text{improvement}(x)] + \lambda\,\text{uncertainty}(x) + \gamma\,\text{diversity}(x)}{\text{estimated compute}(x)}
\]

This is a design abstraction rather than a required exact formula.

---

## 20. Recommended Practical Stack

A practical initial implementation can use:

```text
LLM orchestration
        ↓
Experiment manager
        ↓
Optuna / TPE / Bayesian optimization
        ↓
ASHA / Hyperband pruning
        ↓
PyTorch training framework
        ↓
GAUC / nDCG@5 evaluator
        ↓
Experiment database
```

The system does not need every algorithm on day one.

### Minimal useful version

Start with:

```text
LLM Agent
   ↓
TPE/Random search
   ↓
ASHA
   ↓
Experiment DB
   ↓
LLM review
```

Then add:

- multiple search islands
- BO with uncertainty-based exploration
- evolutionary search
- automatic restarts
- more sophisticated plateau detection
- multi-objective optimization

---

## 21. Safety and Reproducibility Rules

The system must never silently:

- change train/validation/test splits
- change metric definitions
- alter preprocessing without logging it
- bypass `data.py` by loading raw benchmark CSV files in a controller, runner, or model
- compare results produced by incompatible evaluation code
- reuse test-set results for optimization
- overwrite experiment history
- claim an improvement without recording the exact configuration

Every promoted model must have:

1. exact code/version
2. exact configuration
3. data/version identifier
4. metric implementation/version
5. random seed(s)
6. training budget
7. evaluation results
8. parent experiment/hypothesis

The test set should be treated as a **final confirmation**, not as the routine search objective.

---

## 22. Mandatory Action and Iteration Logging

Logging is part of the evaluated autonomous behavior, not an optional dashboard feature. The structured run artifacts are the source of truth, and the Markdown log is the human-readable report generated from them.

The agent must record every material action that changes or evaluates research state, including:

- selecting or rejecting a hypothesis
- creating or modifying a candidate
- changing a configuration
- starting, stopping, timing out, or retrying training
- invoking evaluation
- accepting, rejecting, or restoring a candidate
- detecting a policy, leakage, data-quality, or resource problem
- receiving a manual intervention

Each iteration record must include:

1. Experiment and parent identifiers.
2. The hypothesis: what the agent intended to try and why.
3. The single controlled change and complete configuration.
4. The code diff applied, stored as a patch or an explicit no-code-change configuration diff.
5. Validation GAUC, nDCG@5, primary score, and delta from the accepted parent.
6. Runtime, configured resource budget, random seed, and available compute/token usage.
7. Decision: accepted, rejected, inconclusive, or failed, with the reason.
8. Every error or recovery event and how the agent handled it.
9. Any restoration of the last accepted candidate.
10. Manual interventions associated with the iteration.
11. The semantic review trace: claimed behavior, implementation identifier,
    configuration diff, evidence-stage identifiers, deterministic checks, LLM
    audit verdict when online, and any failed correspondence.
12. For architecture experiments, the selected architecture, declarative
    structural diff, model family, trainable parameter count, and
    `architecture_spec.json` path.

Required outputs:

- append-only JSONL for detailed events and iteration records
- CSV for the metric and decision summary
- Markdown for the readable per-iteration run log
- patch files for code changes
- a short final manual-intervention summary

The manual-intervention summary must report:

- total intervention count, including an explicit zero
- experiment or timestamp for each intervention
- what the human requested or changed
- why intervention was necessary
- its effect on the run or decision

Routine user observation, reading logs, or approving the initial run budget is not an intervention unless it changes an active experiment. The agent must never invent missing metrics, token counts, resource measurements, diffs, errors, recoveries, or interventions; unavailable values must be recorded as unavailable.


---

## 23. Success Criteria

The system is successful when it can demonstrate all of the following:

### Performance

- Find configurations that reliably improve the baseline.
- Preserve improvements under independent re-runs.

### Efficiency

- Use substantially fewer expensive full-training runs than naive grid/random search.
- Stop poor experiments early.
- Minimize LLM token usage by batching search operations.

### Robustness

- Continue exploring after local improvements plateau.
- Maintain multiple plausible search regions.
- Recover from poor early decisions.

### Scientific quality

- Maintain complete experiment lineage.
- Produce reproducible results.
- Distinguish noisy improvements from reliable improvements.
- Keep test evaluation isolated from routine optimization.

---

## 24. Guiding Principle

The system should behave like a scientist with specialized laboratory equipment:

```text
LLM / Agents
    = formulate hypotheses and interpret evidence

Search Algorithms
    = systematically choose experiments

ASHA / Hyperband
    = stop wasting resources on weak experiments

Experiment Runner
    = perform the actual experiment

GAUC / nDCG@5
    = measure the result

Experiment DB
    = preserve scientific memory

Exploration / Restarts / Multiple Islands
    = prevent tunnel vision and local-optimum lock-in
```

The final objective is not merely:

> **"Find the best hyperparameters."**

It is:

> **"Build a closed-loop system that continuously forms better hypotheses, tests them efficiently, learns from the results, explores alternative explanations, and converges toward better models without wasting compute or LLM tokens."**
