# KuaiRand-Pure Starter Kit

## 依赖

Python 3.9+ 和 numpy。**没有别的。** 不需要 torch、pandas、sklearn。

## 数据

从 https://kuairand.com 下载（Zenodo 直链，无需注册）：

```bash
# 在 Starter Kit 目录下执行，解压后得到 ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## 运行

```bash
python3 baseline.py --model fm
```

### LLM research agent

The default research agent is online and deliberately small. On every cycle, an
LLM planner produces a strategy and 3–4 materially different scientific hypotheses
without seeing repository direction names or the executable catalogue. It may
consider model architecture, interaction structure, ranking objectives,
optimisation, features, temporal effects, or diagnostics. One generic LLM
implementer then adopts the expertise required by each assigned hypothesis and
checks it against the full executable registry. It cannot substitute another
hypothesis. Unsupported ideas become explicit diagnostic, capability-build, or
human-approval actions. An independent verifier and deterministic runner enforce
safety, select exact numeric values, train, and score the result.

Copy `.env.example` to `.env`, add `OPENAI_API_KEY`, and optionally change
`OPENAI_MODEL` (the project default is `gpt-5.6-luna`). `OPENAI_PLANNER_TOKEN_BUDGET` optionally sets the per-cycle
token ceiling per planning batch (disabled by default with `0` during development), while `OPENAI_SEMANTIC_CRITIC_TOKEN_BUDGET`
optionally caps the audit call (also disabled by default with `0` during development). Do not paste or commit the key. Each planning
hypothesis batch makes one shared planner call and up to two concurrent generic-implementer
calls. If an implementer returns a capability action, a bounded refill request may
make an additional planning call to keep an execution slot occupied. The agent logs the full
hypothesis slate, selection/defer decision, response IDs, and aggregate token
usage. One additional bounded call independently audits the completed
hypothesis-to-evidence chain. Stable cache keys, low response verbosity, bounded
outputs, and six compact evidence records limit API cost:

The aggregate planning-token guard is disabled during development. Set
`OPENAI_PLANNER_TOKEN_BUDGET` to a positive value to restore it. Per-response
output ceilings and token logging remain active.

```powershell
.\.venv\Scripts\python.exe -m research_agent.run_research --artifact-dir runs_llm
```

The default runs two experiments concurrently with one CPU thread each. Before
each batch, the online LLM planner authors a structured research strategy: the
current phase, focus domains, diagnostic metric emphasis, frozen factors,
worker-domain assignments, evidence rationale, and phase-transition criteria.
It may assign both workers to distinct architecture hypotheses when the evidence
supports architecture discovery, or allocate exploit/explore roles across
different domains. Deterministic code validates and enforces the declared
strategy but does not choose its scientific priority. Strategy decisions are
append-only in `research_strategies.jsonl` and appear in `research_log.md`.
Roles and the shared executable registry are also recorded in planner metadata,
implementer prompts, worker events, stages, and iteration records. Use
`--workers 1` for sequential execution, or `--worker-threads 2` to allow two
threads per worker. Both controls are safety-bounded to a maximum of two:

```bash
.venv/bin/python -m research_agent.run_research --workers 2 --worker-threads 1 --cycles 2 --artifact-dir runs_parallel
```

Capability and diagnostic actions do not consume experiment slots. The
scheduler retains them in the backlog and refills available workers with
distinct executable hypotheses. Safe capability work therefore does not pause
unrelated training. Explicit human authority still pauses after current safe
workers finish, and an action-only batch yields when no executable work remains.

It selects experiments using validation only. It stops successfully at the
configured validation-primary target (default `0.70`), asks the online team for
a diverse restart after three improvements below `0.002`, and has a configurable
persistent lifetime safety cap (default 20 experiments).
Currently executable research domains are pointwise FM optimisation, pairwise
ranking and pairwise optimisation, leakage-safe author affinity, leakage-safe
user-history summaries, weekday features, and human-approved FM architecture
changes.

Architecture experiments require explicit review at invocation time:

```bash
.venv/bin/python -m research_agent.run_research --approve-architecture-experiments --artifact-dir runs_architecture
```

This records a manual intervention and exposes the legacy `deepfm` and
`nfm_residual` structures plus a bounded compositional language to the
generic implementer for assigned architecture hypotheses. The language combines one or two reviewed
embedding-MLP, bi-interaction-MLP, or cross-network residual paths using
bounded width, depth, dropout, and additive or learned-gate fusion. Arbitrary
model code and unbounded dimensions are rejected. The implementer authors the
categorical structure; the search controller retains training hyperparameter
selection. Each run writes
`architecture_spec.json`, checkpoint configuration, parameter count, and an
ordinary configuration patch. These structures remain FM hybrids, use only local training
data, and add no dependency. Without the approval flag, architecture capability
is omitted from the planner manifest.

`--cycles` is an alias for `--round-budget`: it controls how many complete
hypotheses are attempted in one approved research round. `--max-rounds`
pre-approves multiple rounds for one invocation, while
`--max-total-experiments` is the persistent lifetime safety cap. Reusing the
same `--artifact-dir` resumes evidence and state. At an ordinary round boundary,
the agent writes `research_checkpoint.json` and does not access test data.
Final test confirmation requires `--finalize`, unless the validation target has
already been reached.

For example, approve one ten-hypothesis round toward primary 0.70 while retaining
a 40-experiment lifetime ceiling:

```bash
.venv/bin/python -m research_agent.run_research \
  --round-budget 10 --max-rounds 1 --max-total-experiments 40 \
  --target-primary 0.70 --artifact-dir runs_llm
```

One experiment ID represents one hypothesis. Its `low`, `medium`, `full`,
`seed_1`, and `seed_2` executions are recorded in `stages.jsonl` and stored
under `runs/<experiment_id>/<stage_id>/`; only the completed hypothesis is
written to `iterations.jsonl`. Acceptance and parallel reconciliation use the
mean full-fidelity score across seeds 0, 1, and 2; a noisy single seed cannot
replace the incumbent. Any positive robust validation-primary gain replaces
the incumbent. The `0.002` threshold is used only for convergence: three
completed hypotheses without a gain greater than `0.002` trigger a research
direction restart.

Every new experiment clones the complete accepted-incumbent configuration and
then applies exactly one conceptual change. For example, after DeepFM is
accepted, learning-rate, regularization, objective, and feature experiments
remain DeepFM descendants instead of silently reverting to baseline FM.
Architecture approval is inherited only while that accepted structure remains
unchanged; selecting a different structure still requires explicit approval.

Multi-task learning is available as a controlled training-objective factor on
any accepted pointwise architecture. It keeps `long_view` as the sole ranking
and acceptance target while a training-only `is_click` head shares the model's
embeddings. The reviewed objective values use auxiliary-loss weights 0.05,
0.1, or 0.2. Auxiliary outcomes come through `data.py`, are never inference
features, and never alter `evaluate.py`.
When a two-path composed architecture is incumbent, the next architecture
portfolio is restricted to controlled one-path sibling ablations. Both siblings
retain the same frozen parent and all non-architecture settings, allowing the
evidence layer to identify the removed path without overstating causality.

A configuration is durably considered consumed only after it produces a finite
validation `primary` metric. Failed, interrupted, pre-run safety/semantic
rejected, and otherwise unmeasured attempts remain retryable. During concurrent
execution, an in-flight reservation prevents two workers from running the same
configuration; that reservation is released after an unmeasured failure and is
committed to duplicate history only after valid metrics are recorded.

The planner can now emit four first-class decisions: `RUN_EXPERIMENT`,
`RUN_DIAGNOSTIC`, `BUILD_CAPABILITY`, and `REQUEST_HUMAN_APPROVAL`. Exact
implementations proceed to training. Evidence-only diagnostics write
`diagnostics.jsonl` without touching test data. Missing safe implementations
and authority-bound proposals are written append-only to
`capability_backlog.jsonl`. The invocation continues while independent executable
work exists and yields without a persistent terminal stop only when no safe work
remains or human approval is required. The backlog and completed diagnostics are included in
the next planner context, so unsupported hypotheses are retained rather than
silently replaced by an unrelated executable experiment.

Approve only a specifically backlogged human-review request, then resume:

```bash
.venv/bin/python -m research_agent.run_research \
  --approve-capability-gap listwise_new_dependency \
  --artifact-dir runs_llm
```

The gap must currently have `pending_human_approval` status. Approval is scoped
to that ID, appended to the backlog, and recorded as a manual intervention; it
does not itself install dependencies or bypass implementation critics.

Each completed hypothesis also refreshes `evidence_memory.json`. This compact
planner-facing record includes grouped fidelity metrics, seed mean/standard
deviation, training curves, validation log loss, early-stopping cause, score
distribution, feature coverage, model size, failure category, runtime, CPU,
and peak memory. The online planner and generic implementer receive this
evidence with the next hypothesis request.

At each fidelity, every candidate also records validation metrics stratified by
training-derived user activity and categorical feature coverage. Every slice
includes GAUC, nDCG@5, primary, support, and positive rate; fixed boundaries and
training-vocabulary hashes make the evidence reproducible and leakage-auditable.
Recording slices at low fidelity prevents pruning from withholding the evidence
the planner needs to choose its next intervention.

The agent also maintains an append-only `research_tree.jsonl` and a rebuildable
`research_tree.json` snapshot. The tree gives each normalized hypothesis a
stable ID, links executed experiments to their accepted parent, retains
deferred and capability-blocked branches, records failed/rejected outcomes,
and reconstructs the incumbent ancestry. Before planning, the LLM receives a
bounded tree view containing the accepted lineage, near-incumbent rejected
branches, and unresolved hypotheses. Every new planner candidate must label
itself as `continue`, `refine`, `revisit`, or `branch_new` and cite the prior
hypothesis or experiment evidence it depends on.

The complementary `research_coverage.json` answers what the tree cannot: which
model mechanisms, objectives, and feature families are accepted, merely present
inside an accepted combination, tested, untested, pending implementation, or unavailable. It imports only finite,
metric-consistent, semantically approved evidence from non-smoke sibling run
directories, deduplicates scientific configurations, and uses multi-seed means
when stage evidence is available. This lets the planner see that, for example,
DeepFM and the reviewed single-path architectures have baseline evidence;
multi-task objective coverage is tracked separately from model architecture.

Fill only executable architecture coverage gaps without rerunning mechanisms
that already have validated evidence:

```bash
.venv/bin/python -m research_agent.run_research \
  --architecture-coverage --approve-architecture-experiments \
  --workers 2 --worker-threads 1 --cycles 10 \
  --artifact-dir runs_architecture_coverage
```

The LLM still authors the hypothesis and rationale, but the campaign validator
rejects refinements of already tested mechanisms until every executable
architecture has baseline evidence. The run ends when coverage is complete;
unavailable mechanisms are reported without approximation.

Every tree refresh also generates `research_tree.md`, a Mermaid flowchart with
colour-coded incumbent, accepted, rejected, failed, and deferred branches. Open
that file in GitHub or use **Markdown: Open Preview** in VS Code to view the
interactive rendered diagram; the chart is derived only from the JSON/JSONL
evidence and does not create research state of its own. To keep large runs
readable, the diagram shows the twelve most recent unresolved hypotheses and
links to the JSON snapshot for the complete archive.

Regenerate the diagram for an older artifact directory with:

```bash
.venv/bin/python -m research_agent.visualize_tree --artifact-dir runs_full_20_phase7
```

Every experiment is independently audited by a semantic critic. Before
training, it binds the hypothesis and claimed mechanism to a named executable
implementation and the actual configuration diff. After all fidelity and seed
stages, deterministic artifact checks and an independent LLM auditor verify
that the conceptual setting stayed fixed, the patch exists, and the measured
validation evidence belongs to that experiment. The complete
trace is stored in each iteration's `semantic_review` and summarized in planner
memory. A failed semantic audit cannot train at the pre-run gate or replace the
incumbent at the post-run gate. Offline runs retain all deterministic checks and
skip only the LLM judgment.

Planner prompts receive compact tree and coverage summaries. Each generic
implementer receives only its assigned hypothesis, strategy, executable
registry, two recent evidence records, and relevant authority/critic
constraints—not the complete slate or governance history. The LLM semantic
audit receives a compact best-stage correspondence trace and is capped at 1400
output tokens; full diagnostics remain available in local artifacts.

Critic decisions are active planner memory rather than terminal log messages.
Every pre-run and post-run audit appends a compact record to
`critic_feedback.jsonl`; `critic_memory.json` groups alignment repairs,
execution failures, valid negative results, and supported lineages. The next
planner and implementer calls receive those lessons as binding constraints. The
pre-run critic additionally verifies one-factor configuration diffs and online
hypothesis lineage references. The post-run critic independently recomputes
the official primary metric relationship and labels evidence as supporting,
valid-negative, execution failure, or semantic misalignment.

Before each planning batch, the orchestrator also receives compact governance
context: the benchmark contract, architecture-document digest, recorded human
interventions, and recent errors/recoveries. Finalization reconstructs the
architecture recorded in the selected checkpoint and writes `readiness.json`.
The final Markdown report exposes `readiness_passed` and lists missing evidence
instead of presenting an incomplete run as submission-ready.

For a deterministic run that does not use an API key:

```bash
.venv/bin/python -m research_agent.run_research --planner offline --cycles 2 --artifact-dir runs_offline
```

For the required failure/recovery demonstration, use explicit demo mode. The
first candidate intentionally crashes inside the isolated runner; the controller
logs the failure, restores the accepted baseline, and continues normal research:

```bash
.venv/bin/python -m research_agent.run_research --planner offline --demo-failure --cycles 2 --artifact-dir runs_demo
```

`--data_dir` 默认 `./KuaiRand-Pure/data`；数据放在别处时显式指定。

`--model` 可选 `fm`（官方 baseline）/ `pop`（trivial baseline）/ `random`（下界，用于自检评测代码）。
FM 全程约 40 秒（CPU，单核）。

## 任务定义（口径已写死，不要改）

| | |
|---|---|
| 任务 | **用户内排序** —— 每个用户只对其在评测集中的曝光排序，不做全库检索 |
| 相关性标签 | `long_view`（原生列，0/1） |
| 指标 | `GAUC`、`nDCG@5`；**主分 = 两者平均** |
| 数据划分 | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| 零正例用户 | nDCG 记 0.0 并计入平均；GAUC 只统计 `0 < 正例数 < 曝光数` 的用户，按正例数加权 |
| nDCG gain | `2^rel − 1`（二元标签下等价于 identity） |

实现见 `evaluate.py`，全部约定写在文件头注释里。

## Baseline 阶梯

test 集上的分数。**要打败的是 FM 这一行。**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random（下界，自检用） | 0.4996 | 0.4511 | 0.4753 |
| item popularity（trivial） | 0.6308 | 0.5121 | 0.5715 |
| **FM（官方 baseline）** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ 指标的真实区间：nDCG@5 的天花板是 0.729，不是 1.0

test 集 23,875 个用户里：

| | 占比 | 对指标的影响 |
|---|---|---|
| 全负用户（该用户所有曝光都不是 long_view） | **27.1%** | nDCG 恒为 **0**，任何模型都救不了；不计入 GAUC |
| 全正用户 | **9.2%** | nDCG 恒为 **1**；不计入 GAUC |
| 有区分度的用户 | **63.7%** | GAUC 的实际样本 |

所以用真实标签当预测分（oracle，完美排序）也只能拿到：

| | random | FM baseline | **oracle 上限** | FM 已吃掉的区间 |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**评估进展请以 oracle 为分母。** 看到 0.5946 就以为「离满分 1.0 还很远」是误判——
baseline 已经吃掉可用区间的三成，剩余 headroom 是 0.27 而不是 0.41。

FM 在 5 个随机种子上的 std 均为 **0.0008**。据此收敛判据取 **ε = 0.002（≈2.5σ）, N = 3**：
连续 3 轮迭代 validation 主分提升不超过 0.002 即判定收敛。

> 自检：如果你的评测代码跑 `--model random` 得不到 primary ≈ 0.475（±0.001），说明 harness 有问题，先修它。

## 提交格式

CSV，含表头，一行对应评测集的一行：

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| 字段 | 说明 |
|---|---|
| `row_id` | 0 起连续递增，对应 `data.load()[split]` 的行序（确定性：先读 `log_standard_4_08_to_4_21_pure.csv` 再读 `log_standard_4_22_to_5_08_pure.csv`，按 date 过滤后保持原文件顺序） |
| `user_id` / `video_id` | 冗余字段，仅用于校验对齐 |
| `score` | 你的模型给该行打的分，任意实数，只用相对大小；不允许 NaN / Inf |

> **为什么必须带 `row_id`：** `(user_id, video_id)` 在评测集里**不唯一** ——
> test 集有 3.06% 的重复对，最多重复 12 次。所以它不能作为主键。

生成与校验：

```bash
python3 submit.py --make  --split test  submission.csv    # 用官方 FM baseline 生成一份示例提交
python3 submit.py --check --split test  submission.csv    # 校验格式与对齐
python3 submit.py --score --split valid submission.csv    # 校验并打分（本地 valid 可用）
```

`--check` 会拒绝：表头错误、行数不符、`row_id` 跳号、`user_id`/`video_id` 与评测集不对齐、
`score` 非数字或为 NaN/Inf。**提交前请自行跑一遍 `--check`。**

## 从哪里开始改

下面的排序是**实测过的**，不是猜的。组委会已经试过的死路直接标出来，别重复踩。

### 已实测：这两条没有收益，不要浪费迭代

| 试过的 | 结果 |
|---|---|
| **加静态特征** —— 把 CWM 的 13 个特征域全接进来（+`music_id`/`video_type`/`upload_type` + 6 个用户侧粗桶） | primary **0.5940** vs 5 域的 **0.5950**，噪声内无差别，甚至略降 |
| **加模型容量** —— embedding 维度 k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887，几乎不动 |

原因：`user_id × video_id` 的交叉已经吃掉了大部分可学的信号。`follow_user_num_range` 这类粗桶
在 `user_id` 面前是冗余的；而 114 万行数据也撑不起更大的容量。**瓶颈不在特征和容量。**

⚠️ 另外注意：**纯用户侧特征的一阶项对分数贡献恒为 0。** 因为排序在用户内部做，任何在用户内为常数的项
都不改变组内顺序（实测：`item_pop × 用户偏置` 和纯 `item_pop` 的分数一位不差）。用户侧特征只能通过
**与物品侧的交叉项**起作用。

### 未探索：headroom 应该在这里

按我们判断的可能性排序（**这几条组委会没测过，是留给你们的**）：

1. **换损失函数。** 现在是 pointwise logloss，但指标（GAUC / nDCG）是**排序指标**。
   换成 pairwise（BPR）或 listwise（对该用户的曝光做 softmax）—— 目标函数和评测口径对齐，
   这是我们认为最可能有效的一条。
2. **用户历史序列。** 现有特征**完全没用到行为序列**。KuaiRand 每用户在 train 里有上百到上千条交互，
   DIN / SIM 那一类的兴趣建模是完全空白的方向。
3. **多目标。** 日志里还有 `is_click`、`is_like`、`is_follow`、`is_comment`、`is_forward`、`play_time_ms`，
   可以做多任务辅助 `long_view` 主任务。
4. **观看时长的建模。** [CWM](https://github.com/hyz20/CWM) 的贡献正是这条：它把观看时长做**删失回归**
   （视频播完时真实观看时长被截断，所以用单侧损失而非平方误差）。这是个有研究深度的方向。
5. **换模型。** DeepFM / DCN / xDeepFM。鉴于容量实测不是瓶颈，**优先级放在 1-4 之后**。
6. **时间特征与分布漂移。** `hourmin`、`date`，以及 train 与 test 之间的漂移。
7. **无偏验证（进阶）。** `log_random_4_22_to_5_08_pure.csv` 是随机曝光日志（118 万行），
   可作为额外的无偏验证集，检查模型是否只在有偏流量上过拟合。

## 用你自己的模型（包括 CWM）

`evaluate.py` 与模型完全解耦，它只要三个等长数组：

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores 可以来自任何模型
```

- `user_ids`：评测集每一行的 user_id
- `labels`：该行的 `long_view`（0/1）
- `scores`：你的模型给该行打的分（任意实数，只用相对大小）

所以你可以完全不用 `baseline.py`，换成 PyTorch、LightGBM 或 [CWM](https://github.com/hyz20/CWM) 的 xDeepFM，
只要最后把 `scores` 交给 `evaluate()` 即可。**评分口径由 `evaluate.py` 唯一决定。**

> 用 CWM 需注意：它依赖 `torch==1.6.0`（2020 年版本，新 GPU 上大概装不上），
> 且它的损失优化的是 counterfactual watch time、评测标签是自己重建的 `long_view2`。
> 它是一篇时长纠偏论文的研究代码，可以当**进阶参考**，不建议作为起步点。

## 文件

| | |
|---|---|
| `evaluate.py` | 指标实现 + 全部口径约定。**不要改。** |
| `data.py` | 数据加载、官方划分、特征编码。加特征改这里。 |
| `baseline.py` | 三个 baseline。FM 是要打败的那个。 |
| `baseline_scores.json` | 官方发布的分数 + 种子方差 + 收敛参数。 |
| `submit.py` | 生成 / 校验提交文件。 |
| `ablation_features.py` | 特征消融实验，可复现「加特征没有收益」那组数字。 |
