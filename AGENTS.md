# TikTok TechJam 2026 — Project Guidance

## Architecture guidance

- Before designing, implementing, or modifying the autonomous research agent, read `docs/agent-architecture.md`.
- Treat that document as phased architecture guidance. This root `AGENTS.md`, the Starter Kit README, and `evaluate.py` remain authoritative if guidance conflicts.
- The mandatory action, per-iteration, error/recovery, code-diff, metrics, and manual-intervention logging requirements in the architecture document apply to every research run.

## Project goal

Build an Autonomous Machine Learning Research Agent for the KuaiRand-Pure recommender-system challenge. The agent must reproduce the official Factorization Machine (FM), propose and run controlled experiments, evaluate candidates, recover from failures, keep evidence, and produce a valid final prediction file.

Final goal: 
1. How well the recommendation model ranks videos
2. How well the autonomous agent conducts the research process

## Team and deadline

- Submission deadline: 1 September 2026 at 12:00 PM. Assume Asia/Singapore until the organizer confirms the official timezone.
- The team has limited ML experience. Explain unfamiliar ML concepts plainly and prefer small, testable changes.
- Prioritize a complete, reproducible agent loop and submission over bonus datasets, elaborate UI, or advanced models.

## Reference sources

- KuaiRand official dataset and research site: https://kuairand.com/
- Local starter-kit README and `evaluate.py`: benchmark task, permitted data, metrics, and submission contract.

## Fixed benchmark contract

- Required dataset: KuaiRand-Pure. Bonus datasets are out of scope until the required benchmark is complete.
- Task: rank each user's logged video exposures; do not implement full-catalogue retrieval.
- Target label: `long_view` (`1` positive, `0` negative).
- Metrics: GAUC and nDCG@5; `primary = (GAUC + nDCG@5) / 2`.
- Treat `evaluate.py` and the starter-kit README as the scoring source of truth. Do not modify `evaluate.py`.
- Hard rule: train only on the permitted KuaiRand datasets. Do not add, join, augment with, or pre-train on external datasets, and do not use pretrained weights trained on this benchmark's test labels.
- Use training data to fit models and validation data to select experiments. Never generate features from future validation/test labels or tune against a final hidden test.
- Required submission schema: `row_id,user_id,video_id,score`; validate it with `submit.py --check`.

## Verified baseline

The official FM was reproduced on 27 August 2026 with seed 0:

- Rows: train 1,141,112; validation 124,909; test 170,588.
- Features: `user_id`, `video_id`, `author_id`, `tab`, `dur_bucket`.
- Best validation: GAUC 0.6671, nDCG@5 0.5358, primary 0.6015.
- Reference test run: GAUC 0.6621, nDCG@5 0.5286, primary 0.5953.
- Baseline reproduction is complete. The autonomous research agent has not been implemented yet.

## Agent behavior

For each iteration, the agent should:

1. Read the benchmark contract, current best result, and prior experiment history.
2. State one clear hypothesis and choose one controlled change.
3. Create an isolated candidate without overwriting the last accepted version.
4. Train the candidate and evaluate it with the unchanged official evaluator.
5. Check for invalid values, data leakage, crashes, and unreasonable resource use.
6. Accept or reject the complete candidate based on validation evidence.
7. Log the hypothesis, rationale, code diff, configuration, metrics, decision, errors, recovery, runtime, token usage, and human interventions.
8. Restore the last accepted candidate after rejection or unrecoverable failure.
9. Stop after three consecutive iterations without a primary-score improvement greater than 0.002, or when the configured budget is reached.

## Initial experiment scope

- Start with a small approved catalogue: learning-rate or regularization tuning, pairwise ranking loss, leakage-safe author affinity, leakage-safe user-history summaries, time features, and multi-task auxiliary labels.
- Do not prioritize larger FM embeddings or simply adding all static features; the organizer's ablations found no meaningful gain.
- Change one main factor per experiment so results remain interpretable.
- Preserve the official FM as an immutable reference implementation.
- Require human review before adding a new dependency, a substantially different model family, or an experiment outside the approved catalogue.

## Project evidence and interface

- The terminal and structured run artifacts are the source of truth. A dashboard may visualize them but must not fabricate state.
- Prefer JSONL for detailed iteration records, CSV for metric summaries, patch files for code diffs, and explicit model checkpoints.
- Make accepted/rejected decisions, recovery events, score history, resource use, and manual interventions visible.
- The final demo should show at least one real experiment and one failure/recovery case.

## Engineering rules

- Keep secrets in environment variables and a local `.env`; never commit API keys, endpoint credentials, or tokens.
- Do not commit the KuaiRand dataset, archives, checkpoints, generated submissions, or bulky run artifacts.
- Add automated tests for evaluator integration, experiment decisions, logging, recovery, and submission validation.
- Keep setup reproducible and document exact commands in the README.
- Preserve working behavior before refactoring. Prefer the smallest change that tests the current hypothesis.

## Required deliverables to protect time for

- Devpost project description.
- Public code repository with setup and reproduction instructions.
- Per-iteration research logs and manual-intervention summary.
- Best model output/checkpoint and valid prediction CSV.
- Validation results and delta from the official baseline.
- LLM token use and GPU/compute usage.
- Limitations, team contributions, demo material, and presentation rehearsal.
