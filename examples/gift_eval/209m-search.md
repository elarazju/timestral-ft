# Timestral-209M: validation search before the 97-task test

This is a new adaptation experiment. The complete 5M results are unchanged.
The objective is to improve both CRPS and MASE over the original 209M checkpoint;
validation gains do not guarantee test gains or a higher leaderboard rank.

## Fixed data and model

- Base: `mistralai/Timestral-209M`, revision
  `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`.
- Official GIFT source and dataset use the same pinned revisions and verified
  file checksums as the published 5M experiment.
- Each task starts independently from the base, with seed 42 and rank 8,
  alpha 16, dropout 0 LoRA on the attention projections: 1,105,920 parameters.
- Only `Dataset.training_dataset` supplies sampled training windows. Validation
  reads its own official H-step suffix. The search never calls `test_data`.
- Validation uses at most 128 seeded series selected before model evaluation;
  every target in these series and their official H-step suffix is retained.
  The final test includes all official series, targets, and rolling windows.

## Context, budget, and objective

1. Screen training context caps 256, 2048, and 8192 for 200 optimizer steps each.
   Cap the candidates at the maximum training-prefix length rounded to a
   32-step patch; deduplicate the resulting caps for short datasets.
2. Use AdamW, weight decay .01, gradient clipping 1, 40-step warmup, and a cosine
   schedule with maximum 800 steps. A screen is a prefix of this schedule,
   **not** a separately scheduled 200-step run.
3. The initial learning rate is 1e-4. If no screened candidate improves the
   validation selection score by at least 1%, screen 3e-4 at the most promising
   context. This rule is applied to every task, without test feedback.
4. Extend the screening winner to at most 800 steps. Validate every 100 steps;
   stop after six consecutive evaluations without an improvement. Freeze the
   screening decision before continuation so an interrupted run cannot change it.
5. Each step uses four sampled examples: batch 2 with accumulation 2, or batch 1
   with accumulation 4 when context times total input rows exceeds 131,072.
   The latter predeclared rule controls activation memory for many covariates.
6. The training loss remains history-scaled masked pinball over 21 quantiles.
   At each validation point, record that original objective alongside official
   nine-quantile CRPS/WQL and MASE. Selection uses the latter two metrics.

Both validation and test forecasts use the official default mirror averaging,
nonnegative/cap postprocessing, and up to 32,768 historical timestamps. MASE
uses the full input prefix for its seasonal-error denominator. The new validator
uses the official GluonTS metric classes and has been checked against the
unmodified official predictor/evaluation workflow.

Let rC and rM be validation CRPS and MASE divided by the corresponding base
validation metric. The score is sqrt(rC*rM) if both ratios are at most one;
otherwise it is max(rC,rM). The base model at step zero is an eligible candidate
with score one. All tasks actually train, but a task can retain an explicit
identity adapter if no checkpoint improves this criterion. This is reported
separately; it does not imply that all 97 deployed adapters changed the base.

Screening ties use the best accepted score, then the best observed unconstrained
geometric ratio, then longer context. The selected screening candidate continues;
the final checkpoint is selected over all candidates using validation only.

`validation_ablations.csv` compares the chosen steps under pinball, CRPS, MASE,
and Pareto selection at the observed 200/800-step budget caps, per candidate.
Only the screening winner is extended. These are checkpoint-level diagnostics
of observed trajectories, not a complete factorial comparison or independent
training runs for each objective. No selection objective is chosen from test
results.

## Running and resuming

Use the isolated GIFT environment described in [README.md](README.md), with the
source repository in `.cache/upstream` and official GIFT repository/data in their
existing pinned cache locations. The published 5M training receipts supply only
training-prefix shape metadata for device scheduling. No 5M test scores are used.

```sh
.cache/gift-probe-env/bin/python scripts/run_gift209_search.py --stage all
```

Default directories are `runs/gift-eval-209m-search-v1` and
`results/gift_eval_209m_lora`. Protocols and source snapshots are immutable;
changed settings require a new output directory. The training environment uses
MPS for most tasks and CPU for short datasets; CUDA is available by explicit
selection but has not been verified on this machine.

Optional disjoint CPU/MPS workers can run the search concurrently:

```sh
.cache/gift-probe-env/bin/python scripts/run_gift209_search.py --stage search --worker-device cpu
.cache/gift-probe-env/bin/python scripts/run_gift209_search.py --stage search --worker-device mps
# After both workers finish:
.cache/gift-probe-env/bin/python scripts/run_gift209_search.py --stage test
.cache/gift-probe-env/bin/python scripts/run_gift209_search.py --stage report
```

The two worker commands are intended to run in separate terminals. Per-task
locks prevent duplicate claims. The optimizer, scheduler, latest/best adapter,
random state, and screening decision all survive resumption.

## Freeze, audit, and publication

Testing is refused until all 97 task selections exist. Adapter hashes and
selection receipts are frozen in `frozen_selection.json` before the first test
iterator is accessed. A test run cannot initiate a new search. Test results do
not trigger automatic retraining or a new recipe.

The final stage checks raw Arrow lengths against official split cutoffs, data
file checksums, selected adapter/checkpoint equality, finite values, the base
revision/hash, and exact step-zero fallback state. Reproduced base metrics must
match all 11 official metrics within rtol 1e-3, atol 1e-5. The result includes all
regressions, normalized aggregate scores, and the official-format 97-row CSV.
A local adapter ZIP includes licenses and selection receipts, without base
weights, raw data, or optimizer states.

The submission draft is `fine-tuned`. Public model/code availability fields stay
unset until the corresponding artifacts are actually available. The base model's
no-test-leakage declaration is inherited, not independently proven; our own
adaptation and checkpoint selection use training/validation only. The script
never uploads weights, opens a leaderboard PR, or claims an official rank.
