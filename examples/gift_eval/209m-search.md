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
   Cap candidates at the smaller of the rounded maximum training-prefix
   length and a per-example budget of 65,536 context timestamps times input
   rows (targets plus past covariates), rounded down to 32-step patches.
   Deduplicate the resulting caps. Seven-row inputs retain 8192; 21-row inputs
   cap at 3104; 42-row inputs cap at 1536. No input rows are dropped.
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
   with accumulation 4 when context times total input rows exceeds 32,768.
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

Default directories are `runs/gift-eval-209m-search-v2` and
`results/gift_eval_209m_lora_v2`. Protocols and source snapshots are immutable;
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


## Resource revision before the full run

The preliminary v1 screens exposed an impractical large-covariate configuration
on this 48 GB machine: 8192 historical timestamps, seven targets, 35 covariates,
and H=900 required about 76 seconds for one synthetic microbatch and retained
about 59 GB in the Metal driver. This number includes cached driver allocations,
not just live model tensors. No v1 test results were computed.

Version 2 introduces the row/context budget above. The same stress shape with
history capped at 1536 completed three finite optimizer updates in about
1.9 seconds per warmed microbatch, retaining about 14.1 GB in the driver. These
synthetic timing checks say nothing about forecasting quality. Complete v2
training starts in independent directories; v1 preflight artifacts remain
separate. Validation/test inference still uses up to 32,768 timestamps.

## Optional local rank simulation

After the audited full result exists, `scripts/rank_gift209.py` inserts it into a
pinned public GIFT leaderboard snapshot and reproduces the official mean-task-rank
calculation. This is separate from normalized geometric-mean score aggregation.
A snapshot may be a checkout of the public Hugging Face Space repository:

```sh
# Save a public snapshot once, and keep its commit fixed for the comparison.
git clone https://huggingface.co/spaces/Salesforce/GIFT-Eval .cache/gift-leaderboard-current
.cache/gift-probe-env/bin/python scripts/rank_gift209.py
```

Skip the clone when the snapshot directory already exists. The script records
its revision and each input CSV checksum. `--wait` waits for the experiment
supervisor's successful completion before ranking. The resulting
`ranking_simulation.json` is a local insertion/recalculation, not an official
leaderboard place or submission. The adapter archive is unchanged by ranking;
the separate README/ranking JSON contain the added simulation.

The postprocessor also writes `training_diagnostics.json`: total optimizer steps,
parallel-summed training-loop time, supervised wall time, step-zero fallbacks,
and the distribution of selected context caps/learning rates. Its validation
diagnostics compare observed 200/800-step checkpoints within the **same
screening-selected continuation context** for each task. This conditional view
avoids counting all short screens as if each had been extended; it is not an
independently optimized objective comparison or a test-set gain claim.

Final testing uses a separate inference batching budget: up to 32 rows and
1,048,576 row/timestamp tokens per batch, justified by the successful full-history
42-row inference stress check. The official predictor keeps a whole multivariate
item even if it exceeds a soft batch budget. These settings never drop input
rows, shorten history, or alter the frozen training/validation selection. They
are recorded in `test_execution.json`, whose hash accompanies every test record.
A resumed test cannot silently change those settings. Test/report runner repairs
have separate source manifests; the original training source snapshot and
selected weight hashes stay fixed.
