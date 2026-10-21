# Timestral-209M preflight

These are device checks and evaluation-protocol/throughput probes, **not the
completed 97-task adaptation benchmark**. Hardware is Apple M4 Pro, 48 GB unified
memory; FP32, PyTorch 2.14.1. Initial checks use synthetic/validation inputs.
Post-freeze execution probes, identified below, use fixed official test inputs
after all 97 adapters have been selected and frozen.

`hardware_profile.json` records actual forward/backward/optimizer timing after
loading the pinned real 209M checkpoint, on CPU and MPS. Timings include an
initial graph/kernel warmup. Batch is one; these are not complete optimizer-step
or full-benchmark runtime estimates.

`validation_parity.json` compares the new selection validator to the unmodified
pinned official Timestral predictor and GluonTS `evaluate_model`, on official
validation prefixes only. All four cases match CRPS/MASE at rtol 1e-5, atol 1e-6:
short monthly series, seven-variate H=720, past covariates, and M4 yearly data.
A fixed seed chooses up to 128 validation series; the two single-series tasks
include all target dimensions. No test iterator is accessed by these checks.

The standalone verification source and environment manifest accompany these
records. The complete adaptation experiment has separate output directories.

The synthetic large-input check exposed a resource issue before the complete
experiment: 8192 context, seven targets, 35 past covariates, H=900 retained about
59 GB in the Metal driver and took about 76 seconds per microbatch. This was
stopped as an unsuitable production budget; no test data was involved.
`large_shape_unbounded.json` preserves the actual completed timing records.

With the v2 shape-only context budget, that stress case caps context at 1536.
`large_shape_bounded.json` records three finite optimizer updates: about 1.9
seconds per warmed microbatch and 14.1 GB driver reservation. Driver reservation
includes caches, not just live tensors. These timings do not compare forecast
quality. Both standalone scripts are retained for inspection.

`report_smoke_check.json` describes a **synthetic artifact-contract test** of 97
fixtures, raw-length audit, official CSV schema validation, and ZIP packaging.
It is not an actual 97-task training result. `freeze_guard_check.json` verifies
that incomplete task selection refuses testing before reading a test iterator.

`full_context_inference.json` preserves two successful official-default inference
calls with synthetic 32768-step history, seven targets, 35 past covariates and
H=900. Both outputs are finite; retained driver allocations were about
11–12 GB. The training context budget does not shorten validation or test
inference. The accompanying source waits for an accelerator slot between task
jobs; this is a shape/resource check, not a GIFT test score.

`cpu_eval_profile.json` records full official paired-test execution probes for
hospital, hierarchical weekly sales, and car parts, after selection freezing.
All 11 base-model metrics match the pinned public reference at the production
tolerance. Their recorded adapter metrics are correctness/runtime evidence;
they are not used for checkpoint or recipe selection. No complete adaptation
benefit is inferred from these three cases.

[mps_concurrency.md](mps_concurrency.md) and `mps_concurrency.json` preserve the
equal-work one/two-MPS-process probes. The completed ABBA comparisons showed
only about 0.25–2.35% throughput changes and approximately doubled per-worker
Metal driver allocations. A longer electricity repeat did not reproduce the
exploratory short-trial advantage. The supplemental 35-covariate probe was
stopped on the user's decision after three trials and is clearly marked
incomplete. Production testing resumed with the existing one-MPS/one-CPU plan;
no adapters, hyperparameters, test windows, or completed metrics were changed.

[mps_batch_profile.md](mps_batch_profile.md) and `mps_batch_profile.json` record
an equal-work batch-2/4/8/16 inference sweep on 32 fixed real Bitbrains windows.
All sizes run, but larger batches show no throughput gain and fail the strict
cross-batch prediction-parity tolerance. Repeats at each individual size are
bitwise stable. Production keeps its batch-2 budget; both suspended workers
resumed without discarding their active evaluation progress. The exact source
snapshot and difference diagnostics accompany the measurements.
