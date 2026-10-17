# Timestral-209M preflight

These are device and validation-protocol checks, **not test-set adaptation
results**. Hardware is Apple M4 Pro, 48 GB unified memory; FP32, PyTorch 2.14.1.

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
