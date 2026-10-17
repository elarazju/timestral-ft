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
