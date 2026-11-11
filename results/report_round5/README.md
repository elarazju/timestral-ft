# Round 5: observed inference execution

[中文验收](acceptance.zh-CN.md) · [Frozen protocol](protocol.json) ·
[Examples and reproduction](../../examples/report_round5/README.md) · [Independent audit](audit.json)

All 32 fixed 5M runtime cases completed with the requested mode actually active.
CPU BF16 was slower for every measured shape. CPU FP32 compilation helped only
C2048/V7 (1.13×); MPS FP32 compilation helped all four shapes (1.10–1.67×).
These are local measurements, not a claim about all CPUs, GPUs or workloads.

Hardware: Apple M4 Pro, 48 GiB unified memory, macOS; PyTorch 2.14.1, Python 3.12.13,
four CPU threads. Exact platform, dependencies, source/model/config identities and
per-case receipts are retained. CUDA was unavailable and is not hardware-validated.
No training was used in the performance/GIFT comparisons. Larger-model functional
acceptance separately performed one full/LoRA parameter update per case.

## Fixed performance matrix

B1, H96, default symmetric averaging/clipping. Table cells are medians of five
post-warm forecasts. Full samples, first/setup/load/warm costs, throughput, memory,
errors and actual execution receipts are in [all 32 case records](benchmark_cases/)
and the [independently recomputed matrix](benchmark.json).

| Device | C | V | FP32 ms | BF16 ms | Compile ms | BF16+compile ms |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 256 | 1 | 4.820 | 9.323 | 6.591 | 9.349 |
| CPU | 256 | 7 | 8.226 | 20.280 | 10.721 | 22.062 |
| CPU | 2048 | 1 | 7.806 | 19.644 | 17.286 | 37.180 |
| CPU | 2048 | 7 | 40.648 | 157.900 | 35.862 | 153.277 |
| MPS | 256 | 1 | 27.502 | 24.021 | 17.305 | 18.163 |
| MPS | 256 | 7 | 21.946 | 28.362 | 19.973 | 19.775 |
| MPS | 2048 | 1 | 28.007 | 21.137 | 16.748 | 22.209 |
| MPS | 2048 | 7 | 27.885 | 29.147 | 19.166 | 22.563 |

Each case starts a fresh process with an empty private Inductor directory; OS/file/
Metal system caches are not flushed. Input resides on the target device. Timed
forecast calls synchronize before/after execution and exclude final CPU/NumPy
result transfer, DataFrame construction and file IO. Model loading, runtime
construction, first forecast, two extra warmup forecasts and five stable calls are
separate. This is not full application response latency. Five sample ranges are
descriptive, not confidence intervals.

All 16 compiled cases captured one complete normalized-core graph and executed it
16 times (eight full forecasts with default symmetry). Fullgraph admits no graph
breaks; none occurred in these cases. The compiler was actually exercised on MPS;
no blanket claim that MPS compilation is unsupported was assumed. Shape changes
can trigger more graphs and are not covered by one fixed-shape speed number.

CPU C2048/V7 compilation: 40.648→35.862 ms, first forecast 19.553 s, about 4203 useful
calls to amortize startup plus two extra warmups. MPS C256/V1 compilation:
27.502→17.305 ms, first forecast 3.448 s, about 368 calls. Formula uses measured
runtime construction, first and warmup costs and excludes common model loading:
`N = max(1, 1 + ceil((A_candidate - A_fp32)/(t_fp32 - t_candidate)))` when the
candidate's stable latency is lower; otherwise no sustained-speed break-even is
reported. Keep the runtime alive to benefit. These extrapolations assume stable
shape/latency, not an observed multi-thousand-call service trial.

FP32 compile relative L2 residuals against same-device eager FP32 were at most
6.81e-7. BF16 variants ranged from 0.00251 to 0.00972 relative L2 on the fixed
synthetic probes. All outputs were finite/monotone; first and repeated outputs
were actually identical in these cases (predeclared tolerance rtol1e-4/atol1e-5).
This is numerical evidence, not a universal forecast-quality tolerance.

Peak RSS was 318.2–668.8 MiB over each process lifetime, including load/compiler
preparation. Sampled MPS driver peaks were 56.7–74.8 MiB. Driver/current memory was
sampled every 10 ms and can miss short peaks. Unified-memory RSS and MPS counts
must not be added; neither is a precise model-only allocation measurement.

## Fixed six-task GIFT quality control

Official 5M weights only, CPU FP32 versus predeclared CPU BF16 eager. BF16 is kept
as a precision control even when slower; no test-based selection, new training,
three-seed training claim or fourth-round checkpoint choice occurred. Each mode
uses all 2451 official test windows across the six tasks. Predictor uses up to
32768 history points, joint targets/past covariates, default postprocessing,
MAX_ROWS=128 and MAX_TOKENS=2**22 exactly as frozen. Complete data-file manifests
match the pinned GIFT/data revisions in the protocol.

| Task | Complete test windows | FP32 MASE | BF16 MASE | FP32 9q WQL | BF16 9q WQL | WQL change |
|---|---:|---:|---:|---:|---:|---:|
| m4_hourly/H/short | 414 | 0.779742 | 0.782821 | 0.02717430 | 0.02733694 | +0.599% |
| electricity/W/short | 1110 | 1.574676 | 1.575325 | 0.05596001 | 0.05604120 | +0.145% |
| solar/W/short | 137 | 1.345542 | 1.343933 | 0.17910470 | 0.17886136 | -0.136% |
| hospital/M/short | 767 | 0.742706 | 0.742814 | 0.04970510 | 0.04972126 | +0.033% |
| ett1/H/short | 20 | 0.806658 | 0.807207 | 0.17282288 | 0.17291255 | +0.052% |
| ett2/H/long | 3 | 1.006108 | 1.007499 | 0.10011827 | 0.10015418 | +0.036% |

Five tasks have a higher (worse) BF16 weighted loss; solar is slightly lower.
The absolute task scores use different units and cannot be averaged into a raw
MAE ranking. GIFT's nine-quantile CRPS approximation differs from the table API's
21-quantile WQL; the official predictor's `mean` forecast is its median.
This subset is not the complete GIFT leaderboard, a compiled-GIFT validation or
cross-model/domain generalization evidence.

[All 12 task receipts](gift/) retain all eleven official metrics. The
[quality audit](gift_audit.json) repeats those eleven metrics from the saved arrays
and independently computes pooled MAE, median MSE, ND and nine-quantile weighted
loss over every observed label. Full raw forecast arrays remain ignored locally.

## Larger-model and implementation acceptance

[12 current-code cases](acceptance/) cover 28M/123M/209M × CPU/MPS × full/LoRA at
B1/V1/C64/H32, actual forward/backward/update, FP32 and BF16 eager forecast, unchanged
original parameter names and official CPU export reload. All passed;
[the independent artifact audit](acceptance_audit.json) checks the retained arrays
and export files. Largest absolute official-export residual was 4.77e-6.
Large-model compilation, long-context speed, maximum capacity and forecast quality
were not measured. Small-shape success is not extrapolated to those regimes.

Default regression suite: 240 passed, 50 skipped (unavailable/opt-in hardware and
integration cases); ruff passed before source freeze. Runtime tests include
bitwise default parity, input preservation, actual BF16 attention linear output,
unchanged scaling, finite/ordered quantiles, real Dynamo tracing with mock codegen,
failure/fallback and mixed receipts, warmup/cache signatures, parameter references,
training-state handling, dtype/device guards and official state-dict export.
Actual Inductor cost/execution is tested by the real-weight matrix, not imposed on
the default unit suite. Runtime settings are separate from existing checkpoint
schemas; old artifacts need no new inference field.

## Evidence and reproduction

The immutable protocol freezes the complete runtime plus all four producer/audit
scripts, tests and examples. Each case requires exact equality with that source
set and carries the protocol hash; auditor checks actual weights/configs, data and
prediction file hashes, complete case sets, effective modes, throughput and timing
statistics. Raw evidence lives at `runs/report-round5` and includes a source snapshot
with its original protocol plus dependency listings. For later source changes:

```sh
.cache/gift-probe-env/bin/python runs/report-round5/source_snapshot/scripts/audit_report_round5.py \
  --raw /Users/mony/Documents/ts_model/runs/report-round5 \
  --results /Users/mony/Documents/ts_model/results/report_round5
```

The snapshot depends on the retained `.cache` link, pinned data/weights and Python
environment; it is not a self-contained portable bundle. Alternatively check out
this round's committed source, restore the raw evidence and run the audit command
in the example. Historical rounds 1–4 protocols and results were not altered.
The reader article and its figures are local-only under `blog/`, excluded from Git
and wheel/sdist; no external blog publication is part of this delivery.
