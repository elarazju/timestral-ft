# Round 5: inference execution

`InferenceConfig` is separate from `TrainConfig`. Defaults preserve FP32 eager
forecasting. BF16 enables autocast only around the Transformer and quantile head;
patch projection, normalization, quantile reconstruction and refinement preserve
the original FP32/FP64 boundaries. Parameters remain FP32.

```sh
python examples/report_round5/inference.py
timestral-ft predict --model runs/my-model/model --history history.parquet \
  --horizon 96 --inference-config examples/report_round5/bf16.yaml --output forecast.csv
```

The CLI writes `forecast.csv.runtime.json`; tables expose
`predictions.attrs["inference_runtime"]`. `evaluate(..., inference_config=...)` and
`model.backtest(...)` include receipts in evaluation metadata. For the table API:

```python
from timestral_ft import FineTunedModel, InferenceConfig
model = FineTunedModel.from_pretrained("runs/my-model/model")
config = InferenceConfig(precision="bf16", compile=True, warmup_steps=2)
prediction = model.predict("history.parquet", 96, inference_config=config)
print(prediction.attrs["inference_runtime"])
```

Warmup executes the actual forecast `warmup_steps` extra times once per input
shape/dtype/device/options signature; subsequent calls reuse the instance and
compiled graphs. Different values of the same shape do not trigger warmup.
`model.clear_inference_cache()` discards these instances after structural changes.
Low-level runtimes require `model.eval()` and FP32 parameters; device/dtype changes
are rejected. Use one runtime per worker. No runtime is serialized into model
weights or training checkpoints; serialize `config.to_dict()` separately.

Compilation uses actual Inductor kernels for the complete normalized core
(`fullgraph=True`). `compiled_graphs` counts compiler results and
`compiled_executions` counts successful execution of those results. Failure is an
error by default. Explicit `fallback="fp32_eager"` records its reason and actual
mode; receipts include lifetime mode counts and mixed execution status. A changed
shape may compile another graph or fail; warmed signatures do not guarantee every
future shape is supported.

## Reproduction

Formal settings and immutable source/weight/data/environment identities live in
[`results/report_round5/protocol.json`](../../results/report_round5/protocol.json).
Raw arrays, per-process logs, compile caches and source snapshots are ignored under
`runs/report-round5`. Use fresh output paths when changing source. Do not run
benchmarks alongside training or tests.

```sh
python scripts/run_report_round5.py
python scripts/accept_report_round5.py
.cache/gift-probe-env/bin/python scripts/run_report_round5_gift.py
.cache/gift-probe-env/bin/python scripts/audit_report_round5.py
```

The 32 performance cases are 5M, CPU/MPS, B1, C256/2048, H96, V1/7,
FP32/BF16/compile/BF16+compile. Each has a fresh process and Inductor directory;
OS, file and Metal caches are not flushed. Load, runtime construction, first
forecast, two extra warmup forecasts and five stable forecasts are separate.
Inputs already reside on the target device; forecast timing excludes the final
CPU/numpy result transfer and table construction. First/repeat predictions use
predeclared rtol=1e-4, atol=1e-5 repeatability checks, with actual maximum residuals
reported. Mode-to-FP32 numerical residuals are reported without a pass threshold.
CPU RSS covers process lifetime; MPS memory is sampled every 10 ms and may miss
short peaks. Throughput is forecast target/time points per second, excluding the
quantile axis. The sample range is descriptive, not a confidence interval.

GIFT compares fixed official 5M CPU FP32 against CPU BF16 eager on six complete
fixed splits with max context 32768 and official notebook batching. The candidate
is a precision quality control fixed before test metrics; it remains included when
slower. There is no training or seed selection, no compiled GIFT evaluation, no
full-leaderboard or cross-model quality claim. Large-model checks cover only the
small acceptance shape B1/V1/C64/H32, full/LoRA gradients and official CPU reload.
