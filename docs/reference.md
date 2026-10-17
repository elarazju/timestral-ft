# timestral-ft: technical reference

[Project overview](../README.md) · [简体中文](../README_zh.md)

Community full-parameter and LoRA fine-tuning for [Timestral](https://github.com/mistralai/timestral).
Independent of Mistral AI. Python API and CLI, chronological backtesting, numeric and categorical covariates,
and export to the official inference format. See [the Chinese guide](guide.zh-CN.md).

Version 0.1 supports single-device FP32 training on CPU, CUDA, and Apple MPS. CPU and MPS are
hardware-validated locally; CUDA follows the same test suite but requires a NVIDIA GPU to validate.
Distributed training and pretraining are outside v0.1.

## Installation

Requires Python 3.10+ and Git. Python 3.12 is the reference environment.

```sh
uv venv --python 3.12
uv pip install -e '.[dev]'
source .venv/bin/activate
```

For the locked reference dependencies, use `uv sync --locked --extra dev --python 3.12` instead.

The upstream dependency is pinned to commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`.
This repository does not distribute pretrained weights. The first model load downloads weights from
Hugging Face; later runs can use the cache or a local checkpoint directory.

## Quick start

Prepare the pinned ETTh1 dataset and timestamp-derived features:

```sh
python scripts/run_etth1.py --prepare-only
timestral-ft train --config examples/etth1.yaml
timestral-ft train --config examples/etth1.yaml --mode lora --output runs/etth1/lora
timestral-ft evaluate --model runs/etth1/full/model --data data/etth1/test.parquet \
  --history data/etth1/history.parquet --output runs/full-evaluation
```

Or reproduce all three comparisons, including the original model:

```sh
python scripts/run_etth1.py
```

Results and exact settings are in [results/etth1_cpu](../results/etth1_cpu).
The default run trains up to 200 optimizer steps per mode, using one seed. It is an adaptation
demonstration, not a claim that ETTh1 is absent from pretraining or that fine-tuning always helps.

## Python API

```python
from timestral_ft import DataConfig, TrainConfig, finetune, evaluate, FineTunedModel

model = finetune(
    "train.parquet", "validation.parquet",
    model_id="mistralai/Timestral-5M",
    data_config=DataConfig(
        time_column="timestamp", id_column="series_id", target_columns=["demand"],
        past_covariates=["sensor"], known_covariates=["price"],
        categorical_known_covariates=["weekday"], frequency="h",
    ),
    train_config=TrainConfig(mode="lora"),
    output_dir="runs/my-model",
)
result = evaluate(model, "test.parquet", history="train-and-validation.parquet")
print(result.metrics)
loaded = FineTunedModel.from_pretrained("runs/my-model/model")
predictions = loaded.predict("history.parquet", 96, future_covariates="future.parquet")
```

Every table has one row per series and timestamp. Multiple target columns are forecast jointly;
separate `series_id` groups never attend to each other. Omit `id_column` for a single series.
Paths and pandas DataFrames are accepted. CSV and Parquet are supported.

## CPU, CUDA, and Apple MPS

Set `TrainConfig(device="cuda")`, `device="cuda:1"`, or `device="mps"`; the default is `cpu`.
The CLI can override the YAML device:

```sh
timestral-ft train --config examples/etth1.yaml --device mps --output runs/my-mps-run
timestral-ft train --config examples/etth1.yaml --device cuda --output runs/my-cuda-run
timestral-ft evaluate --model runs/my-mps-run/model --device mps --data data/etth1/test.parquet \
  --history data/etth1/history.parquet --output runs/my-mps-evaluation
```

For loading/forecasting, use `FineTunedModel.from_pretrained(path, device="mps")` (or `cuda`).
Unavailable requested devices raise an error; the whole model is never silently moved to CPU.
On MPS the neural network runs in FP32 on Apple GPU, while scaling, categorical target encoding,
and advancing-frame float64 statistics run on CPU. Device copies preserve the refinement gradient.
This keeps the upstream equations without casting double-precision statistics to float32; the
CPU/GPU synchronization adds overhead, so MPS can be slower than CPU for small models/windows.
CUDA uses its native float64 statistics. Category codes stay on CPU before encoding on all devices.
Checkpoints save accelerator RNG state, including MPS dropout, for same-device resume.
Accelerator numerical parity is tested with `rtol=1e-3, atol=1e-4`; bitwise equivalence across devices
is not expected. Exported weights retain the unchanged official CPU/CUDA format. For MPS inference,
use this toolkit's loader, since the upstream inference implementation still uses unsupported MPS float64.

The complete ETTh1 experiment also accepts `python scripts/run_etth1.py --device mps` or `--device cuda`.
An abbreviated 4-step MPS hardware check is recorded in [results/etth1_mps_smoke](../results/etth1_mps_smoke);
it checks the pipeline and is not comparable to the 200-step CPU adaptation experiment.

## Data and leakage rules

- Split chronologically **before** training; pass separate training and validation tables.
  Labels stay inside their split. Validation/test may read earlier history supplied explicitly.
- Each column has one role: target, past numeric, known-future numeric, past categorical,
  or known-future categorical. Known-future fields must genuinely be available at forecast time.
- Duplicate timestamps within a series are errors. Irregular data requires an explicit `frequency`;
  missing grid points become NaN, without interpolating targets. Inf and nonnumeric numeric fields are errors.
- Category vocabularies are fit on training data only and saved with the model. Missing remains NaN;
  unseen categories map to reserved code 0. Single-target categories use upstream's window-local target
  encoding; multivariate targets use the upstream raw-code behavior. Future target values never enter encoding.
- For prediction, `future_covariates` must cover the next horizon timestamps for each series and include
  the configured known-future columns. Missing covariate values retain their masks. Additional future target
  fields are ignored.
- Each series needs enough context and observed labels to form an eligible window. Empty-label batches are
  skipped; non-finite loss/gradients stop training rather than silently corrupting weights.

## Training and resume

The loss is the masked pinball loss over all 21 quantiles, divided by each target history's population
standard deviation. Constant histories use scale 1; targets with no history are excluded. This is our
fine-tuning objective, not a reproduction of Mistral's unpublished training recipe.

Training bypasses `forecast()`'s inference mode and preserves the upstream advancing refine frame.
It does not use test-time symmetry averaging, nonnegative clipping, or extreme-output caps in the loss.
Backtests use identical official default postprocessing for the base and fine-tuned models.

LoRA adapts every attention `wQKV` and `wO`, with rank 8, alpha 16, dropout 0 and zero initial delta.
All other parameters are frozen. Full fine-tuning updates the complete network. Defaults: AdamW,
learning rate 1e-5 (full) / 1e-4 (LoRA), batch 4, accumulation 4, clipping 1, 20 warmup steps and cosine decay.
Validation runs every 50 steps and selects the best checkpoint; three non-improving evaluations stop training.

```sh
timestral-ft train --config examples/etth1.yaml --stop-after-steps 75
timestral-ft train --config examples/etth1.yaml --resume runs/etth1/full/checkpoint.pt
```

`max_steps` defines the original schedule. A pause does not change it or add an off-schedule evaluation.
Resume requires the same data, configuration and base revision, restores optimizer/scheduler/RNG/sampler state,
and continues from the **latest** state. Final exports contain the **best validation** state.
Run directories must be empty for a new run. Only load checkpoints you trust.

## Export and forecasting

Every completed run writes `model/config.json` and `model/model.safetensors`, compatible with:

```python
from timestral import TimestralForecaster
forecaster = TimestralForecaster.from_pretrained("runs/my-model/model")
```

The official interface accepts arrays, not raw tables. Use `FineTunedModel.predict()` to automatically
apply saved column settings and category mappings. Carry the preprocessing files with the model.

```sh
timestral-ft export --model runs/etth1/lora/model --output runs/portable-adapter --adapter-only
timestral-ft export --model runs/portable-adapter --output runs/merged-model
timestral-ft predict --model runs/merged-model --history history.parquet \
  --future future.parquet --horizon 96 --output predictions.csv
```

Adapter-only artifacts require the original base weights and revision. Merged exports are standalone.
Exports preserve upstream weight names. Input schema and training metadata are separate JSON files.
v0.1 trains horizons up to one decode block (1024 steps on released models); official inference handles longer
horizons by feeding predicted medians back as context. Context cannot exceed the selected model's trained limit.

## Evaluation and verification

Backtests save CSV/Parquet predictions, metrics and metadata. MAE, MSE, RMSE and 80% interval coverage pool observed
target/time labels. Point errors use the **input target units** (original units for raw tables); the evaluator
does not apply the train-standardization used by default in
Time-Series-Library. The ETTh1 example predicts OT only with context 256 and stride 96; it must not be
compared directly to seven-target M benchmarks with context 96 and stride 1. The independent
[protocol verification](../results/etth1_protocol_check/verification.json) records both converted OT scores
and a separate zero-shot S/M numeric control. Reproduce it with
`python scripts/verify_etth1_metrics.py --run-standard-control`.
WQL is `2 * sum(pinball) / (21 * sum(abs(actual)))`; zero denominator is reported as null.
Missing labels are excluded. Timing includes the rolling backtest and table construction.

```sh
pytest -q
ruff check src tests scripts
python -m build
TIMESTRAL_INTEGRATION=1 pytest -q -m integration
```

Default tests use a tiny random model and do not download weights. They cover official prediction parity,
finite gradients, full/LoRA updates, isolation between series, leakage, category handling, checkpoint resume,
and export compatibility. The opt-in integration test loads pinned real 5M weights.
Accelerator tests automatically run on available hardware and skip unavailable devices. Require a
particular backend with `TIMESTRAL_TEST_DEVICE=cuda pytest -q tests/test_devices.py` (or `mps`);
these required-device jobs fail instead of skipping if that hardware is missing.
All four sizes (5M/28M/123M/209M) have real-weight CPU/MPS functional acceptance, including four-step
full/LoRA training, resume, adapter/merged reload, official export parity, and seven-target full-parameter backward checks
at L256/H96, L1024/H192 and L256/H1024 (plus LoRA backward at L256/H96). These short runs verify support, not large-model forecast quality
or maximum context/batch capacity. CUDA still needs hardware validation.

The [broader validation](../results/validation/README.zh-CN.md) records three datasets, two horizons and three
seeds, with the standard multivariate test protocol and official DLinear-network reference runs:

```sh
python scripts/benchmark_suite.py --jobs 2
python scripts/transfer_validation.py
python scripts/validate_models.py
python scripts/validate_model_shapes.py
python scripts/audit_validation.py
python scripts/summarize_validation.py
TIMESTRAL_INTEGRATION=1 TIMESTRAL_LARGE_MODELS=1 pytest -q
```

The matrix uses all seven ETT/eight exchange targets, L96, H96/192, train-only standardization and test
stride 1. It retains every seed and regression. DLinear uses the unchanged pinned official network in
an explicit local training loop; different pretraining, optimization objectives, training budgets and
validation aggregation preclude an equal-compute or exact paper-reproduction claim. Dataset sources,
checksums, scaler statistics, run-level scores and audit evidence are saved in `results/validation`.
Median forecasts/labels/origins and training checkpoints are kept under ignored `runs/validation`.
Frozen ETTh1-to-ETTh2/exchange transfer is reported separately: target training-only scalers and array
layouts are available, but no target-dataset labels update network weights. Within-dataset adaptation
and frozen transfer gains must not be conflated.

Unmodified CPU forecast parity keeps `rtol=1e-4, atol=1e-5`. For real-data LoRA merge/export checks only,
algebraic FP32 reordering is compared after dividing by history std (`rtol=1e-4, atol=1e-4`); reports
retain raw absolute and relative L2 residuals. This avoids zero-crossing quantiles in physical units
making a tiny relative numerical difference look like an architectural mismatch.

## GIFT-Eval integration

The optional [GIFT-Eval integration](../examples/gift_eval/README.md) trains independent Timestral-5M
LoRA adapters with the official task cutoffs, variable histories, short-series label masks and
adapter-only resumable checkpoints. It evaluates with the pinned official Predictor and all eleven
metrics, preserving joint targets, past covariates, full inference history and default postprocessing.
Its nine-quantile CRPS approximation differs from the table API's 21-quantile WQL.
Use a separate pinned GIFT environment; the main development environment does not need its legacy
NumPy/pandas dependencies. See [the complete run](../results/gift_eval_5m_lora/README.zh-CN.md)
for task-level results, hardware assignments, numerical audits and the local submission draft.

## License and provenance

Our code is Apache-2.0. The upstream code is separately Apache-2.0; model weights and derivatives remain
subject to their own Modified MIT terms, including applicable commercial restrictions. See the base model's
actual license before distributing a fine-tuned or merged model. Fine-tuning does not relicense weights.
Datasets are downloaded at runtime and are not included in this repository.
