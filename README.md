# timestral-ft

**Fine-tune Timestral on your own time series.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)](pyproject.toml)
[![License](https://img.shields.io/badge/Code%20License-Apache--2.0-green)](LICENSE)

**English** | [简体中文](README_zh.md)

`timestral-ft` is a community toolkit for full-parameter and LoRA fine-tuning of
[Mistral AI's Timestral](https://github.com/mistralai/timestral). It connects tabular
training data, validation, rolling backtests and weight export through a Python API
and four CLI commands: `train`, `evaluate`, `export` and `predict`.

We reuse the official network and provide a separate training layer. Experiments,
including regressions, are recorded alongside the code. This project is independent
of Mistral AI.

[Features](#features) · [Getting started](#getting-started) · [Experiments](#experiments) · [Documentation](#documentation)

## Features

- **Full and LoRA fine-tuning.** Update the whole network or adapt time/variate
  attention projections. Default LoRA: rank 8, alpha 16, zero initial delta.
- **Your data, your schema.** DataFrames, CSV and Parquet; multiple series and joint
  targets; numeric and categorical covariates with past-only or known-future roles.
- **A complete training loop.** Gradient accumulation and clipping, warmup/cosine
  scheduling, validation selection, early stopping and resumable checkpoints.
- **Forecasts and backtests.** Median and all 21 forecast quantiles, rolling-window
  predictions, MAE/MSE/RMSE, WQL and 80% interval coverage.
- **Portable artifacts.** Save preprocessing metadata, export a standalone LoRA
  adapter, or merge weights into the official Timestral inference format.
- **Reproducible evaluation.** Pinned ETTh1 examples, multivariate multi-seed
  benchmarks, frozen-transfer checks and complete 97-task GIFT-Eval LoRA runs
  with Timestral-5M and Timestral-209M.

## Supported models and devices

| Base model | CPU full / LoRA | Apple MPS full / LoRA | Forecast-quality experiments |
|---|---|---|---|
| Timestral-5M | Verified | Verified | ETTh1, multivariate benchmarks, GIFT-Eval |
| Timestral-28M | Verified | Verified | Not yet measured |
| Timestral-123M | Verified | Verified | Not yet measured |
| Timestral-209M | Verified | Verified | GIFT-Eval, all 97 tasks |

“Verified” means real-weight functional checks: short training runs, finite
backpropagation, resume, adapter/merged reload and official-format export.
These functional checks do not establish forecast quality or maximum batch/context capacity.
See the [model acceptance records](results/validation/models).

v0.1 uses **single-device FP32**. Select `cpu`, `mps`, `cuda` or `cuda:1` explicitly;
CPU is the default. CUDA is implemented but has **not been hardware-validated**.
On MPS, the network runs on Apple GPU and float64 statistics run on CPU; small
workloads may be faster on CPU. MPS inference uses this toolkit's loader.
Training horizons are limited to 1,024 steps. Distributed training and pretraining
are outside the current scope.

## Getting started

### 1. Install from source

Clone the repository with Git. Python 3.12 is the reference environment;
package metadata requires Python 3.10+.

```sh
git clone https://github.com/elarazju/timestral-ft.git
cd timestral-ft
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For NVIDIA GPUs, install the appropriate PyTorch build using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) first.
For the locked development environment, use `uv sync --locked --extra dev --python 3.12`.

The official code dependency is pinned to
`a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`. The first model load downloads
weights from Hugging Face; subsequent runs can use the cache or a local checkpoint.
Datasets and pretrained weights are not bundled.

### 2. Prepare data and fine-tune

The ETTh1 preparation script verifies the source checksum, splits data by time and
builds timestamp-derived features. The example predicts `OT`, with historical
`HUFL` and known-future calendar covariates.

```sh
python scripts/run_etth1.py --prepare-only

# LoRA: Timestral-5M, context 256, horizon 96, up to 200 optimizer steps.
timestral-ft train --config examples/etth1.yaml --mode lora --output runs/quickstart/lora

# Full-parameter fine-tuning.
timestral-ft train --config examples/etth1.yaml --mode full --output runs/quickstart/full
```

Add `--device mps` or `--device cuda` to select an accelerator. New runs require
an empty output directory; use `--resume <run>/checkpoint.pt` for an interrupted
run with unchanged data and settings. See [training and resume](docs/reference.md#training-and-resume).

### 3. Evaluate and export

```sh
timestral-ft evaluate --model runs/quickstart/lora/model \
  --data data/etth1/test.parquet --history data/etth1/history.parquet \
  --output runs/quickstart/evaluation

# Save an adapter, then export merged official-compatible weights.
timestral-ft export --model runs/quickstart/lora/model \
  --output runs/quickstart/adapter --adapter-only
timestral-ft export --model runs/quickstart/adapter --output runs/quickstart/merged
```

The evaluation directory contains predictions in CSV/Parquet, metrics and metadata.
Merged exports include `config.json` and `model.safetensors`, loadable by the
upstream `TimestralForecaster` on CPU/CUDA. Keep the preprocessing files for table
inference. Adapter-only exports require the recorded base checkpoint and revision.

## Python API

After preparing the ETTh1 example above:

```python
import pandas as pd
from timestral_ft import DataConfig, FineTunedModel, TrainConfig, evaluate, finetune

model = finetune(
    "data/etth1/train.parquet",
    "data/etth1/validation.parquet",
    model_id="mistralai/Timestral-5M",
    revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    data_config=DataConfig(
        time_column="date",
        target_columns=["OT"],
        past_covariates=["HUFL"],
        known_covariates=["hour_sin", "hour_cos"],
        categorical_known_covariates=["hour_category", "weekday_category"],
        frequency="h",
    ),
    train_config=TrainConfig(mode="lora", device="cpu"),
    output_dir="runs/python-lora",
)

result = evaluate(
    model, "data/etth1/test.parquet",
    history="data/etth1/history.parquet",
    output_dir="runs/python-lora/evaluation",
)
print(result.metrics)

loaded = FineTunedModel.from_pretrained("runs/python-lora/model", device="cpu")
future = pd.read_parquet("data/etth1/test.parquet").iloc[:96]
forecasts = loaded.predict(
    "data/etth1/history.parquet", 96, future_covariates=future,
)
print(forecasts[["timestamp", "target", "median", "q_0.1", "q_0.9"]].head())
```

Future target columns are ignored during prediction; the future table supplies
only configured known-future covariates. Predictions have one row per series,
target and forecast timestamp. Run `timestral-ft predict --help` for the CLI.

## Using your own data

Each row represents one series at one timestamp. Use separate chronological
training, validation and test tables, passed as paths or pandas DataFrames.

| `DataConfig` field | Role |
|---|---|
| `time_column` | Observation timestamp |
| `id_column` | Optional series ID; omit for one series |
| `target_columns` | One or more jointly forecast target columns |
| `past_covariates` | Historical numeric features |
| `known_covariates` | Numeric features known throughout the forecast horizon |
| `categorical_past_covariates` | Historical categorical features |
| `categorical_known_covariates` | Categorical features known throughout the forecast horizon |

Labels stay inside their split; validation/test can read earlier observations as
history. Separate series do not exchange information through attention.
Duplicate timestamps are errors. Irregular observations require an explicit
`frequency`; missing grid points retain NaN masks without target interpolation.
Category vocabularies are fit on training data only and saved with the model;
unseen categories use a reserved code.

The default training objective is a history-scale-balanced masked pinball loss over all
21 quantiles. Set `TrainConfig(loss_space="normalized")` (or the same YAML field) for pinball loss
in the report's advancing per-patch normalized space. A separate `validation_loss_space` can keep
checkpoint selection fixed across objective comparisons. See the
[alignment record](docs/report-alignment.zh-CN.md) and [round-one experiment](results/report_round1/README.md).
The
[technical reference](docs/reference.md) documents normalization, categorical
encoding, numerical tolerances and the inference compatibility layer.

## Experiments

### GIFT-Eval: Timestral-209M, all 97 tasks

**Completed:** validation search, frozen adapter selection and paired official
tests for all 97 tasks. Seed 42; LoRA rank 8, alpha 16; 1,105,920 trainable
parameters. Candidates screen training contexts up to 256/2048/8192 for 200 steps;
the validation-selected winner continues for up to 800 steps. The search used
113,000 optimizer steps in total. Validation selected the step-zero base model
for 21 tasks. Testing retains all official series, targets and rolling windows,
with an inference context limit of 32,768.

| Normalized geometric mean ↓ | Base checkpoint | Selected LoRA / fallback | Relative change | Improved / regressed / unchanged tasks |
|---|---:|---:|---:|---:|
| CRPS approximation | 0.446010 | 0.448549 | +0.569% | 31 / 45 / 21 |
| MASE | 0.657547 | 0.660880 | +0.507% | 33 / 43 / 21 |

**This recipe did not improve aggregate test performance.** The selected
checkpoints improved validation CRPS/MASE by 3.74%/4.06%, but those gains did not
transfer to the test split. Repeated selection on one validation suffix per
series is a possible contributor; this is a diagnostic hypothesis, not a proven
cause. A single seed does not establish statistical significance.

All 97 base replications match all eleven public reference metrics within
rtol=1e-3, atol=1e-5. The experiment's checkpoint/data-integrity audit passes.
A local insertion into the pinned public leaderboard snapshot gives CRPS rank
6/154 and MASE rank 14/154; these are **not official placements**. Neither the
adapter weights nor a leaderboard submission has been published.

[Full report](results/gift_eval_209m_lora_v2/README.md) ·
[Per-task results](results/gift_eval_209m_lora_v2/task_comparison.csv) ·
[Selected contexts](results/gift_eval_209m_lora_v2/context_lengths.csv) ·
[Audit](results/gift_eval_209m_lora_v2/audit.json) ·
[Reproduction protocol](examples/gift_eval/209m-search.md)

### GIFT-Eval: Timestral-5M, all 97 tasks

**Completed:** 97 independent Timestral-5M LoRA adapters, 97 base-model evaluations
and 97 fine-tuned evaluations. Each task uses seed 42 and 200 optimizer steps;
validation is capped at 512 preselected series, while testing covers **all official
series, targets and rolling windows**. Training used CPU for 92 tasks and MPS for 5.

| Normalized geometric mean ↓ | Base checkpoint | LoRA | Relative change | Improved / regressed tasks |
|---|---:|---:|---:|---:|
| CRPS approximation | 0.478864 | 0.479010 | +0.030% | 51 / 46 |
| MASE | 0.704905 | 0.705210 | +0.043% | 45 / 52 |

This fixed-budget recipe produced **no aggregate improvement**; the change is
close to zero and one seed does not establish its stability. Base replications
match all eleven official metrics in 97/97 tasks within the recorded tolerance.
This is a `fine-tuned` experiment using task train/validation partitions.
The submission draft passes the official format validator but **has not been
submitted**. Geometric-mean scores are not the leaderboard's average task ranks.

[Full report](results/gift_eval_5m_lora/README.md) ·
[Per-task comparison](results/gift_eval_5m_lora/task_comparison.csv) ·
[Audit](results/gift_eval_5m_lora/audit.json) ·
[Reproduction guide](examples/gift_eval/README.md)

GIFT-Eval requires a **separate environment** with its recorded NumPy/pandas
versions. Follow the reproduction guide before running `scripts/run_gift_eval.py`.
Adapters and optimizer checkpoints are generated locally; they are not bundled
in the source repository.

### Standard multivariate benchmarks

Timestral-5M, all 7 ETT / 8 exchange-rate targets, context 96, horizon 96/192,
train-only standardization and test stride 1. Fine-tuned results are mean ± sample
standard deviation over seeds 42/43/44; the frozen base is evaluated once.

| Dataset | Horizon | Base MSE ↓ | Full MSE ↓ | LoRA MSE ↓ |
|---|---:|---:|---:|---:|
| ETTh1 | 96 | 0.43385 | 0.41122 ± 0.00098 | 0.41476 ± 0.00096 |
| ETTh1 | 192 | 0.50281 | 0.47749 ± 0.00108 | 0.49542 ± 0.00024 |
| ETTh2 | 96 | 0.30170 | 0.29879 ± 0.00023 | 0.30071 ± 0.00013 |
| ETTh2 | 192 | 0.38884 | 0.38560 ± 0.00011 | 0.38772 ± 0.00001 |
| Exchange Rate | 96 | 0.08178 | 0.07964 ± 0.00015 | 0.08133 ± 0.00019 |
| Exchange Rate | 192 | 0.16748 | 0.16415 ± 0.00063 | 0.16702 ± 0.00037 |

The [complete benchmark](results/validation/benchmark/README.md) includes MAE,
persistence and official DLinear-network reference runs, with their training
settings. Budgets and objectives differ, so this is not an equal-compute ranking
or an exact reproduction of published results. Pretraining overlap is not
independently audited. Frozen ETTh1-to-ETTh2/Exchange transfer is reported
[separately](results/validation/transfer/summary.csv).

The quick-start [ETTh1 covariate example](results/etth1_cpu/README.md) uses only
`OT`, context 256 and stride 96, with MAE in original units. Its scores are not
directly comparable to the multivariate standardized table above.

```sh
# ETTh1 base / full / LoRA covariate example.
python scripts/run_etth1.py

# Multivariate benchmark.
python scripts/benchmark_suite.py --jobs 2
```

For frozen transfer, model acceptance and independent audit commands, see the
[full validation workflow](docs/reference.md#evaluation-and-verification).

## Documentation

| Resource | Contents |
|---|---|
| [Chinese usage guide](docs/guide.zh-CN.md) | Data preparation, configuration, training, resume, prediction and export |
| [English technical reference](docs/reference.md) | Training semantics, device behavior, metrics and compatibility details |
| [Report alignment record](docs/report-alignment.zh-CN.md) | Official mechanisms, implementation choices, round-one loss and remaining tasks |
| [ETTh1 configuration](examples/etth1.yaml) | A runnable example with numeric and categorical covariates |
| [Broader validation](results/validation/README.md) | Multi-seed benchmarks, frozen transfer and four-size acceptance |
| [GIFT-Eval guide](examples/gift_eval/README.md) | Environment setup, all-task training and official evaluation |
| [209M search protocol](examples/gift_eval/209m-search.md) | Completed validation search, frozen 97-task testing and CPU/MPS execution tools |
| [209M device throughput](results/gift_eval_209m_preflight/README.md) | Measured CPU routing, MPS process concurrency and inference batch sizes |
| [Contributing](CONTRIBUTING.md) | Development setup and contribution expectations |

```text
timestral-ft/
├── src/timestral_ft/    # Training, adapters, table inference and GIFT integration
├── examples/           # YAML configuration and GIFT-Eval setup
├── scripts/            # Reproducible experiments, audits and packaging
├── tests/              # Tiny-network tests and opt-in real-weight checks
├── results/            # Recorded metrics, protocols and validation reports
├── docs/               # Usage guides and technical reference
├── README.md
└── README_zh.md
```

## Development

```sh
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests scripts
python -m build
```

Default tests use tiny random networks and do not download model weights. They
cover official prediction parity, gradients, series isolation, future-target
leakage, categorical inputs, resume and adapter/merged export. Real-weight
checks are opt-in:

```sh
TIMESTRAL_INTEGRATION=1 TIMESTRAL_LARGE_MODELS=1 pytest -q
```

Unavailable accelerator tests are skipped. To require a backend, use
`TIMESTRAL_TEST_DEVICE=cuda pytest -q tests/test_devices.py` (or `mps`);
missing required hardware then fails the test job. Large-model functional checks
do not imply large-model forecast-quality validation.

## Acknowledgements and license

Built on [Timestral](https://github.com/mistralai/timestral), with evaluation using
[GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) and
[GluonTS](https://github.com/awslabs/gluonts). The DLinear reference network comes
from [Time-Series-Library](https://github.com/thuml/Time-Series-Library), which
also inspired the README structure.

Toolkit code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for
upstream attribution. Base weights and fine-tuned derivatives retain their own
[Modified MIT model license](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE),
including its commercial-use conditions. Exporting or merging does not relicense
weights.
