# Full GIFT-Eval task adaptation

This experiment trains **97 independent Timestral-5M LoRA adapters**, each starting
from the same pinned base checkpoint. It uses GIFT-Eval training/validation data
and belongs in the `fine-tuned` category, not `zero-shot`.

The runner uses the exact Predictor/metrics from the pinned official Timestral
notebook (Apache-2.0), verifies its checksum, and retains its median-as-mean
convention, nine evaluation quantiles, default forecast postprocessing and full
history up to 32768. The repository is an optional external dependency, not
vendored into this package.

## Environment and data

Use Python 3.12 in a separate environment: GIFT-Eval requires older NumPy/pandas
than our ordinary development lock. The exact working dependency list is saved
in `results/gift_eval_feasibility/requirements-probe.txt`; install this package's
pinned Timestral dependency as well. The local isolated environment has the
pinned upstream source at `.cache/upstream/src`.

For a fresh setup, install the recorded environment and this package into a
separate Python 3.12 virtual environment (use its Python for every later command):

```sh
python3.12 -m venv .venv-gift
.venv-gift/bin/python -m pip install -r results/gift_eval_feasibility/requirements-probe.txt
.venv-gift/bin/python -m pip install -e .
```

```sh
git clone https://github.com/SalesforceAIResearch/gift-eval.git .cache/gift-eval
git -C .cache/gift-eval checkout 8e6da595ac943be01e0848a02b9157af84731284
hf download Salesforce/GiftEval --repo-type dataset \
  --revision 30841734ac5cfddbd0c3bad6d09d2b6b32becbb0 \
  --local-dir .cache/gift-eval-data
```

## Run and resume

From the project root, using the isolated environment's Python:

```sh
python scripts/run_gift_eval.py --stage train
python scripts/run_gift_eval.py --stage evaluate
# CPU training with MPS inference (evaluation device is recorded per task):
python scripts/run_gift_eval.py --stage evaluate --eval-device mps --fast-seasonal-error
# Or run both stages sequentially:
python scripts/run_gift_eval.py
```

Training resumes from per-task `checkpoint.pt` including adapter weights, best
adapter, optimizer, scheduler and all RNG states. Completed tasks are skipped.
Each task records its data fingerprint and base-weight checksum; changes to
training settings require a new output directory. Evaluation records are atomic
and also skipped on restart. Do not edit source/data or change protocol settings
inside an existing experiment. `--task hospital/M/short` scopes a smoke run;
use separate output/results directories for any changed settings.

The CPU reference uses four threads. `--device mps` and `--device cuda` are
available on supported hardware; keep device selection consistent for resume.
The full experiment uses CPU training with five previously unstarted, expensive
long-horizon tasks assigned to MPS, and a mix of CPU/MPS evaluation,
with the inference device retained in each metric record. The optional
`--fast-seasonal-error` replaces GluonTS's repeated contiguous index gathers
with equivalent slices. It preserves the complete seasonal-error history and
mask/reduction semantics; four complete tasks reproduce all eleven metrics
bit for bit, with additional masked/multivariate/axis regression tests.

Gradient clipping uses the standard PyTorch path, with FP64 norm recomputation
if finite FP32 gradients overflow during the squared-norm reduction. Actual
NaN/Inf elements still stop training. The advancing-frame scale floor also has
a finite derivative at zero variance, with exactly unchanged forward values.
The full run's numerical repair/replay evidence is retained with its results.

`train_device_overrides.json` in the run directory fixes these five MPS tasks:
Jena 10-minute medium and hourly long, CloudOps application long, and CloudOps
service medium/long. The map is copied into `protocol.json`. Selection was based
on feature dimensions, horizon and a training-prefix-only timing probe, not test
scores. Already-started CPU tasks keep their original device/checkpoints.
Adapters are initialized on CPU before moving to the selected backend, so the
seeded initial adapter matrices are identical. Actual FP32 trajectories can vary
by backend. Reproduction of this hardware map requires a Mac with MPS.

The runner uses POSIX advisory task locks (macOS/Linux). Several local processes
may handle disjoint `--task` sets; mapped MPS training tasks share one accelerator
lock. Evaluation workers can use `--skip-locked --eval-threads 2` to claim
unoccupied jobs. Rerun evaluation after all workers finish to consolidate CSVs;
incomplete worker passes never masquerade as a full 97-task result.

## Fixed training protocol

- LoRA only on time/variate attention `wQKV`/`wO`; rank 8, alpha 16, dropout 0.
- Seed 42, context up to 256; left NaN padding for shorter histories.
- Up to 200 optimizer steps per task, batch 4, gradient accumulation 4.
- AdamW, learning rate 1e-4, weight decay .01, gradient clip 1; warmup 20,
  then cosine decay.
- Validation every 50 steps, patience 3; best **history-scaled masked
  21-quantile pinball** objective. This is our fine-tuning recipe, not the
  official pretraining recipe or a CRPS-optimal checkpoint guarantee.
- Up to 512 validation series chosen uniformly with seed 42 before model
  evaluation. All training series with at least two timestamps are eligible;
  sample a series uniformly, then a training origin uniformly.
- Where the training prefix is shorter than H+1, sample origin 1 and mask the
  unavailable label suffix. Longer series use full H-step training labels.
  Targets and covariates from validation/test never enter a training window.
- Multivariate targets remain joint targets; numeric past covariates remain
  past-only covariates. No additional calendar/categorical features are added.
- Validation labels are exactly the new H steps after the official training
  prefix. Weights are frozen before all official rolling test windows. Earlier
  observed test windows may supply history, as in the official protocol.
- The validation subset affects checkpoint choice only; **test coverage is
  complete**, with all series, targets and rolling origins in each task.

Independent task adapters are deliberate: different horizons/resamplings of one
source have different cutoffs. Pooling those prefixes without a shared cutoff
can leak a long-horizon task's test period into another task's training.

## Artifacts and submission

`runs/gift-eval-5m-lora/tasks/<key>/` holds adapter weights, checkpoints,
configuration, summaries and history. Load an adapter for native array inference:

```python
from timestral_ft.gift import load_gift_adapter
from timestral_ft.compat import PortableForecaster

model = load_gift_adapter("runs/gift-eval-5m-lora/tasks/hospital__M__short")
fc = PortableForecaster(model, max_context=32768)
forecast = fc.forecast(context, prediction_length=12)  # (batch, targets, history)
```

This artifact uses official GIFT array roles, without the column/vocabulary
metadata expected by the ordinary `FineTunedModel` table API. It is loaded through
`load_gift_adapter`; all 97 adapters require their recorded base revision.
Weights and derivatives retain the base model's license.

`results/gift_eval_5m_lora/` retains protocol/data checksums, per-task training
summaries and all eleven test metrics. Evaluation generates
`baseline_all_results.csv` and `lora_all_results.csv`. Neither is automatically
published. A submission additionally needs public model/code links and accurate
organization/availability fields; these must be supplied when actually publishing.
Official result formatting and the distinction between geometric-mean scores
and average per-task ranks are explained in `docs/gift-eval.zh-CN.md`.

After the complete evaluation, run `python scripts/audit_gift_eval.py` and
`python scripts/summarize_gift_eval.py`. `python scripts/package_gift_adapters.py`
creates a local ZIP of all 97 adapters, their configurations/histories and base
licenses, excluding optimizer states and raw datasets. Nothing is uploaded.

## Timestral-209M validation search

The 209M experiment uses a separate run directory and never changes the recorded
5M run. See [the 209M protocol](209m-search.md) for the fixed search, baseline
fallback, checkpoint-level validation ablations, and test freeze policy.

```sh
.cache/gift-probe-env/bin/python scripts/run_gift209_search.py --stage all
```

This runs the complete search before the paired official tests and independent
report. It does not publish weights or submit a leaderboard entry. The default
backend is MPS; use `--device cpu` or `--device cuda` in a new run directory to
change the protocol. Long experiments checkpoint each optimizer step.


For automatic CPU/MPS worker management, progress receipts, final testing,
audit and packaging, run:

```sh
.cache/gift-probe-env/bin/python scripts/launch_gift209.py
```

The supervisor's current state is saved to
`results/gift_eval_209m_lora_v2/progress.json`. No remote publication is performed.
