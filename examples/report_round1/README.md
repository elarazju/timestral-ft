# Round one: training objective comparison

This runner compares the legacy history-scaled pinball objective with the new
model-normalized objective using Timestral-5M LoRA. The command must run from the
repository root in an environment with the package and development dependencies
installed. A real MPS device is required for the default setting; `--device cpu`
uses CPU instead.

```sh
python scripts/run_report_round1.py --phase preflight
python scripts/run_report_round1.py --phase run
python scripts/run_report_round1.py --phase diagnostics
python scripts/run_report_round1.py --phase summarize
```

`--phase all` performs the complete sequence. Preflight times ten optimizer steps
for each objective, before any test scoring, then freezes a common 200/100/50-step
budget under eight hours including a 40% overhead allowance. The estimate is a
planning approximation, not a runtime guarantee. Both arms use the same
`validation_loss_space="history_scaled"`; the experiment isolates the training
objective while keeping checkpoint selection common. New normalized-space
training remains explicitly opt-in:

```python
from timestral_ft import TrainConfig

config = TrainConfig(
    mode="lora", loss_space="normalized", validation_loss_space="history_scaled",
    context_length=256, prediction_length=96, max_steps=200,
    batch_size=4, gradient_accumulation=4,
    learning_rate=1e-4, lora_rank=8, lora_alpha=16,
    seed=42, device="mps", num_threads=2,
)
```

The paired matrix uses both ETTh1/ETTh2 and seeds 42/43/44. All seven targets are
standardized using only the first 8640 observations. Validation and test each
contain 2880 observations with stride 96: exactly 30 disjoint forecast horizons,
using the preceding observed history for each origin. This is a bounded
experiment, **not the standard stride-one ETT leaderboard protocol**.

Model/source/config/data hashes, selection metadata, metrics and seed aggregates
are saved under `results/report_round1/`. Checkpoints, complete quantile
predictions and diagnostic preflight runs stay under ignored `runs/report-round1/`.
All selected model checksums are frozen before any held-out test forecasting.
Existing files are reused only for matching provenance. To change any frozen
setting or source, choose fresh `--output` and `--results` directories.

The `masked_wql21` metric averages weighted quantile loss over all 21 model knots.
`crps9_approx` uses only .1 through .9, has the same quantile-loss construction as
nine-quantile CRPS reporting, and is reported separately. Numerators and absolute
label denominators aggregate globally over targets, windows and leads. MASE uses
24-hour seasonal differences over the full training prefix. Median MAE/MSE and
80% interval width use externally standardized units. Coverage is inclusive.
There are no label-level confidence intervals; seed standard deviation describes
run variability and does not establish significance. ETT overlap with pretraining
is unknown. Diagnostic objective magnitudes use different units and are not
comparable quality scores.

After completion, verify all 12 selected model artifacts and 28 saved split scores:

```sh
python scripts/audit_report_round1.py
```

The audit checks frozen source/data/config/model/prediction hashes, forecast
origins and labels, and recomputes every saved metric from the ignored quantile
predictions. The metric formulas also have independent small-array unit tests.

Always rerun the auditor after resuming or regenerating summaries. A matching
protocol alone is insufficient: the auditor additionally requires each row to
match its selected model ledger, artifact metadata and recomputed aggregate
CSV tables. Audit failures must be resolved before publishing experimental
conclusions; existing mismatched provenance must never be overwritten.
