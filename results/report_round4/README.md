# Round four: synthetic priors and staged continued training

A project-defined synthetic generator and a resumable sample-level mixture/curriculum interface, evaluated from the released Timestral-5M checkpoint. This is **not** a reproduction of the unpublished official synthetic recipe or from-scratch pretraining.

In this fixed experiment, real-only adaptation had lower mean WQL than either synthetic recipe in all four dataset/horizon conditions. The staged plan selected steps200/150/50 across its seeds; the last selected model had only experienced its pure-synthetic phase. Negative results and selected-prefix exposure are retained.

- [Frozen protocol](protocol.json), [device timing decision](device_preflight.json)
- [Selected checkpoint ledger](selected_models.json): all nine adapters selected on ETTh1 validation and frozen before held-out forecasting; total-plan and selected-prefix source exposure are separate.
- [All runs](runs.csv), [mean / sample-standard-deviation summary](summary.csv), [Chinese interpretation](report.zh-CN.md)
- [Independent numerical/provenance audit](integrity_audit.json)
- [Functional and capacity acceptance](acceptance.zh-CN.md), [machine-readable cases](acceptance.json)
- [Runnable Python / nested YAML examples](../../examples/report_round4/README.md)

The three arms use the same 200 optimizer steps and C128→C256 schedule, seven targets, H192, normalized DHF loss, LoRA r8/alpha16, learning rate1e-4, batch4×accum4, and seeds42/43/44. Synthetic probabilities are real-only0→0, mixed.5→.5, and staged1→.5. Selection always uses ordinary C256/H192 forecasting with the same `history_scaled` loss. No missingness or dynamic roles are enabled in this experiment.

Only ETTh1 trains or selects weights. Frozen ETTh2 transfer uses the same selected models; the ETTh2 training partition supplies fixed standardization and the seasonal MASE denominator, so this does not claim complete absence of target-domain statistics. Both datasets use nonoverlapping H96/H192 validation/test forecasts (30/15 origins), not a stride-one leaderboard protocol.

WQL21 pools pinball loss and absolute-label normalization over all labels and 21 knots. CRPS9 is a separate nine-knot normalized approximation; MAE/MSE use median forecasts in train-standardized units, MASE24 uses each target's training-prefix seasonal denominator, and coverage80/width80 use q10–q90. Standard deviations describe three training seeds; overlapping histories do not yield independent statistical replicates.

```sh
python scripts/run_report_round4.py --phase preflight --device cpu
python scripts/run_report_round4.py --phase run --device cpu
python scripts/audit_report_round4.py
python scripts/accept_report_round4.py --output runs/report-round4-acceptance-final
ruff check src tests scripts
pytest -q
```

Raw predictions, actual contexts/labels, checkpoints and the frozen source tree live under ignored `runs/report-round4`; fixed input CSVs live in `data/validation`. The run checks code, data, model and schedule identity before reuse. Future source changes require checking out this round's commit or using its saved `source_snapshot`, rather than applying new code to an old frozen protocol. The audit separately reconstructs scaling, seven metrics, source/phase exposures, layout draws, and final independent data RNG states; generator execution is reused solely to advance its versioned generation stream.

Long-context acceptance is B1/V1/H32, one single-origin forward/backward/update at8192/32768. It establishes finite execution for those shapes, not long-context DHF training quality. CPU/MPS short stage-boundary resume and official-format exports also pass. CUDA remains unvalidated locally.
