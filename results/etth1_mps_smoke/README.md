# ETTh1 MPS adaptation experiment

Reproduce with `python scripts/run_etth1.py --device mps --steps 4 --output runs/etth1-mps-smoke --results results/etth1_mps_smoke` from the repository root.

Timestral-5M; context 256; horizon/stride 96; seed 42; up to 4 optimizer steps per mode.
Training uses only the train split; checkpoints are selected on validation loss.

| Mode | MAE (original OT units) ↓ | WQL ↓ | 80% coverage | Trainable parameters | Training seconds | Best step |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot | 1.897857 | 0.275712 | 62.60% | 0 | 0.00 | 0 |
| full | 1.895544 | 0.275337 | 62.67% | 5,208,960 | 3.84 | 4 |
| lora | 1.897013 | 0.275580 | 62.60% | 92,160 | 1.56 | 4 |

Actual step budget: 4. Test labels: 2,880 over 30 rolling origins.

The standard ETT 12/4/4 split uses blocks of 30 days, not calendar months.
Train [0,8640), validation [8640,11520), test [11520,14400).
Target OT; historical HUFL; known-future hour sine/cosine and hour/weekday categories.

MAE is in original OT units, not the train-standardized MAE/MSE reported by default
in Time-Series-Library. This OT-only, context-256, stride-96 experiment is not the
common seven-target M, context-96, stride-1 benchmark. Do not compare those values directly.

Data revision: `1d16c8f4f943005d613b5bc962e9eeb06058cf07`. SHA256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`.
Base weights revision: `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`.

MAE and coverage pool observed labels. WQL = 2*sum(pinball)/(21*sum(abs(actual))).
All models use the same official default postprocessing. Training timing includes validation
and checkpoint writes but excludes data preparation/model loading. Timing depends on hardware and load.
See environment.json for exact software versions and device; JSON files contain run metadata.
Detailed rolling forecasts are generated under runs/etth1-mps-smoke/evaluations and are not versioned.

This is one dataset and one seed, not a general ranking. ETTh1 may overlap with pretraining data.
Hyperparameters and checkpoints are not selected using the test split. Improvements are not guaranteed.
