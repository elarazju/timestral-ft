# Round two: native multi-block DHF

The executable protocol is generated from `scripts/run_report_round2.py`. Run from
the repository root with the installed development environment. Published results are frozen;
use fresh output/results directories for a new experiment:

```sh
python scripts/run_report_round2.py --phase preflight --supplementary --output runs/my-dhf-comparison --results results/my-dhf-comparison
python scripts/run_report_round2.py --phase run --supplementary --output runs/my-dhf-comparison --results results/my-dhf-comparison
python scripts/audit_report_round2.py --output runs/my-dhf-comparison --results results/my-dhf-comparison
python scripts/analyze_report_round2_exposure.py --output runs/my-dhf-comparison --results results/my-dhf-comparison
```

The first invocation measures ten optimizer steps per strategy and picks a common
200/100/50-step budget before held-out scoring. `--supplementary` also requests a
fixed, approximately label-matched DHF control if its projected cost fits the
8-hour planning envelope. This is an estimate, not an enforced runtime cutoff. Keep the same options when resuming. Never delete or overwrite
a frozen protocol to force reuse after source/config changes; choose fresh output
and results directories instead.

Both strategies use normalized loss, LoRA rank 8/alpha 16, training horizon 192,
context configuration 256, batch 4 with accumulation 4, learning rate 1e-4, and
seeds 42/43/44. Model selection always uses ordinary single-origin, history-scaled
validation at horizon/stride 192. Every adapter is frozen before either strategy
is evaluated on the held-out test split at horizons 96 and 192.

The same 448-point sampled sequence contains 192 supervised steps per target for
single-window training and 416 for DHF (one 32-point prefix patch). DHF contexts
and forecast distances vary by block; equal optimizer steps do **not** mean equal
labels, context, computation or information distribution. The optional control
uses `floor(main_steps * 192 / 416)` DHF optimizer steps; its learning-rate schedule
also differs, so it does not isolate the effect of shared encoding. A separate
checkpointed layout generator preserves identical sampled sequence-window
indices across the main strategies on this fully observed dataset.

Use `--device cpu` on a machine without MPS. All weight files, checkpoints,
execution source snapshots and forecast arrays stay under ignored `runs/`.
The tracked results contain only lightweight configuration, checksums and tables.
