# Multi-dataset / multivariate validation

M task: all columns, L=96, H=96/192, test stride=1, training-only per-column standardization.
Timestral uses median forecasts, symmetry averaging, nonnegative clipping disabled for signed standardized inputs.
All test origins are included. Training selects checkpoints on validation only; test scores never choose a run.
Three seeds (42/43/44); mean ± sample standard deviation describes seed variability, not statistical significance.

| Dataset | H | Method | Runs | MAE mean ± std | MSE mean ± std | MAE gain vs base | MSE gain vs base |
|---|---:|---|---:|---:|---:|---:|---:|
| ETTh1 | 96 | dlinear | 3 | 0.41166 ± 0.00110 | 0.39677 ± 0.00091 | -2.61% | +8.55% |
| ETTh1 | 96 | full | 3 | 0.39148 ± 0.00021 | 0.41122 ± 0.00098 | +2.42% | +5.22% |
| ETTh1 | 96 | lora | 3 | 0.39236 ± 0.00020 | 0.41476 ± 0.00096 | +2.20% | +4.40% |
| ETTh1 | 96 | persistence | 1 | 0.71318 ± — | 1.29437 ± — | -77.76% | -198.35% |
| ETTh1 | 96 | zero-shot | 1 | 0.40121 ± — | 0.43385 ± — | +0.00% | +0.00% |
| ETTh1 | 192 | dlinear | 3 | 0.44098 ± 0.00134 | 0.44554 ± 0.00114 | -0.75% | +11.39% |
| ETTh1 | 192 | full | 3 | 0.42791 ± 0.00027 | 0.47749 ± 0.00108 | +2.24% | +5.03% |
| ETTh1 | 192 | lora | 3 | 0.43452 ± 0.00003 | 0.49542 ± 0.00024 | +0.73% | +1.47% |
| ETTh1 | 192 | persistence | 1 | 0.73310 ± — | 1.32488 ± — | -67.49% | -163.50% |
| ETTh1 | 192 | zero-shot | 1 | 0.43770 ± — | 0.50281 ± — | +0.00% | +0.00% |
| ETTh2 | 96 | dlinear | 3 | 0.40297 ± 0.00139 | 0.35136 ± 0.00164 | -20.77% | -16.46% |
| ETTh2 | 96 | full | 3 | 0.33161 ± 0.00007 | 0.29879 ± 0.00023 | +0.62% | +0.97% |
| ETTh2 | 96 | lora | 3 | 0.33295 ± 0.00011 | 0.30071 ± 0.00013 | +0.22% | +0.33% |
| ETTh2 | 96 | persistence | 1 | 0.42162 ± — | 0.43166 ± — | -26.36% | -43.08% |
| ETTh2 | 96 | zero-shot | 1 | 0.33368 ± — | 0.30170 ± — | +0.00% | +0.00% |
| ETTh2 | 192 | dlinear | 3 | 0.47684 ± 0.00372 | 0.47801 ± 0.00643 | -23.03% | -22.93% |
| ETTh2 | 192 | full | 3 | 0.38566 ± 0.00003 | 0.38560 ± 0.00011 | +0.49% | +0.83% |
| ETTh2 | 192 | lora | 3 | 0.38681 ± 0.00001 | 0.38772 ± 0.00001 | +0.20% | +0.29% |
| ETTh2 | 192 | persistence | 1 | 0.47254 ± — | 0.53372 ± — | -21.92% | -37.26% |
| ETTh2 | 192 | zero-shot | 1 | 0.38757 ± — | 0.38884 ± — | +0.00% | +0.00% |
| exchange_rate | 96 | dlinear | 3 | 0.22658 ± 0.00095 | 0.09385 ± 0.00051 | -14.13% | -14.77% |
| exchange_rate | 96 | full | 3 | 0.19618 ± 0.00014 | 0.07964 ± 0.00015 | +1.18% | +2.61% |
| exchange_rate | 96 | lora | 3 | 0.19801 ± 0.00022 | 0.08133 ± 0.00019 | +0.26% | +0.54% |
| exchange_rate | 96 | persistence | 1 | 0.19636 ± — | 0.08113 ± — | +1.10% | +0.79% |
| exchange_rate | 96 | zero-shot | 1 | 0.19853 ± — | 0.08178 ± — | +0.00% | +0.00% |
| exchange_rate | 192 | dlinear | 3 | 0.32383 ± 0.00204 | 0.18445 ± 0.00193 | -11.87% | -10.13% |
| exchange_rate | 192 | full | 3 | 0.28712 ± 0.00035 | 0.16415 ± 0.00063 | +0.81% | +1.99% |
| exchange_rate | 192 | lora | 3 | 0.28913 ± 0.00028 | 0.16702 ± 0.00037 | +0.12% | +0.28% |
| exchange_rate | 192 | persistence | 1 | 0.28868 ± — | 0.16712 ± — | +0.28% | +0.22% |
| exchange_rate | 192 | zero-shot | 1 | 0.28948 ± — | 0.16748 ± — | +0.00% | +0.00% |

Positive gain means lower error; all regressions are retained.

Timestral: full/LoRA up to 200 steps, batch4 × accumulation4, AdamW, original toolkit defaults.
Its validation uses stride H and scaled pinball; DLinear uses stride1 validation MSE.
DLinear: unchanged official TSLib network; local training loop, Adam lr1e-4, batch32, 10 epochs, type1 decay, patience3.
DLinear validation uses sequential label-weighted MSE; TSLib shuffles validation and averages batch means. Both retain the final partial batch.
DLinear is a reference run using the official network, not a paper reproduction.
Training budgets/objectives differ; this is a matched test-protocol comparison, not an equal-compute ranking.
ETTh1/2 use standard 8640/2880/2880 splits. Exchange uses 70/10/20% with integer train/test boundaries.
Exchange timestamps are synthetic ordered indices; no calendar covariates are used by any method.
Only two domains and short adaptation budgets are covered. Pretraining overlap is unknown.
Overlapping test windows are not independent samples; no significance claims or naive label-level confidence intervals.
Per-case protocol JSON records source hashes and scaler statistics; per-run JSON includes per-target/lead scores.
All median predictions, labels and origins are saved under runs/validation; these large files are not packaged.
