# Timestral-FT-5M-LoRA on GIFT-Eval

Complete 97-task fine-tuning and paired official-protocol evaluation. See the
[full report](README.zh-CN.md), [task comparison](task_comparison.csv),
[independent audit](audit.json) and [reproduction guide](../../examples/gift_eval/README.md).

| Normalized geometric mean | Base | LoRA | Relative change |
|---|---:|---:|---:|
| CRPS approximation | 0.478864 | 0.479010 | +0.030% |
| MASE | 0.704905 | 0.705210 | +0.043% |

Lower is better. This is one seed, up to 200 optimizer steps per independent
task adapter, validation capped at 512 preselected series, and **complete official
test coverage**. All gains and regressions are retained. Hardware is recorded
per task; five previously unstarted heavy tasks use MPS and the rest use CPU.
Base replications match the official eleven metrics in 97/97 tasks
within rtol=1e-3 / atol=1e-5. Geometric-mean scores are not official ranks.

The local submission draft passed the official schema validator. It has **not**
been submitted or published; model/code links and organization fields remain
unfilled. Weights retain the base license. Training on GIFT partitions makes
this a **fine-tuned** method, not a zero-shot claim; base pretraining provenance
was not independently audited. CUDA hardware has not been validated.

The original experiment source is archived at commit
[59fbbbf](https://github.com/elarazju/timestral-ft/tree/59fbbbf);
`source_manifest.json` retains its file checksums. The subsequent release adds
runtime detection and a GPU bias-add workaround for affected M1/macOS 26 linear
operations. Experiment metrics and adapters are unchanged. On the original M4
host the native path remains active, the probe preserves RNG state, and a real
5M forecast is bitwise unchanged; see [the compatibility check](release_compatibility_check.json).
