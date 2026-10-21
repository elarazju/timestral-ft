# MPS inference batch-size sweep: Timestral-209M

Decision: retain the existing batch-2 production budget for the remaining Bitbrains evaluation. All tested sizes run successfully, but none of the larger sizes improves throughput on this workload. Larger batches also fail the strict prediction-parity check.

| Batch | 32-window time (s, median) | Windows/s | Throughput change | Sampled Metal driver GiB | Prediction parity |
|---:|---:|---:|---:|---:|:---|
| 2 | 29.87 | 1.071 | +0.0% | 2.07 | Pass |
| 4 | 30.42 | 1.052 | -1.8% | 3.10 | Fail |
| 8 | 30.87 | 1.036 | -3.3% | 5.06 | Fail |
| 16 | 35.15 | 0.910 | -15.0% | 10.71 | Fail |

The sweep uses the same 32 actual input windows from bitbrains_fast_storage/5T/short, uniformly selected with seed 42 before profiling. Context is 7776–8544, with two targets, five historical covariates, and horizon 48. It loads the validation-frozen LoRA adapter after all 97 selections were frozen. Test labels and test scores are not used for selection. The pinned official predictor, full context policy, FP32, mirror averaging and default clipping are retained.

Trials run in forward/reverse order (2, 4, 8, 16, 16, 8, 4, 2). Each size is warmed before timing; model loading, warmup and output comparison are excluded. Each timed trial predicts all 32 windows. This is a throughput sample, not a full-dataset timing or an evaluation of training batch sizes. Only two repeats per size were measured.

Parity uses rtol=1e-4, atol=1e-5 against the first batch-2 trial. The final batch-2 repeat is bitwise identical to that baseline; both repeats of each larger size are also bitwise identical to each other. For batches 4/8/16, about 0.45%/0.68%/0.75% of returned values exceed tolerance, respectively; maximum absolute differences are 0.0298/0.0601/0.0884. Mean absolute differences are approximately 0.000135/0.000230/0.000258. These differences do not by themselves establish a change in benchmark accuracy.

The official predictor sorts inputs and pads each batch to its longest history; changing batch size changes padding and GPU tensor shapes. Differences also occur for windows whose padded length is unchanged. This probe does not isolate the numerical cause, so no precise root cause is asserted.

Memory is the maximum Metal driver allocation sampled after warmup and after predictor calls, including allocator caches. It is neither whole-system memory nor a guaranteed instantaneous peak. The already-running production MPS process retained its own memory during profiling. No allocation failure occurred up to batch 16; the maximum feasible size was not searched.

Production MPS and CPU workers were suspended without discarding partial progress, then resumed in finally with an independent watchdog. The pause ran from 2026-10-20T10:35:34.757020+00:00 to approximately 2026-10-20T10:40:55.457313+00:00. This deliberate pause is included in the active production task wall timers. Completed metrics, frozen adapters and production batch settings were unchanged.

Reproduce with scripts/profile_gift209_batch.py in the pinned GIFT environment. Raw JSON records timings, budgets, source/input/adapter hashes, sampled origin indices and comparison diagnostics; the exact source snapshot accompanies it.
