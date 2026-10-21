# One versus two MPS processes: Timestral-209M

Decision: retain **one MPS worker plus one CPU worker** for the frozen 97-task test. Two MPS processes provide only small throughput changes in these probes and approximately double Metal driver allocations.

| Workload | Context | Targets / past covariates | Batches per trial | One process (s) | Two processes (s) | Throughput change | Metal driver GiB: one / two |
|---|---:|---:|---:|---:|---:|---:|---:|
| electricity/15T/short | 32768 | 1 / 0 | 24 | 21.57 | 21.52 | +0.25% | 2.06 / 4.11 |
| ett1/15T/long | 32768 | 7 / 0 | 4 | 8.47 | 8.28 | +2.35% | 2.09 / 4.19 |
| bizitobs_l2c/5T/long | 28368 | 7 / 2 | 4 | 19.92 | 19.51 | +2.10% | 3.06 / 6.13 |

Each workload uses the same total number of batches for one and two processes. Two independent models are loaded and warmed, then a shared start barrier releases their equal shards. Trials run in ABBA order (1, 2, 2, 1); the table compares median total wall times. Each process has four CPU threads. Initialization is excluded from the warmed throughput column but recorded in the raw JSON.

Inputs are actual fixed GIFT test-input batches, after all 97 adapters were frozen. The official context limit (32768), default mirror/clipping, and original batch budgets remain unchanged. The first workload batches three univariate inputs; the other workloads batch one multivariate input each. All nine quantile outputs match across repeated calls and process counts at rtol=1e-4, atol=1e-5. These timings do not measure complete datasets or guarantee a corresponding wall-time change for the full benchmark.

The exploratory four-batch electricity trial showed +7.08% throughput. Extending to 24 batches did not reproduce that apparent advantage; both raw records are retained. Small point-estimate differences are not evidence of a statistically established speedup.

A supplemental 35-past-covariate workload completed three trials: one process 44.37 s, two processes 43.65 s and 43.47 s for eight batches. The final single-process repeat was stopped when the user chose to retain the current execution plan. These incomplete ABBA results are preserved separately and excluded from the summary table.

Memory is the sum of per-worker peak Metal driver allocations, including allocator caches; it is not system-wide resident memory. The suspended production process retained its model memory equally during both trial types. The production CPU side worker was not running during these isolation measurements.

The production evaluator and pending replacement supervisor were suspended during profiling and resumed afterwards, with a separate watchdog. The production evaluation in progress includes this deliberate pause in its raw wall timer. Approximate pause intervals are recorded in mps_concurrency.json; previously completed metrics/receipts are untouched.

Reproduce with scripts/profile_gift209_mps.py; use --case to select a workload and --jobs for an even equal-work count. Use --pause-pid and --pause-pending-pid only with the matching experiment processes. Raw timing, parity, source/input/adapter hashes and both source snapshots accompany this report.
