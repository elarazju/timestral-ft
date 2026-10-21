# Timestral-209M GIFT-Eval adaptation

Completed all 97 tasks. **This recipe did not improve aggregate test performance.**

| Metric | Base | Adapted | Change |
|---|---:|---:|---:|
| CRPS | 0.44601008 | 0.44854910 | +0.5693% |
| MASE | 0.65754709 | 0.66087957 | +0.5068% |

Validation selected step-zero identity adapters for 21 tasks. All tasks trained; fallback is disclosed. These are normalized geometric means, not official ranks. Weights and leaderboard submission have not been published by this run.

Training screens are prefixes of the longer cosine schedule. Only the chosen candidate is extended; validation_ablations.csv describes observed checkpoints, not a complete factorial search. All three selection objectives are compared on validation only, and no test-driven retry is performed.

Final testing used one MPS worker alongside a CPU worker. The initial CPU queue covered short histories; later CPU extension rounds also processed heavier multivariate and Bitbrains tasks. See parallel_test_execution.json, cpu_extension*_plan.json and each test receipt for routing, source hashes and devices. All frozen adapters, official test windows, context limits, and prediction settings were retained.


Local rank simulation against public snapshot `23078cb5bffd`: CRPS **6/154**, MASE **14/154**. This is a local insertion/recalculation, not an official leaderboard placement.

Formal search used 113,000 optimizer steps. [Training costs and choices](training_diagnostics.json) · [Conditional validation checkpoint diagnostics](validation_diagnostics.md).

## Result records and interpretation

- [Per-task comparison](task_comparison.csv), [all base metrics](baseline_all_results.csv), and [all adapted metrics](lora_all_results.csv) retain all 97 official configurations.
- [Selected training contexts](context_lengths.csv) includes the best step, identity fallback and available training history. Validation and test inference use at most 32768 available timestamps; the training context column is a window width, including missing-value padding for short histories.
- [Generalization diagnostics](generalization_diagnostics.json) compares the selected validation checkpoint with test performance. Selected validation CRPS/MASE improved by about 3.74%/4.06%; test metrics regressed by about 0.57%/0.51%. Repeated use of one official validation suffix per selected series is a possible source of selection overfitting, not an isolated causal finding. These post-test diagnostics do not select replacement adapters.
- The largest CRPS regression (ett2/D/short, about +19.14%) was reproduced with the same frozen adapter on CPU and MPS. All 21 identity fallbacks reproduce every base metric exactly. These checks do not prove the absence of every possible implementation defect.
- [Original training provenance](code_provenance.json) identifies the committed training implementation. [Historical execution sources](execution_sources/manifest.json) preserve the actual inference helper snapshots before later publication edits; runtime source hashes in the test/report manifests refer to those historical files.
- [Device throughput probes](../gift_eval_209m_preflight/README.md) record one/two-MPS-process and inference batch-size measurements. Production retained one MPS worker and the original batch budget. Recorded task wall times include deliberate profiling pauses and interrupted execution; they are not pure accelerator compute times.

The local adapter ZIP is recorded by checksum in package.json and is not stored in Git. The submission directory is a format-validated draft with incomplete publication metadata, not a published leaderboard entry.
