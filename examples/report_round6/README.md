# Context representations (exploratory)

Run `python examples/report_round6/encode.py --device cpu` (or `mps`) after installing this package. It downloads the pinned released 5M checkpoint if absent. The output contains separate patch/variable vectors, validity masks and valid-patch mean pooling. `pooling="last"` takes the last observed patch. Invalid vectors are zero.

Only numeric target arrays are supported. The API is FP32 eager and does not use `InferenceConfig`, forecast queries, the quantile head, symmetry averaging or inverse scaling. Covariate/future arguments raise. An artifact configured with covariates requires explicit `artifact.encode(x, targets_only=True)`; this means targets alone, not covariate-conditioned features. The array interface never applies the artifact's tabular preprocessor.

`lengths=[...]` gives the real prefix length of each batch example; trailing storage is ignored. The most recent `max_context` points are retained before left patch padding. `patch_start`/`patch_end` give half-open coordinates in the original supplied time axis. Partial patches count once in mean pooling. A patch sees all its own points and all earlier patches; full-window pooling is offline, not online pointwise detection.

The formal protocol and lightweight results are in `results/report_round6/`. Raw data and auditable arrays stay in ignored `runs/report-round6/`. From a fresh checkout:

```sh
python scripts/run_report_round6.py --run
python scripts/run_report_round6.py --accept
python scripts/audit_report_round6.py
```

The committed protocol is already frozen. `--freeze` is only for creating the initial protocol before results exist; it refuses to overwrite it. To audit on another machine, copy the retained `runs/report-round6/` evidence (including `source_snapshot`) together with the corresponding source checkout. `--run` validates the source set and checksums before scoring, downloads the exact data when needed, and refuses to overwrite existing formal arrays. CPU runs the formal experiment; acceptance checks CPU and MPS on a Mac with MPS support. The published three classification methods are prespecified; detection thresholds come only from the unmodified calibration segment. No real-fault performance claim is made.

To re-audit the frozen source snapshot itself (absolute paths avoid ambiguity):

```sh
python runs/report-round6/source_snapshot/scripts/audit_report_round6.py \
  --source-root "$PWD/runs/report-round6/source_snapshot" \
  --evidence-root "$PWD/runs/report-round6" \
  --results-root "$PWD/results/report_round6"
```

This requires the same installed pinned upstream package, NumPy/Pandas and checkpoint cache. The auditor compares complete source/array key sets and reconstructs inputs, injection, distances, thresholds and metrics independently; it validates hashes of the neural arrays rather than running the backbone again.
