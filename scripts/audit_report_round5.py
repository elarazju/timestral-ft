"""Independent recomputation from saved runtime predictions and raw timing samples.

The benchmark audit uses no producer statistics. GIFT additionally repeats the
pinned official evaluator on saved forecasts and independently computes pooled
MAE, median MSE, ND and nine-quantile weighted loss from the complete test labels.
"""

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from run_report_round5 import sources

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(record):
    assert record["source_hashes"] == sources()
    assert record["protocol_sha256"] == sha(ROOT / "results/report_round5/protocol.json")
    if record.get("status") == "process_failure":
        return
    protocol = json.loads((ROOT / "results/report_round5/protocol.json").read_text())
    frozen = protocol["weights"][record.get("size", "5M")]
    assert record["weights_sha256"] == frozen["weights_sha256"]
    assert record["config_sha256"] == frozen["config_sha256"]
    assert sha(Path(record["weights_path"]) / "model.safetensors") == record["weights_sha256"]
    assert sha(Path(record["weights_path"]) / "config.json") == record["config_sha256"]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def benchmark(args):
    rows = []
    for device, context, variables, mode in itertools.product(
            ["cpu", "mps"], [256, 2048], [1, 7], ["fp32", "bf16", "compile", "bf16_compile"]):
        name = f"{device}_{context}_{variables}_{mode}"
        folder = args.raw / "benchmark" / name
        record = json.loads((folder / "record.json").read_text())
        verify_sources(record)
        assert record["case"] == f"{device}:{context}:{variables}:{mode}"
        if record["status"] != "completed":
            rows.append({"case": name, "status": record["status"], "error": record.get("error")})
            continue
        assert record["shape"] == [1, variables, context, 96]
        assert record["runtime"]["status"] == "supported"
        assert record["runtime"]["effective"] == {"precision": "bf16" if "bf16" in mode else "fp32",
                                                   "compile": "compile" in mode}
        assert record["runtime"]["compiled_executed"] == ("compile" in mode)
        assert record["runtime"]["forecast_calls"] == 8
        assert sum(record["runtime"]["successful_calls_by_mode"].values()) == 8
        reference_folder = args.raw / "benchmark" / f"{device}_{context}_{variables}_fp32"
        reference = json.loads((reference_folder / "record.json").read_text())
        assert sha(folder / "input.npy") == record["input_sha256"] == reference["input_sha256"]
        assert sha(folder / "first.npy") == record["first_sha256"]
        assert sha(folder / "predictions.npy") == record["predictions_sha256"]
        actual = np.load(folder / "predictions.npy").astype(np.float64)
        expected = np.load(reference_folder / "predictions.npy").astype(np.float64)
        assert actual.shape == (5, 1, variables, 96, 21)
        assert np.isfinite(actual).all() and (np.diff(actual, axis=-1) >= 0).all()
        first = np.load(folder / "first.npy").astype(np.float64)
        np.testing.assert_allclose(actual[0], first, rtol=1e-4, atol=1e-5)
        for sample in actual:
            np.testing.assert_allclose(sample, actual[0], rtol=1e-4, atol=1e-5)
        repeat_max = float(max(np.abs(actual - actual[0]).max(), np.abs(first - actual[0]).max()))
        samples = np.asarray(record["stable_seconds"])
        assert samples.shape == (5,) and (samples > 0).all()
        median = float(np.median(samples))
        baseline = float(np.median(reference["stable_seconds"]))
        assert median == record["median_seconds"]
        assert variables * 96 / median == record["throughput_forecast_points_per_second"]
        delta = actual - expected
        startup = record["runtime_setup_seconds"] + record["first_seconds"] + sum(record["warmup_seconds"])
        base_startup = reference["runtime_setup_seconds"] + reference["first_seconds"] + sum(reference["warmup_seconds"])
        breakeven = max(1, 1 + math.ceil((startup - base_startup) / (baseline - median))) if median < baseline else None
        row = {"case": name, "device": device, "context": context, "variables": variables, "mode": mode,
               "status": "completed", "repeat_max_abs_difference": repeat_max, "median_seconds": median, "min_seconds": float(samples.min()),
               "max_seconds": float(samples.max()), "speedup_vs_fp32": baseline / median,
               "throughput_forecast_points_per_second": variables * 96 / median,
               "first_seconds": record["first_seconds"], "load_seconds": record["load_seconds"],
               "runtime_setup_seconds": record["runtime_setup_seconds"], "warmup_seconds": record["warmup_seconds"],
               "break_even_useful_calls_including_first": breakeven,
               "max_abs_error": float(np.abs(delta).max()),
               "relative_l2_error": float(np.linalg.norm(delta) / np.linalg.norm(expected)),
               "rss_peak_bytes": record["rss_peak_bytes"],
               "mps_driver_peak_sampled_bytes": record["mps_driver_peak_sampled_bytes"],
               "runtime": record["runtime"], "record_sha256": sha(folder / "record.json"),
               "predictions_sha256": sha(folder / "predictions.npy"),
               "source_hashes": record["source_hashes"]}
        rows.append(row)
    write(args.results / "benchmark.json", rows)
    return {"cases": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
            "independent_recomputation": "timing median/range, throughput, speedup, amortization, all-quantile residuals; repeatability within predeclared rtol1e-4/atol1e-5 (max difference reported), actual modes, source/input hashes"}


def gift(args):
    # Reconstruct official forecast objects from saved arrays; no model execution.
    sys.path.insert(0, str(ROOT / "scripts"))
    from gluonts.model.evaluation import evaluate_forecasts
    from run_report_round5_gift import TASKS, official_namespace
    ns = official_namespace()
    checks = []
    for task, mode in itertools.product(TASKS, ["fp32", "bf16"]):
        record = json.loads((args.results / "gift" / (task.replace("/", "__") + "." + mode + ".json")).read_text())
        verify_sources(record)
        path = Path(record["prediction_path"])
        assert sha(path) == record["prediction_sha256"]
        arrays = np.load(path)
        dataset = ns["Dataset"](record["dataset_name"], term=record["term"], to_univariate=False)
        inputs, labels = list(dataset.test_data.input), list(dataset.test_data.label)
        assert len(inputs) == len(labels) == len(arrays) == record["test_windows"]
        assert record["runtime"]["effective"] == {"precision": mode, "compile": False}
        assert not record["runtime"]["mixed_execution"]
        manifest = json.loads((ROOT / "results/gift_eval_5m_lora/data_manifest.json").read_text())
        expected_files = {f: info["sha256"] for f, info in manifest["files"].items()
                          if f.startswith(record["dataset_name"] + "/")}
        assert record["data_files"] and record["data_files"] == expected_files
        for name, digest in record["data_files"].items():
            assert sha(ROOT / ".cache/gift-eval-data" / name) == digest
        forecasts = [ns["QuantileForecast"](forecast_arrays=a,
                    forecast_keys=[str(q) for q in ns["QUANTILE_LEVELS"]] + ["mean"],
                    start_date=e["start"] + np.asarray(e["target"]).shape[-1], item_id=e.get("item_id"))
                     for a, e in zip(arrays, inputs)]
        official = evaluate_forecasts(forecasts, test_data=dataset.test_data, metrics=ns["METRICS"],
                                      batch_size=512, axis=None, mask_invalid_label=True,
                                      allow_nan_forecast=False, seasonality=ns["get_seasonality"](dataset.freq))
        repeated = {k: float(official[k].iloc[0]) for k in official.columns}
        for name, value in repeated.items():
            np.testing.assert_allclose(value, record["metrics"][name], rtol=1e-12, atol=1e-12)
        count = 0
        absolute = squared = denominator = pinball = 0.0
        for a, e in zip(arrays, labels):
            target = np.asarray(e["target"], dtype=np.float64)
            if target.ndim == 2:
                target = target.T
            observed = np.isfinite(target)
            point = a[4].astype(np.float64)
            error = (target - point)[observed]
            absolute += np.abs(error).sum()
            squared += np.square(error).sum()
            denominator += np.abs(target[observed]).sum()
            count += observed.sum()
            for qi, q in enumerate(ns["QUANTILE_LEVELS"]):
                residual = (target - a[qi].astype(np.float64))[observed]
                pinball += np.maximum(residual * q, residual * (q - 1)).sum()
        independent = {"MAE[0.5]": float(absolute / count), "MSE[0.5]": float(squared / count),
                       "ND[0.5]": float(absolute / denominator),
                       "mean_weighted_sum_quantile_loss": float(2 * pinball / (9 * denominator))}
        for name, value in independent.items():
            np.testing.assert_allclose(value, repeated[name], rtol=2e-6, atol=1e-6)
        checks.append({"task": task, "mode": mode, "test_windows": len(inputs), "observed_labels": int(count),
                       "official_repeated": repeated, "independent": independent,
                       "max_independent_abs_error": max(abs(v - repeated[k]) for k, v in independent.items())})
    write(args.results / "gift_audit.json", checks)
    return {"complete_task_mode_cases": len(checks), "official_metrics_repeated": 11, "independent_metrics": 4}


def acceptance(args):
    checks = []
    for size, device, mode in itertools.product(["28M", "123M", "209M"], ["cpu", "mps"], ["full", "lora"]):
        record = json.loads((args.results / "acceptance" / f"{size}_{device}_{mode}.json").read_text())
        verify_sources(record)
        path = Path(record["export_path"])
        assert sha(path / "model.safetensors") == record["export_sha256"]
        assert sha(path / "config.json") == record["export_config_sha256"]
        assert sha(path / "predictions.npz") == record["predictions_sha256"]
        assert sha(path / "official.npy") == record["official_sha256"]
        arrays = np.load(path / "predictions.npz")
        official = np.load(path / "official.npy")
        assert record["finite_gradient_tensors"] > 0
        tolerance = (1e-3, 1e-4) if device == "mps" else (1e-4, 1e-5)
        np.testing.assert_allclose(arrays["fp32"], official, rtol=tolerance[0], atol=tolerance[1])
        assert np.isfinite(arrays["bf16"]).all() and (np.diff(arrays["bf16"], axis=-1) >= 0).all()
        for precision, receipt in record["runtime"].items():
            assert receipt["effective"] == {"precision": precision, "compile": False}
        checks.append({"case": record["case"], "official_export_max_abs_error": float(np.abs(arrays["fp32"] - official).max())})
    write(args.results / "acceptance_audit.json", checks)
    return {"cases": len(checks), "shape": [1, 1, 64, 32]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "runs/report-round5")
    p.add_argument("--results", type=Path, default=ROOT / "results/report_round5")
    p.add_argument("--stage", choices=["benchmark", "gift", "acceptance", "all"], default="all")
    args = p.parse_args()
    summary = {"auditor_sha256": sha(__file__), "checks": {}}
    for name, function in [("benchmark", benchmark), ("gift", gift), ("acceptance", acceptance)]:
        if args.stage in (name, "all"):
            summary["checks"][name] = function(args)
    write(args.results / ("audit.json" if args.stage == "all" else f"audit_{args.stage}.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
