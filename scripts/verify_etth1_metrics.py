"""Independently reconcile original-unit OT scores with TSLib's standardized metrics.

Optional control evaluates the untouched 5M base model using TSLib-style M/S
numeric inputs, train-only StandardScaler statistics, L=H=96 and stride=1.
It does not train a TSLib model or reproduce a particular paper's full setup.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import PortableForecaster

TSLIB_REVISION = "4e938a1767106324dd753b2a44832bf870a0252e"
MODEL_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


def reconcile(raw, evaluations):
    training = raw.iloc[:8640]
    mean = float(training["OT"].mean())
    scale = float(training["OT"].std(ddof=0))
    lookup = raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")["OT"]
    rows = []
    for mode in ["zero-shot", "full", "lora"]:
        path = Path(evaluations) / mode / "predictions.parquet"
        saved = pd.read_parquet(path)
        assert saved["target"].unique().tolist() == ["OT"]
        assert saved["origin"].nunique() == 30
        assert len(saved) == saved["timestamp"].nunique() == 2880
        times = pd.to_datetime(saved["timestamp"])
        expected_times = pd.to_datetime(raw["date"].iloc[11520:14400])
        assert pd.DatetimeIndex(times).equals(pd.DatetimeIndex(expected_times))
        np.testing.assert_allclose(saved["actual"], lookup.reindex(times), rtol=0, atol=2e-6)
        prediction = saved["median"].to_numpy(dtype=float)
        actual = saved["actual"].to_numpy(dtype=float)
        error = prediction - actual
        standardized_error = (prediction - mean) / scale - (actual - mean) / scale
        row = {"mode": mode, "raw_mae": float(np.abs(error).mean()),
               "raw_mse": float(np.square(error).mean()),
               "standardized_mae": float(np.abs(standardized_error).mean()),
               "standardized_mse": float(np.square(standardized_error).mean()),
               "windows": 30, "stride": 96, "context": 256, "horizon": 96,
               "target_columns": ["OT"], "prediction_file_sha256": file_sha256(path)}
        np.testing.assert_allclose(row["standardized_mae"], row["raw_mae"] / scale, rtol=1e-12)
        np.testing.assert_allclose(row["standardized_mse"], row["raw_mse"] / scale**2, rtol=1e-12)
        rows.append(row)
    return {"OT_train_mean": mean, "OT_train_std_ddof0": scale, "rows": rows,
            "verified": "All 2880 saved labels and timestamps per mode match raw ETTh1 OT test data"}


def standard_control(raw, batch_size=16):
    torch.set_num_threads(4)
    forecaster = PortableForecaster.from_pretrained("mistralai/Timestral-5M", revision=MODEL_REVISION,
                                                    device="cpu", max_context=96)
    rows = []
    for feature_mode in ["S", "M"]:
        columns = ["OT"] if feature_mode == "S" else raw.columns[1:].tolist()
        values = raw[columns].to_numpy(dtype=float)
        mean = values[:8640].mean(axis=0)
        scale = values[:8640].std(axis=0, ddof=0)
        values = ((values - mean) / scale).astype(np.float32)
        origins = list(range(11520, 14400 - 96 + 1))
        absolute, squared, count = 0.0, 0.0, 0
        started = time.perf_counter()
        for offset in range(0, len(origins), batch_size):
            selected = origins[offset:offset + batch_size]
            x = np.stack([values[t - 96:t].T for t in selected])
            y = np.stack([values[t:t + 96].T for t in selected]).astype(float)
            prediction = forecaster.forecast(x, 96, clip_nonnegative=False).median.cpu().numpy().astype(float)
            error = prediction - y
            absolute += float(np.abs(error).sum())
            squared += float(np.square(error).sum())
            count += error.size
        row = {"model": "Timestral-5M base (zero-shot)", "features": feature_mode,
               "columns": columns, "context": 96, "horizon": 96, "stride": 1,
               "windows": len(origins), "scored_values_including_overlaps": count,
               "standardized_mae": absolute / count, "standardized_mse": squared / count,
               "seconds": time.perf_counter() - started,
               "input_scale": "Per-column train [0,8640) population mean/std, then float32",
               "point_estimate": "median", "symmetric_averaging": True,
               "clip_nonnegative": False, "calendar_covariates": False}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/etth1/ETTh1.csv")
    parser.add_argument("--evaluations", default="runs/etth1/evaluations")
    parser.add_argument("--output", default="results/etth1_protocol_check/verification.json")
    parser.add_argument("--run-standard-control", action="store_true")
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    result = {"tslib_revision": TSLIB_REVISION, "model_revision": MODEL_REVISION,
              "data_sha256": file_sha256(args.data), "existing_experiment": reconcile(raw, args.evaluations),
              "scope": "Metric/protocol verification; not a reproduction of a TSLib model's training",
              "source_paths": ["data_provider/data_loader.py", "exp/exp_long_term_forecasting.py",
                               "utils/metrics.py", "scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh"],
              "important_differences": ["Original-unit MAE versus train-standardized MSE/MAE",
                                        "OT plus extra covariates versus M seven-target / S OT-only tasks",
                                        "Context 256 versus 96 in the cited TimesNet script",
                                        "Stride 96, 30 windows versus stride 1, 2785 overlapping windows",
                                        "Median forecast / pinball adaptation versus point models trained with MSE"]}
    if args.run_standard_control:
        result["standard_numeric_control"] = standard_control(raw)
        result["control_limits"] = ("Uses TSLib numeric task/split/scaling/window settings, but the model is the "
                                    "unmodified Timestral base. No TSLib model training or calendar embeddings "
                                    "are reproduced; these are not direct head-to-head leaderboard scores.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result["existing_experiment"], indent=2))


if __name__ == "__main__":
    main()
