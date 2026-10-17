"""Reproducible multi-dataset M-task fine-tuning benchmarks (not a paper leaderboard).

Test every origin at stride 1; standardize using training statistics only. Keep
all seeds and negative results. Official DLinear is downloaded at a pinned commit;
its network is unchanged, while the training loop here is deliberately explicit.
"""

import argparse
import copy
import gc
import gzip
import importlib
import io
import json
import logging
import platform
import random
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, finetune
from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import PortableForecaster, load_model
from timestral_ft.data import DataProcessor

MODEL_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
TSLIB_REVISION = "4e938a1767106324dd753b2a44832bf870a0252e"
ETT_REVISION = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
EXCHANGE_REVISION = "7f402f185cc2435b5e66aed13a3b560ed142e023"
MODEL_ID = "mistralai/Timestral-5M"
DATASETS = {
    "ETTh1": {"sha256": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
              "url": f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_REVISION}/ETT-small/ETTh1.csv"},
    "ETTh2": {"sha256": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
              "url": f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_REVISION}/ETT-small/ETTh2.csv"},
    "exchange_rate": {"sha256": "6a4fbb561dcdee43f071a83badbeec9223a335552278348656738b556f1c2e9d",
                      "url": f"https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/{EXCHANGE_REVISION}/exchange_rate/exchange_rate.txt.gz"},
}
DLINEAR_SHA256 = {"models/DLinear.py": "e03acaf97e70c85ea6bda6776b3a62550f84470afa7bc53dffccb6121626e7eb",
                  "layers/Autoformer_EncDec.py": "48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e"}


def training_standardize(raw, train_end):
    mean, scale = raw[:train_end].mean(0), raw[:train_end].std(0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("Constant training column")
    return ((raw - mean) / scale).astype(np.float32), mean, scale


def prepare_dataset(name, directory):
    spec = DATASETS[name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + (".txt.gz" if name == "exchange_rate" else ".csv"))
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(spec["url"], timeout=60).read())
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    if name == "exchange_rate":
        values = np.loadtxt(io.BytesIO(gzip.decompress(path.read_bytes())), delimiter=",")
        frame = pd.DataFrame(values, columns=[str(i) for i in range(values.shape[1])])
        # Source contains observations, no dates. Synthetic DAILY indices preserve
        # order only; no calendar covariates or inferred real dates are used.
        frame.insert(0, "date", pd.date_range("2000-01-01", periods=len(frame), freq="D"))
        train_end = int(len(frame) * .7)
        validation_end = len(frame) - int(len(frame) * .2)
        frequency = "D"
    else:
        frame = pd.read_csv(path, parse_dates=["date"]).iloc[:14400].copy()
        train_end, validation_end, frequency = 8640, 11520, "h"
    columns = frame.columns[1:].tolist()
    raw = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or frame["date"].duplicated().any():
        raise ValueError("Benchmark expects finite, uniquely indexed observations")
    standardized, mean, scale = training_standardize(raw, train_end)
    frame[columns] = standardized
    manifest = {"dataset": name, **spec, "rows": len(frame), "columns": columns,
                "train": [0, train_end], "validation": [train_end, validation_end],
                "test": [validation_end, len(frame)], "mean": mean.tolist(), "std_ddof0": scale.tolist(),
                "standardization": "fit training partition only; cast to float32 after transform",
                "timestamps": "synthetic observation indices, no calendar inputs" if name == "exchange_rate"
                else "original hourly dates", "frequency": frequency}
    return frame, standardized, manifest


def origins(start, stop, horizon):
    return np.arange(start, stop - horizon + 1, dtype=np.int64)


def windows(values, indices, context, horizon):
    x = np.stack([values[t - context:t].T for t in indices])
    y = np.stack([values[t:t + horizon].T for t in indices])
    return x, y


def score_forecaster(predict, values, indices, context, horizon, output, *, batch_size=32):
    """Save all median forecasts and independent sufficient statistics, not millions of dicts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions, actuals, window_mae, window_mse = [], [], [], []
    covered, has_intervals = 0, False
    started = time.perf_counter()
    for offset in range(0, len(indices), batch_size):
        x, y = windows(values, indices[offset:offset + batch_size], context, horizon)
        pred, interval = predict(x)
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise FloatingPointError(f"Invalid predictions: {pred.shape} versus {y.shape}")
        error = pred.astype(np.float64) - y.astype(np.float64)
        predictions.append(pred)
        actuals.append(y)
        window_mae.extend(np.abs(error).mean((1, 2)))
        window_mse.extend(np.square(error).mean((1, 2)))
        if interval is not None:
            has_intervals = True
            covered += int(((y >= interval[0]) & (y <= interval[1])).sum())
    prediction = np.concatenate(predictions)
    actual = np.concatenate(actuals)
    error = prediction.astype(np.float64) - actual.astype(np.float64)
    metric = {"mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean()),
              "rmse": float(np.sqrt(np.square(error).mean())),
              "coverage_80": covered / error.size if has_intervals else None,
              "windows": len(indices), "scored_values_including_overlaps": error.size,
              "per_target_mae": np.abs(error).mean((0, 2)).tolist(),
              "per_target_mse": np.square(error).mean((0, 2)).tolist(),
              "per_lead_mae": np.abs(error).mean((0, 1)).tolist(),
              "per_lead_mse": np.square(error).mean((0, 1)).tolist(),
              "evaluation_seconds": time.perf_counter() - started}
    np.savez_compressed(output / "predictions.npz", prediction=prediction, actual=actual, origins=indices,
                        window_mae=np.asarray(window_mae), window_mse=np.asarray(window_mse))
    metric["predictions_sha256"] = file_sha256(output / "predictions.npz")
    write_json(output / "metrics.json", metric)
    return metric


def timestral_predictor(model, context):
    forecaster = PortableForecaster(model.eval(), max_context=context)

    def predict(x):
        result = forecaster.forecast(x, predict.horizon, symmetric_averaging=True, clip_nonnegative=False)
        q = result.quantiles.detach().cpu().numpy()
        return q[..., 10], (q[..., 2], q[..., 18])

    return predict


def prepare_dlinear(directory):
    root = Path(directory).resolve()
    for relative in ["models/DLinear.py", "layers/Autoformer_EncDec.py", "LICENSE"]:
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/thuml/Time-Series-Library/{TSLIB_REVISION}/{relative}"
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        if relative in DLINEAR_SHA256 and file_sha256(path) != DLINEAR_SHA256[relative]:
            raise ValueError(f"Official DLinear source checksum mismatch: {path}")
    write_json(root / "provenance.json", {"revision": TSLIB_REVISION,
               "files": {p: file_sha256(root / p) for p in ["models/DLinear.py", "layers/Autoformer_EncDec.py"]}})
    return root


def train_dlinear(values, split, context, horizon, seed, directory, source, threads):
    """Official network; Adam/MSE, 10 epochs, lr=1e-4, type1 LR, patience=3.

    Train shuffled, including the final partial batch (TSLib drop_last=False).
    Validation is stride 1, sequential and label-weighted; TSLib shuffles
    validation and averages batch means. The distinction is recorded explicitly.
    """
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, str(source))
    cls = importlib.import_module("models.DLinear").Model
    model = cls(SimpleNamespace(task_name="long_term_forecast", seq_len=context, pred_len=horizon,
                                moving_avg=25, enc_in=values.shape[1]))
    train_end, val_end = split["train"][1], split["validation"][1]
    tensor = torch.from_numpy(values)
    xs = tensor.unfold(0, context, 1).permute(0, 2, 1)
    ys = tensor.unfold(0, horizon, 1).permute(0, 2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, bad, best_state, history = float("inf"), 0, None, []
    started = time.perf_counter()
    for epoch in range(10):
        model.train()
        selected = torch.randperm(train_end - context - horizon + 1, generator=generator) + context
        losses = []
        for offset in range(0, len(selected), 32):
            idx = selected[offset:offset + 32]
            optimizer.zero_grad(set_to_none=True)
            error = model(xs[idx - context], None, None, None) - ys[idx]
            loss = error.square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        squared, count = 0., 0
        with torch.inference_mode():
            for offset in range(train_end, val_end - horizon + 1, 256):
                idx = torch.arange(offset, min(offset + 256, val_end - horizon + 1))
                error = model(xs[idx - context], None, None, None) - ys[idx]
                squared += float(error.double().square().sum())
                count += error.numel()
        validation = squared / count
        history.append({"epoch": epoch + 1, "training_mse": float(np.mean(losses)), "validation_mse": validation})
        if validation < best:
            best, bad, best_state = validation, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= 3:
            break
        for group in optimizer.param_groups:
            group["lr"] = 1e-4 * .5 ** epoch
    model.load_state_dict(best_state)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, directory / "weights.pt")
    meta = {"training_seconds": time.perf_counter() - started, "epochs": len(history), "history": history,
            "trainable_parameters": sum(p.numel() for p in model.parameters()), "official_revision": TSLIB_REVISION,
            "network_source": "unchanged models/DLinear.py and layers/Autoformer_EncDec.py",
            "training": "Adam MSE lr1e-4 batch32 10epochs type1 schedule patience3; shuffled, final partial batch included",
            "validation_difference_from_tslib": "sequential label-weighted MSE; TSLib shuffles validation and averages batch means",
            "test_protocol": "M, all columns, train-only standardization, stride1; no calendar inputs"}
    write_json(directory / "training.json", meta)

    def predict(x):
        with torch.inference_mode():
            result = model(torch.from_numpy(x.transpose(0, 2, 1).copy()), None, None, None)
        return result.numpy().transpose(0, 2, 1), None

    return predict, meta


def run_case(name, horizon, settings):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(settings["threads"])
    frame, values, manifest = prepare_dataset(name, settings["data"])
    root = Path(settings["output"]) / f"{name}_h{horizon}"
    root.mkdir(parents=True, exist_ok=True)
    report = Path(settings["results"]) / f"{name}_h{horizon}"
    report.mkdir(parents=True, exist_ok=True)
    case_spec = {**manifest, "horizon": horizon, "context": 96, "test_stride": 1,
                 "seeds": settings["seeds"], "steps": settings["steps"], "threads": settings["threads"],
                 "model_revision": MODEL_REVISION, "device": "cpu", "features": "M",
                 "covariates": [], "point_estimate": "median", "clip_nonnegative": False,
                 "symmetric_averaging": True,
                 "timestral_validation": "masked scaled pinball, stride horizon (toolkit default)",
                 "scope": "matched test protocol; not identical pretraining, training budgets or objectives"}
    existing = report / "protocol.json"
    if existing.exists() and json.loads(existing.read_text()) != case_spec:
        raise ValueError(f"Existing benchmark configuration mismatch: {report}")
    write_json(existing, case_spec)
    indices = origins(manifest["test"][0], len(frame), horizon)
    train_end, val_end = manifest["train"][1], manifest["validation"][1]
    data_config = DataConfig(time_column="date", target_columns=manifest["columns"], frequency=manifest["frequency"])
    records = []
    # Every row is recoverable. A completed row is reused only for this exact protocol.
    for mode in settings.get("modes", ["persistence", "zero-shot", "dlinear", "full", "lora"]):
        for seed in ([None] if mode in {"persistence", "zero-shot"} else settings["seeds"]):
            label = mode if seed is None else f"{mode}_s{seed}"
            row_path = report / f"{label}.json"
            if row_path.exists():
                records.append(json.loads(row_path.read_text()))
                continue
            logging.info("START %s h%s %s", name, horizon, label)
            run = root / label
            metadata = {"training_seconds": 0., "trainable_parameters": 0}
            if mode == "persistence":
                def predict(x):
                    return np.repeat(x[..., -1:], horizon, axis=-1), None
            elif mode == "dlinear":
                predict, metadata = train_dlinear(values, manifest, 96, horizon, seed, run,
                                                   Path(settings["dlinear_source"]), settings["threads"])
            else:
                cfg = TrainConfig(mode="full" if mode == "zero-shot" else mode, context_length=96,
                                  prediction_length=horizon, max_steps=settings["steps"], seed=seed or 42,
                                  num_threads=settings["threads"])
                if mode == "zero-shot":
                    artifact = FineTunedModel(load_model(MODEL_ID, revision=MODEL_REVISION),
                                             DataProcessor(data_config).fit(frame.iloc[:train_end]), cfg, metadata)
                elif (run / "model/finetune.json").exists():
                    artifact = FineTunedModel.from_pretrained(run / "model")
                else:
                    artifact = finetune(frame.iloc[:train_end], frame.iloc[train_end:val_end],
                                        model_id=MODEL_ID, revision=MODEL_REVISION, data_config=data_config,
                                        train_config=cfg, output_dir=run,
                                        resume=run / "checkpoint.pt" if (run / "checkpoint.pt").exists() else None)
                metadata = artifact.metadata
                predict = timestral_predictor(artifact.model, 96)
                predict.horizon = horizon
            metrics = score_forecaster(predict, values, indices, 96, horizon, root / "evaluations" / label)
            row = {"dataset": name, "horizon": horizon, "mode": mode, "seed": seed, **metrics,
                   "training_seconds": metadata["training_seconds"],
                   "trainable_parameters": metadata["trainable_parameters"],
                   "best_step": metadata.get("best_step"), "actual_steps": metadata.get("steps"),
                   "predictions": str(root / "evaluations" / label / "predictions.npz")}
            write_json(row_path, row)
            records.append(row)
            logging.info("DONE %s h%s %s MAE=%.6f MSE=%.6f train=%.1fs", name, horizon, label,
                         row["mae"], row["mse"], row["training_seconds"])
            del predict
            if mode in {"zero-shot", "full", "lora"}:
                del artifact
            gc.collect()
    return records


def summarize(directory):
    directory = Path(directory)
    rows = [json.loads(path.read_text()) for path in directory.glob("*_h*/*.json")
            if path.name != "protocol.json"]
    if not rows:
        raise ValueError("No completed benchmark rows")
    df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
    df.to_csv(directory / "runs.csv", index=False)
    grouped = df.groupby(["dataset", "horizon", "mode"], sort=True)
    summary = grouped.agg(runs=("mae", "size"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                          mse_mean=("mse", "mean"), mse_std=("mse", "std"),
                          training_seconds_mean=("training_seconds", "mean")).reset_index()
    base = df[df["mode"] == "zero-shot"].set_index(["dataset", "horizon"])
    summary["mae_gain_pct_vs_base"] = [100 * (1 - row.mae_mean / base.loc[(row.dataset, row.horizon), "mae"])
                                        for row in summary.itertuples()]
    summary["mse_gain_pct_vs_base"] = [100 * (1 - row.mse_mean / base.loc[(row.dataset, row.horizon), "mse"])
                                        for row in summary.itertuples()]
    summary.to_csv(directory / "summary.csv", index=False)
    lines = ["# Multi-dataset / multivariate validation", "",
             "M task: all columns, L=96, H=96/192, test stride=1, training-only per-column standardization.",
             "Timestral uses median forecasts, symmetry averaging, nonnegative clipping disabled for signed standardized inputs.",
             "All test origins are included. Training selects checkpoints on validation only; test scores never choose a run.",
             "Three seeds (42/43/44); mean ± sample standard deviation describes seed variability, not statistical significance.", "",
             "| Dataset | H | Method | Runs | MAE mean ± std | MSE mean ± std | MAE gain vs base | MSE gain vs base |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    def spread(x):
        return "—" if pd.isna(x) else f"{x:.5f}"

    for row in summary.itertuples():
        lines.append(f"| {row.dataset} | {row.horizon} | {row.mode} | {row.runs} | "
                     f"{row.mae_mean:.5f} ± {spread(row.mae_std)} | {row.mse_mean:.5f} ± {spread(row.mse_std)} | "
                     f"{row.mae_gain_pct_vs_base:+.2f}% | {row.mse_gain_pct_vs_base:+.2f}% |")
    lines += ["", "Positive gain means lower error; all regressions are retained.", "",
              "Timestral: full/LoRA up to 200 steps, batch4 × accumulation4, AdamW, original toolkit defaults.",
              "Its validation uses stride H and scaled pinball; DLinear uses stride1 validation MSE.",
              "DLinear: unchanged official TSLib network; local training loop, Adam lr1e-4, batch32, 10 epochs, type1 decay, patience3.",
              "DLinear validation uses sequential label-weighted MSE; TSLib shuffles validation and averages batch means. Both retain the final partial batch.",
              "DLinear is a reference run using the official network, not a paper reproduction.",
              "Training budgets/objectives differ; this is a matched test-protocol comparison, not an equal-compute ranking.",
              "ETTh1/2 use standard 8640/2880/2880 splits. Exchange uses 70/10/20% with integer train/test boundaries.",
              "Exchange timestamps are synthetic ordered indices; no calendar covariates are used by any method.",
              "Only two domains and short adaptation budgets are covered. Pretraining overlap is unknown.",
              "Overlapping test windows are not independent samples; no significance claims or naive label-level confidence intervals.",
              "Per-case protocol JSON records source hashes and scaler statistics; per-run JSON includes per-target/lead scores.",
              "All median predictions, labels and origins are saved under runs/validation; these large files are not packaged.", ""]
    (directory / "README.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--data", default="data/validation")
    parser.add_argument("--output", default="runs/validation/benchmark")
    parser.add_argument("--results", default="results/validation/benchmark")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dlinear-only", action="store_true", help="Run only DLinear references with the same protocol")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    Path(args.results).mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(summarize(args.results).to_string(index=False))
        return
    source = prepare_dlinear(".cache/validation-sources/tslib")
    for dataset in args.datasets:
        prepare_dataset(dataset, args.data)
    settings = {**vars(args), "dlinear_source": str(source)}
    if args.dlinear_only:
        settings["modes"] = ["dlinear"]
    environment_settings = {**settings, "dlinear_source": ".cache/validation-sources/tslib"}
    write_json(Path(args.results) / ("dlinear_environment.json" if args.dlinear_only else "environment.json"), {
        "python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pandas": pd.__version__, "settings": environment_settings,
        "model_revision": MODEL_REVISION, "tslib_revision": TSLIB_REVISION,
        "dlinear_sources": json.loads((source / "provenance.json").read_text()),
    })
    cases = [(name, horizon) for name in args.datasets for horizon in args.horizons]
    if args.jobs == 1:
        for name, horizon in cases:
            run_case(name, horizon, settings)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_case, name, horizon, settings) for name, horizon in cases]
            for future in as_completed(futures):
                future.result()
    print(summarize(args.results).to_string(index=False))


if __name__ == "__main__":
    main()
