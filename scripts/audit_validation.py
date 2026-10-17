"""Independently re-read raw data and saved arrays to audit the completed matrix."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def audit_benchmark(root, data, names=None):
    root, data = Path(root), Path(data)
    checked, paired, dataset_shapes = [], [], []
    names = names or ["ETTh1", "ETTh2", "exchange_rate"]
    for name in names:
        if name == "exchange_rate":
            raw_path = data / f"{name}.txt.gz"
            raw = pd.read_csv(io.BytesIO(gzip.decompress(raw_path.read_bytes())), header=None)
            train_end, val_end = int(len(raw) * .7), len(raw) - int(len(raw) * .2)
        else:
            raw_path = data / f"{name}.csv"
            raw = pd.read_csv(raw_path).iloc[:14400, 1:]
            train_end, val_end = 8640, 11520
        # pandas column reductions provide an independent implementation of the
        # runner's NumPy axis reductions; compare after float32 conversion.
        mean = raw.iloc[:train_end].mean().to_numpy(dtype=float)
        std = raw.iloc[:train_end].std(ddof=0).to_numpy(dtype=float)
        values = ((raw.to_numpy(dtype=float) - mean) / std).astype(np.float32)
        dataset_shapes.append({"dataset": name, "raw_rows": len(raw), "targets": raw.shape[1],
                               "train": [0, train_end], "validation": [train_end, val_end],
                               "test": [val_end, len(raw)]})
        for horizon in [96, 192]:
            folder = root / f"{name}_h{horizon}"
            protocol = read_json(folder / "protocol.json")
            digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            check(digest == protocol["sha256"], f"Raw source checksum {name}")
            np.testing.assert_allclose(mean, protocol["mean"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(std, protocol["std_ddof0"], rtol=1e-12, atol=1e-12)
            expected_origins = np.arange(val_end, len(raw) - horizon + 1)
            baseline = read_json(folder / "zero-shot.json")
            for mode in ["persistence", "zero-shot", "dlinear", "full", "lora"]:
                for seed in ([None] if mode in ["persistence", "zero-shot"] else [42, 43, 44]):
                    label = mode if seed is None else f"{mode}_s{seed}"
                    row = read_json(folder / f"{label}.json")
                    path = Path(row["predictions"])
                    check(hashlib.sha256(path.read_bytes()).hexdigest() == row["predictions_sha256"],
                          f"Prediction file checksum {path}")
                    with np.load(path) as arrays:
                        np.testing.assert_array_equal(arrays["origins"], expected_origins)
                        prediction, actual = arrays["prediction"], arrays["actual"]
                        check(prediction.shape == actual.shape == (len(expected_origins), raw.shape[1], horizon),
                              f"Shape {label}")
                        check(np.isfinite(prediction).all(), f"Finite predictions {label}")
                        truth = np.stack([values[t:t + horizon].T for t in expected_origins])
                        np.testing.assert_allclose(actual, truth, rtol=1e-7, atol=1e-7)
                        residual = prediction.astype(float) - actual.astype(float)
                        absolute, squared = np.abs(residual), residual * residual
                        independent_mae = float(absolute.sum(dtype=np.float64) / residual.size)
                        independent_mse = float(squared.sum(dtype=np.float64) / residual.size)
                        np.testing.assert_allclose([row["mae"], row["mse"], row["rmse"]],
                                                   [independent_mae, independent_mse, np.sqrt(independent_mse)],
                                                   rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(arrays["window_mae"], absolute.mean((1, 2)), rtol=1e-12)
                        np.testing.assert_allclose(arrays["window_mse"], squared.mean((1, 2)), rtol=1e-12)
                        np.testing.assert_allclose(row["per_target_mse"], squared.mean((0, 2)), rtol=1e-12)
                        np.testing.assert_allclose(row["per_lead_mse"], squared.mean((0, 1)), rtol=1e-12)
                        if mode == "persistence":
                            last = np.stack([values[t - 1] for t in expected_origins])[:, :, None]
                            np.testing.assert_array_equal(prediction, np.repeat(last, horizon, axis=-1))
                        check(row["observed_labels"] == residual.size if "observed_labels" in row else
                              row["scored_values_including_overlaps"] == residual.size, "Label count")
                    checked.append({"dataset": name, "horizon": horizon, "mode": mode, "seed": seed,
                                    "windows": len(expected_origins), "scored_values": prediction.size,
                                    "all_labels_and_origins_verified": True, "metrics_verified": True})
                    if mode in ["full", "lora"]:
                        metadata = read_json(path.parents[2] / label / "model/finetune.json")
                        cfg = metadata["train_config"]
                        check(cfg["seed"] == seed and cfg["max_steps"] == 200 and cfg["context_length"] == 96
                              and cfg["prediction_length"] == horizon and cfg["mode"] == mode, "Training configuration")
                        history_path = path.parents[2] / label / "history.jsonl"
                        history = [json.loads(line) for line in history_path.read_text().splitlines()]
                        vals = [r for r in history if "validation_loss" in r]
                        best = min(vals, key=lambda r: r["validation_loss"])
                        check(best["step"] == metadata["best_step"], "Validation-selected checkpoint")
                        paired.append({"dataset": name, "horizon": horizon, "seed": seed, "mode": mode,
                                       "mae_gain_pct": 100 * (1 - row["mae"] / baseline["mae"]),
                                       "mse_gain_pct": 100 * (1 - row["mse"] / baseline["mse"])})
    check(len(checked) == len(names) * 22 and len(paired) == len(names) * 12, "Complete benchmark matrix")
    summary = {}
    for mode in ["full", "lora"]:
        selected = [r for r in paired if r["mode"] == mode]
        summary[mode] = {"runs": len(selected),
                         "mae_wins": sum(r["mae_gain_pct"] > 0 for r in selected),
                         "mse_wins": sum(r["mse_gain_pct"] > 0 for r in selected),
                         "mae_gain_pct_range": [min(r["mae_gain_pct"] for r in selected), max(r["mae_gain_pct"] for r in selected)],
                         "mse_gain_pct_range": [min(r["mse_gain_pct"] for r in selected), max(r["mse_gain_pct"] for r in selected)]}
    return {"passed": True, "benchmark_rows": checked, "dataset_shapes": dataset_shapes,
            "paired_adaptation_results": paired, "adaptation_summary": summary,
            "independence": "raw data/scaler checked using pandas reductions; all labels/origins and median metrics independently re-read from saved NPZ arrays",
            "coverage_limit": "interval coverage is streamed by runner, not independently re-audited from median-only saved arrays",
            "statistical_limit": "wins across related datasets/horizons/seeds are descriptive, not independent significance tests"}


def audit_models(root, artifacts):
    torch.set_num_threads(2)
    root = Path(root)
    rows = []
    for size in ["5M", "28M", "123M", "209M"]:
        for device in ["cpu", "mps"]:
            functional = read_json(root / f"{size}_{device}.json")
            shapes = read_json(root / f"{size}_{device}_shapes.json")
            check(functional["passed"] and shapes["passed"], "Model acceptance")
            check(all(functional["modes"][mode]["passed"] for mode in ["full", "lora"]), "Model training modes")
            check(len(shapes["shapes"]) == 3 and shapes["lora_256_96_backward"], "Representative and boundary shapes")
            for mode in ["full", "lora"]:
                run = Path(artifacts) / f"{size}_{device}" / mode
                direct = torch.load(run / "direct/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
                resumed = torch.load(run / "resumed/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
                check(direct["step"] == resumed["step"] == 4, "Latest resume step")
                check(direct["scheduler"] == resumed["scheduler"], "Scheduler state restoration")
                check(direct["optimizer"]["param_groups"] == resumed["optimizer"]["param_groups"], "Optimizer hyperparameters")
                for section in ["model", "optimizer"]:
                    a, b = direct[section], resumed[section]
                    if section == "optimizer":
                        a, b = a["state"], b["state"]
                    check(a.keys() == b.keys(), "Resume state keys")
                    for key in a:
                        aa, bb = a[key], b[key]
                        for av, bv in ([(aa, bb)] if isinstance(aa, torch.Tensor) else
                                       [(aa[k], bb[k]) for k in aa]):
                            torch.testing.assert_close(av, bv, rtol=0 if device == "cpu" else 1e-5,
                                                       atol=0 if device == "cpu" else 1e-6)
                check(direct["rng"]["python"] == resumed["rng"]["python"], "Python RNG")
                check(direct["rng"]["numpy"] == resumed["rng"]["numpy"], "NumPy RNG")
                for key in ["torch", "sampler", "mps"]:
                    if key in direct["rng"] and direct["rng"][key] is not None:
                        torch.testing.assert_close(direct["rng"][key], resumed["rng"][key], rtol=0, atol=0)
                base_path = hf_hub_download(f"mistralai/Timestral-{size}", "model.safetensors",
                                            revision=functional["revision"], local_files_only=True)
                changed = 0
                with safe_open(base_path, framework="pt", device="cpu") as base:
                    for key, tensor in direct["model"].items():
                        if key.endswith(("lora_A", "lora_B")):
                            continue
                        original = base.get_tensor(key.replace(".base.", "."))
                        different = not torch.equal(tensor, original)
                        if mode == "lora":
                            check(not different, "Real-weight LoRA frozen base")
                        changed += different
                if mode == "full":
                    check(changed > 0, "Real full-parameter updates")
                else:
                    check(any(t.abs().sum() > 0 for key, t in direct["model"].items() if key.endswith("lora_B")),
                          "Real LoRA adapter updated from zero")
                del direct, resumed
            rows.append({"size": size, "device": device, "functional": True, "shape_checks": True,
                         "latest_checkpoint_model_optimizer_scheduler_rng_verified": True,
                         "real_full_updates_and_lora_frozen_base_verified": True})
    return {"passed": True, "rows": rows, "cuda": "unverified; no NVIDIA hardware",
            "limits": "functional correctness and finite gradients at selected shapes; not large-model quality or maximum-capacity testing"}


def audit_transfer(root, data, benchmark_artifacts):
    root, data = Path(root), Path(data)
    checked, paired = [], []
    for name in ["ETTh2", "exchange_rate"]:
        if name == "exchange_rate":
            raw = pd.read_csv(io.BytesIO(gzip.decompress((data / f"{name}.txt.gz").read_bytes())), header=None)
            train_end, start = int(len(raw) * .7), len(raw) - int(len(raw) * .2)
        else:
            raw = pd.read_csv(data / f"{name}.csv").iloc[:14400, 1:]
            train_end, start = 8640, 11520
        values = ((raw.to_numpy(dtype=float) - raw.iloc[:train_end].mean().to_numpy())
                  / raw.iloc[:train_end].std(ddof=0).to_numpy()).astype(np.float32)
        for horizon in [96, 192]:
            folder = root / f"ETTh1_to_{name}_h{horizon}"
            baseline = read_json(folder / "zero-shot.json")
            protocol = read_json(folder / "protocol.json")
            check(protocol["target_dataset_optimizer_steps"] == 0, "Frozen transfer")
            expected_origins = np.arange(start, len(raw) - horizon + 1)
            truth = np.stack([values[t:t + horizon].T for t in expected_origins])
            for mode in ["zero-shot", "full", "lora"]:
                for seed in ([None] if mode == "zero-shot" else [42, 43, 44]):
                    label = mode if seed is None else f"{mode}_s{seed}"
                    row = read_json(folder / f"{label}.json")
                    path = Path(row["predictions"])
                    check(hashlib.sha256(path.read_bytes()).hexdigest() == row["predictions_sha256"], "Transfer prediction checksum")
                    with np.load(path) as a:
                        np.testing.assert_array_equal(a["origins"], expected_origins)
                        np.testing.assert_allclose(a["actual"], truth, rtol=1e-7, atol=1e-7)
                        error = a["prediction"].astype(float) - a["actual"].astype(float)
                        check(np.isfinite(error).all(), "Transfer finite predictions")
                        np.testing.assert_allclose([row["mae"], row["mse"]],
                                                   [np.abs(error).mean(), (error * error).mean()], rtol=1e-12)
                    if mode != "zero-shot":
                        source = Path(benchmark_artifacts) / f"ETTh1_h{horizon}" / label / "model/model.safetensors"
                        check(hashlib.sha256(source.read_bytes()).hexdigest() == row["source_weights_sha256"],
                              "Frozen source weight provenance")
                        paired.append({"target_dataset": name, "horizon": horizon, "mode": mode, "seed": seed,
                                       "mae_gain_pct": 100 * (1 - row["mae"] / baseline["mae"]),
                                       "mse_gain_pct": 100 * (1 - row["mse"] / baseline["mse"])})
                    checked.append({"target_dataset": name, "horizon": horizon, "mode": mode, "seed": seed,
                                    "all_labels_origins_and_metrics_verified": True})
    check(len(checked) == 28 and len(paired) == 24, "Complete frozen-transfer matrix")
    return {"passed": True, "rows": checked, "paired_results": paired,
            "limit": "Target training-only scalers are available, but target dataset labels never update network weights"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/validation")
    parser.add_argument("--data", default="data/validation")
    parser.add_argument("--model-artifacts", default="runs/validation/models")
    parser.add_argument("--benchmark-artifacts", default="runs/validation/benchmark")
    args = parser.parse_args()
    root = Path(args.root)
    result = {"benchmark": audit_benchmark(root / "benchmark", args.data),
              "models": audit_models(root / "models", args.model_artifacts),
              "transfer": audit_transfer(root / "transfer", args.data, args.benchmark_artifacts)}
    write = root / "audit.json"
    write.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"benchmark_rows_verified": len(result["benchmark"]["benchmark_rows"]),
                      "adaptation": result["benchmark"]["adaptation_summary"],
                      "frozen_transfer_rows_verified": len(result["transfer"]["rows"]),
                      "model_device_cases": len(result["models"]["rows"])}, indent=2))


if __name__ == "__main__":
    main()
