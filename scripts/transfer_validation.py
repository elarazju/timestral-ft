"""Frozen ETTh1-adapted weights on other datasets; no target-dataset optimization.

Each target uses its own training-only scaler and array schema. This is network
weight transfer, not reusing ETTh1's raw-table schema or its normalization stats.
"""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from benchmark_suite import (
    MODEL_ID,
    MODEL_REVISION,
    origins,
    prepare_dataset,
    score_forecaster,
    timestral_predictor,
)

from timestral_ft import FineTunedModel
from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="runs/validation/benchmark")
    parser.add_argument("--data", default="data/validation")
    parser.add_argument("--output", default="runs/validation/transfer")
    parser.add_argument("--results", default="results/validation/transfer")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result_root = Path(args.results)
    result_root.mkdir(parents=True, exist_ok=True)
    for target in ["ETTh2", "exchange_rate"]:
        frame, values, manifest = prepare_dataset(target, args.data)
        for horizon in [96, 192]:
            label = f"ETTh1_to_{target}_h{horizon}"
            report, output = result_root / label, Path(args.output) / label
            report.mkdir(parents=True, exist_ok=True)
            write_json(report / "protocol.json", {**manifest, "source_dataset": "ETTh1", "target_dataset": target,
                       "context": 96, "horizon": horizon, "stride": 1, "seeds": [42, 43, 44],
                       "target_dataset_optimizer_steps": 0,
                       "preprocessing": "target training-only scaler and target column layout; no source raw-table schema reuse",
                       "postprocessing": "median, symmetry=True, nonnegative clipping=False"})
            indices = origins(manifest["test"][0], len(frame), horizon)
            for mode in ["zero-shot", "full", "lora"]:
                for seed in ([None] if mode == "zero-shot" else [42, 43, 44]):
                    name = mode if seed is None else f"{mode}_s{seed}"
                    row_path = report / f"{name}.json"
                    if row_path.exists():
                        continue
                    if mode == "zero-shot":
                        model = load_model(MODEL_ID, revision=MODEL_REVISION)
                        source_hash = None
                    else:
                        source = Path(args.source) / f"ETTh1_h{horizon}" / name / "model"
                        model = FineTunedModel.from_pretrained(source).model
                        source_hash = file_sha256(source / "model.safetensors")
                    predict = timestral_predictor(model, 96)
                    predict.horizon = horizon
                    metrics = score_forecaster(predict, values, indices, 96, horizon, output / name)
                    row = {"source_dataset": "ETTh1", "target_dataset": target, "horizon": horizon,
                           "mode": mode, "seed": seed, "source_weights_sha256": source_hash, **metrics,
                           "predictions": str(output / name / "predictions.npz")}
                    write_json(row_path, row)
                    print(target, horizon, name, row["mae"], row["mse"], flush=True)
                    del predict, model
                    gc.collect()
    rows = [json.loads(p.read_text()) for p in result_root.glob('*_h*/*.json') if p.name != 'protocol.json']
    frame = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows])
    frame.to_csv(result_root / "runs.csv", index=False)
    summary = frame.groupby(["target_dataset", "horizon", "mode"]).agg(
        runs=("mae", "size"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        mse_mean=("mse", "mean"), mse_std=("mse", "std")).reset_index()
    summary.to_csv(result_root / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
