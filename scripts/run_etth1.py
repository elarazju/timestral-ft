"""Pinned, single-device domain adaptation demonstration (not a zero-shot leaderboard)."""

import argparse
import hashlib
import json
import logging
import platform
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, evaluate, finetune
from timestral_ft.artifacts import write_json
from timestral_ft.compat import check_device, load_model
from timestral_ft.data import DataProcessor

DATA_REVISION = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
DATA_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
MODEL_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
MODEL_ID = "mistralai/Timestral-5M"
URL = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{DATA_REVISION}/ETT-small/ETTh1.csv"


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "ETTh1.csv"
    if not raw.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            raw.write_bytes(response.read())
    if hashlib.sha256(raw.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("ETTh1 checksum mismatch; remove the cached CSV and retry")
    frame = pd.read_csv(raw, parse_dates=["date"]).iloc[:14400].copy()
    hour = frame["date"].dt.hour
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["hour_category"] = hour.astype(str)
    frame["weekday_category"] = frame["date"].dt.dayofweek.astype(str)
    partitions = {"train": frame.iloc[:8640], "validation": frame.iloc[8640:11520],
                  "test": frame.iloc[11520:14400], "history": frame.iloc[:11520]}
    for name, part in partitions.items():
        part.to_parquet(directory / f"{name}.parquet", index=False)
    write_json(directory / "manifest.json", {
        "url": URL, "revision": DATA_REVISION, "sha256": DATA_SHA256,
        "split_rows": {"train": [0, 8640], "validation": [8640, 11520], "test": [11520, 14400]},
        "split_convention": "12/4/4 blocks of 30 days, hourly (standard ETT convention)",
        "known_features": "hour sine/cosine and hour/weekday categories derived only from timestamps",
    })
    return partitions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/etth1")
    parser.add_argument("--output")
    parser.add_argument("--results")
    parser.add_argument("--device", default="cpu", help="cpu, cuda[:index], or mps")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    frames = prepare(args.data_dir)
    if args.prepare_only:
        print(f"Prepared verified ETTh1 data in {args.data_dir}")
        return
    data_config = DataConfig(time_column="date", target_columns=["OT"], past_covariates=["HUFL"],
                             known_covariates=["hour_sin", "hour_cos"],
                             categorical_known_covariates=["hour_category", "weekday_category"],
                             frequency="h")
    check_device(args.device)
    train_config = TrainConfig(max_steps=args.steps, device=args.device)
    torch.set_num_threads(train_config.num_threads)
    processor = DataProcessor(data_config).fit(frames["train"])
    baseline = FineTunedModel(load_model(MODEL_ID, revision=MODEL_REVISION, device=args.device), processor, train_config,
                             {"base_model": MODEL_ID, "base_revision": MODEL_REVISION,
                              "trainable_parameters": 0, "training_seconds": 0, "mode": "zero-shot"})
    device_slug = args.device.replace(":", "-")
    root = Path(args.output or ("runs/etth1" if args.device == "cpu" else f"runs/etth1_{device_slug}"))
    result_root = Path(args.results or f"results/etth1_{device_slug}")
    result_root.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for name in ["zero-shot", "full", "lora"]:
        if name == "zero-shot":
            artifact = baseline
        else:
            run = root / name
            summary_path = run / "training_summary.json"
            if summary_path.exists() and json.loads(summary_path.read_text())["completed"]:
                artifact = FineTunedModel.from_pretrained(run / "model", device=args.device)
                if artifact.train_config.max_steps != args.steps or artifact.train_config.device != args.device:
                    raise ValueError("Existing run has a different step budget/device; choose a new output directory")
            else:
                artifact = finetune(frames["train"], frames["validation"], model_id=MODEL_ID,
                                    revision=MODEL_REVISION, data_config=data_config,
                                    train_config=TrainConfig(mode=name, max_steps=args.steps, device=args.device), output_dir=run,
                                    resume=run / "checkpoint.pt" if (run / "checkpoint.pt").exists() else None)
        logging.info("Backtesting %s", name)
        result = evaluate(artifact, frames["test"], history=frames["history"], stride=96,
                          output_dir=root / "evaluations" / name)
        row = {"mode": name, **result.metrics, "training_seconds": artifact.metadata["training_seconds"],
               "trainable_parameters": artifact.metadata["trainable_parameters"],
               "best_step": artifact.metadata.get("best_step", 0)}
        comparisons.append(row)
        write_json(result_root / f"{name}.json", {"metrics": row, "metadata": result.metadata})
        logging.info("%s", json.dumps(row))
    pd.DataFrame(comparisons).to_csv(result_root / "comparison.csv", index=False)
    write_json(result_root / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
        "device": args.device, "threads": train_config.num_threads, "model_revision": MODEL_REVISION,
        "data_sha256": DATA_SHA256, "max_steps": args.steps,
        "notes": "Single seed; adaptation demonstration. Dataset may overlap with pretraining."
    })
    command = (f"python scripts/run_etth1.py --device {args.device} --steps {args.steps} "
               f"--output {root} --results {result_root}")
    lines = [f"# ETTh1 {args.device.upper()} adaptation experiment", "",
             f"Reproduce with `{command}` from the repository root.", "",
             f"Timestral-5M; context 256; horizon/stride 96; seed 42; up to {args.steps} optimizer steps per mode.",
             "Training uses only the train split; checkpoints are selected on validation loss.", "",
             "| Mode | MAE (original OT units) ↓ | WQL ↓ | 80% coverage | Trainable parameters | Training seconds | Best step |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['mode']} | {row['mae']:.6f} | {row['wql']:.6f} | "
                     f"{row['coverage_80']:.2%} | {row['trainable_parameters']:,} | "
                     f"{row['training_seconds']:.2f} | {row['best_step']} |")
    lines += ["", f"Actual step budget: {args.steps}. Test labels: 2,880 over 30 rolling origins.", "",
              "The standard ETT 12/4/4 split uses blocks of 30 days, not calendar months.",
              "Train [0,8640), validation [8640,11520), test [11520,14400).",
              "Target OT; historical HUFL; known-future hour sine/cosine and hour/weekday categories.", "",
              "MAE is in original OT units. It is not the train-standardized MAE/MSE reported by default",
              "in Time-Series-Library. This OT-only, context-256, stride-96 experiment is not the common",
              "seven-target M, context-96, stride-1 benchmark. Do not compare those values directly.", "",
              f"Data revision: `{DATA_REVISION}`. SHA256: `{DATA_SHA256}`.",
              f"Base weights revision: `{MODEL_REVISION}`.", "",
              "MAE and coverage pool observed labels. WQL = 2*sum(pinball)/(21*sum(abs(actual))).",
              "All models use the same official default postprocessing. Training timing includes validation",
              "and checkpoint writes but excludes data preparation/model loading. Timing depends on hardware and load.",
              "See environment.json for exact software versions and device; JSON files contain run metadata.",
              f"Detailed rolling forecasts are generated under {root}/evaluations and are not versioned.", "",
              "This is one dataset and one seed, not a general ranking. ETTh1 may overlap with pretraining data.",
              "Hyperparameters and checkpoints are not selected using the test split. Improvements are not guaranteed.", ""]
    (result_root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
