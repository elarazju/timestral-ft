"""Thin CLI over the public Python API."""

import argparse
import json
import logging
from pathlib import Path

import yaml

from . import DataConfig, FineTunedModel, TrainConfig, evaluate, export, finetune


def main(argv=None):
    parser = argparse.ArgumentParser(description="Community full/LoRA fine-tuning for Timestral")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train using YAML configuration")
    train.add_argument("--config", required=True)
    train.add_argument("--mode", choices=["full", "lora"])
    train.add_argument("--device", help="cpu, cuda[:index], or mps; overrides YAML")
    train.add_argument("--output")
    train.add_argument("--resume")
    train.add_argument("--stop-after-steps", type=int)
    backtest = sub.add_parser("evaluate", help="Run a chronological rolling backtest")
    backtest.add_argument("--model", required=True)
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--history")
    backtest.add_argument("--stride", type=int)
    backtest.add_argument("--device", default="cpu")
    backtest.add_argument("--output", required=True)
    save = sub.add_parser("export", help="Export merged official-compatible weights or an adapter")
    save.add_argument("--model", required=True)
    save.add_argument("--output", required=True)
    save.add_argument("--adapter-only", action="store_true")
    predict = sub.add_parser("predict", help="Forecast using a saved model and its preprocessing")
    predict.add_argument("--model", required=True)
    predict.add_argument("--history", required=True)
    predict.add_argument("--future")
    predict.add_argument("--horizon", type=int)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "train":
            payload = yaml.safe_load(Path(args.config).read_text())
            training = payload.get("training", {})
            if args.mode:
                training["mode"] = args.mode
            if args.device:
                training["device"] = args.device
            result = finetune(payload["train_data"], payload["validation_data"],
                              model_id=payload.get("model_id", "mistralai/Timestral-5M"),
                              revision=payload.get("revision"), data_config=DataConfig(**payload["data"]),
                              train_config=TrainConfig(**training),
                              output_dir=args.output or payload.get("output_dir", "runs/finetune"),
                              resume=args.resume, stop_after_steps=args.stop_after_steps)
            print(json.dumps(result.metadata, indent=2))
        elif args.command == "evaluate":
            model = FineTunedModel.from_pretrained(args.model, device=args.device)
            result = evaluate(model, args.data, history=args.history, stride=args.stride,
                              output_dir=args.output)
            print(json.dumps(result.metrics, indent=2))
        elif args.command == "export":
            print(export(args.model, args.output, adapter_only=args.adapter_only))
        elif args.command == "predict":
            model = FineTunedModel.from_pretrained(args.model, device=args.device)
            predictions = model.predict(args.history, args.horizon, future_covariates=args.future)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() in {".parquet", ".pq"}:
                predictions.to_parquet(output, index=False)
            else:
                predictions.to_csv(output, index=False)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
