"""Rolling backtests using identical official inference settings for every model."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import FineTunedModel, prediction_frame, write_json
from .compat import PortableForecaster
from .data import WindowDataset, collate


@dataclass
class EvaluationResult:
    metrics: dict
    predictions: pd.DataFrame
    metadata: dict

    def save(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.predictions.to_parquet(output / "predictions.parquet", index=False)
        self.predictions.to_csv(output / "predictions.csv", index=False)
        write_json(output / "metrics.json", self.metrics)
        write_json(output / "metadata.json", self.metadata)
        return output


def evaluate(model, data, *, history=None, stride=None, output_dir=None,
             symmetric_averaging=True, clip_nonnegative=True):
    if not isinstance(model, FineTunedModel):
        model = FineTunedModel.from_pretrained(model)
    cfg = model.train_config
    stride = cfg.prediction_length if stride is None else stride
    dataset = WindowDataset(model.processor, data, cfg.context_length, cfg.prediction_length,
                            history=history, stride=stride)
    device = next(model.model.parameters()).device
    forecaster = PortableForecaster(model.model.eval(), max_context=cfg.context_length)
    records = []
    started = time.perf_counter()
    for i in range(len(dataset)):
        batch = collate([dataset[i]], device)
        labels = batch.pop("labels")[0].cpu().numpy()
        supported = np.isfinite(batch["context"][0].cpu().numpy()).any(-1)
        labels[~supported] = np.nan
        forecast = forecaster.forecast(prediction_length=cfg.prediction_length, **batch,
                                       symmetric_averaging=symmetric_averaging,
                                       clip_nonnegative=clip_nonnegative).quantiles[0].cpu().numpy()
        group_index, origin = dataset.index[i]
        records.extend(prediction_frame(model.processor, dataset.groups[group_index], origin,
                                        cfg.prediction_length, forecast, labels))
    elapsed = time.perf_counter() - started
    predictions = pd.DataFrame(records)
    valid = predictions["actual"].notna()
    target = predictions.loc[valid, "actual"].to_numpy(dtype=float)
    q = np.array(model.model.config.quantile_knots)
    quantiles = predictions.loc[valid, [f"q_{x:g}" for x in q]].to_numpy(dtype=float)
    if not len(target) or not np.isfinite(quantiles).all():
        raise FloatingPointError("Backtest contains no labels or non-finite predictions")
    error = target[:, None] - quantiles
    pinball = np.maximum(error * q, error * (q - 1))
    denominator = np.abs(target).sum()
    point_error = target - predictions.loc[valid, "median"].to_numpy()
    metrics = {"mae": float(np.abs(point_error).mean()),
               "mse": float(np.square(point_error).mean()),
               "rmse": float(np.sqrt(np.square(point_error).mean())),
               "wql": float(2 * pinball.sum() / (len(q) * denominator)) if denominator > 0 else None,
               "coverage_80": float(((target >= predictions.loc[valid, "q_0.1"].to_numpy())
                                      & (target <= predictions.loc[valid, "q_0.9"].to_numpy())).mean()),
               "observed_labels": len(target), "windows": len(dataset), "backtest_seconds": elapsed}
    metadata = {"model": model.metadata, "data_fingerprint": dataset.fingerprint(),
                "evaluation_device": str(device),
                "point_metric_scale": "input target units (evaluator applies no dataset-level standardization)",
                "point_estimate": "median",
                "stride": stride, "context_length": cfg.context_length,
                "prediction_length": cfg.prediction_length,
                "symmetric_averaging": symmetric_averaging, "clip_nonnegative": clip_nonnegative,
                "quantile_real_cap_k": 1e4,
                "aggregation": "pooled observed labels; WQL = 2*sum(pinball)/(21*sum(abs(y)))"}
    result = EvaluationResult(metrics, predictions, metadata)
    if output_dir is not None:
        result.save(output_dir)
    return result
