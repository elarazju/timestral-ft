"""Portable model artifacts, adapters, metadata, and tabular forecasting."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file
from timestral.loading import resolve_checkpoint

from .compat import PortableForecaster, check_device, load_model
from .config import UPSTREAM_REVISION, TrainConfig
from .data import INTERNAL_ID, DataProcessor, read_frame
from .lora import adapter_state, apply_lora, merge_lora


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_frame(processor, group, origin, prediction_length, forecast, actual=None):
    cfg = processor.config
    output = []
    for variable, name in enumerate(cfg.target_columns):
        for step in range(prediction_length):
            record = {"timestamp": group.iloc[origin + step][cfg.time_column], "target": name,
                      "origin": group.iloc[origin - 1][cfg.time_column], "step": step + 1,
                      "median": float(forecast[variable, step, 10])}
            if cfg.id_column:
                record[cfg.id_column] = group.iloc[origin][cfg.id_column]
            if actual is not None:
                value = actual[variable, step]
                record["actual"] = float(value) if np.isfinite(value) else np.nan
            for qi, quantile in enumerate((.01, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5,
                                           .55, .6, .65, .7, .75, .8, .85, .9, .95, .99)):
                record[f"q_{quantile:g}"] = float(forecast[variable, step, qi])
            output.append(record)
    return output


class FineTunedModel:
    def __init__(self, model, processor, train_config, metadata):
        self.model = model
        self.processor = processor
        self.train_config = train_config
        self.metadata = metadata

    @classmethod
    def from_pretrained(cls, path, *, device="cpu", prefer_adapter=False):
        device = check_device(device)
        path = Path(path)
        meta = json.loads((path / "finetune.json").read_text())
        processor = DataProcessor.from_dict(json.loads((path / "preprocessing.json").read_text()))
        config = TrainConfig(**meta["train_config"])
        if meta["artifact_type"] == "adapter" or (prefer_adapter and (path / "adapter.safetensors").exists()):
            base = meta["base_model"]
            checkpoint = resolve_checkpoint(base, revision=meta["base_revision"])
            if (meta.get("base_weights_sha256")
                    and file_sha256(checkpoint / "model.safetensors") != meta["base_weights_sha256"]):
                raise ValueError("Base model weights have changed; the adapter requires the original base")
            model = load_model(checkpoint, device=device)
            apply_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout)
            adapter = load_file(path / "adapter.safetensors", device="cpu")
            expected = set(adapter_state(model))
            if set(adapter) != expected:
                raise ValueError("Adapter keys do not match the base model")
            model.load_state_dict(adapter, strict=False)
        else:
            model = load_model(path, device=device)
        return cls(model, processor, config, meta)

    def export(self, output_dir, *, adapter_only=False):
        output = Path(output_dir)
        has_adapter = bool(adapter_state(self.model))
        if adapter_only and not has_adapter:
            raise ValueError("This model has no unmerged LoRA adapter")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Export directory must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        meta = dict(self.metadata)
        meta.update(train_config=self.train_config.to_dict(), upstream_revision=UPSTREAM_REVISION,
                    artifact_type="adapter" if adapter_only else "merged")
        if has_adapter:
            save_file(adapter_state(self.model), output / "adapter.safetensors")
            write_json(output / "adapter_config.json", {
                "base_model": meta["base_model"], "base_revision": meta["base_revision"],
                "rank": self.train_config.lora_rank, "alpha": self.train_config.lora_alpha,
                "dropout": self.train_config.lora_dropout, "targets": ["wQKV", "wO"],
            })
        if not adapter_only:
            merged = merge_lora(copy.deepcopy(self.model).cpu())
            merged.config.save(output / "config.json")
            save_file({key: value.detach().contiguous() for key, value in merged.state_dict().items()},
                      output / "model.safetensors")
        write_json(output / "finetune.json", meta)
        write_json(output / "preprocessing.json", self.processor.to_dict())
        base = meta["base_model"]
        license_reference = (f"https://huggingface.co/{base}/blob/{meta['base_revision'] or 'main'}/LICENSE"
                             if not Path(base).is_absolute() else f"Consult the license in {base}")
        (output / "WEIGHTS_LICENSE.txt").write_text(
            "Model weights and derivatives retain the base model's license.\n"
            f"Base model: {meta['base_model']}\nRevision: {meta['base_revision']}\n"
            f"{license_reference}\n"
            "The Apache-2.0 license of timestral-ft applies to its code, not model weights.\n"
        )
        if meta.get("base_license_text"):
            (output / "BASE_MODEL_LICENSE").write_text(meta["base_license_text"])
        return output

    def predict(self, history, prediction_length=None, *, future_covariates=None,
                symmetric_averaging=True, clip_nonnegative=True):
        """Return a long prediction table. Future target columns are never read."""
        processor, cfg = self.processor, self.processor.config
        horizon = self.train_config.prediction_length if prediction_length is None else prediction_length
        if horizon < 1:
            raise ValueError("prediction_length must be positive")
        historical = processor.normalize(history)
        known = cfg.known_covariates + cfg.categorical_known_covariates
        if known and future_covariates is None:
            raise ValueError("Configured known-future covariates require a future table")
        future = processor.normalize(future_covariates, columns=known) if future_covariates is not None else None
        records = []
        forecaster = PortableForecaster(self.model.eval(), max_context=self.train_config.context_length)
        for sid, past in historical.groupby(processor.id_column, sort=False):
            past = past.sort_values(cfg.time_column)
            dates = pd.date_range(past[cfg.time_column].iloc[-1], periods=horizon + 1,
                                  freq=processor.frequency)[1:]
            if future is not None:
                selected = future[future[processor.id_column] == sid].sort_values(cfg.time_column)
                if len(selected) != horizon or not pd.DatetimeIndex(selected[cfg.time_column]).equals(dates):
                    raise ValueError(f"Future covariates must cover exactly the next {horizon} timestamps: {sid}")
                selected = selected.copy()
                # Ignore any supplied future targets/past-only features.
                selected = selected[[processor.id_column, cfg.time_column] + known]
            else:
                selected = pd.DataFrame({cfg.time_column: dates, processor.id_column: sid})
            combined = pd.concat([past, selected], ignore_index=True)
            window = processor.window(combined, len(past), self.train_config.context_length, horizon)
            window = {name: value[None] for name, value in window.items()}
            forecast = forecaster.forecast(prediction_length=horizon, **window,
                                           symmetric_averaging=symmetric_averaging,
                                           clip_nonnegative=clip_nonnegative).quantiles[0].cpu().numpy()
            records.extend(prediction_frame(processor, combined, len(past), horizon, forecast))
        return pd.DataFrame(records)


def export(model, output_dir, *, adapter_only=False):
    if not isinstance(model, FineTunedModel):
        model = FineTunedModel.from_pretrained(model, prefer_adapter=adapter_only)
    return model.export(output_dir, adapter_only=adapter_only)


def concat_public(processor, *frames):
    combined = pd.concat([read_frame(frame) for frame in frames], ignore_index=True)
    if processor.config.id_column is None and INTERNAL_ID in combined:
        combined = combined.drop(columns=INTERNAL_ID)
    return combined
