"""Leakage-safe, variable-history sampling and resumable GIFT-Eval adaptation.

GIFT-Eval itself is an optional, separately pinned evaluation dependency. This
module consumes its official Dataset rather than interpreting the HF train name.
"""

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from timestral.loading import resolve_checkpoint

from .artifacts import file_sha256
from .compat import check_device, differentiable_forecast, load_model, pinball_loss
from .config import UPSTREAM_REVISION
from .lora import adapter_state, apply_lora
from .trainer import _restore_rng, _rng_state, clip_gradients

GIFT_REVISION = "8e6da595ac943be01e0848a02b9157af84731284"
DATA_REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
BASE_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


def bounded_contexts(requested, max_history, input_rows, row_context_budget=65536, patch_size=32):
    """Patch-aligned training caps determined only by history shape/resources.

    This limits context times target+covariate rows in one example. It never
    removes variates or shortens validation/test inference history.
    """
    if min(max_history, input_rows, row_context_budget, patch_size) < 1 or not requested:
        raise ValueError("Context requests, history, rows and budget must be positive")
    if any(c < patch_size or c % patch_size for c in requested):
        raise ValueError("Requested contexts must align to a complete patch")
    resource_cap = (row_context_budget // input_rows // patch_size) * patch_size
    if resource_cap < patch_size:
        raise ValueError("Row budget cannot accommodate even one history patch")
    cap = min(math.ceil(max_history / patch_size) * patch_size, resource_cap)
    return sorted({min(c, cap) for c in requested})


def seasonal_error_sliced(time_series, seasonality, time_axis=0):
    """GluonTS seasonal_error, replacing contiguous range gathers with slices.

    Identical full-history inputs, masks and mean reduction; avoids rebuilding
    long Python index ranges for every rolling origin. No history is truncated.
    """
    length = time_series.shape[time_axis]
    if seasonality > length:
        seasonality = 1
    current, previous = [slice(None)] * time_series.ndim, [slice(None)] * time_series.ndim
    current[time_axis] = slice(seasonality, length)
    previous[time_axis] = slice(0, length - seasonality)
    return np.abs(time_series[tuple(current)] - time_series[tuple(previous)]).mean(
        axis=time_axis, keepdims=True)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _matrix(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2:
        raise ValueError("Expected (variates, time) arrays")
    return value


def _padded(array, length):
    result = np.full((array.shape[0], length), np.nan, dtype=np.float32)
    if array.shape[-1]:
        result[:, -min(length, array.shape[-1]):] = array[:, -length:]
    return result


class GiftWindows:
    """Keep only official training prefixes; sample series/origins on demand.

    All series with >=2 timestamps are eligible. For a training prefix shorter
    than H+1, labels beyond its end are masked, never borrowed from validation.
    Validation uses the one official H-step suffix and its preceding history.
    """

    def __init__(self, dataset, context_length, validation_limit=512, seed=42):
        self.context_length = context_length
        self.horizon = dataset.prediction_length
        self.train, self.validation = [], []
        self.validation_indices = []
        digest = hashlib.sha256()
        for entry in dataset.training_dataset:
            target = _matrix(entry["target"])
            cov = entry.get("past_feat_dynamic_real")
            cov = _matrix(cov) if cov is not None else None
            if cov is not None and cov.shape[-1] != target.shape[-1]:
                raise ValueError("Past covariates must end at the training cutoff")
            # Copies release any view of withheld values in the source array.
            target = target.copy()
            cov = cov.copy() if cov is not None else None
            self.train.append((target, cov))
            for array in (target, cov):
                if array is not None:
                    digest.update(str(array.shape).encode())
                    digest.update(array.tobytes())
        if not self.train:
            raise ValueError("Empty official training partition")
        self.eligible = [i for i, (x, _) in enumerate(self.train) if x.shape[-1] >= 2]
        if not self.eligible:
            raise ValueError("No series with at least one history and one training label")
        generator = torch.Generator().manual_seed(seed)
        selected = set(torch.randperm(len(self.train), generator=generator)[:validation_limit].tolist())
        count = 0
        for i, entry in enumerate(dataset.validation_dataset):
            count += 1
            if i not in selected:
                continue
            target = _matrix(entry["target"])
            training, cov = self.train[i]
            cutoff = training.shape[-1]
            if target.shape[-1] != cutoff + self.horizon:
                raise ValueError("Validation must append exactly H labels to the training prefix")
            if not np.array_equal(target[:, :cutoff], training, equal_nan=True):
                raise ValueError("Official training and validation prefixes disagree")
            labels = target[:, cutoff:].copy()
            self.validation_indices.append(i)
            self.validation.append(self.window(training, cov, cutoff, labels=labels))
            digest.update(labels.tobytes())
        if count != len(self.train):
            raise ValueError("Training/validation series counts disagree")
        self.fingerprint = digest.hexdigest()
        self.summary = {
            "series": len(self.train), "eligible_training_series": len(self.eligible),
            "validation_series": len(self.validation), "horizon": self.horizon,
            "train_length_min": min(x.shape[-1] for x, _ in self.train),
            "train_length_max": max(x.shape[-1] for x, _ in self.train),
            "validation_sampling": "seeded uniform series, fixed before any model evaluation",
            "series_sampling": "uniform series then uniform origin; partial labels only for T < H+1",
        }

    def window(self, target, covariates, origin, labels=None):
        if not 1 <= origin <= target.shape[-1]:
            raise ValueError("Origin must follow observed history")
        context = _padded(target[:, max(0, origin - self.context_length):origin], self.context_length)
        if labels is None:
            labels = target[:, origin:origin + self.horizon]
        result = {"context": context, "labels": np.full(
            (target.shape[0], self.horizon), np.nan, dtype=np.float32)}
        result["labels"][:, :labels.shape[-1]] = labels
        if covariates is not None:
            result["past_covariates"] = _padded(
                covariates[:, max(0, origin - self.context_length):origin], self.context_length)
        return result

    def sample(self, generator):
        i = self.eligible[int(torch.randint(len(self.eligible), (), generator=generator))]
        target, cov = self.train[i]
        last_origin = max(1, target.shape[-1] - self.horizon)
        origin = int(torch.randint(1, last_origin + 1, (), generator=generator))
        return self.window(target, cov, origin)


def gift_batch(windows, device):
    return {key: torch.from_numpy(np.stack([w[key] for w in windows])).to(device)
            for key in windows[0]}


def _reconcile_history(path, checkpoint_step):
    """Roll back log writes made after the last atomically committed checkpoint."""
    records = {}
    lines = path.read_text().splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            break  # an interrupted final log write cannot have a committed checkpoint
        if row["step"] <= checkpoint_step:
            previous = records.get(row["step"])
            if previous is not None and any(previous[k] != row[k] for k in previous.keys() & row.keys()):
                raise ValueError("Conflicting training log records")
            records[row["step"]] = row
    if set(records) != set(range(1, checkpoint_step + 1)):
        raise ValueError("Training history does not cover the committed checkpoint")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(records[i], allow_nan=False) + "\n"
                                 for i in range(1, checkpoint_step + 1)))
    temporary.replace(path)


@dataclass
class GiftTrainConfig:
    context_length: int = 256
    max_steps: int = 200
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    gradient_clip: float = 1.0
    eval_every: int = 50
    patience: int = 3
    validation_limit: int = 512
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 4
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self):
        for key in ("context_length", "max_steps", "batch_size", "gradient_accumulation",
                    "eval_every", "patience", "validation_limit", "num_threads", "rank"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if self.warmup_steps < 0 or self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid warmup or adapter settings")


def validation_loss(model, windows, cfg):
    model.eval()
    numerator, denominator = 0., 0
    with torch.no_grad():
        for start in range(0, len(windows.validation), cfg.batch_size):
            batch = gift_batch(windows.validation[start:start + cfg.batch_size], cfg.device)
            labels = batch.pop("labels")
            predicted = differentiable_forecast(model, prediction_length=windows.horizon, **batch)
            loss = pinball_loss(predicted, labels, batch["context"], model.config.quantile_knots)
            if loss is not None:
                weight = int((torch.isfinite(labels)
                              & torch.isfinite(batch["context"]).any(-1)[..., None]).sum())
                numerator += float(loss) * weight
                denominator += weight
    if not denominator or not math.isfinite(numerator):
        raise FloatingPointError("No finite validation objective")
    return numerator / denominator


def load_gift_adapter(directory, device="cpu"):
    directory = Path(directory)
    metadata = json.loads((directory / "adapter_config.json").read_text())
    path = resolve_checkpoint(metadata["base_model"], revision=metadata["base_revision"])
    if file_sha256(path / "model.safetensors") != metadata["base_weights_sha256"]:
        raise ValueError("Base checkpoint checksum differs")
    model = load_model(path, device=device)
    cfg = GiftTrainConfig(**metadata["train_config"])
    apply_lora(model, cfg.rank, cfg.alpha, cfg.dropout)
    state = load_file(directory / "adapter.safetensors")
    if set(state) != set(adapter_state(model)):
        raise ValueError("Adapter keys differ from the configured network")
    model.load_state_dict(state, strict=False)
    return model.eval()


def train_gift_task(dataset, *, task_key, output_dir, config=None,
                    model_id="mistralai/Timestral-5M", revision=BASE_REVISION,
                    stop_after_steps=None, validation_fn=None, validation_metadata=None,
                    include_initial=False, retain_validation_checkpoints=False):
    """Train one independent LoRA; exact-resume only its adapter/optimizer state."""
    cfg = config or GiftTrainConfig()
    check_device(cfg.device)
    torch.set_num_threads(cfg.num_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("mps"):
        torch.mps.manual_seed(cfg.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    windows = GiftWindows(dataset, cfg.context_length, cfg.validation_limit, cfg.seed)
    path = resolve_checkpoint(model_id, revision=revision)
    base_model = str(Path(model_id).resolve()) if Path(model_id).is_dir() else model_id
    base_revision = path.name if path.parent.name == "snapshots" else revision
    metadata = {
        "task": task_key, "base_model": base_model, "base_revision": base_revision,
        "base_weights_sha256": file_sha256(path / "model.safetensors"),
        "upstream_revision": UPSTREAM_REVISION, "gift_revision": GIFT_REVISION,
        "data_revision": DATA_REVISION, "train_config": asdict(cfg),
        "data_fingerprint": windows.fingerprint, "data": windows.summary,
        "selection_objective": "history-scaled masked pinball over all 21 quantiles",
    }
    if validation_fn is not None:
        if not validation_metadata:
            raise ValueError("Custom validation requires reproducible validation_metadata")
        metadata["selection_objective"] = validation_metadata
    if include_initial:
        metadata["include_initial_checkpoint"] = True
    if retain_validation_checkpoints:
        metadata["retain_validation_checkpoints"] = True
    # Initialize adapters on CPU for the same seeded A matrices on every backend.
    # Completed CPU tasks used this same initialization order already.
    model = load_model(path, device="cpu")
    if not 1 <= windows.horizon <= model.config.patch_size * model.config.block_size:
        raise ValueError("Horizon exceeds model decode block")
    if cfg.context_length > model.config.context_length:
        raise ValueError("Context exceeds model capacity")
    apply_lora(model, cfg.rank, cfg.alpha, cfg.dropout)
    model.to(cfg.device)
    model.transformer.rope.rebuild_inv_freq()
    parameters = [p for p in model.parameters() if p.requires_grad]
    metadata["trainable_parameters"] = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def schedule(step):
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1, max(0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator().manual_seed(cfg.seed)
    step, best_step, bad, elapsed, best = 0, 0, 0, 0., math.inf
    best_state = None
    checkpoint_path = output / "checkpoint.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["metadata"] != metadata:
            raise ValueError("Resume metadata changed; use a new output directory")
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"], generator)
        step, best_step, bad = state["step"], state["best_step"], state["bad"]
        elapsed, best, best_state = state["elapsed"], state["best"], state["best_adapter"]
        _reconcile_history(output / "history.jsonl", step)
    started = time.perf_counter()

    def assess():
        result = (validation_fn(model, windows, cfg) if validation_fn is not None
                  else validation_loss(model, windows, cfg))
        result = result if isinstance(result, dict) else {"score": result}
        if "score" not in result or not all(math.isfinite(float(v)) for v in result.values()):
            raise FloatingPointError("Custom validation returned a non-finite or missing score")
        return {k: float(v) for k, v in result.items()}

    def retain(at_step):
        if retain_validation_checkpoints:
            checkpoints = output / "validation_checkpoints"
            checkpoints.mkdir(exist_ok=True)
            temporary = checkpoints / f"step_{at_step:05d}.safetensors.tmp"
            save_file(adapter_state(model), temporary)
            temporary.replace(checkpoints / f"step_{at_step:05d}.safetensors")

    if step == 0 and include_initial:
        initial = assess()
        best = initial["score"]
        best_state = {k: v.clone() for k, v in adapter_state(model).items()}
        atomic_json(output / "initial_validation.json", dict(step=0, validation_loss=best,
                    validation_metrics=initial))
        retain(0)
        print(f"TRAIN {task_key} step=0 val={best:.6g} best=0", flush=True)
    stop = cfg.max_steps if stop_after_steps is None else min(cfg.max_steps, stop_after_steps)
    if stop < step:
        raise ValueError("Pause precedes resumed step")
    with (output / "history.jsonl").open("a") as log:
        while step < stop and bad < cfg.patience:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, attempts = [], 0
            while len(losses) < cfg.gradient_accumulation:
                attempts += 1
                if attempts > cfg.gradient_accumulation * 100:
                    raise ValueError("Too many sampled batches without valid labels")
                batch = gift_batch([windows.sample(generator) for _ in range(cfg.batch_size)], cfg.device)
                labels = batch.pop("labels")
                prediction = differentiable_forecast(model, prediction_length=windows.horizon, **batch)
                loss = pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots)
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
                (loss / cfg.gradient_accumulation).backward()
                losses.append(float(loss.detach()))
            gradient_norm = clip_gradients(parameters, cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            record = {"step": step, "train_loss": float(np.mean(losses)),
                      "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": gradient_norm}
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                metrics = assess()
                value = metrics["score"]
                record["validation_loss"] = value
                if validation_fn is not None:
                    record["validation_metrics"] = metrics
                retain(step)
                if value < best:
                    best, best_step, bad = value, step, 0
                    best_state = {k: v.clone() for k, v in adapter_state(model).items()}
                else:
                    bad += 1
                print(f"TRAIN {task_key} step={step} val={value:.6g} best={best_step}", flush=True)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            state = {"metadata": metadata, "adapter": adapter_state(model),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "rng": _rng_state(generator, cfg.device), "step": step, "best_step": best_step,
                     "bad": bad, "best": best, "best_adapter": best_state,
                     "elapsed": elapsed + time.perf_counter() - started}
            torch.save(state, output / "checkpoint.tmp")
            (output / "checkpoint.tmp").replace(checkpoint_path)
    elapsed += time.perf_counter() - started
    completed = step >= cfg.max_steps or bad >= cfg.patience
    if completed and best_state is None:
        raise RuntimeError("Completed training without validation")
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        save_file(best_state, output / "adapter.safetensors.tmp")
        (output / "adapter.safetensors.tmp").replace(output / "adapter.safetensors")
    atomic_json(output / "adapter_config.json", metadata)
    summary = dict(metadata, steps=step, best_step=best_step, completed=completed,
                   validation_loss=best if math.isfinite(best) else None, training_seconds=elapsed,
                   torch_version=str(torch.__version__))
    atomic_json(output / "training_summary.json", summary)
    return model.eval(), summary
