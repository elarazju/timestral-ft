"""Single-device full/LoRA training with exact local checkpoint resume."""

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from timestral.loading import resolve_checkpoint

from .artifacts import FineTunedModel, file_sha256, write_json
from .compat import check_device, differentiable_forecast, load_model, pinball_loss
from .config import UPSTREAM_REVISION, DataConfig, TrainConfig
from .data import DataProcessor, WindowDataset, collate
from .lora import apply_lora

logger = logging.getLogger(__name__)


def clip_gradients(parameters, max_norm):
    """Clip finite FP32 gradients even when their squared norm overflows FP32.

    Keep the usual PyTorch path; its non-finite norm guard executes before
    modifying gradients. Finite-element overflow is recomputed in CPU FP64.
    Transfer MPS tensors before converting dtype; a fused device/dtype copy
    may silently corrupt values. Real NaN/Inf gradients still terminate.
    """
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.
    try:
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True))
    except RuntimeError:
        pass
    total = torch.linalg.vector_norm(torch.stack([
        # Transfer first: fused MPS->CPU float64 conversion is not supported
        # reliably; CPU conversion must occur after the device copy.
        torch.linalg.vector_norm(g.detach().cpu().to(torch.float64)) for g in grads]))
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite gradient elements or FP64 norm") from None
    coefficient = min(1., max_norm / (float(total) + 1e-6))
    with torch.no_grad():
        for grad in grads:
            grad.mul_(coefficient)
    return float(total)


def _rng_state(generator, device="cpu"):
    npstate = np.random.get_state()
    return {"python": random.getstate(), "numpy": [npstate[0], npstate[1].tolist(),
            npstate[2], npstate[3], npstate[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else [],
            "mps": torch.mps.get_rng_state() if torch.device(device).type == "mps" else None,
            "sampler": generator.get_state()}


def _restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("mps") is not None:
        torch.mps.set_rng_state(state["mps"])
    generator.set_state(state["sampler"])


def _cpu_state(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}


def _validation_loss(model, dataset, cfg):
    model.eval()
    numerator, denominator = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(dataset), cfg.batch_size):
            batch = collate([dataset[i] for i in range(start, min(start + cfg.batch_size, len(dataset)))],
                            cfg.device)
            labels = batch.pop("labels")
            prediction = differentiable_forecast(model, prediction_length=cfg.prediction_length, **batch)
            loss = pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots)
            if loss is not None:
                valid = torch.isfinite(labels) & torch.isfinite(batch["context"]).any(-1)[..., None]
                weight = int(valid.sum())
                numerator += float(loss) * weight
                denominator += weight
    if denominator == 0 or not math.isfinite(numerator):
        raise FloatingPointError("Validation produced no finite loss")
    return numerator / denominator


def finetune(train_data, validation_data, *, model_id="mistralai/Timestral-5M",
             data_config=None, train_config=None, output_dir="runs/finetune", revision=None,
             resume=None, stop_after_steps=None):
    """Fine-tune and return the best model; export is written to output_dir/model.

    `stop_after_steps` pauses a fixed training schedule without changing max_steps,
    allowing exact resume. It counts total optimizer steps, not additional steps.
    """
    cfg = train_config or TrainConfig()
    data_cfg = data_config or DataConfig()
    check_device(cfg.device)
    torch.set_num_threads(cfg.num_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Training directory already has files; use resume or a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    processor = DataProcessor(data_cfg).fit(train_data)
    training = WindowDataset(processor, train_data, cfg.context_length, cfg.prediction_length)
    validation = WindowDataset(processor, validation_data, cfg.context_length, cfg.prediction_length,
                               history=train_data, stride=cfg.prediction_length)
    fingerprints = {"train": training.fingerprint(), "validation": validation.fingerprint()}
    checkpoint_path = resolve_checkpoint(model_id, revision=revision)
    # Resolve main to an immutable Hub snapshot before saving adapter references.
    base_revision = checkpoint_path.name if checkpoint_path.parent.name == "snapshots" else revision
    base_model = str(Path(model_id).resolve()) if Path(model_id).is_dir() else str(model_id)
    license_path = checkpoint_path / "LICENSE"
    if not Path(model_id).is_dir() and not license_path.exists():
        license_path = Path(hf_hub_download(str(model_id), "LICENSE", revision=base_revision))
    model = load_model(checkpoint_path, device=cfg.device)
    if cfg.context_length > model.config.context_length:
        raise ValueError("context_length exceeds this model's supported history")
    if cfg.prediction_length > model.config.block_size * model.config.patch_size:
        raise ValueError("prediction_length exceeds this model's single decode block")
    if cfg.mode == "lora":
        apply_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    else:
        model.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def schedule(step):
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1, max(0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator().manual_seed(cfg.seed)
    metadata = {"base_model": base_model, "base_revision": base_revision,
                "base_weights_sha256": file_sha256(checkpoint_path / "model.safetensors"),
                "base_license_text": license_path.read_text() if license_path.exists() else None,
                "upstream_revision": UPSTREAM_REVISION, "train_config": cfg.to_dict(),
                "preprocessing": processor.to_dict(), "fingerprints": fingerprints,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "torch_version": str(torch.__version__), "validated_device": cfg.device}
    step, best, bad, best_step, elapsed = 0, math.inf, 0, 0, 0.0
    best_state = None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        for key in ("base_model", "base_revision", "base_weights_sha256", "upstream_revision", "train_config",
                    "preprocessing", "fingerprints"):
            if state["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}; use the same data, base model and schedule")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"], generator)
        step, best, bad = state["step"], state["best_loss"], state["bad_evaluations"]
        best_step, elapsed, best_state = state["best_step"], state["elapsed"], state["best_model"]
    started = time.perf_counter()
    stop = min(cfg.max_steps, stop_after_steps) if stop_after_steps is not None else cfg.max_steps
    if stop < step:
        raise ValueError("stop_after_steps precedes the resumed checkpoint")
    with (output / "history.jsonl").open("a") as log:
        while step < stop and bad < cfg.patience:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated, attempts, losses = 0, 0, []
            while accumulated < cfg.gradient_accumulation:
                attempts += 1
                if attempts > cfg.gradient_accumulation * 100:
                    raise ValueError("Too many batches without valid labels")
                indices = torch.randint(len(training), (cfg.batch_size,), generator=generator).tolist()
                batch = collate([training[i] for i in indices], cfg.device)
                labels = batch.pop("labels")
                prediction = differentiable_forecast(model, prediction_length=cfg.prediction_length, **batch)
                loss = pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots)
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
                (loss / cfg.gradient_accumulation).backward()
                losses.append(float(loss.detach()))
                accumulated += 1
            clip_gradients(parameters, cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            record = {"step": step, "train_loss": float(np.mean(losses)),
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            # A pause does not add an evaluation: resuming keeps exactly the same schedule.
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                value = _validation_loss(model, validation, cfg)
                record["validation_loss"] = value
                if value < best:
                    best, best_step, bad = value, step, 0
                    best_state = {k: v.clone() for k, v in _cpu_state(model).items()}
                else:
                    bad += 1
                logger.info("step=%s train=%.6f validation=%.6f", step, record["train_loss"], value)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            checkpoint = {"metadata": metadata, "model": _cpu_state(model),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "rng": _rng_state(generator, cfg.device), "step": step,
                          "best_loss": best, "best_step": best_step, "bad_evaluations": bad,
                          "best_model": best_state, "elapsed": elapsed + time.perf_counter() - started}
            # Replace atomically so an interrupted write leaves the last checkpoint intact.
            torch.save(checkpoint, output / "checkpoint.tmp")
            (output / "checkpoint.tmp").replace(output / "checkpoint.pt")
    elapsed += time.perf_counter() - started
    metadata.update(steps=step, best_step=best_step, best_validation_loss=best if math.isfinite(best) else None,
                    training_seconds=elapsed, completed=step >= cfg.max_steps or bad >= cfg.patience)
    write_json(output / "training_summary.json", metadata)
    write_json(output / "preprocessing.json", processor.to_dict())
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        save_file(best_state, output / "best.safetensors")
    result = FineTunedModel(model.eval(), processor, cfg, metadata)
    # Exports only on a completed schedule; interrupted runs resume from checkpoint.pt.
    if metadata["completed"]:
        result.export(output / "model")
    return result
