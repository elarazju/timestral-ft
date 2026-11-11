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
from timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

from .artifacts import FineTunedModel, file_sha256, write_json
from .augmentation import prepare_augmented_batch
from .compat import (
    check_device,
    differentiable_forecast,
    differentiable_normalized_forecast,
    load_model,
    normalized_pinball_loss,
    pinball_loss,
)
from .config import UPSTREAM_REVISION, DataConfig, TrainConfig
from .data import DataProcessor, DHFSequenceDataset, WindowDataset, collate
from .dhf import dhf_pinball_loss, differentiable_dhf_forecast, layout_statistics, sample_block_layout
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


def _rng_state(generator, device="cpu", layout_generator=None, missing_generator=None, role_generator=None):
    npstate = np.random.get_state()
    return {"python": random.getstate(), "numpy": [npstate[0], npstate[1].tolist(),
            npstate[2], npstate[3], npstate[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else [],
            "mps": torch.mps.get_rng_state() if torch.device(device).type == "mps" else None,
            "sampler": generator.get_state(),
            "dhf_layout": layout_generator.get_state() if layout_generator is not None else None,
            "missingness": missing_generator.get_state() if missing_generator is not None else None,
            "roles": role_generator.get_state() if role_generator is not None else None}


def _restore_rng(state, generator, layout_generator=None, missing_generator=None, role_generator=None):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("mps") is not None:
        torch.mps.set_rng_state(state["mps"])
    generator.set_state(state["sampler"])
    if layout_generator is not None:
        if state.get("dhf_layout") is None:
            raise ValueError("DHF checkpoint is missing its forecast-layout RNG state")
        layout_generator.set_state(state["dhf_layout"])
    for key, rng in (("missingness", missing_generator), ("roles", role_generator)):
        if rng is not None:
            if state.get(key) is None:
                raise ValueError(f"Checkpoint is missing its {key} RNG state")
            rng.set_state(state[key])


def _cpu_state(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}


def _batch_loss(model, batch, labels, cfg, *, loss_space=None):
    space = loss_space or cfg.loss_space
    if space == "normalized":
        prediction = differentiable_normalized_forecast(model, prediction_length=cfg.prediction_length, **batch)
        return normalized_pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots)
    if space != "history_scaled":
        raise ValueError("loss_space must be history_scaled or normalized")
    prediction = differentiable_forecast(model, prediction_length=cfg.prediction_length, **batch)
    return pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots)



def _augmented_window_loss(model, augmented, batch, cfg):
    """Variable target cardinality: weight per-example losses by actual labels."""
    total, count, hidden_count = None, 0, 0
    origin = cfg.context_length
    for i, roles in enumerate(augmented.role_ids):
        targets = torch.where(roles == ROLE_TARGET)[0]
        context = augmented.inputs[i:i + 1, targets, :origin]
        labels = augmented.labels[i:i + 1, targets, origin:]
        covariates = {}
        for key, role, end in (("past_covariates", ROLE_PAST_COV, origin),
                                ("known_covariates", ROLE_KNOWN_COV, augmented.inputs.shape[-1])):
            columns = torch.where(roles == role)[0]
            if len(columns):
                covariates[key] = augmented.inputs[i:i + 1, columns, :end]
        for key in ("categorical_past_covariates", "categorical_known_covariates"):
            if key in batch:
                covariates[key] = batch[key][i:i + 1]
        valid = torch.isfinite(labels) & torch.isfinite(context).any(-1)[..., None]
        weight = int(valid.sum())
        if weight:
            loss = _batch_loss(model, {"context": context, **covariates}, labels, cfg)
            total = loss * weight if total is None else total + loss * weight
            count += weight
            hidden_count += int((augmented.hidden_mask[i:i + 1, targets, origin:] & valid).sum())
    return (total / count if count else None), count, hidden_count

def _validation_loss(model, dataset, cfg):
    model.eval()
    numerator, denominator = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(dataset), cfg.batch_size):
            batch = collate([dataset[i] for i in range(start, min(start + cfg.batch_size, len(dataset)))],
                            cfg.device)
            labels = batch.pop("labels")
            loss = _batch_loss(model, batch, labels, cfg,
                               loss_space=cfg.validation_loss_space or cfg.loss_space)
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
    validation = WindowDataset(processor, validation_data, cfg.context_length, cfg.prediction_length,
                               history=train_data, stride=cfg.prediction_length)
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
    dhf_metadata = None
    if cfg.training_strategy == "dhf":
        ps = model.config.patch_size
        sequence_length = cfg.context_length + cfg.prediction_length
        n_patches = math.ceil(sequence_length / ps)
        max_block = cfg.dhf_max_block_patches or math.ceil(cfg.prediction_length / ps)
        if max_block > model.config.block_size:
            raise ValueError("dhf_max_block_patches exceeds this model's single decode block")
        if cfg.dhf_min_context_patches >= n_patches or cfg.dhf_min_context_patches * ps >= sequence_length:
            raise ValueError("DHF initial context must leave at least one real target step")
        if (n_patches - 1) * ps > model.config.context_length:
            raise ValueError("DHF longest possible history exceeds the model's supported context length")
        training = DHFSequenceDataset(processor, train_data, cfg.context_length, cfg.prediction_length,
                                      min_context_steps=cfg.dhf_min_context_patches * ps)
        dhf_metadata = {"sequence_length": sequence_length, "padded_sequence_length": n_patches * ps,
                        "minimum_context_patches": cfg.dhf_min_context_patches, "max_block_patches": max_block,
                        "block_length_distribution": "uniform_1_to_max",
                        "potential_supervised_steps_per_target": sequence_length - cfg.dhf_min_context_patches * ps,
                        "possible_history_steps": [cfg.dhf_min_context_patches * ps, (n_patches - 1) * ps],
                        "padding": "right_nan", "layout_rng": "separate_checkpointed_generator",
                        "layout_seed": cfg.seed ^ 0x444846}
    else:
        training = WindowDataset(processor, train_data, cfg.context_length, cfg.prediction_length)
    fingerprints = {"train": training.fingerprint(), "validation": validation.fingerprint()}
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
    layout_generator = (torch.Generator().manual_seed(cfg.seed ^ 0x444846)
                        if cfg.training_strategy == "dhf" else None)
    missing_generator = (torch.Generator().manual_seed(cfg.seed ^ 0x4D495353) if cfg.missing_probability else None)
    role_generator = (torch.Generator().manual_seed(cfg.seed ^ 0x524F4C45) if cfg.dynamic_role_sampling else None)
    use_augmentation = bool(cfg.missing_probability or cfg.dynamic_role_sampling)
    metadata = {"base_model": base_model, "base_revision": base_revision,
                "base_weights_sha256": file_sha256(checkpoint_path / "model.safetensors"),
                "base_license_text": license_path.read_text() if license_path.exists() else None,
                "upstream_revision": UPSTREAM_REVISION, "train_config": cfg.to_dict(),
                "preprocessing": processor.to_dict(), "fingerprints": fingerprints,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "torch_version": str(torch.__version__), "validated_device": cfg.device,
                "training_loss_space": cfg.loss_space,
                "training_strategy": cfg.training_strategy,
                "validation_loss_space": cfg.validation_loss_space or cfg.loss_space}
    if use_augmentation:
        metadata["augmentation"] = {
            "missingness": "one interval per numeric row with configured Bernoulli probability; uniform length1..max_span clipped to window, uniform valid start",
            "role_sampling": "iid target/past/known on originally declared targets only; uniformly repair zero-target examples; fixed covariates retain roles",
            "missing_seed": cfg.seed ^ 0x4D495353, "role_seed": cfg.seed ^ 0x524F4C45,
            "labels": "original finite observations; artificial gaps affect inputs only; eligibility requires observed target prefix"}
    if dhf_metadata is not None:
        metadata["dhf"] = dhf_metadata
    step, best, bad, best_step, elapsed = 0, math.inf, 0, 0, 0.0
    best_state = None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        for key in ("base_model", "base_revision", "base_weights_sha256", "upstream_revision", "train_config",
                    "preprocessing", "fingerprints"):
            saved = state["metadata"][key]
            if key == "train_config":
                # Pre-loss-space checkpoints default to the exact legacy objective.
                saved = TrainConfig(**saved).to_dict()
            if saved != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}; use the same data, base model and schedule")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"], generator, layout_generator, missing_generator, role_generator)
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
            supervision = {"supervised_points": 0, "forecast_blocks": 0, "block_length_histogram": {}}
            while accumulated < cfg.gradient_accumulation:
                attempts += 1
                if attempts > cfg.gradient_accumulation * 100:
                    raise ValueError("Too many batches without valid labels")
                indices = torch.randint(len(training), (cfg.batch_size,), generator=generator).tolist()
                batch = collate([training[i] for i in indices], cfg.device)
                augmented = (prepare_augmented_batch(batch, cfg, missing_generator, role_generator)
                             if use_augmentation else None)
                if cfg.training_strategy == "dhf":
                    anchors = sample_block_layout(cfg.batch_size, n_patches, cfg.dhf_min_context_patches,
                                                  max_block, layout_generator)
                    if augmented is None:
                        prediction = differentiable_dhf_forecast(model, block_anchors=anchors, **batch)
                        training_labels = batch["sequence"]
                    else:
                        categories = {key: value for key, value in batch.items() if key.startswith("categorical_")}
                        prediction = differentiable_dhf_forecast(model, augmented.inputs, anchors,
                            numeric_role_ids=augmented.role_ids, labels=augmented.labels, **categories)
                        training_labels = augmented.labels
                    loss = dhf_pinball_loss(prediction, training_labels, model.config.quantile_knots)
                elif augmented is not None:
                    loss, window_count, hidden_supervised = _augmented_window_loss(model, augmented, batch, cfg)
                else:
                    labels = batch.pop("labels")
                    loss = _batch_loss(model, batch, labels, cfg)
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
                (loss / cfg.gradient_accumulation).backward()
                if cfg.training_strategy == "dhf":
                    stats = layout_statistics(prediction)
                    for name in ("supervised_points", "forecast_blocks"):
                        supervision[name] += stats[name]
                    for length, count in stats["block_length_histogram"].items():
                        histogram = supervision["block_length_histogram"]
                        histogram[length] = histogram.get(length, 0) + count
                elif augmented is not None:
                    supervision["supervised_points"] += window_count
                else:
                    supervision["supervised_points"] += int((torch.isfinite(labels)
                        & torch.isfinite(batch["context"]).any(-1)[..., None]).sum())
                if augmented is not None:
                    for name, value in augmented.statistics().items():
                        supervision[name] = supervision.get(name, 0) + value
                    hidden_count = (int((augmented.hidden_mask & prediction.supervision_mask).sum())
                                    if cfg.training_strategy == "dhf" else hidden_supervised)
                    supervision["hidden_supervised_points"] = supervision.get("hidden_supervised_points", 0) + hidden_count
                losses.append(float(loss.detach()))
                accumulated += 1
            clip_gradients(parameters, cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            record = {"step": step, "train_loss": float(np.mean(losses)),
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            if cfg.training_strategy == "dhf" or use_augmentation:
                record.update(supervision)
            else:
                record["supervised_points"] = supervision["supervised_points"]
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
                          "rng": _rng_state(generator, cfg.device, layout_generator, missing_generator, role_generator), "step": step,
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
