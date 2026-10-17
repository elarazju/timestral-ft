"""The only layer coupled to the pinned upstream's private inference helpers.

We use its differentiable block decoder directly, never its inference_mode entrypoint.
Refinement is left in the graph, including dependence on earlier predicted medians.
"""

import torch
from safetensors.torch import load_file
from timestral.config import TimestralConfig
from timestral.forecaster import TimestralForecaster
from timestral.loading import resolve_checkpoint
from timestral.model import TimestralModel
from timestral.scaler import clamp_nonfinite, cumulative_welford, welford_combine


def _finite_floor_sqrt(variance, minimum_scale):
    """Exact upstream forward with a finite derivative on the flat floor.

    sqrt(var).clamp_min(s) is constant when var < s**2, but PyTorch's
    composed backward can evaluate 0 / sqrt(0) and produce NaN. Move the
    floor before sqrt for the gradient only; preserve the original FP64
    forward value bit for bit. No median/statistic dependence is detached.
    """
    original = variance.sqrt().clamp_min(minimum_scale)
    safe = variance.clamp_min(minimum_scale**2).sqrt()
    return original.detach() + (safe - safe.detach())


def _advancing_frames_with_finite_gradients(context, observation_mask, sinh_median, minimum_scale):
    """Pinned upstream advancing_block_frames with only the floor derivative fixed.

    Adapted from mistralai/timestral scaler.py (Apache-2.0), revision in config.py.
    Keep FP64 moments and dependence on each previous predicted median.
    """
    r, n_pred, ps = sinh_median.shape
    blank = sinh_median.new_zeros(r, n_pred, ps)
    values = torch.cat([context.unflatten(-1, (-1, ps)).to(torch.float64), blank], dim=1)
    weights = torch.cat([observation_mask.unflatten(-1, (-1, ps)).to(torch.float64), blank], dim=1)
    cnt = weights.sum(-1)
    mean = (values * weights).sum(-1) / cnt.clamp(min=1.0)
    m2 = (weights * (values - mean.unsqueeze(-1)).square()).sum(-1)
    is_zero = cnt == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    cnt, mean, m2 = (s[:, -n_pred - 1] for s in cumulative_welford(cnt, mean, m2))
    offset = mean
    clip = torch.finfo(torch.float32).max ** 0.5 / 1e3
    locs, scales = [], []
    for h in range(n_pred):
        var = (m2 / (cnt - 1.0).clamp(min=1.0)).clamp(min=0.0)
        scale = _finite_floor_sqrt(var, minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = (sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None]).clamp(-clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = (folded - b_mean[:, None]).square().sum(-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, torch.full_like(cnt, ps), b_mean, b_m2)
    return torch.stack(locs, dim=1), torch.stack(scales, dim=1)


def _refine_with_finite_gradients(forecaster, quantiles, context, mask, target_rows):
    ps = forecaster.patch_size
    n_pred = quantiles.shape[1] // ps
    if n_pred <= 1:
        return quantiles
    _, loc, scale = forecaster.model.scale(context, mask)
    loc0, scale0 = loc[:, -1], scale[:, -1]
    median = quantiles[..., forecaster.median_index].unflatten(-1, (n_pred, ps)).to(torch.float64)
    sinh_median = (median - loc0[:, None, None]) / scale0[:, None, None]
    loc_j, scale_j = _advancing_frames_with_finite_gradients(
        context, mask, sinh_median, forecaster.config.minimum_scale)
    first = torch.zeros(loc_j.shape, dtype=torch.bool, device=loc_j.device)
    first[:, 0] = True
    loc_j = torch.where(first, loc0[:, None], loc_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
    scale_j = torch.where(first, scale0[:, None], scale_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
    sinh_norm = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
    refined = clamp_nonfinite(sinh_norm * scale_j[..., None] + loc_j[..., None])
    if target_rows is not None:
        refined = torch.where(target_rows[:, None, None], refined, quantiles)
    return refined


def check_device(device: str) -> torch.device:
    result = torch.device(device)
    if result.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Supported devices are cpu, cuda[:index], and mps")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if result.type == "cuda" and result.index is not None and result.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is unavailable: {result.index}")
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if result.type == "mps" and result.index not in {None, 0}:
        raise ValueError("Only mps or mps:0 is supported")
    return result


class PortableTimestralModel(TimestralModel):
    """Same parameter names/architecture; keep double-precision statistics off MPS.

    Neural network compute remains on MPS. Only scaling/refinement statistics use
    CPU float64, preserving the upstream forward equations and autograd graph.
    """

    def scale(self, inputs, observed):
        if inputs.device.type != "mps":
            return super().scale(inputs, observed)
        result = super().scale(inputs.cpu(), observed.cpu())
        return tuple(value.to(inputs.device) for value in result)


class PortableForecaster(TimestralForecaster):
    @classmethod
    def from_pretrained(cls, path_or_repo_id, device="cpu", max_context=None,
                        per_shot_horizon_patches=32, quantile_real_cap_k=1e4, **hub_kwargs):
        model = load_model(path_or_repo_id, device=device, **hub_kwargs)
        return cls(model, max_context=max_context, per_shot_horizon_patches=per_shot_horizon_patches,
                   quantile_real_cap_k=quantile_real_cap_k)

    def _encode_categorical(self, codes, target_context):
        if target_context is not None and target_context.device.type == "mps":
            target_context = target_context.cpu()
        return super()._encode_categorical(codes, target_context)

    def _as_categorical(self, values, batch, length, name):
        if isinstance(values, torch.Tensor):
            values = values.cpu()
        return super()._as_categorical(values, batch, length, name)

    def _refine_block(self, quantiles, context, mask, target_rows):
        if torch.is_grad_enabled() and quantiles.requires_grad:
            device = quantiles.device
            if device.type == "mps":
                quantiles, context, mask = quantiles.cpu(), context.cpu(), mask.cpu()
                target_rows = target_rows.cpu() if target_rows is not None else None
            return _refine_with_finite_gradients(self, quantiles, context, mask, target_rows).to(device)
        if quantiles.device.type != "mps":
            return super()._refine_block(quantiles, context, mask, target_rows)
        # Tensor.to('cpu') remains differentiable. Do not detach the predicted
        # medians: later horizon patches depend on their accumulated statistics.
        refined = super()._refine_block(quantiles.cpu(), context.cpu(), mask.cpu(),
                                        target_rows.cpu() if target_rows is not None else None)
        return refined.to(quantiles.device)

    def _cap_quantiles(self, quantiles, inputs, observation_mask):
        if quantiles.device.type != "mps":
            return super()._cap_quantiles(quantiles, inputs, observation_mask)
        return super()._cap_quantiles(quantiles.cpu(), inputs.cpu(), observation_mask.cpu()).to(quantiles.device)


def load_model(path_or_repo_id, *, device="cpu", **hub_kwargs):
    """Load unchanged upstream weights into the backend-compatible subclass."""
    device = check_device(device)
    checkpoint = resolve_checkpoint(path_or_repo_id, **hub_kwargs)
    model = PortableTimestralModel(TimestralConfig.load(checkpoint / "config.json"))
    model.load_state_dict(load_file(checkpoint / "model.safetensors", device="cpu"), strict=True)
    model.to(device=device, dtype=torch.float32)
    model.transformer.rope.rebuild_inv_freq()
    return model.eval()


def differentiable_forecast(model, context, prediction_length, **covariates):
    """Return (B,V,H,Q) predictions without augmentation or output clipping."""
    if not 1 <= prediction_length <= model.config.block_size * model.config.patch_size:
        raise ValueError("Training horizon must fit in one model decode block")
    if context.shape[-1] > model.config.context_length:
        raise ValueError("Training context exceeds the model's supported context length")
    forecaster = PortableForecaster(model, quantile_real_cap_k=0)
    context, observed = forecaster._as_context(context, None)
    b, _, t = context.shape
    # Category target encoding is window-local and reads only the context target.
    numeric = {}
    for name, length in (("past_covariates", t), ("known_covariates", t + prediction_length)):
        numeric[name] = forecaster._as_covariates(covariates.get(name), b, length, name)
    target = forecaster._encoding_target(context, observed)
    for category_name, numeric_name, length in (
        ("categorical_past_covariates", "past_covariates", t),
        ("categorical_known_covariates", "known_covariates", t + prediction_length),
    ):
        codes = forecaster._as_categorical(covariates.get(category_name), b, length, category_name)
        encoded = forecaster._encode_categorical(codes, target)
        if encoded is not None:
            previous = numeric[numeric_name]
            numeric[numeric_name] = encoded if previous is None else torch.cat([previous, encoded], dim=1)
    if numeric["known_covariates"] is not None and not forecaster.supports_known_covariates:
        raise ValueError("This model cannot consume known-future covariates")
    rows, mask, role_ids, future, future_mask, n_target = forecaster._assemble_rows(
        context, observed, prediction_length, **numeric
    )
    # _block_ar_decode preserves the official advancing refine frame. No eval() or
    # inference-mode call is made here, so LoRA dropout and gradients remain active.
    predictions = forecaster._block_ar_decode(rows, mask, prediction_length, role_ids, future, future_mask)
    return predictions[:, :n_target]


def pinball_loss(predictions, labels, context, quantile_levels):
    """Mean over valid labels/quantiles, scaled by each history's population std.

    A history with no observations excludes that target. Constant histories use a
    scale of one, preventing an arbitrary 1e6 multiplier. None means no labels.
    """
    observed = torch.isfinite(context)
    count = observed.sum(-1)
    history = torch.where(observed, context, 0.0)
    mean = history.sum(-1) / count.clamp_min(1)
    variance = torch.where(observed, (history - mean[..., None]).square(), 0.0).sum(-1)
    scale = (variance / count.clamp_min(1)).sqrt()
    scale = torch.where(scale > 1e-6, scale, 1.0).detach()
    valid = torch.isfinite(labels) & (count > 0)[..., None]
    if not valid.any():
        return None
    active = predictions[valid]
    if not torch.isfinite(active).all():
        raise FloatingPointError("Non-finite forecast at an observed label")
    error = (labels[valid, None] - active) / scale[..., None].expand_as(labels)[valid, None]
    q = predictions.new_tensor(quantile_levels)
    return torch.maximum(q * error, (q - 1) * error).mean()
