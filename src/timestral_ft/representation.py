"""Context-only representations of the pinned Timestral Transformer.

This is an exploratory frozen-feature API, not a trained classifier/detector.
Only numeric target rows are supported. A token sees its entire patch and earlier
patches (including other target variates in its example), not later patches.
"""

from dataclasses import dataclass

import torch
from timestral.masks import (
    build_forecast_positions,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    enc_causal_mask,
    group_first_observed_patch,
)
from timestral.model import ROLE_TARGET


@dataclass(frozen=True)
class EncodedContext:
    """Tensors on the model device; invalid representations are exactly zero.

    patches: (B,V,P,D); patch_valid: (B,V,P), at least one observation.
    pooled: (B,V,D), or None; valid: (B,V), at least one valid patch.
    patch_start/end: (B,P), half-open original input coordinates, clipped to the
    retained real interval; wholly padded slots have start=end=retained_start.
    lengths/retained_start: (B,), true prefix length and crop boundary.
    Left padding is NOT an observed zero. Partially observed patches count once
    in mean pooling, regardless of how many points they contain.
    """

    patches: torch.Tensor
    patch_valid: torch.Tensor
    pooled: torch.Tensor | None
    valid: torch.Tensor
    patch_start: torch.Tensor
    patch_end: torch.Tensor
    lengths: torch.Tensor
    retained_start: torch.Tensor
    pooling: str | None


@torch.inference_mode()
def encode(model, context, *, observation_mask=None, lengths=None, max_context=None,
           pooling="mean", role_ids=None, **unsupported):
    """Extract context Transformer outputs without forecast queries or head.

    context accepts (T,), (B,T), (B,V,T). Nonfinite/masked points are missing.
    lengths is an integer (B,) vector: each example occupies [0,lengths[b]);
    trailing storage is ignored. Keep the most recent max_context real points
    (default model.config.context_length), then left-pad to a shared patch grid.
    Supply all variables of an example together: variate attention mixes them.
    mean/last pool only observed patches; all-missing rows return zeros, valid=False.
    model must be in eval mode; neither its state nor forecast behavior is changed.
    role_ids, if supplied, must be (V,) or (B,V) and entirely ROLE_TARGET (0).
    Covariates, categorical roles and future values are deliberately unsupported.
    No symmetry averaging, head, forecasting refinement or output inversion runs.
    """
    if unsupported:
        raise ValueError("encode supports numeric targets only; unsupported arguments: " + ", ".join(sorted(unsupported)))
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("encode requires FP32 model parameters")
    if model.training:
        raise ValueError("encode requires model.eval(); it does not alter training state")
    if pooling not in {None, "mean", "last"}:
        raise ValueError("pooling must be None, mean or last")
    cap = model.config.context_length if max_context is None else max_context
    if type(cap) is not int or cap < 1:
        raise ValueError("max_context must be a positive integer")
    device = next(model.parameters()).device
    x = torch.as_tensor(context, dtype=torch.float32, device=device)
    original_shape = x.shape
    if x.ndim == 1:
        x = x[None, None]
    elif x.ndim == 2:
        x = x[:, None]
    if x.ndim != 3 or any(s == 0 for s in x.shape):
        raise ValueError("context must be nonempty (T,), (B,T) or (B,V,T)")
    b, v, t = x.shape
    if role_ids is not None:
        roles = torch.as_tensor(role_ids, device=device)
        if roles.shape not in {(v,), (b, v)} or not bool((roles == ROLE_TARGET).all()):
            raise ValueError("role_ids must be (V,) or (B,V), containing numeric target role 0 only")
    observed = torch.isfinite(x)
    if observation_mask is not None:
        mask = torch.as_tensor(observation_mask, device=device)
        if mask.shape != original_shape or mask.dtype != torch.bool:
            raise ValueError("observation_mask must be boolean and match context shape exactly")
        observed &= mask.reshape(x.shape)
    if lengths is None:
        lens = torch.full((b,), t, dtype=torch.long, device=device)
    else:
        lens = torch.as_tensor(lengths, device=device)
        if lens.shape != (b,) or lens.dtype not in {torch.int32, torch.int64}:
            raise ValueError("lengths must be an integer vector of shape (B,)")
        if bool(((lens < 1) | (lens > t)).any()):
            raise ValueError("lengths must be between 1 and the stored time dimension")
        lens = lens.long()
    kept = lens.clamp(max=cap)
    starts = lens - kept
    ps = model.config.patch_size
    width = ((int(kept.max()) + ps - 1) // ps) * ps
    left = width - kept
    coords = torch.arange(width, device=device)[None] - left[:, None] + starts[:, None]
    real = (coords >= starts[:, None]) & (coords < lens[:, None])
    indices = coords.clamp(0, t - 1)[:, None].expand(b, v, width)
    mask = observed.gather(-1, indices) & real[:, None]
    packed = torch.where(mask, x.gather(-1, indices), 0).reshape(b * v, width)
    obs = mask.reshape(b * v, width)
    n = width // ps
    patch_valid = obs.reshape(b * v, n, ps).any(-1)
    key_valid = compute_enc_key_validity(patch_valid, model.config.attend_missing_patches)
    groups = torch.arange(b, device=device).repeat_interleave(v)
    first = group_first_observed_patch(patch_valid, groups, b)[groups]
    positions = build_forecast_positions(n, 0, first)
    scaled, _, _ = model.scale(packed, obs)
    hidden = model.transformer(model.representation(scaled, obs),
                               enc_causal_mask(key_valid).unsqueeze(1),
                               build_two_stream_variate_mask(groups, key_valid, 0), positions)
    hidden = torch.where(patch_valid[..., None], hidden, 0).reshape(b, v, n, -1)
    patch_valid = patch_valid.reshape(b, v, n)
    valid = patch_valid.any(-1)
    pooled = None
    if pooling == "mean":
        pooled = hidden.sum(-2) / patch_valid.sum(-1).clamp_min(1)[..., None]
    elif pooling == "last":
        last = (patch_valid.long() * torch.arange(n, device=device)).amax(-1)
        pooled = hidden.gather(-2, last[..., None, None].expand(b, v, 1, hidden.shape[-1])).squeeze(-2)
        pooled = torch.where(valid[..., None], pooled, 0)
    boundaries = torch.arange(n + 1, device=device)[None] * ps - left[:, None] + starts[:, None]
    boundaries = boundaries.maximum(starts[:, None]).minimum(lens[:, None])
    return EncodedContext(hidden, patch_valid, pooled, valid, boundaries[:, :-1], boundaries[:, 1:],
                          lens, starts, pooling)
