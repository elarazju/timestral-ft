"""Native shared-context DHF training (report section 3.1, Figure 5).

The full observed sequence is encoded once. Independent forecast blocks tile
all patches after an initial prefix; labels never enter the forecast tokens.
Uniform block lengths are a project choice, not a disclosed official recipe.
"""

import math
from dataclasses import dataclass

import torch
from timestral.masks import (
    build_two_stream_positions,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    enc_causal_mask,
    group_first_observed_patch,
)
from timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

from .compat import NormalizedForecast, _normalized_frames, normalized_pinball_loss


def sample_block_layout(batch_size, n_patches, min_context, max_block, generator):
    """Return CPU `(B,N)` anchors; -1 denotes the initial unscored prefix.

    Lengths are uniform on 1..max_block, truncated at the end of the sequence.
    Every variate in one example uses this same layout. All randomness uses the
    caller's checkpointed CPU generator, independently of validation.
    """
    if not 1 <= min_context < n_patches or max_block < 1:
        raise ValueError("DHF needs a positive initial prefix, forecast suffix and maximum block length")
    anchors = torch.full((batch_size, n_patches), -1, dtype=torch.long)
    for row in range(batch_size):
        start = min_context
        while start < n_patches:
            length = int(torch.randint(1, max_block + 1, (), generator=generator))
            end = min(n_patches, start + length)
            anchors[row, start:end] = start
            start = end
    return anchors


def validate_block_layout(anchors, max_block):
    """Reject gaps, overlaps, unequal initial prefixes and invalid block anchors."""
    if anchors.ndim != 2 or anchors.dtype != torch.long:
        raise ValueError("block_anchors must be a (batch, patches) int64 tensor")
    first = None
    for row in anchors.detach().cpu().tolist():
        start = next((i for i, anchor in enumerate(row) if anchor != -1), len(row))
        if not 1 <= start < len(row) or (first is not None and start != first):
            raise ValueError("DHF layouts require the same positive initial prefix and nonempty suffix")
        first = start
        while start < len(row):
            if row[start] != start:
                raise ValueError("DHF blocks must tile the suffix with their starting patch as anchor")
            end = start + 1
            while end < len(row) and row[end] == start:
                end += 1
            if end - start > max_block:
                raise ValueError("DHF forecast block exceeds the model's maximum block length")
            start = end


def build_dhf_time_mask(anchors, key_valid, known_rows=None):
    """Dense `(R,1,2N,2N)` query-to-key permissions for sampled blocks.

    Context is causal (known-future rows are bidirectional), and never reads
    decoder tokens. Each active decoder reads only context before its anchor
    and earlier/current decoder tokens in its own block. Inactive tokens read
    only themselves, keeping every softmax row defined.
    """
    rows, n = anchors.shape
    device = anchors.device
    active = anchors >= 0
    index = torch.arange(n, device=device)
    eye = torch.eye(n, device=device, dtype=torch.bool)
    enc_enc = enc_causal_mask(key_valid)
    if known_rows is not None:
        enc_bi = key_valid[:, None, :].expand(rows, n, n) | eye[None]
        enc_enc = torch.where(known_rows[:, None, None], enc_bi, enc_enc)
    dec_enc = active[:, :, None] & (index[None, None, :] < anchors[:, :, None]) & key_valid[:, None, :]
    dec_dec = (anchors[:, :, None] == anchors[:, None, :]) & active[:, :, None] & active[:, None, :]
    dec_dec = (dec_dec & (index[None, :, None] >= index[None, None, :])) | eye[None]
    enc_dec = torch.zeros(rows, n, n, device=device, dtype=torch.bool)
    return torch.cat([torch.cat([enc_enc, enc_dec], -1),
                      torch.cat([dec_enc, dec_dec], -1)], -2)[:, None]


@dataclass(frozen=True)
class DHFForecast:
    forecast: NormalizedForecast
    supervision_mask: torch.Tensor
    block_anchors: torch.Tensor


def differentiable_dhf_forecast(model, sequence, block_anchors, *, past_covariates=None,
                                known_covariates=None, **unsupported):
    """Forecast every active sequence position using one shared transformer call.

    sequence and numeric covariates are `(B,V,T)` over the same full training
    window. T need not divide the patch size: trailing NaNs pad the final patch
    and never become labels. Returned predictions are cropped to the real T.
    """
    if unsupported:
        raise ValueError("DHF currently supports only numeric covariates; categorical covariates are deferred")
    if sequence.ndim != 3 or sequence.shape[-1] < 2:
        raise ValueError("DHF sequence must have shape (batch, targets, time)")
    c, ps = model.config, model.config.patch_size
    batch, n_target, length = sequence.shape
    n = math.ceil(length / ps)
    if block_anchors.shape != (batch, n):
        raise ValueError("block_anchors shape must match batch size and padded sequence patches")
    validate_block_layout(block_anchors, c.block_size)
    if (n - 1) * ps > c.context_length:
        raise ValueError("DHF longest possible history exceeds the model's supported context length")
    if known_covariates is not None and not c.supports_known_covariates:
        raise ValueError("This model cannot consume known-future covariates")
    if (past_covariates is not None or known_covariates is not None) and not c.covariate_enabled:
        raise ValueError("This model cannot consume covariates")
    features, roles = [sequence], [ROLE_TARGET] * n_target
    for values, role in ((past_covariates, ROLE_PAST_COV), (known_covariates, ROLE_KNOWN_COV)):
        if values is not None:
            if values.ndim != 3 or values.shape[0] != batch or values.shape[-1] != length:
                raise ValueError("DHF numeric covariates must share the full sequence batch and time dimensions")
            features.append(values)
            roles.extend([role] * values.shape[1])
    full = torch.cat(features, dim=1)
    v = full.shape[1]
    full = torch.nn.functional.pad(full, (0, n * ps - length), value=float("nan"))
    observed = torch.isfinite(full).reshape(batch * v, n * ps)
    inputs = torch.where(torch.isfinite(full), full, 0.).reshape(batch * v, n * ps)
    role_ids = torch.tensor(roles, device=sequence.device).repeat(batch)
    known_rows = role_ids == ROLE_KNOWN_COV
    group_ids = torch.arange(batch, device=sequence.device).repeat_interleave(v)
    scaled, loc, scale = model.scale(inputs, observed)
    # Whole-window normalization is used only for genuinely known-future rows.
    # The upstream variate firewall prevents target/past values entering them.
    scaled, _, _ = model._known_cov_scaling(inputs, observed, scaled, loc, scale, known_rows)
    x_enc = model.representation(scaled, observed)
    x_dec = model.representation.forecast_tokens(batch * v, n)
    obs_per_patch = observed.unflatten(-1, (n, ps)).any(-1)
    key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
    anchors = block_anchors.to(sequence.device).repeat_interleave(v, dim=0)
    time_mask = build_dhf_time_mask(anchors, key_valid, known_rows)
    variate_mask = build_two_stream_variate_mask(
        group_ids, key_valid, n, target_rows=role_ids == ROLE_TARGET, bidirectional_rows=known_rows)
    first_obs = group_first_observed_patch(obs_per_patch, group_ids, batch)[group_ids]
    positions = build_two_stream_positions(n, first_obs)
    quantiles = model._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
    quantiles = quantiles.reshape(batch, v, n * ps, -1)[:, :n_target]
    target_inputs = inputs.reshape(batch, v, -1)[:, :n_target]
    target_observed = observed.reshape(batch, v, -1)[:, :n_target]
    all_loc, all_scale, all_valid = [], [], []
    for i, layout in enumerate(block_anchors.detach().cpu().tolist()):
        prefix = next(j for j, anchor in enumerate(layout) if anchor >= 0)
        loc_parts = [sequence.new_zeros(n_target, prefix * ps)]
        scale_parts = [sequence.new_ones(n_target, prefix * ps)]
        valid_parts = [torch.zeros(n_target, prefix * ps, device=sequence.device, dtype=torch.bool)]
        start = prefix
        while start < n:
            end = start + 1
            while end < n and layout[end] == start:
                end += 1
            history = target_inputs[i, :, :start * ps]
            history_mask = target_observed[i, :, :start * ps]
            loc_j, scale_j = _normalized_frames(model, quantiles[i, :, start * ps:end * ps], history, history_mask)
            loc_parts.append(loc_j)
            scale_parts.append(scale_j)
            valid_parts.append(history_mask.any(-1)[:, None].expand(-1, (end - start) * ps))
            start = end
        all_loc.append(torch.cat(loc_parts, -1))
        all_scale.append(torch.cat(scale_parts, -1))
        all_valid.append(torch.cat(valid_parts, -1))
    forecast = NormalizedForecast(quantiles[..., :length, :], torch.stack(all_loc)[..., :length],
                                  torch.stack(all_scale)[..., :length], c.use_asinh)
    valid = torch.stack(all_valid)[..., :length] & torch.isfinite(sequence)
    return DHFForecast(forecast, valid, block_anchors)


def dhf_pinball_loss(output, sequence, quantile_levels):
    labels = sequence.masked_fill(~output.supervision_mask, float("nan"))
    return normalized_pinball_loss(output.forecast, labels, sequence, quantile_levels)


def layout_statistics(output):
    """JSON-safe realized supervision and block counts for experiment audits."""
    histogram = {}
    for row in output.block_anchors.detach().cpu().tolist():
        for anchor in sorted(set(row) - {-1}):
            length = str(row.count(anchor))
            histogram[length] = histogram.get(length, 0) + 1
    return {"supervised_points": int(output.supervision_mask.sum()),
            "forecast_blocks": sum(histogram.values()), "block_length_histogram": histogram}
