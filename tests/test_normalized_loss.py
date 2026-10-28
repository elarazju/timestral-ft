import copy
from dataclasses import replace

import pytest
import torch
from timestral.forecaster import TimestralForecaster

from timestral_ft.compat import (
    NormalizedForecast,
    _normalized_frames,
    differentiable_normalized_forecast,
    normalized_pinball_loss,
)
from timestral_ft.config import TrainConfig
from timestral_ft.lora import apply_lora


@pytest.mark.parametrize("n_target", [1, 2])
@pytest.mark.parametrize("covariates", [False, True])
@pytest.mark.parametrize("horizon", [3, 7, 12])
def test_direct_head_and_raw_forecast_parity(model, n_target, covariates, horizon):
    context = torch.randn(2, n_target, 15)
    context[0, 0, 3:6] = torch.nan
    kwargs = {}
    if covariates:
        kwargs = dict(past_covariates=torch.randn(2, 1, 15),
                      known_covariates=torch.randn(2, 1, 15 + horizon),
                      categorical_past_covariates=torch.arange(15).remainder(3).expand(2, 1, -1),
                      categorical_known_covariates=torch.arange(15 + horizon).remainder(5).expand(2, 1, -1))
    captured = []
    handle = model.head.register_forward_hook(lambda module, args, output: captured.append(output))
    forecast = differentiable_normalized_forecast(model, context, horizon, **kwargs)
    handle.remove()
    assert len(captured) == 1
    head = captured[0].reshape(2, -1, captured[0].shape[-2], 21)
    # Covariate encoding has decoder tokens over the context and horizon.
    offset = 16 if covariates else 0
    torch.testing.assert_close(forecast.quantiles, head[:, :n_target, offset:offset + horizon], rtol=0, atol=0)
    with torch.no_grad():
        official = TimestralForecaster(model, quantile_real_cap_k=0).forecast(
            context, horizon, symmetric_averaging=False, clip_nonnegative=False, **kwargs).quantiles.clone()
    restored = forecast.quantiles.sinh() * forecast.scale[..., None] + forecast.loc[..., None]
    torch.testing.assert_close(restored, official, rtol=1e-4, atol=1e-5)


def test_frames_match_manual_causal_reference_and_keep_median_gradients(model):
    context = torch.tensor([[1., 3., 2., 4., 6., 8., 5., 7.]])
    observed = torch.ones_like(context, dtype=torch.bool)
    observed[:, 1] = False
    q = torch.zeros(1, 12, 21, requires_grad=True)
    with torch.no_grad():
        q[..., 10] = torch.linspace(-.4, .6, 12)
    loc, scale = _normalized_frames(model, q, context, observed)
    values = context[observed].double()
    expected_loc, expected_scale = [], []
    for start in range(0, 12, 4):
        mean, std = values.mean(), values.std().clamp_min(model.config.minimum_scale)
        expected_loc.extend([float(mean)] * 4)
        expected_scale.extend([float(std)] * 4)
        predicted = q.detach()[0, start:start + 4, 10].double().sinh() * std + mean
        values = torch.cat([values, predicted])
    torch.testing.assert_close(loc, torch.tensor([expected_loc]))
    torch.testing.assert_close(scale, torch.tensor([expected_scale]))
    labels = torch.full((1, 1, 12), torch.nan)
    labels[..., 8:] = 9.
    output = NormalizedForecast(q[:, None], loc[:, None], scale[:, None])
    normalized_pinball_loss(output, labels, context[:, None], model.config.quantile_knots).backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[0, :8, 10].abs().sum() > 0  # No direct labels on these head positions.
    assert q.grad[0, :8, :10].count_nonzero() == 0
    changed = labels.clone()
    changed[..., 8:] = -1000.
    normalized_pinball_loss(output, changed, context[:, None], model.config.quantile_knots)
    # Labels enter only the target transform, never the frame construction.
    again = _normalized_frames(model, q, context, observed)
    torch.testing.assert_close(again[0], loc, rtol=0, atol=0)
    torch.testing.assert_close(again[1], scale, rtol=0, atol=0)


def test_hand_computed_masked_pinball(model):
    q = torch.zeros(2, 1, 3, 21, requires_grad=True)
    loc, scale = torch.full((2, 1, 3), 3.), torch.full((2, 1, 3), 2.)
    labels = torch.full((2, 1, 3), torch.nan)
    labels[0, 0, 0] = 3 + 2 * torch.tensor(1.).sinh()
    labels[0, 0, 2] = 3 + 2 * torch.tensor(-2.).sinh()
    labels[1] = 100  # No observed history: all three labels are excluded.
    context = torch.tensor([[[1., 2.]], [[torch.nan, torch.nan]]])
    output = NormalizedForecast(q, loc, scale)
    loss = normalized_pinball_loss(output, labels, context, model.config.quantile_knots)
    torch.testing.assert_close(loss, torch.tensor(.75))
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[0, 0, 1].count_nonzero() == 0
    assert q.grad[1].count_nonzero() == 0
    assert normalized_pinball_loss(output, torch.full_like(labels, torch.nan),
                                  context, model.config.quantile_knots) is None
    bad = q.detach().clone()
    bad[0, 0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="Non-finite normalized forecast"):
        normalized_pinball_loss(NormalizedForecast(bad, loc, scale), labels, context,
                                model.config.quantile_knots)


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("kind", ["constant", "missing", "extreme"])
def test_normalized_finite_backward_and_update(model, mode, kind):
    if mode == "lora":
        apply_lora(model, rank=2, alpha=4)
    context = torch.ones(2, 2, 16)
    labels = torch.ones(2, 2, 12)
    if kind == "missing":
        context[0, 0] = torch.nan
        context[1, 0, ::2] = torch.nan
        labels[0, 1, ::2] = torch.nan
    if kind == "extreme":
        context = torch.randn_like(context) * 1e30
        labels = torch.randn_like(labels) * 1e30
        labels[0, 0, 0] = torch.finfo(torch.float32).max
    prior = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
    output = differentiable_normalized_forecast(model, context, 12)
    loss = normalized_pinball_loss(output, labels, context, model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(v, prior[k]) for k, v in model.state_dict().items())


def test_large_offset_retains_direct_head_precision(model):
    captured = []
    handle = model.head.register_forward_hook(lambda module, args, output: captured.append(output))
    output = differentiable_normalized_forecast(model, torch.full((1, 1, 16), 1e20), 12)
    handle.remove()
    torch.testing.assert_close(output.quantiles, captured[0][:, None], rtol=0, atol=0)
    assert output.quantiles.std() > 0
    loss = normalized_pinball_loss(output, torch.full((1, 1, 12), -1e20),
                                  torch.full((1, 1, 16), 1e20), model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_affine_only_checkpoint_uses_affine_targets(model):
    model.config = replace(model.config, use_asinh=False)
    context = torch.randn(1, 2, 16)
    output = differentiable_normalized_forecast(model, context, 7)
    assert output.use_asinh is False
    with torch.no_grad():
        expected = TimestralForecaster(model, quantile_real_cap_k=0).forecast(
            context, 7, symmetric_averaging=False, clip_nonnegative=False).quantiles.clone()
    torch.testing.assert_close(output.quantiles * output.scale[..., None] + output.loc[..., None],
                               expected, rtol=1e-4, atol=1e-5)
    labels = torch.randn(1, 2, 7)
    loss = normalized_pinball_loss(output, labels, context, model.config.quantile_knots)
    transformed = (labels.double() - output.loc.double()) / output.scale.double()
    errors = transformed.float()[..., None] - output.quantiles
    levels = torch.tensor(model.config.quantile_knots)
    reference = torch.maximum(levels * errors, (levels - 1) * errors).mean()
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)


def test_constant_history_extreme_opposite_label_uses_finite_transform(model):
    context = torch.full((1, 1, 16), -torch.finfo(torch.float32).max)
    labels = torch.full((1, 1, 12), torch.finfo(torch.float32).max)
    output = differentiable_normalized_forecast(model, context, 12)
    loss = normalized_pinball_loss(output, labels, context, model.config.quantile_knots)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_advancing_frame_gradient_matches_finite_difference(model):
    context = torch.tensor([[1., 3., 2., 4., 6., 8., 5., 7.]])
    observed = torch.ones_like(context, dtype=torch.bool)
    q = torch.full((1, 12, 21), .2, dtype=torch.float64, requires_grad=True)
    labels = torch.full((1, 1, 12), torch.nan, dtype=torch.float64)
    labels[..., 8:] = 12.

    def objective(predictions):
        loc, scale = _normalized_frames(model, predictions, context, observed)
        return normalized_pinball_loss(NormalizedForecast(predictions[:, None], loc[:, None], scale[:, None]),
                                       labels, context[:, None], model.config.quantile_knots)

    objective(q).backward()
    epsilon = 1e-5
    plus, minus = q.detach().clone(), q.detach().clone()
    plus[0, 0, 10] += epsilon
    minus[0, 0, 10] -= epsilon
    numeric = (objective(plus) - objective(minus)) / (2 * epsilon)
    assert q.grad[0, 0, 10].abs() > 0
    torch.testing.assert_close(q.grad[0, 0, 10], numeric, rtol=1e-5, atol=1e-8)


def test_loss_space_config_defaults_and_validation():
    assert TrainConfig().loss_space == "history_scaled"
    assert TrainConfig().validation_loss_space is None
    assert TrainConfig(loss_space="normalized", validation_loss_space="history_scaled").to_dict()[
        "loss_space"] == "normalized"
    for kwargs in ({"loss_space": "raw"}, {"validation_loss_space": "raw"}):
        with pytest.raises(ValueError, match="loss_space"):
            TrainConfig(**kwargs)
