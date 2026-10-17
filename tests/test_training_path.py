import copy

import pytest
import torch
from timestral.forecaster import TimestralForecaster

from timestral_ft.compat import PortableForecaster, _finite_floor_sqrt, differentiable_forecast, pinball_loss
from timestral_ft.lora import apply_lora, merge_lora
from timestral_ft.trainer import clip_gradients


@pytest.mark.parametrize("n_target", [1, 2])
@pytest.mark.parametrize("covariates", [False, True])
def test_official_parity_and_gradients(model, n_target, covariates):
    x = torch.randn(2, n_target, 15)
    x[0, 0, 3:6] = torch.nan
    kwargs = {}
    if covariates:
        kwargs = dict(past_covariates=torch.randn(2, 1, 15),
                      known_covariates=torch.randn(2, 1, 22),
                      categorical_past_covariates=torch.arange(15).remainder(3).expand(2, 1, -1),
                      categorical_known_covariates=torch.arange(22).remainder(5).expand(2, 1, -1))
    official = TimestralForecaster(model, quantile_real_cap_k=0).forecast(
        x, 7, symmetric_averaging=False, clip_nonnegative=False, **kwargs
    ).quantiles.clone()
    model.train()
    predictions = differentiable_forecast(model, x, 7, **kwargs)
    torch.testing.assert_close(predictions, official, rtol=1e-4, atol=1e-5)
    loss = pinball_loss(predictions, torch.randn(2, n_target, 7), x, model.config.quantile_knots)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_lora_only_updates_adapters_and_merges(model):
    x = torch.randn(1, 2, 16)
    original = differentiable_forecast(model, x, 8).detach()
    apply_lora(model, rank=2, alpha=4)
    torch.testing.assert_close(differentiable_forecast(model, x, 8), original)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    loss = pinball_loss(differentiable_forecast(model, x, 8), torch.randn(1, 2, 8),
                        x, model.config.quantile_knots)
    loss.backward()
    optimizer.step()
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n])
    before = differentiable_forecast(model, x, 8).detach()
    merge_lora(model)
    torch.testing.assert_close(differentiable_forecast(model, x, 8), before, rtol=1e-4, atol=1e-5)


def test_no_cross_series_attention(model):
    x = torch.randn(2, 2, 16)
    before = differentiable_forecast(model, x, 8).detach()
    x[1] *= 100
    after = differentiable_forecast(model, x, 8)
    torch.testing.assert_close(before[0], after[0])


def test_constant_series_and_empty_labels(model):
    x = torch.ones(2, 1, 16)
    out = differentiable_forecast(model, x, 8)
    loss = pinball_loss(out, torch.ones(2, 1, 8), x, model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert pinball_loss(out, torch.full((2, 1, 8), torch.nan), x, model.config.quantile_knots) is None


def test_zero_variance_refinement_keeps_exact_forward_and_finite_gradients(model):
    q = torch.zeros(1, 12, 21, requires_grad=True)
    context = torch.zeros(1, 16)
    mask = torch.ones_like(context, dtype=torch.bool)
    official = TimestralForecaster(model)
    portable = PortableForecaster(model)
    with torch.no_grad():
        expected = official._refine_block(q, context, mask, None)
    output = portable._refine_block(q, context, mask, None)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad.count_nonzero() > 0
    variance = torch.tensor([0., 1e-20, .1, 1.], dtype=torch.float64, requires_grad=True)
    result = _finite_floor_sqrt(variance, 1e-6)
    torch.testing.assert_close(result, variance.sqrt().clamp_min(1e-6), rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(variance.grad, torch.tensor([0., 0., .5/.1**.5, .5], dtype=torch.float64))


def test_missing_labels_do_not_poison_gradients(model):
    x = torch.randn(2, 2, 17)
    x[0, 1] = torch.nan
    labels = torch.randn(2, 2, 7)
    labels[1, 0, :3] = torch.nan
    out = differentiable_forecast(model, x, 7)
    loss = pinball_loss(out, labels, x, model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    out = out.detach().clone()
    out[0, 0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="Non-finite"):
        pinball_loss(out, labels, x, model.config.quantile_knots)


def test_full_optimizer_updates(model):
    previous = copy.deepcopy(model.state_dict())
    x = torch.randn(1, 1, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    pinball_loss(differentiable_forecast(model, x, 8), torch.randn(1, 1, 8),
                 x, model.config.quantile_knots).backward()
    optimizer.step()
    assert any(not torch.equal(v, previous[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_finite_gradient_norm_overflow_is_clipped_but_nan_terminates(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    parameter = torch.nn.Parameter(torch.zeros(2, device=device))
    parameter.grad = torch.full_like(parameter, 1e30)
    norm = clip_gradients([parameter], 1.)
    assert norm == pytest.approx(2**.5 * 1e30)
    torch.testing.assert_close(parameter.grad.cpu().double().norm(), torch.tensor(1., dtype=torch.float64))
    parameter.grad[0] = torch.nan
    with pytest.raises(FloatingPointError, match="gradient elements"):
        clip_gradients([parameter], 1.)
