"""Opt-in real-weight integration: no network during default tests."""

import os

import pytest
import torch
from timestral.forecaster import TimestralForecaster

from timestral_ft.compat import differentiable_forecast, load_model, pinball_loss
from timestral_ft.lora import apply_lora, merge_lora


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("TIMESTRAL_INTEGRATION") != "1", reason="Set TIMESTRAL_INTEGRATION=1")
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("size,revision", [
    ("5M", "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"),
    ("28M", "180698bbf5a50dda3516aa6419798437fb03a430"),
    ("123M", "5a147287addda85f6acd8da3fb8858fe9af7e8f7"),
    ("209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"),
])
def test_real_model_parity_backward_and_lora(device, size, revision):
    if size != "5M" and os.getenv("TIMESTRAL_LARGE_MODELS") != "1":
        pytest.skip("Set TIMESTRAL_LARGE_MODELS=1 for larger real-weight tests")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    model = load_model(f"mistralai/Timestral-{size}", revision=revision)
    context = torch.randn(2, 2, 63)
    context[0, 0, 10:15] = torch.nan
    kwargs = dict(past_covariates=torch.randn(2, 1, 63), known_covariates=torch.randn(2, 1, 104),
                  categorical_known_covariates=torch.arange(104).remainder(7).expand(2, 1, -1))
    expected = TimestralForecaster(model, quantile_real_cap_k=0).forecast(
        context, 41, symmetric_averaging=False, clip_nonnegative=False, **kwargs
    ).quantiles.clone()
    model.to(device)
    model.transformer.rope.rebuild_inv_freq()
    context = context.to(device)
    kwargs = {key: value if key.startswith("categorical_") else value.to(device)
              for key, value in kwargs.items()}
    prediction = differentiable_forecast(model, context, 41, **kwargs)
    rtol, atol = (1e-4, 1e-5) if device == "cpu" else (1e-3, 1e-4)
    torch.testing.assert_close(prediction.cpu(), expected, rtol=rtol, atol=atol)
    loss = pinball_loss(prediction, torch.randn(2, 2, 41, device=device), context, model.config.quantile_knots)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    apply_lora(model)
    adapted = differentiable_forecast(model, context, 41, **kwargs)
    torch.testing.assert_close(adapted.cpu(), expected, rtol=rtol, atol=atol)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    pinball_loss(adapted, torch.randn(2, 2, 41, device=device), context, model.config.quantile_knots).backward()
    optimizer.step()
    before_merge = differentiable_forecast(model, context, 41, **kwargs).detach()
    merge_lora(model)
    torch.testing.assert_close(differentiable_forecast(model, context, 41, **kwargs),
                               before_merge, rtol=1e-4, atol=1e-5)
