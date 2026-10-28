"""Normalized training acceptance on every available backend and pinned weights."""

import copy
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from timestral.forecaster import TimestralForecaster

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, finetune
from timestral_ft.compat import (
    PortableTimestralModel,
    differentiable_normalized_forecast,
    load_model,
    normalized_pinball_loss,
)
from timestral_ft.lora import apply_lora


def require_backend(device):
    available = {"cpu": True, "mps": torch.backends.mps.is_available(),
                 "cuda": torch.cuda.is_available()}[device]
    if not available:
        if os.environ.get("TIMESTRAL_TEST_DEVICE") == device:
            pytest.fail(f"Required accelerator {device} is unavailable")
        pytest.skip(f"{device} hardware unavailable")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("kind", ["random", "constant"])
def test_normalized_backend_parity_and_updates(model, device, mode, kind):
    require_backend(device)
    context = torch.randn(2, 2, 15) if kind == "random" else torch.ones(2, 2, 15)
    context[0, 0, 3:6] = torch.nan
    labels = torch.randn(2, 2, 7)
    labels[0, 0, ::2] = torch.nan
    covariates = dict(past_covariates=torch.randn(2, 1, 15),
                      known_covariates=torch.randn(2, 1, 22),
                      categorical_past_covariates=torch.arange(15).remainder(3).expand(2, 1, -1),
                      categorical_known_covariates=torch.arange(22).remainder(5).expand(2, 1, -1))
    reference = differentiable_normalized_forecast(model, context, 7, **covariates)
    expected_loss = normalized_pinball_loss(reference, labels, context, model.config.quantile_knots)
    portable = PortableTimestralModel(model.config)
    portable.load_state_dict(model.state_dict())
    if mode == "lora":
        apply_lora(portable, rank=2, alpha=4)
    portable.to(device)
    other_covariates = {name: value if name.startswith("categorical_") else value.to(device)
                        for name, value in covariates.items()}
    actual = differentiable_normalized_forecast(portable, context.to(device), 7, **other_covariates)
    for name in ("quantiles", "loc", "scale"):
        torch.testing.assert_close(getattr(actual, name).cpu(), getattr(reference, name),
                                   rtol=1e-3, atol=1e-4)
    loss = normalized_pinball_loss(actual, labels.to(device), context.to(device), model.config.quantile_knots)
    torch.testing.assert_close(loss.cpu(), expected_loss, rtol=1e-3, atol=1e-4)
    before = {name: value.detach().clone() for name, value in portable.named_parameters()}
    optimizer = torch.optim.AdamW((p for p in portable.parameters() if p.requires_grad), lr=1e-3)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in portable.parameters())
    optimizer.step()
    assert any(not torch.equal(value, before[name]) for name, value in portable.named_parameters())
    if mode == "lora":
        assert all(torch.equal(value, before[name]) for name, value in portable.named_parameters()
                   if not value.requires_grad)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_normalized_backend_resume_and_reload(model, frame, tmp_path, device):
    require_backend(device)
    base = tmp_path / "base"
    base.mkdir()
    model.config.save(base / "config.json")
    save_file(model.state_dict(), base / "model.safetensors")
    cfg = TrainConfig(mode="lora", loss_space="normalized", validation_loss_space="history_scaled",
                      context_length=16, prediction_length=7, max_steps=4, batch_size=2,
                      gradient_accumulation=1, warmup_steps=1, eval_every=2, num_threads=2,
                      lora_rank=2, lora_alpha=4, lora_dropout=.1, device=device)
    data = DataConfig(past_covariates=["past"], known_covariates=["known"],
                      categorical_past_covariates=["catpast"], categorical_known_covariates=["catknown"])
    kwargs = dict(model_id=base, data_config=data, train_config=cfg)
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "direct", **kwargs)
    finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "resume", stop_after_steps=1, **kwargs)
    resumed = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "resume",
                       resume=tmp_path / "resume/checkpoint.pt", **kwargs)
    for name, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    loaded = FineTunedModel.from_pretrained(tmp_path / "direct/model", device=device)
    assert loaded.train_config.loss_space == "normalized"
    future = frame.iloc[128:135]
    reference = direct.predict(frame.iloc[:128], future_covariates=future)
    reloaded = loaded.predict(frame.iloc[:128], future_covariates=future)
    np.testing.assert_allclose(reloaded["median"], reference["median"], rtol=1e-3, atol=1e-4)
    official = TimestralForecaster.from_pretrained(tmp_path / "direct/model")
    assert torch.isfinite(official.forecast(torch.randn(16), 7).quantiles).all()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("TIMESTRAL_INTEGRATION") != "1", reason="Set TIMESTRAL_INTEGRATION=1")
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_real_5m_normalized_head_and_backward(device):
    require_backend(device)
    model = load_model("mistralai/Timestral-5M", revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f")
    context = torch.randn(1, 2, 63)
    context[0, 0, 7:11] = torch.nan
    covariates = dict(past_covariates=torch.randn(1, 1, 63), known_covariates=torch.randn(1, 1, 104))
    expected = TimestralForecaster(model, quantile_real_cap_k=0).forecast(
        context, 41, symmetric_averaging=False, clip_nonnegative=False, **covariates).quantiles.clone()
    portable = copy.deepcopy(model).to(device)
    portable.transformer.rope.rebuild_inv_freq()
    forecast = differentiable_normalized_forecast(
        portable, context.to(device), 41, **{k: v.to(device) for k, v in covariates.items()})
    restored = forecast.quantiles.sinh() * forecast.scale[..., None] + forecast.loc[..., None]
    torch.testing.assert_close(restored.cpu(), expected, rtol=1e-3, atol=1e-4)
    labels = torch.randn(1, 2, 41, device=device)
    labels[0, 0, 5:8] = torch.nan
    loss = normalized_pinball_loss(forecast, labels, context.to(device), model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in portable.parameters())
