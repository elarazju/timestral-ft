"""Independent block-by-block gradient reference and hardware acceptance."""

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
from timestral_ft.dhf import dhf_pinball_loss, differentiable_dhf_forecast
from timestral_ft.lora import apply_lora


def require_backend(device):
    available = {"cpu": True, "mps": torch.backends.mps.is_available(),
                 "cuda": torch.cuda.is_available()}[device]
    if not available:
        if os.environ.get("TIMESTRAL_TEST_DEVICE") == device:
            pytest.fail(f"Required accelerator {device} unavailable")
        pytest.skip(f"{device} unavailable")


def test_shared_dhf_gradient_matches_independent_prefix_computations(model):
    reference = copy.deepcopy(model)
    sequence = torch.randn(2, 2, 28)
    sequence[0, 0, 1] = torch.nan
    sequence[1, 1, 20:22] = torch.nan
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4], [-1, 1, 1, 3, 3, 5, 5]])
    shared = differentiable_dhf_forecast(model, sequence, anchors)
    shared_loss = dhf_pinball_loss(shared, sequence, model.config.quantile_knots)
    shared_loss.backward()
    terms, count = [], 0
    for sample, layout in enumerate(anchors.tolist()):
        for start in sorted(set(layout) - {-1}):
            stop = start + layout.count(start)
            context = sequence[sample:sample + 1, :, :start * 4]
            labels = sequence[sample:sample + 1, :, start * 4:stop * 4]
            predicted = differentiable_normalized_forecast(reference, context, (stop - start) * 4)
            loss = normalized_pinball_loss(predicted, labels, context, model.config.quantile_knots)
            weight = int((torch.isfinite(labels) & torch.isfinite(context).any(-1)[..., None]).sum())
            terms.append(loss * weight)
            count += weight
    independent_loss = torch.stack(terms).sum() / count
    independent_loss.backward()
    torch.testing.assert_close(shared_loss, independent_loss, rtol=3e-5, atol=3e-6)
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert actual.grad is not None and expected.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-3, atol=2e-5, msg=name)


def test_current_block_outputs_have_zero_future_target_input_gradient(model):
    sequence = torch.randn(1, 2, 28, requires_grad=True)
    past = torch.randn(1, 1, 28, requires_grad=True)
    known = torch.randn(1, 1, 28, requires_grad=True)
    output = differentiable_dhf_forecast(
        model, sequence, torch.tensor([[-1, 1, 2, 2, 4, 4, 4]]),
        past_covariates=past, known_covariates=known)
    output.forecast.quantiles[..., 8:16, :].sum().backward()
    assert sequence.grad[..., 8:].count_nonzero() == 0
    assert past.grad[..., 8:].count_nonzero() == 0
    assert sequence.grad[..., :8].abs().sum() > 0
    assert known.grad[..., 8:].abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("kind", ["random", "constant"])
def test_dhf_device_forward_backward(model, device, mode, kind):
    require_backend(device)
    sequence = torch.randn(2, 2, 28) if kind == "random" else torch.ones(2, 2, 28)
    sequence[0, 0, 5:7] = torch.nan
    layout = torch.tensor([[-1, 1, 2, 2, 4, 4, 4], [-1, 1, 1, 3, 3, 5, 5]])
    covariates = dict(past_covariates=torch.randn(2, 1, 28), known_covariates=torch.randn(2, 1, 28))
    expected = differentiable_dhf_forecast(model, sequence, layout, **covariates)
    portable = PortableTimestralModel(model.config)
    portable.load_state_dict(model.state_dict())
    if mode == "lora":
        apply_lora(portable, rank=2, alpha=4)
    portable.to(device)
    actual = differentiable_dhf_forecast(portable, sequence.to(device), layout,
                                         **{k: v.to(device) for k, v in covariates.items()})
    assert torch.equal(actual.supervision_mask.cpu(), expected.supervision_mask)
    for name in ("quantiles", "loc", "scale"):
        torch.testing.assert_close(getattr(actual.forecast, name).cpu(), getattr(expected.forecast, name),
                                   rtol=1e-3, atol=1e-4)
    loss = dhf_pinball_loss(actual, sequence.to(device), model.config.quantile_knots)
    parameters = {name: value.detach().clone() for name, value in portable.named_parameters()}
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in portable.parameters())
    torch.optim.AdamW((p for p in portable.parameters() if p.requires_grad), lr=.001).step()
    assert any(not torch.equal(value, parameters[name]) for name, value in portable.named_parameters())
    if mode == "lora":
        assert all(torch.equal(value, parameters[name]) for name, value in portable.named_parameters()
                   if not value.requires_grad)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_dhf_hardware_resume_export(model, frame, tmp_path, device, mode):
    require_backend(device)
    base = tmp_path / "base"
    base.mkdir()
    model.config.save(base / "config.json")
    save_file(model.state_dict(), base / "model.safetensors")
    cfg = TrainConfig(mode=mode, training_strategy="dhf", loss_space="normalized",
                      validation_loss_space="history_scaled", context_length=16, prediction_length=8,
                      dhf_max_block_patches=3, max_steps=4, batch_size=2, gradient_accumulation=1,
                      warmup_steps=1, eval_every=2, num_threads=2, lora_rank=2, lora_alpha=4,
                      lora_dropout=.1 if mode == "lora" else 0., device=device)
    kwargs = dict(model_id=base, data_config=DataConfig(past_covariates=["past"], known_covariates=["known"]),
                  train_config=cfg)
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "direct", **kwargs)
    finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "resume", stop_after_steps=1, **kwargs)
    resumed = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "resume",
                       resume=tmp_path / "resume/checkpoint.pt", **kwargs)
    for key, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert direct.metadata["best_validation_loss"] == resumed.metadata["best_validation_loss"]
    loaded = FineTunedModel.from_pretrained(tmp_path / "direct/model", device=device)
    assert loaded.train_config.training_strategy == "dhf"
    future = frame.iloc[128:136]
    expected = direct.predict(frame.iloc[:128], future_covariates=future)
    actual = loaded.predict(frame.iloc[:128], future_covariates=future)
    np.testing.assert_allclose(actual["median"], expected["median"], rtol=1e-3, atol=1e-4)
    official = TimestralForecaster.from_pretrained(tmp_path / "direct/model")
    assert torch.isfinite(official.forecast(torch.randn(16), 8).quantiles).all()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("TIMESTRAL_INTEGRATION") != "1", reason="Set TIMESTRAL_INTEGRATION=1")
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_real_5m_shared_dhf_and_prefix_reference(device):
    require_backend(device)
    model = load_model("mistralai/Timestral-5M", revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f")
    sequence = torch.randn(1, 2, 224)
    sequence[..., 2:5] = torch.nan
    layout = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]])
    reference = differentiable_normalized_forecast(model, sequence[..., :128], 96)
    portable = copy.deepcopy(model).to(device)
    portable.transformer.rope.rebuild_inv_freq()
    actual = differentiable_dhf_forecast(portable, sequence.to(device), layout)
    for name in ("quantiles", "loc", "scale"):
        value = getattr(actual.forecast, name)
        value = value[..., 128:, :] if name == "quantiles" else value[..., 128:]
        torch.testing.assert_close(value.cpu(), getattr(reference, name), rtol=1e-3, atol=1e-4)
    loss = dhf_pinball_loss(actual, sequence.to(device), model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in portable.parameters())
