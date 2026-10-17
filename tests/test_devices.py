"""Run on available accelerators; CUDA cases stay visible as skips on Apple hosts.

Set TIMESTRAL_TEST_DEVICE=mps/cuda to require that accelerator in a hardware job.
"""

import copy
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from timestral.forecaster import TimestralForecaster

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, evaluate, export, finetune
from timestral_ft import compat as compatibility
from timestral_ft.compat import (
    PortableForecaster,
    PortableTimestralModel,
    differentiable_forecast,
    pinball_loss,
)
from timestral_ft.data import collate
from timestral_ft.trainer import _restore_rng, _rng_state


@pytest.fixture(params=["mps", "cuda"])
def accelerator(request):
    name = request.param
    required = os.environ.get("TIMESTRAL_TEST_DEVICE")
    if required and required not in {"mps", "cuda"}:
        pytest.fail("TIMESTRAL_TEST_DEVICE must be mps or cuda")
    if required and required != name:
        pytest.skip("Another accelerator selected")
    available = torch.backends.mps.is_available() if name == "mps" else torch.cuda.is_available()
    if not available:
        if required == name:
            pytest.fail(f"Required accelerator {name} is unavailable")
        pytest.skip(f"{name} hardware unavailable")
    return name


@pytest.mark.parametrize("n_target", [1, 2])
def test_accelerator_parity_and_gradients(model, accelerator, n_target):
    context = torch.randn(2, n_target, 15)
    context[0, 0, :3] = torch.nan
    covariates = dict(past_covariates=torch.randn(2, 1, 15), known_covariates=torch.randn(2, 1, 22),
                      categorical_past_covariates=torch.arange(15).remainder(3).expand(2, 1, -1),
                      categorical_known_covariates=torch.arange(22).remainder(5).expand(2, 1, -1))
    expected = TimestralForecaster(model, quantile_real_cap_k=0).forecast(
        context, 7, symmetric_averaging=False, clip_nonnegative=False, **covariates
    ).quantiles.clone()
    portable = PortableTimestralModel(model.config)
    portable.load_state_dict(model.state_dict())
    portable.to(accelerator)
    batch = collate([dict(context=context[i], **{k: v[i] for k, v in covariates.items()})
                     for i in range(2)], accelerator)
    assert batch["categorical_known_covariates"].device.type == "cpu"
    prediction = differentiable_forecast(portable, prediction_length=7, **batch)
    assert prediction.device.type == accelerator
    try:
        torch.testing.assert_close(prediction.cpu(), expected, rtol=1e-3, atol=1e-4)
    except AssertionError as error:
        pytest.fail(f"{error}\nBackend diagnostics: {_parity_diagnostics(model, portable, context, covariates, batch)}")
    labels = torch.randn(2, n_target, 7, device=accelerator)
    loss = pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in portable.parameters())


def _parity_diagnostics(cpu_model, device_model, context, covariates, batch):
    """On failure, locate the first CPU/accelerator divergence without weakening parity."""
    snapshots = [{}, {}]
    handles = []

    def capture(store, name):
        def hook(module, inputs, output):
            if inputs and isinstance(inputs[0], torch.Tensor):
                store.setdefault(name + ":input", []).append(inputs[0].detach().cpu().clone())
            if isinstance(output, torch.Tensor):
                store.setdefault(name, []).append(output.detach().cpu().clone())
        return hook

    def difference(left, right):
        return float(torch.nan_to_num((left - right).abs(), nan=float("inf")).max())

    try:
        for model, store in zip((cpu_model, device_model), snapshots):
            for name, module in model.named_modules():
                if name in {"representation", "transformer", "head"} or isinstance(module, torch.nn.Linear):
                    handles.append(module.register_forward_hook(capture(store, name)))
        expected = TimestralForecaster(cpu_model, quantile_real_cap_k=0).forecast(
            context, 7, symmetric_averaging=False, clip_nonnegative=False, **covariates).quantiles
        actual = differentiable_forecast(device_model, prediction_length=7, **batch).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    stages = {}
    for name, values in snapshots[0].items():
        other = snapshots[1].get(name, [])
        stages[name] = ([difference(a, b) for a, b in zip(values, other)]
                        if len(values) == len(other) else "call count differs")
    inference = PortableForecaster(device_model, quantile_real_cap_k=0).forecast(
        batch["context"], 7, symmetric_averaging=False, clip_nonnegative=False,
        **{key: value for key, value in batch.items() if key != "context"}).quantiles.cpu()
    weights = {name: difference(value, device_model.state_dict()[name].cpu())
               for name, value in cpu_model.state_dict().items()}
    return {"torch": torch.__version__, "weights_max_abs": max(weights.values()), "stages_max_abs": stages,
            "gradient_forecast_max_abs": difference(actual, expected),
            "inference_forecast_max_abs": difference(inference, expected)}


def test_accelerator_rng_restore(accelerator):
    generator = torch.Generator().manual_seed(123)
    saved = _rng_state(generator, accelerator)
    expected = torch.rand(50, device=accelerator)
    _restore_rng(saved, generator)
    torch.testing.assert_close(torch.rand(50, device=accelerator), expected, rtol=0, atol=0)


@pytest.mark.parametrize("force_workaround", [False, True])
def test_accelerator_linear_bias_and_gradients(accelerator, monkeypatch, force_workaround):
    if force_workaround:
        monkeypatch.setattr(compatibility, "_mps_linear_omits_bias", lambda: True)
    cpu = compatibility._PortableLinear(8, 16)
    device = copy.deepcopy(cpu).to(accelerator)
    # Zero input makes an omitted bias directly observable, independent of matmul accuracy.
    zeros = torch.zeros(10, 6, 8, device=accelerator)
    expected_bias = cpu.bias.detach().expand(10, 6, -1)
    torch.testing.assert_close(device(zeros).cpu(), expected_bias, rtol=1e-5, atol=1e-6)
    inputs = torch.randn(3, 2, 8).transpose(0, 1).requires_grad_()
    other = inputs.detach().to(accelerator).requires_grad_()
    expected, actual = cpu(inputs), device(other)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-3, atol=1e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for reference, candidate in ((inputs, other), (cpu.weight, device.weight), (cpu.bias, device.bias)):
        torch.testing.assert_close(candidate.grad.cpu(), reference.grad, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_accelerator_train_resume_export(model, frame, tmp_path, accelerator, mode):
    base = tmp_path / "base"
    base.mkdir()
    model.config.save(base / "config.json")
    save_file(model.state_dict(), base / "model.safetensors")
    data_cfg = DataConfig(past_covariates=["past"], known_covariates=["known"],
                          categorical_past_covariates=["catpast"],
                          categorical_known_covariates=["catknown"])
    cfg = TrainConfig(mode=mode, device=accelerator, context_length=16, prediction_length=7,
                      max_steps=4, batch_size=2, gradient_accumulation=1, warmup_steps=1,
                      eval_every=2, num_threads=2, lora_rank=2, lora_alpha=4,
                      lora_dropout=.1 if mode == "lora" else 0.)
    args = dict(model_id=base, data_config=data_cfg, train_config=cfg)
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "direct", **args)
    finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "resumed", stop_after_steps=1, **args)
    resumed = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / "resumed",
                       resume=tmp_path / "resumed/checkpoint.pt", **args)
    for key, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=1e-5, atol=1e-6)
    kwargs = dict(history=frame.iloc[:128], future_covariates=frame.iloc[128:135])
    loaded = FineTunedModel.from_pretrained(tmp_path / "direct/model", device=accelerator)
    expected = direct.predict(**kwargs)["median"].to_numpy()
    np.testing.assert_allclose(loaded.predict(**kwargs)["median"], expected, rtol=1e-3, atol=1e-4)
    backtest = evaluate(loaded, frame.iloc[128:], history=frame.iloc[:128])
    assert np.isfinite(backtest.metrics["mae"])
    if mode == "lora":
        exported = export(direct, tmp_path / "adapter", adapter_only=True)
        adapter = FineTunedModel.from_pretrained(exported, device=accelerator)
        np.testing.assert_allclose(adapter.predict(**kwargs)["median"], expected, rtol=1e-3, atol=1e-4)
    # Device-specific training never changes the public, CPU-compatible weight format.
    official = TimestralForecaster.from_pretrained(tmp_path / "direct/model")
    assert torch.isfinite(official.forecast(torch.randn(16), 7).quantiles).all()
