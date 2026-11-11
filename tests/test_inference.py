"""Fast mechanism tests; real Inductor acceptance is an opt-in experiment."""

import copy
import json

import pytest
import torch

from timestral_ft import InferenceConfig, InferenceRuntime
from timestral_ft.compat import PortableForecaster, PortableTimestralModel


def test_config():
    config = InferenceConfig(precision="bf16", compile=True, warmup_steps=2)
    assert InferenceConfig(**json.loads(json.dumps(config.to_dict()))) == config
    for kwargs in ({"precision": "float16"}, {"compile": "false"}, {"warmup_steps": -1},
                   {"warmup_steps": 1.5}, {"fallback": "silent"}):
        with pytest.raises(ValueError):
            InferenceConfig(**kwargs)


def test_default_and_warmup(model):
    model.eval()
    context = torch.randn(1, 2, 17)
    original = context.clone()
    weights = copy.deepcopy(model.state_dict())
    expected = PortableForecaster(model).forecast(context=context, prediction_length=7).quantiles
    runtime = InferenceRuntime(model, InferenceConfig(warmup_steps=2))
    for _ in range(2):
        actual = runtime.forecast(context=context, prediction_length=7).quantiles
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.metadata["warmup_calls"] == 2
    runtime.forecast(context=context[..., :-1], prediction_length=7)
    assert runtime.metadata["warmup_calls"] == 4
    assert runtime.metadata["warmed_signatures"] == 2
    assert runtime.metadata["effective"] == {"precision": "fp32", "compile": False}
    torch.testing.assert_close(original, context, rtol=0, atol=0)
    assert weights.keys() == model.state_dict().keys()
    for key in weights:
        torch.testing.assert_close(weights[key], model.state_dict()[key], rtol=0, atol=0)
    assert "_run_two_stream" not in model.__dict__
    model.train()
    with pytest.raises(ValueError, match="model.eval"):
        runtime.forecast(context=context, prediction_length=7)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_bf16_preserves_scaling(model, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    portable = PortableTimestralModel(model.config).to(device).eval()
    portable.load_state_dict(model.state_dict())
    context = (10000 + torch.randn(1, 1, 16)).to(device)
    observed = torch.ones_like(context, dtype=torch.bool)
    before = portable.scale(context.flatten(0, 1), observed.flatten(0, 1))
    runtime = InferenceRuntime(portable, InferenceConfig(precision="bf16"))
    linear_dtypes = []
    handles = [module.register_forward_hook(lambda module, inputs, output: linear_dtypes.append(output.dtype))
               for name, module in portable.named_modules() if name.endswith("wQKV")]
    try:
        prediction = runtime.forecast(context=context, prediction_length=8).quantiles
    finally:
        for handle in handles:
            handle.remove()
    assert linear_dtypes and set(linear_dtypes) == {torch.bfloat16}
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert (prediction.diff(dim=-1) >= 0).all()
    assert runtime.metadata["autocast_probe_dtype"] == "torch.bfloat16"
    assert runtime.metadata["core_output_dtypes"] == ["torch.float32"]
    after = portable.scale(context.flatten(0, 1), observed.flatten(0, 1))
    for x, y in zip(before, after):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert all(p.dtype == torch.float32 for p in portable.parameters())


def test_compile_execution_and_reuse(model, monkeypatch):
    # Exercise Dynamo for real, substituting only costly Inductor codegen.
    monkeypatch.setattr(torch._inductor, "compile", lambda graph, inputs: graph.forward)
    runtime = InferenceRuntime(model.eval(), InferenceConfig(compile=True, warmup_steps=1))
    assert not runtime.metadata["compiled_executed"]
    context = torch.randn(1, 1, 16)
    runtime.forecast(context=context, prediction_length=8, symmetric_averaging=False)
    first = runtime.metadata
    assert first["compiled_executions"] == 2
    assert first["compiled_graphs"] >= 1
    runtime.forecast(context=context, prediction_length=8, symmetric_averaging=False)
    assert runtime.metadata["compiled_graphs"] == first["compiled_graphs"]
    assert runtime.metadata["warmup_calls"] == 1


@pytest.mark.parametrize("fallback", ["error", "fp32_eager"])
def test_compile_failure(model, monkeypatch, fallback):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate compiler failure")
    monkeypatch.setattr(torch._inductor, "compile", fail)
    runtime = InferenceRuntime(model.eval(), InferenceConfig(compile=True, fallback=fallback))
    args = {"context": torch.randn(1, 1, 16), "prediction_length": 8}
    if fallback == "error":
        with pytest.raises(RuntimeError, match="deliberate compiler failure"):
            runtime.forecast(**args)
        assert runtime.metadata["effective"] is None
    else:
        assert torch.isfinite(runtime.forecast(**args).quantiles).all()
        assert runtime.metadata["effective"] == {"precision": "fp32", "compile": False}
        assert runtime.metadata["status"] == "fallback"
    assert "deliberate compiler failure" in runtime.metadata["failure_reason"]
    assert not runtime.metadata["compiled_executed"]


def test_views_isolated_and_weights_live(model):
    model.eval()
    normal = InferenceRuntime(model)
    low = InferenceRuntime(model, InferenceConfig(precision="bf16"))
    assert normal._view._modules is not model._modules
    assert low._view._modules is not normal._view._modules
    assert next(low._view.parameters()) is next(model.parameters())
    context = torch.randn(1, 1, 16)
    expected = normal.forecast(context, 8).quantiles
    low.forecast(context, 8)
    torch.testing.assert_close(normal.forecast(context, 8).quantiles, expected, rtol=0, atol=0)
    with torch.no_grad():
        for p in model.head.parameters():
            p.add_(.1)
    changed = normal.forecast(context, 8).quantiles
    assert not torch.equal(changed, expected)
    torch.testing.assert_close(changed, PortableForecaster(model).forecast(context, 8).quantiles, rtol=0, atol=0)


def test_arraylike_signature_and_dtype(model):
    import numpy as np
    model.eval()
    runtime = InferenceRuntime(model, InferenceConfig(warmup_steps=1))
    for value in (np.ones((1, 1, 16)), np.zeros((1, 1, 16)),
                  np.ones((1, 1, 16)).tolist(), np.zeros((1, 1, 16)).tolist()):
        runtime.forecast(value, 8)
    assert runtime.metadata["warmup_calls"] == 1
    model.bfloat16()
    with pytest.raises(ValueError, match="FP32 model parameters"):
        runtime.forecast(torch.ones(1, 1, 16), 8)
    with pytest.raises(ValueError, match="FP32 model parameters"):
        InferenceRuntime(model)


def test_mixed_fallback_receipt(model):
    runtime = InferenceRuntime(model.eval(), InferenceConfig(precision="bf16", fallback="fp32_eager"))
    runtime.forecast(torch.ones(1, 1, 16), 8)

    def fail(*args, **kwargs):
        raise RuntimeError("unsupported second shape")

    runtime._core = fail
    runtime.forecast(torch.ones(1, 1, 20), 8)
    assert runtime.metadata["mixed_execution"]
    assert runtime.metadata["successful_calls_by_mode"] == {"bf16_eager": 1, "fp32_eager": 1}
    assert runtime.metadata["effective"] == {"precision": "fp32", "compile": False}
    assert runtime.metadata["status"] == "fallback"


def test_table_cache_and_export(model, frame, tmp_path):
    from safetensors.torch import load_file

    from timestral_ft import DataConfig, FineTunedModel, TrainConfig
    from timestral_ft.data import DataProcessor
    processor = DataProcessor(DataConfig(time_column="timestamp", target_columns=["target"], frequency="h"))
    processor.fit(frame.iloc[:96])
    artifact = FineTunedModel(model, processor, TrainConfig(context_length=16, prediction_length=8),
                             {"base_model": "fixture", "base_revision": None})
    config = InferenceConfig(warmup_steps=1)
    first = artifact.predict(frame.iloc[:32], inference_config=config)
    runtime = artifact.inference_runtime(config)
    model.train()
    second = artifact.predict(frame.iloc[:32], inference_config=config)
    assert first.equals(second)
    assert runtime is artifact.inference_runtime(config)
    assert runtime.metadata["warmup_calls"] == 1
    artifact.predict(frame.iloc[:32], inference_config=InferenceConfig(precision="bf16"))
    artifact.export(tmp_path / "export")
    saved = load_file(tmp_path / "export/model.safetensors")
    assert saved.keys() == model.state_dict().keys()
    for key in saved:
        torch.testing.assert_close(saved[key], model.state_dict()[key], rtol=0, atol=0)
    artifact.model = copy.deepcopy(model)
    assert artifact.inference_runtime(config) is not runtime
    artifact.clear_inference_cache()
    assert not artifact._inference_runtimes
