"""Information boundaries and independent forecast-hook references for encoding."""

import copy
from types import SimpleNamespace

import pytest
import torch
from timestral.forecaster import pad_to_patch

from timestral_ft import FineTunedModel, encode
from timestral_ft.compat import PortableForecaster, PortableTimestralModel


def test_official_hidden_reference_and_forecast_unchanged(model):
    model.eval()
    x = torch.randn(2, 3, 17)
    x[0, 1, :5] = float("nan")
    x[1, 2] = float("nan")
    before = copy.deepcopy(model.state_dict())
    forecast = PortableForecaster(model)
    expected = forecast.forecast(x, 8, symmetric_averaging=False).quantiles
    observed = torch.isfinite(x)
    padded, mask = pad_to_patch(torch.where(observed, x, 0), observed, 4)
    captured = []
    hook = model.transformer.register_forward_hook(lambda m, a, y: captured.append(y.detach().clone()))
    try:
        model.predict_block(padded.flatten(0, 1), mask.flatten(0, 1), torch.arange(2).repeat_interleave(3), 2)
    finally:
        hook.remove()
    result = encode(model, x)
    reference = captured[0][:, :5].reshape(2, 3, 5, 16)
    torch.testing.assert_close(result.patches[result.patch_valid], reference[result.patch_valid], rtol=2e-6, atol=2e-6)
    assert torch.count_nonzero(result.patches[~result.patch_valid]) == 0
    assert not result.valid[1, 2]
    assert torch.count_nonzero(result.pooled[1, 2]) == 0
    actual = forecast.forecast(x, 8, symmetric_averaging=False).quantiles
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert before.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_masks_coordinates_pooling_and_batch_composition(model):
    model.eval()
    x = torch.randn(2, 2, 23)
    x[0, 0, 2:6] = float("nan")
    x[0, 1, :6] = float("nan")
    # True prefix lengths 6/23; crop the second to [14,23).
    result = encode(model, x, lengths=[6, 23], max_context=9)
    assert result.patch_start.tolist() == [[0, 0, 2], [14, 15, 19]]
    assert result.patch_end.tolist() == [[0, 2, 6], [15, 19, 23]]
    assert result.patch_valid[0].tolist() == [[False, True, False], [False, False, False]]
    for b in range(2):
        for v in range(2):
            selected = result.patches[b, v][result.patch_valid[b, v]]
            expected = selected.mean(0) if len(selected) else torch.zeros(16)
            torch.testing.assert_close(result.pooled[b, v], expected)
    last = encode(model, x, lengths=[6, 23], max_context=9, pooling="last")
    torch.testing.assert_close(last.pooled[0, 0], result.patches[0, 0, 1])
    assert last.pooled[0, 1].count_nonzero() == 0
    single = encode(model, x[:1, :, :6], max_context=9)
    torch.testing.assert_close(single.patches, result.patches[:1, :, -2:], rtol=2e-6, atol=2e-6)
    changed = x.clone()
    changed[1] *= 100
    changed[0, :, 6:] = 1e6  # Outside the real prefix cannot influence output.
    again = encode(model, changed, lengths=[6, 23], max_context=9)
    torch.testing.assert_close(again.patches[0], result.patches[0], rtol=0, atol=0)
    # A partial first patch is valid: genuine values are never rounded away.
    assert result.patch_valid[1, :, 0].all()


def test_patch_causality_and_within_patch_visibility(model):
    model.eval()
    x = torch.randn(1, 2, 16)
    a = encode(model, x, pooling=None)
    y = x.clone()
    y[..., 8:] += 20 * torch.randn_like(y[..., 8:])
    b = encode(model, y, pooling=None)
    torch.testing.assert_close(a.patches[..., :2, :], b.patches[..., :2, :], rtol=0, atol=0)
    assert a.pooled is None
    y = x.clone()
    y[..., 3] += 20
    c = encode(model, y)
    assert not torch.allclose(a.patches[..., 0, :], c.patches[..., 0, :])
    # The output retains a separate vector per target variable.
    assert a.patches.shape == (1, 2, 4, 16)
    assert not torch.allclose(a.patches[:, 0], a.patches[:, 1])


def test_validation_and_artifact_interface(model):
    model.eval()
    x = torch.randn(12)
    for kwargs in ({"known_covariates": x}, {"past_covariates": x}, {"role_ids": [1]},
                   {"role_ids": [0, 0]}, {"pooling": "max"}, {"max_context": 0},
                   {"lengths": [0]}, {"lengths": [1.2]}, {"lengths": [13]},
                   {"observation_mask": torch.ones(12)}, {"observation_mask": torch.ones(1, 12, dtype=torch.bool)}):
        with pytest.raises(ValueError):
            encode(model, x, **kwargs)
    for bad in (torch.zeros(0), torch.zeros(2, 3, 4, 5)):
        with pytest.raises(ValueError):
            encode(model, bad)
    cfg = SimpleNamespace(past_covariates=["cov"], known_covariates=[],
                          categorical_past_covariates=[], categorical_known_covariates=[])
    artifact = FineTunedModel(model, SimpleNamespace(config=cfg), SimpleNamespace(context_length=8), {})
    with pytest.raises(ValueError, match="targets_only"):
        artifact.encode(x)
    torch.testing.assert_close(artifact.encode(x, targets_only=True).patches,
                               encode(model, x, max_context=8, role_ids=[0]).patches)
    model.bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        encode(model, x)
    model.float()
    model.train()
    with pytest.raises(ValueError, match="model.eval"):
        encode(model, x)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_portable_encoder_finite(model, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    portable = PortableTimestralModel(model.config).to(device).eval()
    portable.load_state_dict(model.state_dict())
    x = torch.randn(2, 3, 13)
    x[0, 1] = float("nan")
    result = encode(portable, x)
    assert torch.isfinite(result.patches).all()
    assert torch.isfinite(result.pooled).all()
    assert result.patches.device.type == device


def test_export_files_unchanged_after_encoding(model, frame, tmp_path):
    from timestral_ft import DataConfig, TrainConfig
    from timestral_ft.data import DataProcessor

    model.eval()
    processor = DataProcessor(DataConfig(target_columns=["target"])).fit(frame)
    artifact = FineTunedModel(model, processor, TrainConfig(context_length=16),
                              {"base_model": "test/base", "base_revision": "fixed"})
    artifact.export(tmp_path / "before")
    artifact.encode(torch.randn(3, 13))
    artifact.export(tmp_path / "after")
    before = {p.name: p.read_bytes() for p in (tmp_path / "before").iterdir()}
    after = {p.name: p.read_bytes() for p in (tmp_path / "after").iterdir()}
    assert before == after
