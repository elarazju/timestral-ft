import copy

import pytest
import torch
from timestral.forecaster import TimestralForecaster

from timestral_ft.compat import differentiable_normalized_forecast
from timestral_ft.dhf import (
    build_dhf_time_mask,
    dhf_pinball_loss,
    differentiable_dhf_forecast,
    layout_statistics,
    sample_block_layout,
    validate_block_layout,
)
from timestral_ft.lora import apply_lora


def test_figure5_dense_mask_manual():
    # Figure 5 uses forecast blocks [1], [2,3], [4,5,6].
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]])
    key_valid = torch.ones(1, 7, dtype=torch.bool)
    mask = build_dhf_time_mask(anchors, key_valid)[0, 0]
    expected = torch.zeros(14, 14, dtype=torch.bool)
    for query in range(7):
        expected[query, :query + 1] = True
    expected[7, 7] = True
    for query, block in ((1, 1), (2, 2), (3, 2), (4, 4), (5, 4), (6, 4)):
        expected[7 + query, :block] = True
        expected[7 + query, 7 + block:7 + query + 1] = True
    assert torch.equal(mask, expected)
    key_valid[0, 0] = False
    missing = build_dhf_time_mask(anchors, key_valid)[0, 0]
    assert not missing[1:, 0].any()
    assert missing[0, 0]
    known = build_dhf_time_mask(anchors, key_valid, torch.tensor([True]))[0, 0]
    assert known[0, 6]
    assert not known[:7, 7:].any()


def test_sampler_complete_tiling_and_exact_rng_restore():
    generator = torch.Generator().manual_seed(123)
    first = sample_block_layout(30, 14, 1, 6, generator)
    state = generator.get_state()
    second = sample_block_layout(30, 14, 1, 6, generator)
    generator.set_state(state)
    assert torch.equal(second, sample_block_layout(30, 14, 1, 6, generator))
    validate_block_layout(first, 6)
    assert (first[:, 0] == -1).all() and (first[:, 1:] >= 1).all()
    observed_lengths = set()
    for row in first.tolist():
        for anchor in set(row) - {-1}:
            observed_lengths.add(row.count(anchor))
    assert observed_lengths == set(range(1, 7))
    with pytest.raises(ValueError, match="tile"):
        validate_block_layout(torch.tensor([[-1, 1, 1, 2]]), 6)
    with pytest.raises(ValueError, match="maximum"):
        validate_block_layout(torch.tensor([[-1, 1, 1, 1]]), 2)


@pytest.mark.parametrize("targets", [1, 2])
def test_single_block_normalized_and_official_raw_parity(model, targets):
    model.eval()
    sequence = torch.randn(2, targets, 28)
    sequence[0, 0, 2:4] = torch.nan
    anchors = torch.tensor([[-1, -1, -1, -1, 4, 4, 4]]).expand(2, -1)
    actual = differentiable_dhf_forecast(model, sequence, anchors)
    expected = differentiable_normalized_forecast(model, sequence[..., :16], 12)
    for name in ("quantiles", "loc", "scale"):
        torch.testing.assert_close(getattr(actual.forecast, name)[..., 16:, :] if name == "quantiles"
                                   else getattr(actual.forecast, name)[..., 16:], getattr(expected, name),
                                   rtol=2e-5, atol=2e-6)
    raw = actual.forecast.quantiles[..., 16:, :].sinh() * actual.forecast.scale[..., 16:, None]
    raw = raw + actual.forecast.loc[..., 16:, None]
    official = TimestralForecaster(model, quantile_real_cap_k=0).forecast(
        sequence[..., :16], 12, symmetric_averaging=False, clip_nonnegative=False).quantiles.clone()
    torch.testing.assert_close(raw, official, rtol=1e-4, atol=2e-5)


def test_multiblock_matches_independent_prefixes_and_single_transformer_call(model):
    sequence = torch.randn(2, 2, 28)
    sequence[0, 0, :4] = torch.nan
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4], [-1, 1, 1, 3, 3, 5, 5]])
    calls = []
    handle = model.transformer.register_forward_hook(lambda module, args, out: calls.append(1))
    actual = differentiable_dhf_forecast(model, sequence, anchors)
    handle.remove()
    assert len(calls) == 1
    for i, layout in enumerate(anchors.tolist()):
        for start in sorted(set(layout) - {-1}):
            end = start + layout.count(start)
            reference = differentiable_normalized_forecast(model, sequence[i:i + 1, :, :start * 4],
                                                           (end - start) * 4)
            torch.testing.assert_close(actual.forecast.quantiles[i:i + 1, :, start * 4:end * 4],
                                       reference.quantiles, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(actual.forecast.loc[i:i + 1, :, start * 4:end * 4],
                                       reference.loc, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(actual.forecast.scale[i:i + 1, :, start * 4:end * 4],
                                       reference.scale, rtol=3e-5, atol=3e-6)


def test_future_target_counterfactual_and_cross_series_firewall(model):
    x = torch.randn(2, 2, 28)
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]]).expand(2, -1)
    before = differentiable_dhf_forecast(model, x, anchors)
    x[0, :, 8:] = torch.randn_like(x[0, :, 8:]) * 200
    x[1] *= -100
    after = differentiable_dhf_forecast(model, x, anchors)
    torch.testing.assert_close(before.forecast.quantiles[0, :, 4:16], after.forecast.quantiles[0, :, 4:16])
    torch.testing.assert_close(before.forecast.loc[0, :, 4:16], after.forecast.loc[0, :, 4:16])
    assert not torch.allclose(before.forecast.quantiles[0, :, 16:], after.forecast.quantiles[0, :, 16:])
    # Modify only another series: all outputs of the first remain unchanged.
    y = x.clone()
    y[1] = 1000
    isolated = differentiable_dhf_forecast(model, y, anchors)
    torch.testing.assert_close(after.forecast.quantiles[0], isolated.forecast.quantiles[0], rtol=0, atol=0)


def test_context_never_reads_decoder_and_blocks_do_not_mix(model):
    x = torch.randn(1, 2, 28)
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]])
    captured = []
    handle = model.transformer.register_forward_hook(lambda module, args, out: captured.append(out.detach().clone()))
    baseline = differentiable_dhf_forecast(model, x, anchors)
    original = model.representation.forecast_tokens

    def altered(rows, n):
        tokens = original(rows, n).clone()
        tokens[:, 2:4] += 10 * torch.randn_like(tokens[:, 2:4])
        return tokens

    model.representation.forecast_tokens = altered
    modified = differentiable_dhf_forecast(model, x, anchors)
    handle.remove()
    torch.testing.assert_close(captured[0][:, :7], captured[1][:, :7], rtol=0, atol=0)
    torch.testing.assert_close(baseline.forecast.quantiles[..., 4:8, :], modified.forecast.quantiles[..., 4:8, :])
    torch.testing.assert_close(baseline.forecast.quantiles[..., 16:, :], modified.forecast.quantiles[..., 16:, :])
    assert not torch.allclose(baseline.forecast.quantiles[..., 8:16, :], modified.forecast.quantiles[..., 8:16, :])


def test_padding_missing_and_exactly_one_label_per_real_position(model):
    x = torch.randn(2, 2, 25)
    x[0, 0, :8] = torch.nan
    x[1, 1, 12] = torch.nan
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]]).expand(2, -1)
    result = differentiable_dhf_forecast(model, x, anchors)
    assert result.forecast.quantiles.shape == (2, 2, 25, 21)
    expected = torch.isfinite(x)
    expected[..., :4] = False
    expected[0, 0, :16] = False  # Prefix of block2 is unobserved; block4 has history.
    assert torch.equal(result.supervision_mask, expected)
    assert layout_statistics(result)["supervised_points"] == int(expected.sum())
    assert layout_statistics(result)["block_length_histogram"] == {"1": 2, "2": 2, "3": 2}
    empty = torch.full_like(x, torch.nan)
    assert dhf_pinball_loss(differentiable_dhf_forecast(model, empty, anchors), empty,
                            model.config.quantile_knots) is None


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("kind", ["constant", "extreme", "missing"])
def test_finite_gradients_and_update(model, mode, kind):
    if mode == "lora":
        apply_lora(model, rank=2, alpha=4, dropout=.1)
    x = torch.ones(2, 2, 28)
    if kind == "extreme":
        x = torch.randn_like(x) * 1e30
        x[0, 0, 12] = torch.finfo(torch.float32).max
    if kind == "missing":
        x[0, 0, :8] = torch.nan
        x[1, 0, ::2] = torch.nan
    prior = copy.deepcopy(model.state_dict())
    output = differentiable_dhf_forecast(model, x, torch.tensor([[-1, 1, 2, 2, 4, 4, 4]]).expand(2, -1))
    loss = dhf_pinball_loss(output, x, model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001).step()
    assert any(not torch.equal(value, prior[name]) for name, value in model.state_dict().items())


def test_numeric_covariate_roles_and_future_information_firewall(model):
    x = torch.randn(1, 2, 28)
    past = torch.randn(1, 1, 28)
    known = torch.randn(1, 1, 28)
    layout = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]])
    before = differentiable_dhf_forecast(model, x, layout, past_covariates=past, known_covariates=known)
    changed_x, changed_past = x.clone(), past.clone()
    changed_x[..., 8:] *= 100
    changed_past[..., 8:] *= 100
    changed = differentiable_dhf_forecast(model, changed_x, layout,
                                         past_covariates=changed_past, known_covariates=known)
    torch.testing.assert_close(before.forecast.quantiles[..., 8:16, :], changed.forecast.quantiles[..., 8:16, :])
    changed_known = known.clone()
    changed_known[..., 8:] += 100
    conditioned = differentiable_dhf_forecast(model, x, layout, past_covariates=past, known_covariates=changed_known)
    assert not torch.allclose(before.forecast.quantiles[..., 8:16, :], conditioned.forecast.quantiles[..., 8:16, :])
    dhf_pinball_loss(before, x, model.config.quantile_knots).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
