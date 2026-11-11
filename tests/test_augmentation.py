import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_dhf_api import base_checkpoint, config
from timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, finetune
from timestral_ft.augmentation import augment_sequence, prepare_augmented_batch, sample_roles
from timestral_ft.compat import differentiable_normalized_forecast
from timestral_ft.dhf import dhf_pinball_loss, differentiable_dhf_forecast


def rng(seed):
    return torch.Generator().manual_seed(seed)


def test_gap_masks_labels_contiguity_and_rng():
    x = torch.arange(120.).reshape(2, 3, 20)
    x[0, 0, 10] = torch.nan
    roles = torch.full((2, 3), ROLE_TARGET)
    cfg = TrainConfig(missing_probability=1, missing_max_span=7)
    gen = rng(42)
    state = gen.get_state()
    a = augment_sequence(x, roles, cfg, gen, None)
    gen.set_state(state)
    b = augment_sequence(x, roles, cfg, gen, None)
    assert torch.equal(a.hidden_mask, b.hidden_mask)
    torch.testing.assert_close(a.labels, x, equal_nan=True)
    assert not a.original_mask[0, 0, 10] and not a.hidden_mask[0, 0, 10]
    assert torch.equal(a.input_mask, a.original_mask & ~a.hidden_mask)
    assert torch.equal(torch.isfinite(a.inputs), a.input_mask)
    assert a.hidden_mask.any()
    for row in a.hidden_mask.flatten(0, 1)[1:]:
        positions = torch.where(row)[0]
        assert len(positions) <= 7
        assert len(positions) == int(positions[-1] - positions[0] + 1)
    default = augment_sequence(x, roles, TrainConfig(), None, None)
    torch.testing.assert_close(default.inputs, x, equal_nan=True)


def test_role_sampler_guarantees_targets_and_keeps_fixed_covariates():
    generator = rng(3)
    state = generator.get_state()
    roles = sample_roles(50, 7, [.01, .49, .5], generator)
    generator.set_state(state)
    assert torch.equal(roles, sample_roles(50, 7, [.01, .49, .5], generator))
    assert (roles == ROLE_TARGET).any(-1).all()
    batch = {'sequence': torch.randn(50, 3, 28), 'past_covariates': torch.randn(50, 2, 28),
             'known_covariates': torch.randn(50, 1, 28)}
    result = prepare_augmented_batch(batch, TrainConfig(dynamic_role_sampling=True), None, rng(9))
    assert (result.role_ids[:, :3] == ROLE_TARGET).any(-1).all()
    assert (result.role_ids[:, 3:5] == ROLE_PAST_COV).all()
    assert (result.role_ids[:, 5] == ROLE_KNOWN_COV).all()
    assert torch.unique(result.role_ids[:, :3]).numel() == 3


def test_hidden_targets_still_scored_natural_nan_excluded_and_no_history_skipped(model):
    labels = torch.randn(1, 2, 28)
    labels[0, 0, 9] = torch.nan
    inputs = labels.clone()
    inputs[0, 0, 8:12] = torch.nan
    inputs[0, 1, :8] = torch.nan
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]])
    out = differentiable_dhf_forecast(model, inputs, anchors, labels=labels)
    assert out.supervision_mask[0, 0, 8] and out.supervision_mask[0, 0, 10]
    assert not out.supervision_mask[0, 0, 9]
    assert not out.supervision_mask[0, 1, :16].any()
    assert out.supervision_mask[0, 1, 16:].all()
    loss = dhf_pinball_loss(out, labels, model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_dynamic_roles_future_firewall_and_known_future_effect(model):
    x = torch.randn(2, 3, 28)
    roles = torch.tensor([[ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV],
                          [ROLE_PAST_COV, ROLE_TARGET, ROLE_KNOWN_COV]])
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]]).expand(2, -1)
    original = differentiable_dhf_forecast(model, x, anchors, numeric_role_ids=roles)
    altered = x.clone()
    altered[:, :2, 8:] *= 100
    safe = differentiable_dhf_forecast(model, altered, anchors, numeric_role_ids=roles)
    for key in ('quantiles', 'loc', 'scale'):
        a, b = getattr(original.forecast, key), getattr(safe.forecast, key)
        torch.testing.assert_close(a[:, :2, 4:16], b[:, :2, 4:16], rtol=1e-5, atol=1e-6)
    altered = x.clone()
    altered[:, 2, 8:] += 100
    known = differentiable_dhf_forecast(model, altered, anchors, numeric_role_ids=roles)
    assert not torch.allclose(original.forecast.quantiles[0, 0, 8:16], known.forecast.quantiles[0, 0, 8:16])
    assert not original.supervision_mask[roles != ROLE_TARGET].any()
    # Role-conditioned normalization/attention agrees with fixed declaration.
    fixed = differentiable_dhf_forecast(model, x[:1, :1], anchors[:1],
                                       past_covariates=x[:1, 1:2], known_covariates=x[:1, 2:3])
    torch.testing.assert_close(original.forecast.quantiles[:1, :1], fixed.forecast.quantiles)
    torch.testing.assert_close(original.forecast.loc[:1, :1], fixed.forecast.loc)


@pytest.mark.parametrize('dynamic', [False, True])
def test_categorical_dhf_reencodes_each_origin_and_blocks_future_target_leak(model, dynamic):
    x = torch.randn(1, 3 if dynamic else 2, 27)
    cats = ((torch.arange(27, dtype=torch.float64) % 3) + 2 ** 30)[None, None]
    anchors = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]])
    roles = torch.tensor([[ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV]]) if dynamic else None
    kwargs = {'numeric_role_ids': roles} if dynamic else {}
    actual = differentiable_dhf_forecast(model, x, anchors, categorical_known_covariates=cats, **kwargs)
    targets = slice(0, 1) if dynamic else slice(None)
    for origin, end in [(4, 8), (8, 16), (16, 27)]:
        cov = {'past_covariates': x[:, 1:2, :origin], 'known_covariates': x[:, 2:3, :end]} if dynamic else {}
        expected = differentiable_normalized_forecast(model, x[:, targets, :origin], end-origin,
            categorical_known_covariates=cats[..., :end], **cov)
        torch.testing.assert_close(actual.forecast.quantiles[:, targets, origin:end], expected.quantiles)
    changed = x.clone()
    changed[:, :2, 8:] *= 99
    changed_cats = cats.clone()
    changed_cats[..., 16:] += 100
    safe = differentiable_dhf_forecast(model, changed, anchors, categorical_known_covariates=changed_cats, **kwargs)
    torch.testing.assert_close(actual.forecast.quantiles[:, targets, 4:16], safe.forecast.quantiles[:, targets, 4:16])
    loss = dhf_pinball_loss(actual, x, model.config.quantile_knots)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('strategy', ['dhf', 'window'])
@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_augmented_exact_resume_export_and_independent_rng(model, frame, tmp_path, strategy, mode, device):
    from test_dhf_acceptance import require_backend
    require_backend(device)
    base = base_checkpoint(model, tmp_path)
    cfg = replace(config(mode), device=device, training_strategy=strategy, missing_probability=.7, missing_max_span=7,
                  dynamic_role_sampling=True, lora_dropout=.1 if mode == 'lora' else 0.)
    data = DataConfig(target_columns=['target', 'other'], past_covariates=['past'],
                      known_covariates=['known'], categorical_known_covariates=['catknown'])
    kwargs = dict(model_id=base, data_config=data, train_config=cfg)
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path/'direct', **kwargs)
    finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path/'resumed', stop_after_steps=1, **kwargs)
    resumed = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path/'resumed',
                       resume=tmp_path/'resumed/checkpoint.pt', **kwargs)
    for key, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    a = torch.load(tmp_path/'direct/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path/'resumed/checkpoint.pt', weights_only=True)
    for key in ['sampler', 'missingness', 'roles'] + (['dhf_layout'] if strategy == 'dhf' else []):
        assert torch.equal(a['rng'][key], b['rng'][key])
    records = [json.loads(line) for line in (tmp_path/'direct/history.jsonl').read_text().splitlines()]
    assert all(row['artificially_hidden_points'] > 0 for row in records)
    loaded = FineTunedModel.from_pretrained(tmp_path/'direct/model')
    kwargs = dict(history=frame.iloc[:128], future_covariates=frame.iloc[128:140])
    np.testing.assert_allclose(loaded.predict(**kwargs)['median'], direct.predict(**kwargs)['median'], rtol=1e-4, atol=1e-5)
    if mode == 'lora':
        direct.export(tmp_path/'adapter', adapter_only=True)
        adapter = FineTunedModel.from_pretrained(tmp_path/'adapter')
        np.testing.assert_allclose(adapter.predict(**kwargs)['median'], direct.predict(**kwargs)['median'], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('changes', [{'missing_probability': float('nan')}, {'missing_probability': 1.1},
    {'missing_max_span': 0}, {'missing_max_span': True}, {'dynamic_role_sampling': 1},
    {'role_probabilities': [0, .5, .5]}, {'role_probabilities': [.5, .5, .5]}])
def test_invalid_augmentation_configuration(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes)


def test_categorical_past_future_codes_cannot_change_current_block(model):
    x = torch.randn(1, 1, 28)
    cats = (torch.arange(28, dtype=torch.float64) % 4)[None, None]
    layout = torch.tensor([[-1, 1, 2, 2, 4, 4, 4]])
    first = differentiable_dhf_forecast(model, x, layout, categorical_past_covariates=cats)
    changed = cats.clone()
    changed[..., 8:] += 2 ** 35
    second = differentiable_dhf_forecast(model, x, layout, categorical_past_covariates=changed)
    torch.testing.assert_close(first.forecast.quantiles[..., 4:16, :], second.forecast.quantiles[..., 4:16, :])


def test_augmentation_rng_streams_independent_and_missing_state_fails():
    from timestral_ft.trainer import _restore_rng, _rng_state
    cfg = TrainConfig(missing_probability=.5, dynamic_role_sampling=True)
    x = torch.randn(2, 4, 28)
    roles = torch.full((2, 4), ROLE_TARGET)
    a, b = rng(2), rng(3)
    joint = augment_sequence(x, roles, cfg, a, b)
    missing = augment_sequence(x, roles, replace(cfg, dynamic_role_sampling=False), rng(2), None)
    sampled = augment_sequence(x, roles, replace(cfg, missing_probability=0), None, rng(3))
    assert torch.equal(joint.hidden_mask, missing.hidden_mask)
    assert torch.equal(joint.role_ids, sampled.role_ids)
    state = _rng_state(rng(0), 'cpu', rng(1), a, b)
    next_expected = augment_sequence(x, roles, cfg, a, b)
    _restore_rng(state, rng(0), rng(1), a, b)
    next_actual = augment_sequence(x, roles, cfg, a, b)
    assert torch.equal(next_expected.hidden_mask, next_actual.hidden_mask)
    assert torch.equal(next_expected.role_ids, next_actual.role_ids)
    state.pop('roles')
    with pytest.raises(ValueError, match='roles RNG'):
        _restore_rng(state, rng(0), rng(1), a, b)


def test_window_augmentation_preserves_legacy_loss_choice(model, frame, tmp_path):
    base = base_checkpoint(model, tmp_path)
    cfg = replace(config(), training_strategy='window', loss_space='history_scaled',
                  missing_probability=.5, missing_max_span=7, dynamic_role_sampling=True)
    result = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base,
                      data_config=DataConfig(target_columns=['target', 'other']),
                      train_config=cfg, output_dir=tmp_path/'run')
    assert result.metadata['training_loss_space'] == 'history_scaled'
    assert np.isfinite(result.metadata['best_validation_loss'])
