import json
from dataclasses import replace

import pytest
import torch
import yaml
from test_api import base_checkpoint
from timestral.forecaster import TimestralForecaster

from timestral_ft import DataConfig, FineTunedModel, SyntheticConfig, TrainConfig, TrainingStage, finetune
from timestral_ft.synthetic import generate_sequence, stage_at


def test_generator_reproducible_independent_and_diverse():
    cfg = SyntheticConfig(count_probability=0)
    rng = torch.Generator().manual_seed(42)
    global_state = torch.get_rng_state()
    state = rng.get_state()
    first = generate_sequence(128, 3, cfg, rng)
    rng.set_state(state)
    assert torch.equal(first, generate_sequence(128, 3, cfg, rng))
    assert torch.equal(global_state, torch.get_rng_state())
    assert torch.isfinite(first).all()
    for change in [dict(trend_scale=0), dict(seasonal_scale=0), dict(noise_scale=0),
                   dict(shift_probability=1), dict(event_probability=1), dict(dependency_strength=0),
                   dict(nonlinear_probability=1), dict(count_probability=1)]:
        rng.set_state(state)
        assert not torch.equal(first, generate_sequence(128, 3, replace(cfg, **change), rng))
    counts = generate_sequence(128, 3, replace(cfg, count_probability=1), rng)
    assert (counts >= 0).all() and torch.equal(counts, counts.round())


def config(mode, strategy):
    return TrainConfig(mode=mode, training_strategy=strategy, loss_space='normalized',
        validation_loss_space='history_scaled', context_length=16, prediction_length=7,
        max_steps=4, batch_size=2, gradient_accumulation=2, eval_every=2, warmup_steps=1,
        num_threads=2, lora_rank=2, lora_alpha=4, lora_dropout=.2,
        missing_probability=.5, dynamic_role_sampling=True,
        synthetic=SyntheticConfig(), stages=[TrainingStage(2, 8, 1), TrainingStage(2, 16, .5)])


@pytest.mark.parametrize('mode', ['full', 'lora'])
@pytest.mark.parametrize('strategy', ['window', 'dhf'])
@pytest.mark.parametrize('pause', [1, 2, 3])
def test_stage_resume_all_streams_and_export(model, frame, tmp_path, mode, strategy, pause):
    base = base_checkpoint(model, tmp_path)
    cfg = config(mode, strategy)
    kw = dict(model_id=base, train_config=cfg, data_config=DataConfig(target_columns=['target', 'other']))
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / 'direct', **kw)
    finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / 'resume', stop_after_steps=pause, **kw)
    resumed = finetune(frame.iloc[:96], frame.iloc[96:128], output_dir=tmp_path / 'resume',
                       resume=tmp_path / 'resume/checkpoint.pt', **kw)
    a, b = [torch.load(tmp_path / name / 'checkpoint.pt', weights_only=True) for name in ['direct', 'resume']]
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key])
    for key in ['sampler', 'torch', 'mixture', 'synthetic', 'missingness', 'roles', 'dhf_layout']:
        if a['rng'][key] is not None:
            assert torch.equal(a['rng'][key], b['rng'][key])
    assert a['exposure'] == b['exposure'] == direct.metadata['exposure']
    assert a['stage_progress'] == [2, 0]
    assert direct.metadata['best_validation_loss'] == resumed.metadata['best_validation_loss']
    assert (tmp_path / 'direct/history.jsonl').read_text() == (tmp_path / 'resume/history.jsonl').read_text()
    loaded = FineTunedModel.from_pretrained(tmp_path / 'resume/model')
    assert loaded.train_config.to_dict() == cfg.to_dict()
    assert TimestralForecaster.from_pretrained(tmp_path / 'resume/model').forecast(torch.arange(16.).numpy(), 7).quantiles.shape[-2] == 7
    if mode == 'lora':
        resumed.export(tmp_path / 'adapter', adapter_only=True)
        FineTunedModel.from_pretrained(tmp_path / 'adapter')


def test_nested_yaml_and_validation():
    cfg = config('lora', 'dhf')
    for payload in [json.loads(json.dumps(cfg.to_dict())), yaml.safe_load(yaml.safe_dump(cfg.to_dict()))]:
        assert TrainConfig(**payload).to_dict() == cfg.to_dict()
    with pytest.raises(ValueError, match='sum'):
        replace(cfg, max_steps=5)
    with pytest.raises(ValueError, match='explicit'):
        replace(cfg, synthetic=None)
    assert stage_at(cfg.stages, 2) == (1, 0)
    assert TrainConfig(**{}).stages == []


def test_real_stages_covariates_and_mixed_rejection(model, frame, tmp_path):
    base = base_checkpoint(model, tmp_path)
    cfg = replace(config('full', 'dhf'), stages=[TrainingStage(2, 8), TrainingStage(2, 16)])
    data_cfg = DataConfig(past_covariates=['past'], known_covariates=['known'],
        categorical_past_covariates=['catpast'], categorical_known_covariates=['catknown'])
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg,
        data_config=data_cfg, output_dir=tmp_path / 'real')
    with pytest.raises(ValueError, match='declared covariates'):
        finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base,
            train_config=config('full', 'dhf'), data_config=data_cfg, output_dir=tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_lagged_dependency_uses_no_future_statistics():
    from timestral_ft.synthetic import lagged_parent
    original = torch.arange(32.)
    changed = original.clone()
    changed[16:] = 10000
    assert torch.equal(lagged_parent(original, 3)[:19], lagged_parent(changed, 3)[:19])
    assert not torch.equal(lagged_parent(original, 3)[19:], lagged_parent(changed, 3)[19:])


@pytest.mark.parametrize('strategy', ['window', 'dhf'])
def test_synthetic_only_stage_does_not_require_long_real_windows(model, frame, tmp_path, strategy):
    base = base_checkpoint(model, tmp_path)
    cfg = replace(config('lora', strategy), stages=[TrainingStage(2, 96, 1), TrainingStage(2, 16, .5)])
    args = dict(model_id=base, train_config=cfg, output_dir=tmp_path / 'run')
    finetune(frame.iloc[:48], frame.iloc[48:80], stop_after_steps=2, **args)
    result = finetune(frame.iloc[:48], frame.iloc[48:80], resume=tmp_path / 'run/checkpoint.pt', **args)
    assert result.metadata['fingerprints']['stages'][0].startswith('synthetic_only:')
    assert result.metadata['exposure']['synthetic_samples'] >= 8


def test_disabled_mixture_preserves_legacy_streams(model, frame, tmp_path):
    base = base_checkpoint(model, tmp_path)
    cfg = replace(config('lora', 'dhf'), stages=[], synthetic=None)
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg, output_dir=tmp_path / 'old')
    phased = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base,
        train_config=replace(cfg, stages=[TrainingStage(2, 16), TrainingStage(2, 16)]), output_dir=tmp_path / 'phased')
    for key in direct.model.state_dict():
        assert torch.equal(direct.model.state_dict()[key], phased.model.state_dict()[key])
