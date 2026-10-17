"""Selection, baseline fallback, and resume must not depend on test data."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from test_gift import dataset

from timestral_ft.gift import GiftTrainConfig, train_gift_task
from timestral_ft.gift_validation import GiftValidation, selection_score
from timestral_ft.lora import adapter_state


def test_pareto_score_and_zero_candidate():
    assert selection_score(2, 3, 2, 3) == 1
    assert selection_score(1, 2, 2, 3) < 1
    assert selection_score(.1, 3.01, 2, 3) > 1
    with pytest.raises(FloatingPointError):
        selection_score(float('nan'), 1, 1, 1)


def test_custom_selection_zero_fallback_and_exact_resume(model, tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    model.config.save(base / 'config.json')
    save_file(model.state_dict(), base / 'model.safetensors')
    cfg = GiftTrainConfig(context_length=16, max_steps=4, batch_size=1,
                          gradient_accumulation=1, eval_every=2, patience=4)

    def rejecting_validation(m, windows, config):
        assert not hasattr(windows, 'test_data')
        changed = any(bool(v.count_nonzero()) for k, v in adapter_state(m).items() if k.endswith('lora_B'))
        return {'score': 2. if changed else 1., 'crps': 3., 'mase': 4.}

    kwargs = dict(model_id=base, config=cfg, validation_fn=rejecting_validation,
                  validation_metadata={'name': 'rejection-test'}, include_initial=True,
                  retain_validation_checkpoints=True, task_key='test/M/short')
    full, summary = train_gift_task(dataset(), output_dir=tmp_path / 'full', **kwargs)
    assert summary['best_step'] == 0 and summary['steps'] == 4 and summary['validation_loss'] == 1
    assert all(not bool(v.count_nonzero()) for k, v in adapter_state(full).items() if k.endswith('lora_B'))
    train_gift_task(dataset(), output_dir=tmp_path / 'resume', stop_after_steps=1, **kwargs)
    resumed, _ = train_gift_task(dataset(), output_dir=tmp_path / 'resume', **kwargs)
    for k, v in adapter_state(full).items():
        torch.testing.assert_close(v, adapter_state(resumed)[k], rtol=0, atol=0)
    state = torch.load(tmp_path / 'full/checkpoint.pt', weights_only=True)
    resumed_state = torch.load(tmp_path / 'resume/checkpoint.pt', weights_only=True)
    for key, value in state['adapter'].items():
        torch.testing.assert_close(value, resumed_state['adapter'][key], rtol=0, atol=0)
    assert any(bool(v.count_nonzero()) for k, v in state['adapter'].items() if k.endswith('lora_B'))
    assert len(list((tmp_path / 'full/validation_checkpoints').glob('*.safetensors'))) == 3
    for key, value in load_file(tmp_path / 'full/validation_checkpoints/step_00004.safetensors').items():
        torch.testing.assert_close(value, state['adapter'][key], rtol=0, atol=0)
    assert json.loads((tmp_path / 'full/initial_validation.json').read_text())['step'] == 0


def test_validation_reference_rejects_protocol_change(model, tmp_path, monkeypatch):
    from timestral_ft.gift import GiftWindows
    from timestral_ft.lora import apply_lora
    apply_lora(model)
    windows = GiftWindows(dataset(), 16)
    cfg = GiftTrainConfig()
    validator = GiftValidation(tmp_path / 'baseline.json', base_identity={'revision': 'one'})
    monkeypatch.setattr(validator, 'measure', lambda *args: {'crps': 2., 'mase': 3., 'scaled_pinball': .5})
    assert validator(model, windows, cfg)['score'] == 1
    validator.base_identity = {'revision': 'two'}
    with pytest.raises(ValueError, match='different protocol'):
        validator(model, windows, cfg)


def test_official_metric_reductions_if_installed(model, tmp_path):
    pytest.importorskip('gluonts')
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

    from timestral_ft.compat import PortableForecaster
    from timestral_ft.gift import GiftWindows, seasonal_error_sliced
    from timestral_ft.gift_validation import LEVELS, GiftValidationConfig
    from timestral_ft.lora import apply_lora
    apply_lora(model)
    ds = dataset()
    windows = GiftWindows(ds, 16)
    cfg = GiftTrainConfig(context_length=16, batch_size=1)
    validator = GiftValidation(tmp_path / 'baseline.json', GiftValidationConfig(context_length=128))
    actual = validator.measure(model, windows, cfg)
    context = np.asarray(ds.training_dataset[0]['target'])[None, None]
    cov = ds.training_dataset[0]['past_feat_dynamic_real'][None]
    fc = PortableForecaster(model, max_context=128)
    predictions = fc.forecast(context, 5, past_covariates=cov).quantiles.cpu().numpy()
    data = {'label': np.ma.masked_invalid(np.arange(7, 12, dtype=np.float32)[None, None]),
            'seasonal_error': np.array([seasonal_error_sliced(context[0], 1, -1)])}
    for q in LEVELS:
        i = next(i for i, level in enumerate(fc.quantile_levels) if abs(level-q) < 1e-9)
        data[str(q)] = predictions[..., i]
    for definition, key in [(MASE(), 'mase'), (MeanWeightedSumQuantileLoss(LEVELS), 'crps')]:
        metric = definition(axis=None)
        metric.update(data)
        assert actual[key] == pytest.approx(float(metric.get()), rel=1e-6, abs=1e-7)


def test_search_resume_after_continuation_commit(model, tmp_path, monkeypatch):
    """A completed 800-step continuation must not be replayed as a 200-step screen."""
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    import timestral_ft.gift as gift
    from timestral_ft.artifacts import file_sha256

    script = Path(__file__).resolve().parents[1] / 'scripts/run_gift209_search.py'
    spec = importlib.util.spec_from_file_location('gift209_resume_test', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    base = tmp_path / 'base'
    base.mkdir()
    model.config.save(base / 'config.json')
    save_file(model.state_dict(), base / 'model.safetensors')
    monkeypatch.setattr(runner, 'BASE', str(base))
    ds = dataset()
    ds.training_dataset[0]['target'] = np.arange(100, dtype=np.float32)
    ds.training_dataset[0]['past_feat_dynamic_real'] = np.arange(100, dtype=np.float32)[None]
    ds.validation_dataset[0]['target'] = np.arange(105, dtype=np.float32)
    ds.freq, ds.target_dim, ds.past_feat_dynamic_real_dim = 'D', 1, 1
    data_module = ModuleType('gift_eval.data')
    data_module.Dataset = lambda *a, **kw: ds
    frequency_module = ModuleType('gluonts.time_feature')
    frequency_module.get_seasonality = lambda _: 1
    monkeypatch.setitem(sys.modules, 'gift_eval.data', data_module)
    monkeypatch.setitem(sys.modules, 'gluonts.time_feature', frequency_module)

    def measure(self, m, windows, cfg):
        changed = any(bool(v.count_nonzero()) for k, v in adapter_state(m).items() if k.endswith('lora_B'))
        return dict(crps=.9 if changed else 1., mase=.9 if changed else 1., scaled_pinball=.2)

    monkeypatch.setattr(GiftValidation, 'measure', measure)
    args = SimpleNamespace(run_dir=tmp_path/'run', results_dir=tmp_path/'results', contexts=[32, 64], row_context_budget=65536,
                           max_steps=4, screen_steps=2, eval_every=1, validation_limit=1, threads=2)
    args.results_dir.mkdir()
    key = 'test/D/short'
    protocol = dict(train_devices={key: 'cpu'}, base_identity={'weights_sha256': file_sha256(base/'model.safetensors')})
    task = dict(key=key, dataset='test', term='short')
    original = gift.atomic_json

    def fail_before_receipt(path, payload):
        if Path(path).name == 'selection.json':
            raise RuntimeError('interrupted after committing continued model')
        return original(path, payload)

    monkeypatch.setattr(gift, 'atomic_json', fail_before_receipt)
    with pytest.raises(RuntimeError, match='after committing'):
        runner.search_task(args, protocol, task, base)
    output = runner.task_directory(args, key)
    screening = json.loads((output/'screen_selection.json').read_text())
    winner = args.run_dir / screening['winner']
    assert json.loads((winner/'training_summary.json').read_text())['steps'] == 4
    hashes = {p: file_sha256(p/'checkpoint.pt') for p in (output/'candidates').iterdir()}
    monkeypatch.setattr(gift, 'atomic_json', original)
    runner.search_task(args, protocol, task, base)
    assert (output/'selection.json').exists()
    assert len((winner/'history.jsonl').read_text().splitlines()) == 4
    # The losing screen must remain a two-step run; resumption cannot extend it.
    for path in (output/'candidates').iterdir():
        assert json.loads((path/'training_summary.json').read_text())['steps'] == (4 if path == winner else 2)
        if path != winner:
            assert file_sha256(path/'checkpoint.pt') == hashes[path]


def test_context_budget_keeps_covariate_rows_and_bounds_training_memory():
    from timestral_ft.gift import bounded_contexts
    assert bounded_contexts([256, 2048, 8192], 100000, 1) == [256, 2048, 8192]
    assert bounded_contexts([256, 2048, 8192], 100000, 7) == [256, 2048, 8192]
    assert bounded_contexts([256, 2048, 8192], 100000, 42) == [256, 1536]
    assert bounded_contexts([256, 2048, 8192], 60, 1) == [64]
    assert bounded_contexts([256, 2048, 8192], 100000, 21) == [256, 2048, 3104]
    with pytest.raises(ValueError, match='one history patch'):
        bounded_contexts([256], 100, 3000)


def test_final_test_batch_budget_is_separate_from_validation(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    from timestral_ft.gift_validation import GiftValidationConfig

    spec = importlib.util.spec_from_file_location('gift209_cli_test',
              Path(__file__).resolve().parents[1]/'scripts/run_gift209_search.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(sys, 'argv', ['run_gift209_search.py', '--stage', 'test'])
    args = runner.cli()
    assert (args.eval_max_rows, args.eval_max_tokens) == (32, 2**20)
    # Validation/checkpoint selection keeps the original, frozen runtime budget.
    cfg = GiftValidationConfig()
    assert (cfg.max_rows, cfg.max_tokens) == (16, 2**17)
