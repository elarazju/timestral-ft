"""Bounded protocol, source freeze/resume, counts, and independent reporting."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/run_report_round3.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('report_round3', SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_horizons_are_nonoverlapping_and_confined_to_split():
    for horizon, count in [(96, 30), (192, 15)]:
        for start, stop in [(8640, 11520), (11520, 14400)]:
            origins = report.sparse_origins(start, stop, horizon)
            assert len(origins) == count
            assert origins[0] == start and origins[-1] + horizon == stop
            np.testing.assert_array_equal(np.diff(origins), horizon)


def test_four_arms_only_differ_in_declared_augmentation_and_keep_paired_matrix():
    settings = SimpleNamespace(device='cpu', threads=2)
    configs = report.training_configs(settings, {'selected_steps': 200})
    shared = []
    for name, cfg in configs.items():
        cfg = cfg.copy()
        assert cfg.pop('missing_probability') == (.5 if name in ('missing', 'both') else 0.)
        assert cfg.pop('dynamic_role_sampling') == (name in ('roles', 'both'))
        shared.append(cfg)
    assert all(cfg == shared[0] for cfg in shared)
    protocol = {'training': configs}
    assert len(report.expected_labels(protocol)) == 24
    assert len(report.expected_rows(protocol)) == 104


def test_exposure_uses_actual_history_counts_and_rejects_duplicate_steps(tmp_path):
    records = [{'step': 1, 'supervised_points': 100, 'forecast_blocks': 4, 'block_length_histogram': {'2': 4}},
               {'step': 2, 'supervised_points': 200, 'forecast_blocks': 3, 'block_length_histogram': {'2': 1, '3': 2}}]
    path = tmp_path / 'history.jsonl'
    path.write_text(''.join(json.dumps(x) + '\n' for x in records))
    metadata = {'steps': 2, 'train_config': {'batch_size': 4, 'gradient_accumulation': 4, 'training_strategy': 'dhf', 'missing_probability': .5, 'dynamic_role_sampling': True}}
    result = report.exposure_record(tmp_path, metadata)
    assert result['sampled_windows'] == 32
    assert result['supervised_points'] == 300
    assert result['block_length_histogram'] == {'2': 5, '3': 2}
    records[1]['step'] = 1
    path.write_text(''.join(json.dumps(x) + '\n' for x in records))
    with pytest.raises(ValueError, match='Noncontiguous'):
        report.exposure_record(tmp_path, metadata)


def test_freeze_resume_keeps_original_head_but_rejects_source_change(tmp_path, monkeypatch):
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'model.safetensors').write_bytes(b'weights')
    (weights / 'config.json').write_text('{}')
    settings = SimpleNamespace(results=tmp_path / 'results', output=tmp_path / 'runs', device='cpu', threads=2)
    settings.results.mkdir()
    settings.output.mkdir()
    monkeypatch.setattr(report, 'resolve_checkpoint', lambda *a, **kw: weights)
    monkeypatch.setattr(report, 'source_identity', lambda: {})
    monkeypatch.setattr(report.platform, 'platform', lambda: 'test-platform')
    monkeypatch.setattr(report, 'training_configs', lambda *a: {'single_window': {}})
    monkeypatch.setattr(report.subprocess, 'check_output', lambda *a, **kw: 'oldhead\n')
    first = report.freeze(settings, {'selected_steps': 200}, {})
    monkeypatch.setattr(report.subprocess, 'check_output', lambda *a, **kw: 'newhead\n')
    assert report.freeze(settings, {'selected_steps': 200}, {}) == first
    assert first['environment']['git_head_before_round3'] == 'oldhead'
    monkeypatch.setattr(report, 'source_identity', lambda: {'changed.py': 'different'})
    with pytest.raises(ValueError, match='Frozen provenance mismatch'):
        report.freeze(settings, {'selected_steps': 200}, {})


def fake_rows():
    scores = {k: 1. for k in report.METRICS if k != 'training_seconds'}
    scores.update(evaluation_seconds=1., predictions_sha256='hash', windows=15, scored_values=15 * 7 * 192)
    return [{'dataset': 'ETTh1', 'strategy': 'dhf', 'seed': seed, 'horizon': 192, 'condition': 'gap64',
             'training_seconds': 2., 'best_step': 50, 'actual_steps': 50, 'validation': scores, 'test': scores}
            for seed in [42, 43, 44]]


def test_independent_aggregate_audit_rejects_tampering(tmp_path):
    from audit_report_round3 import validate_aggregate_tables

    rows = fake_rows()
    runs, summary = report.aggregate_frames(rows)
    runs.to_csv(tmp_path / 'runs.csv', index=False)
    summary.to_csv(tmp_path / 'summary.csv', index=False)
    assert validate_aggregate_tables(rows, tmp_path) == (6, 2)
    changed = pd.read_csv(tmp_path / 'summary.csv')
    changed.loc[0, 'mae_mean'] = 8
    changed.to_csv(tmp_path / 'summary.csv', index=False)
    with pytest.raises(AssertionError):
        validate_aggregate_tables(rows, tmp_path)


def test_row_audit_rejects_stale_history_or_model_identity():
    from audit_report_round3 import validate_row_identity

    selected = {'model_sha256': 'model', 'history_sha256': 'history', 'best_step': 50}
    row = {'strategy': 'dhf', **selected}
    validate_row_identity(row, selected, {})
    for key in selected:
        with pytest.raises(ValueError, match='Row/selected model mismatch'):
            validate_row_identity({**row, key: 'stale'}, selected, {})


def test_summary_refuses_partial_matrix(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'training': {'single_window': {}, 'dhf': {}}}))
    with pytest.raises(ValueError, match='incomplete/extra'):
        report.summarize(tmp_path)


def test_counter_replay_keeps_paired_windows_and_layouts_and_accounts_for_hidden_labels():
    import torch
    from audit_report_round3 import replay_training_counts

    values = np.arange(900 * 7, dtype=np.float32).reshape(900, 7)
    settings = SimpleNamespace(device='cpu', threads=2)
    reports = {strategy: replay_training_counts(values, report.config(strategy, 42, 200, settings), 2)
               for strategy in report.STRATEGIES}
    baseline, states = reports['baseline']
    assert baseline['supervised_points'] == 2 * 16 * 7 * 416
    assert baseline['original_observed_points'] == baseline['input_observed_points']
    assert baseline['artificially_hidden_points'] == 0
    for counts, replay_states in reports.values():
        assert counts['input_observed_points'] + counts['artificially_hidden_points'] == counts['original_observed_points']
        assert counts['sampled_target_rows'] + counts['sampled_past_rows'] + counts['sampled_known_rows'] == 2 * 16 * 7
        assert torch.equal(states['sampler'], replay_states['sampler'])
        assert torch.equal(states['dhf_layout'], replay_states['dhf_layout'])
    assert reports['missing'][0]['hidden_supervised_points'] > 0
    assert reports['roles'][0]['supervised_points'] < baseline['supervised_points']
    assert torch.equal(reports['roles'][1]['roles'], reports['both'][1]['roles'])
    assert torch.equal(reports['missing'][1]['missingness'], reports['both'][1]['missingness'])
