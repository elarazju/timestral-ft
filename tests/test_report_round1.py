"""Arithmetic, split boundaries, and immutable experiment provenance."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/run_report_round1.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('report_round1', SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_sparse_origins_cover_30_disjoint_horizons_without_crossing_split():
    for start, stop in [(8640, 11520), (11520, 14400)]:
        indices = report.sparse_origins(start, stop)
        assert len(indices) == 30
        assert indices[0] == start
        assert indices[-1] + 96 == stop
        assert np.all(np.diff(indices) == 96)


def test_metric_arithmetic_and_masked_labels():
    truth = np.array([[[1., np.nan, 1.]]])
    prediction = np.zeros((1, 1, 3, 21))
    prediction[0, 0, 1] = np.nan  # Inactive label must not enter numerators.
    result = report.score_arrays(prediction, truth, np.array([2.]))
    assert result['masked_wql21'] == pytest.approx(1)
    assert result['crps9_approx'] == pytest.approx(1)
    assert result['mae'] == result['mse'] == 1
    assert result['mase24'] == .5
    assert result['coverage80'] == result['width80'] == 0
    assert result['scored_values'] == 2
    np.testing.assert_allclose(result['per_quantile_wql'], 2 * report.LEVELS)


def test_metric_uses_global_label_weighting_not_mean_target_ratios():
    actual = np.array([[[1.], [9.]]])
    prediction = np.zeros((1, 2, 1, 21))
    prediction[0, 1] = 9
    result = report.score_arrays(prediction, actual, np.ones(2))
    assert result['masked_wql21'] == pytest.approx(.1)
    assert result['mase24'] == .5


def test_seasonal_scale_uses_full_training_prefix():
    training = np.arange(40, dtype=float)[:, None]
    np.testing.assert_array_equal(report.seasonal_scales(training, 24), [24])
    with pytest.raises(ValueError, match='positive finite'):
        report.seasonal_scales(np.ones((40, 2)), 24)


def test_frozen_provenance_reuses_equal_and_refuses_changed_files(tmp_path):
    path = tmp_path / 'protocol.json'
    report.immutable_json(path, {'seed': 42})
    before = path.read_bytes()
    report.immutable_json(path, {'seed': 42})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='Frozen provenance mismatch'):
        report.immutable_json(path, {'seed': 43})
    assert json.loads(path.read_text()) == {'seed': 42}


def test_nonfinite_active_forecast_and_zero_wql_denominator_fail():
    prediction = np.zeros((1, 1, 1, 21))
    with pytest.raises(ValueError, match='zero absolute'):
        report.score_arrays(prediction, np.zeros((1, 1, 1)), np.ones(1))
    prediction[..., 0] = np.nan
    with pytest.raises(FloatingPointError, match='non-finite'):
        report.score_arrays(prediction, np.ones((1, 1, 1)), np.ones(1))


def test_paired_training_configs_only_change_training_objective():
    from types import SimpleNamespace

    settings = SimpleNamespace(device='cpu', threads=2)
    legacy = report.config('history_scaled', 42, 200, settings).to_dict()
    normalized = report.config('normalized', 42, 200, settings).to_dict()
    assert legacy['validation_loss_space'] == normalized['validation_loss_space'] == 'history_scaled'
    assert legacy.pop('loss_space') == 'history_scaled'
    assert normalized.pop('loss_space') == 'normalized'
    assert legacy == normalized


def test_auditor_rejects_stale_row_identity_even_with_matching_protocol():
    from audit_report_round1 import validate_row_identity

    selected = {'model_sha256': 'a', 'adapter_sha256': 'b', 'best_step': 50}
    row = {'mode': 'normalized', **selected}
    validate_row_identity(row, selected, {'model_sha256': 'base'})
    for key in selected:
        changed = {**row, key: 'stale'}
        with pytest.raises(ValueError, match='Row/selected ledger mismatch'):
            validate_row_identity(changed, selected, {'model_sha256': 'base'})
    with pytest.raises(ValueError, match='Base row model SHA'):
        validate_row_identity({'mode': 'base', 'model_sha256': 'stale'}, None, {'model_sha256': 'base'})


def test_auditor_recomputes_aggregate_tables_and_rejects_tampering(tmp_path):
    import pandas as pd
    from audit_report_round1 import validate_aggregate_tables

    scores = {k: 1. for k in ['masked_wql21', 'crps9_approx', 'mae', 'mse', 'mase24', 'coverage80',
                              'width80', 'evaluation_seconds']}
    rows = []
    for seed in [42, 43, 44]:
        row = {'dataset': 'ETTh1', 'mode': 'normalized', 'seed': seed, 'training_seconds': 2.,
               'best_step': 50, 'actual_steps': 50, 'validation': scores, 'test': scores}
        (tmp_path / f'ETTh1_normalized_s{seed}.json').write_text(json.dumps(row))
        rows.append(row)
    report.summarize(tmp_path)
    assert validate_aggregate_tables(rows, tmp_path) == (6, 2)
    table = pd.read_csv(tmp_path / 'summary.csv')
    table.loc[0, 'mae_mean'] = 100
    table.to_csv(tmp_path / 'summary.csv', index=False)
    with pytest.raises(AssertionError):
        validate_aggregate_tables(rows, tmp_path)


def test_independent_auditor_metrics_match_analytic_asymmetric_quantile_fixture():
    from audit_report_round1 import independently_score

    actual = np.array([[[2., np.nan, -2.]]])
    prediction = np.zeros((1, 1, 3, 21))
    prediction[0, 0, 1] = np.nan
    result = independently_score(prediction, actual, np.array([4.]))
    # Equal positive/negative labels cancel quantile asymmetry: every WQL=1.
    assert result['masked_wql21'] == pytest.approx(1)
    assert result['crps9_approx'] == pytest.approx(1)
    assert result['mae'] == 2
    assert result['mse'] == 4
    assert result['mase24'] == .5
    assert result['coverage80'] == result['width80'] == 0
