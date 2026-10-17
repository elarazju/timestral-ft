"""Independent arithmetic and public-path checks for the dense benchmark runner."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, evaluate
from timestral_ft.data import DataProcessor

spec = importlib.util.spec_from_file_location("benchmark_suite", Path(__file__).parents[1] / "scripts/benchmark_suite.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_training_scaler_is_independent_of_future_and_uses_population_std():
    raw = np.array([[1., 3.], [3., 7.], [8., 20.], [12., 30.]])
    scaled, mean, std = benchmark.training_standardize(raw, 2)
    changed = raw.copy()
    changed[2:] = 1e9
    other, other_mean, other_std = benchmark.training_standardize(changed, 2)
    np.testing.assert_array_equal(mean, [2., 5.])
    np.testing.assert_array_equal(std, [1., 2.])
    np.testing.assert_array_equal(scaled[:2], [[-1., -1.], [1., 1.]])
    np.testing.assert_array_equal(other[:2], scaled[:2])
    np.testing.assert_array_equal(other_mean, mean)
    np.testing.assert_array_equal(other_std, std)


def test_dense_scoring_matches_public_evaluation(model, frame, tmp_path):
    columns = ["target", "other"]
    values = frame[columns].to_numpy(dtype=np.float32)
    cfg = DataConfig(target_columns=columns)
    artifact = FineTunedModel(model, DataProcessor(cfg).fit(frame.iloc[:96]),
                             TrainConfig(context_length=16, prediction_length=7), {})
    dense = benchmark.timestral_predictor(model, 16)
    dense.horizon = 7
    result = benchmark.score_forecaster(dense, values, benchmark.origins(128, 160, 7), 16, 7, tmp_path)
    public = evaluate(artifact, frame.iloc[128:], history=frame.iloc[:128], stride=1,
                      clip_nonnegative=False)
    assert result["windows"] == public.metrics["windows"] == 26
    for key in ["mae", "mse", "rmse", "coverage_80"]:
        assert result[key] == pytest.approx(public.metrics[key], rel=1e-5, abs=1e-6)
    saved = np.load(tmp_path / "predictions.npz")
    assert saved["prediction"].shape == (26, 2, 7)
    expected = np.stack([values[t:t + 7].T for t in range(128, 154)])
    np.testing.assert_array_equal(expected, saved["actual"])
    error = saved["prediction"].astype(float) - expected
    assert result["mae"] == pytest.approx(sum(abs(x) for x in error.flat) / error.size)
    assert result["mse"] == pytest.approx(sum(x * x for x in error.flat) / error.size)
    np.testing.assert_allclose(result["per_target_mse"], (error * error).mean((0, 2)))
    np.testing.assert_allclose(result["per_lead_mse"], (error * error).mean((0, 1)))


def test_zero_coverage_is_not_missing_and_invalid_predictions_fail(tmp_path):
    values = np.arange(32, dtype=np.float32).reshape(16, 2)

    def missed(x):
        pred = np.zeros((len(x), 2, 3), dtype=np.float32)
        return pred, (pred - 1, pred)

    result = benchmark.score_forecaster(missed, values, benchmark.origins(8, 16, 3), 4, 3, tmp_path)
    assert result["coverage_80"] == 0

    def invalid(x):
        return np.full((len(x), 2, 3), np.nan), None

    with pytest.raises(FloatingPointError, match="Invalid predictions"):
        benchmark.score_forecaster(invalid, values, benchmark.origins(8, 16, 3), 4, 3, tmp_path)
