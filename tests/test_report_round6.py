"""Arithmetic edge cases whose errors would change the frozen probe's claims."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("file", ["run_report_round6.py", "audit_report_round6.py"])
def test_tied_ap_and_threshold(file):
    m = module(file)
    fn = m.metric if hasattr(m, "metric") else m.summary
    # Equal scores must not be ordered by labels; threshold equality is negative.
    result = fn(np.array([1, 0, 1, 0]), np.array([3., 3., 1., 0.]), 3.)
    assert result["auprc_ap"] == pytest.approx(.5 * .5 + .5 * (2 / 3))
    assert result["tp"] == 0 and result["fp"] == 0 and result["fn"] == 2
    result = fn(np.array([1, 0]), np.array([1., 0.]), .5)
    assert result["auprc_ap"] == result["f1"] == 1
    assert result["fpr"] == 0


@pytest.mark.parametrize("file", ["run_report_round6.py", "audit_report_round6.py"])
def test_nearest_tie_and_nontrivial_statistics(file):
    m = module(file)
    fn = m.nearest if hasattr(m, "nearest") else m.nn
    index, distance = fn(np.array([[0., 0.], [2., 0.], [3., 4.]]), np.array([[1., 0.], [3., 3.]]))
    np.testing.assert_array_equal(index, [0, 2])
    np.testing.assert_array_equal(distance, [1., 1.])
    features = m.stats if hasattr(m, "stats") else m.statistical_features
    row = features(np.array([[0., 3., 0., -3.]]))[0]
    np.testing.assert_allclose(row, [0, np.sqrt(4.5), -3, 3, 3, 3])
