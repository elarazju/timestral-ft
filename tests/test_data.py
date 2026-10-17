import numpy as np
import pandas as pd
import pytest
import torch

from timestral_ft import DataConfig
from timestral_ft.data import DataProcessor, WindowDataset


def all_features():
    return DataConfig(past_covariates=["past"], known_covariates=["known"],
                      categorical_past_covariates=["catpast"], categorical_known_covariates=["catknown"])


def test_no_future_target_leakage_and_chronological_boundaries(frame):
    cfg = all_features()
    processor = DataProcessor(cfg).fit(frame.iloc[:100])
    ds = WindowDataset(processor, frame.iloc[100:], 16, 8, history=frame.iloc[:100], stride=8)
    assert len(ds) == 7
    g, origin = ds.index[0]
    assert origin == 100
    changed = frame.copy()
    changed.loc[100:, "target"] = 9999
    other = WindowDataset(processor, changed.iloc[100:], 16, 8, history=changed.iloc[:100], stride=8)
    for key, value in ds[0].items():
        if key != "labels":
            torch.testing.assert_close(value, other[0][key])
    assert not torch.equal(ds[0]["labels"], other[0]["labels"])
    assert ds.groups[g].iloc[origin + 7]["timestamp"] <= frame["timestamp"].iloc[-1]


def test_unknown_categories_do_not_fit_validation(frame):
    cfg = all_features()
    processor = DataProcessor(cfg).fit(frame.iloc[:100])
    payload = processor.to_dict()
    changed = frame.iloc[100:].copy()
    changed["catknown"] = "new"
    ds = WindowDataset(processor, changed, 16, 8, history=frame.iloc[:100])
    assert torch.equal(ds[0]["categorical_known_covariates"][:, -8:], torch.zeros(1, 8, dtype=torch.float64))
    assert processor.to_dict() == payload


def test_numeric_category_identity_survives_missing_value_promotion(frame):
    cfg = DataConfig(categorical_known_covariates=["code"])
    frame["code"] = np.arange(len(frame), dtype=float) % 7
    frame.loc[0, "code"] = np.nan
    processor = DataProcessor(cfg).fit(frame.iloc[:100])
    future = frame.iloc[100:].copy()
    future["code"] = future["code"].astype(int)
    encoded = processor.feature_array(future, ["code"], categorical=True)
    assert (encoded > 0).all()


def test_duplicates_missing_grid_and_no_interpolation(frame):
    with pytest.raises(ValueError, match="Duplicate"):
        DataProcessor(DataConfig()).fit(pd.concat([frame, frame.iloc[:1]]))
    gap = frame.drop(index=20)
    with pytest.raises(ValueError, match="frequency"):
        DataProcessor(DataConfig()).fit(gap)
    processor = DataProcessor(DataConfig(frequency="h")).fit(gap)
    restored = processor.normalize(gap)
    assert len(restored) == len(frame)
    assert pd.isna(restored.loc[20, "target"])


def test_multiple_series_and_targets(frame):
    data = pd.concat([frame.assign(id="a"), frame.assign(id="b")], ignore_index=True)
    cfg = DataConfig(id_column="id", target_columns=["target", "other"])
    ds = WindowDataset(DataProcessor(cfg).fit(data), data, 16, 8)
    assert len(ds.groups) == 2
    assert ds[0]["context"].shape == (2, 16)


def test_history_overlap_and_empty_labels(frame):
    processor = DataProcessor(DataConfig()).fit(frame)
    with pytest.raises(ValueError, match="overlaps"):
        WindowDataset(processor, frame.iloc[100:], 16, 8, history=frame.iloc[:101])
    frame["target"] = np.nan
    with pytest.raises(ValueError, match="No eligible"):
        WindowDataset(processor, frame, 16, 8)
