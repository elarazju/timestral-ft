"""Chronological tabular data preparation and leakage-safe forecast windows."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig

INTERNAL_ID = "__timestral_series__"
COVARIATE_ROLES = ("past_covariates", "known_covariates", "categorical_past_covariates",
                   "categorical_known_covariates")


def read_frame(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Expected a DataFrame, CSV, or Parquet path")


def _category_key(value):
    if isinstance(value, np.generic):
        value = value.item()
    # CSV missing values can promote integer category columns to float. Preserve
    # category identity when the future table is read back as integers.
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        value = int(value)
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


class DataProcessor:
    def __init__(self, config: DataConfig, vocabulary=None, frequency=None):
        self.config = config
        self.vocabulary = vocabulary or {}
        self.frequency = frequency or config.frequency

    @property
    def id_column(self):
        return self.config.id_column or INTERNAL_ID

    def normalize(self, data, *, columns=None):
        df = read_frame(data)
        cfg = self.config
        required = [cfg.time_column] + (columns if columns is not None else
                                       cfg.target_columns + cfg.covariate_columns)
        if cfg.id_column:
            required.append(cfg.id_column)
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Data must contain at least one row")
        if not cfg.id_column:
            if INTERNAL_ID in df:
                raise ValueError(f"{INTERNAL_ID} is reserved")
            df[INTERNAL_ID] = "single"
        if df[self.id_column].isna().any():
            raise ValueError("Series IDs cannot be missing")
        df[cfg.time_column] = pd.to_datetime(df[cfg.time_column], errors="raise")
        if df[cfg.time_column].isna().any():
            raise ValueError("Timestamps cannot be missing")
        if df.duplicated([self.id_column, cfg.time_column]).any():
            raise ValueError("Duplicate timestamps within a series")
        numeric = cfg.target_columns + cfg.past_covariates + cfg.known_covariates
        if columns is not None:
            numeric = [name for name in numeric if name in columns]
        for column in numeric:
            if column in df:
                df[column] = pd.to_numeric(df[column], errors="raise")
                if np.isinf(df[column].to_numpy(dtype=float)).any():
                    raise ValueError(f"Infinite values are not supported: {column}")
        frames = []
        for series_id, group in df.groupby(self.id_column, sort=False):
            group = group.sort_values(cfg.time_column).set_index(cfg.time_column)
            if self.frequency is None:
                inferred = pd.infer_freq(group.index) if len(group) >= 3 else None
                if inferred is None:
                    raise ValueError("Cannot infer a regular frequency; set DataConfig.frequency explicitly")
                self.frequency = inferred
            index = pd.date_range(group.index[0], group.index[-1], freq=self.frequency)
            if not group.index.isin(index).all():
                raise ValueError("Timestamps do not align with the configured sampling frequency")
            # For an implicit frequency, gaps are errors rather than silent assumptions.
            if cfg.frequency is None and len(index) != len(group):
                raise ValueError("Irregular timestamps; set DataConfig.frequency to fill the time grid")
            group = group.reindex(index)
            group[self.id_column] = series_id
            group.index.name = cfg.time_column
            frames.append(group.reset_index())
        return pd.concat(frames, ignore_index=True)

    def fit(self, data):
        df = self.normalize(data)
        self.vocabulary = {}
        for column in self.config.categorical_columns:
            keys = sorted({_category_key(v) for v in df[column].dropna().unique()})
            # 0 is reserved for unseen categories; missing remains NaN.
            self.vocabulary[column] = {key: i + 1 for i, key in enumerate(keys)}
        return self

    def feature_array(self, group, columns, categorical=False):
        if not columns:
            return None
        if not categorical:
            return group[columns].to_numpy(dtype=np.float32).T.copy()
        values = []
        for column in columns:
            mapping = self.vocabulary[column]
            values.append(np.array([np.nan if pd.isna(x) else mapping.get(_category_key(x), 0)
                                    for x in group[column]], dtype=np.float64))
        return np.stack(values)

    def window(self, group, origin, context_length, prediction_length):
        cfg = self.config
        context = group.iloc[max(0, origin - context_length):origin]
        window = {"context": self.feature_array(context, cfg.target_columns)}
        for role in COVARIATE_ROLES:
            columns = getattr(cfg, role)
            end = origin + prediction_length if "known" in role else origin
            source = group.iloc[max(0, origin - context_length):end]
            values = self.feature_array(source, columns, categorical=role.startswith("categorical"))
            if values is not None:
                window[role] = values
        return window

    def to_dict(self):
        return {"data_config": self.config.to_dict(), "frequency": self.frequency,
                "vocabulary": self.vocabulary}

    @classmethod
    def from_dict(cls, payload):
        return cls(DataConfig(**payload["data_config"]), payload["vocabulary"], payload["frequency"])


class WindowDataset(Dataset):
    """Labels belong entirely to `data`; `history` supplies earlier context only."""

    def __init__(self, processor, data, context_length, prediction_length, *, history=None, stride=1):
        if stride < 1:
            raise ValueError("stride must be positive")
        self.processor = processor
        self.context_length = context_length
        self.prediction_length = prediction_length
        labels = processor.normalize(data)
        cfg, id_col = processor.config, processor.id_column
        bounds = {sid: (g[cfg.time_column].min(), g[cfg.time_column].max())
                  for sid, g in labels.groupby(id_col, sort=False)}
        if history is not None:
            historical = processor.normalize(history)
            for sid, g in historical.groupby(id_col, sort=False):
                if sid in bounds and g[cfg.time_column].max() >= bounds[sid][0]:
                    raise ValueError("History overlaps or follows the label interval")
            # normalize accepts only public columns when it creates the internal ID.
            combined = pd.concat([historical, labels], ignore_index=True)
            if not cfg.id_column:
                combined = combined.drop(columns=INTERNAL_ID)
            combined = processor.normalize(combined)
        else:
            combined = labels
        self.groups = []
        self.index = []
        for sid, group in combined.groupby(id_col, sort=False):
            if sid not in bounds:
                continue
            group = group.sort_values(cfg.time_column).reset_index(drop=True)
            start, end = bounds[sid]
            first_label = int(group[cfg.time_column].searchsorted(start))
            first_origin = max(context_length, first_label)
            group_index = len(self.groups)
            self.groups.append(group)
            for origin in range(first_origin, len(group) - prediction_length + 1, stride):
                if group.iloc[origin + prediction_length - 1][cfg.time_column] > end:
                    break
                if group.iloc[origin][cfg.time_column] < start:
                    continue
                context = group.iloc[origin - context_length:origin][cfg.target_columns]
                future = group.iloc[origin:origin + prediction_length][cfg.target_columns]
                if (context.notna().any(axis=0) & future.notna().any(axis=0)).any():
                    self.index.append((group_index, origin))
        if not self.index:
            raise ValueError("No eligible windows: check history length, split size, and observed labels")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        group_index, origin = self.index[index]
        group = self.groups[group_index]
        item = self.processor.window(group, origin, self.context_length, self.prediction_length)
        item["labels"] = self.processor.feature_array(
            group.iloc[origin:origin + self.prediction_length], self.processor.config.target_columns
        )
        return {k: torch.from_numpy(v) for k, v in item.items()}

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps(self.processor.to_dict(), sort_keys=True).encode())
        digest.update(str((self.context_length, self.prediction_length, self.index)).encode())
        for group in self.groups:
            digest.update(pd.util.hash_pandas_object(group, index=False).values.tobytes())
        return digest.hexdigest()


def collate(items, device="cpu"):
    # Categories are encoded on CPU in float64, including codes above 2**24.
    # Moving them to MPS first would either fail or lose category identity.
    return {key: torch.stack([item[key] for item in items]).to(
        "cpu" if key.startswith("categorical_") else device
    ) for key in items[0]}
