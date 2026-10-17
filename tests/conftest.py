import numpy as np
import pandas as pd
import pytest
import torch
from timestral.config import TimestralConfig
from timestral.model import TimestralModel


@pytest.fixture(autouse=True)
def seeds():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    np.random.seed(7)


@pytest.fixture
def model():
    return TimestralModel(TimestralConfig(
        patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=32,
        block_schedule=("time_variate", "time_variate"),
        quantile_knots=(.01, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5,
                        .55, .6, .65, .7, .75, .8, .85, .9, .95, .99),
        block_size=8, context_length=128,
    ))


@pytest.fixture
def frame():
    n = 160
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "target": np.sin(np.arange(n) / 8) + 2,
        "other": np.cos(np.arange(n) / 8) + 2,
        "past": np.arange(n) / n,
        "known": np.sin(np.arange(n) / 24),
        "catpast": np.where(np.arange(n) % 2, "a", "b"),
        "catknown": (np.arange(n) % 7).astype(str),
    })
