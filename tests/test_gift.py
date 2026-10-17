"""GIFT integration invariants without downloading weights or GIFT dependencies."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from timestral_ft.compat import differentiable_forecast
from timestral_ft.gift import (
    GiftTrainConfig,
    GiftWindows,
    load_gift_adapter,
    seasonal_error_sliced,
    train_gift_task,
)
from timestral_ft.lora import adapter_state


def dataset():
    train = [{"target": np.arange(7, dtype=np.float32),
              "past_feat_dynamic_real": np.arange(7, dtype=np.float32)[None]}]
    validation = [{"target": np.arange(12, dtype=np.float32),
                   "past_feat_dynamic_real": np.arange(12, dtype=np.float32)[None]}]
    return SimpleNamespace(prediction_length=5, training_dataset=train, validation_dataset=validation)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("seasonality", [1, 4, 99])
def test_seasonal_metric_slice_is_exactly_the_official_gather(axis, masked, seasonality):
    x = np.random.randn(5, 13).astype(np.float32)
    if masked:
        x[0, :3] = np.nan
        x = np.ma.masked_invalid(x)
    n = x.shape[axis]
    lag = 1 if seasonality > n else seasonality
    expected = np.abs(np.take(x, range(lag, n), axis=axis)
                      - np.take(x, range(n-lag), axis=axis)).mean(axis=axis, keepdims=True)
    actual = seasonal_error_sliced(x, seasonality, axis)
    np.testing.assert_array_equal(actual, expected)


def test_short_history_partial_labels_and_boundary():
    ds = dataset()
    windows = GiftWindows(ds, 16)
    assert windows.summary["eligible_training_series"] == 1
    val = windows.validation[0]
    np.testing.assert_equal(val["labels"], np.arange(7, 12)[None])
    np.testing.assert_equal(val["context"][0, -7:], np.arange(7))
    assert np.isnan(val["context"][0, :-7]).all()
    np.testing.assert_equal(val["past_covariates"][0, -7:], np.arange(7))
    ds.prediction_length = 10
    ds.validation_dataset[0]["target"] = np.arange(17, dtype=np.float32)
    windows = GiftWindows(ds, 16)
    sample = windows.sample(torch.Generator().manual_seed(3))
    np.testing.assert_equal(sample["labels"][0, :6], np.arange(1, 7))
    assert np.isnan(sample["labels"][0, 6:]).all()
    # Mutation of validation labels never changes training history/label samples.
    ds.validation_dataset[0]["target"][7:] = -999
    altered = GiftWindows(ds, 16)
    other = altered.sample(torch.Generator().manual_seed(3))
    for key in sample:
        np.testing.assert_equal(other[key], sample[key])


def test_malformed_split_is_rejected():
    ds = dataset()
    ds.validation_dataset[0]["target"][0] = 999
    with pytest.raises(ValueError, match="prefixes disagree"):
        GiftWindows(ds, 16)


def test_exact_resume_and_adapter_roundtrip(model, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    model.config.save(base / "config.json")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, base / "model.safetensors")
    cfg = GiftTrainConfig(context_length=16, max_steps=4, batch_size=2, gradient_accumulation=2,
                          warmup_steps=1, eval_every=2, validation_limit=2, num_threads=2)
    ds = dataset()
    full, summary = train_gift_task(ds, task_key="test/M/short", output_dir=tmp_path / "full",
                                   config=cfg, model_id=base)
    assert summary["completed"] and summary["steps"] == 4
    paused, status = train_gift_task(ds, task_key="test/M/short", output_dir=tmp_path / "resume",
                                    config=cfg, model_id=base, stop_after_steps=1)
    assert not status["completed"]
    # A process can stop after flushing the next log row but before committing
    # its checkpoint. Resume must discard that uncommitted tail.
    with (tmp_path / "resume/history.jsonl").open("a") as log:
        log.write('{"step":2,"train_loss":999}\n{"step":')
    resumed, _ = train_gift_task(ds, task_key="test/M/short", output_dir=tmp_path / "resume",
                                config=cfg, model_id=base)
    for k, v in adapter_state(full).items():
        torch.testing.assert_close(v, adapter_state(resumed)[k], rtol=0, atol=0)
    loaded = load_gift_adapter(tmp_path / "full")
    x = torch.arange(16, dtype=torch.float32)[None, None]
    with torch.no_grad():
        torch.testing.assert_close(differentiable_forecast(loaded, x, 5),
                                   differentiable_forecast(full, x, 5), rtol=0, atol=0)
    changed = copy.copy(cfg)
    changed.seed = 123
    with pytest.raises(ValueError, match="Resume metadata changed"):
        train_gift_task(ds, task_key="test/M/short", output_dir=tmp_path / "resume",
                        config=changed, model_id=base)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_adapter_seed_is_identical_before_cpu_and_mps_updates(model, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    model.config.save(base / "config.json")
    save_file(model.state_dict(), base / "model.safetensors")
    states = []
    for device in ("cpu", "mps"):
        cfg = GiftTrainConfig(context_length=16, device=device, num_threads=2)
        m, _ = train_gift_task(dataset(), task_key="test/M/short", output_dir=tmp_path / device,
                               config=cfg, model_id=base, stop_after_steps=0)
        states.append(adapter_state(m))
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
