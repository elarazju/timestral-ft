import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from timestral.forecaster import TimestralForecaster

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, evaluate, export, finetune
from timestral_ft.cli import main
from timestral_ft.data import read_frame


def base_checkpoint(model, tmp_path):
    path = tmp_path / "base"
    path.mkdir()
    model.config.save(path / "config.json")
    save_file(model.state_dict(), path / "model.safetensors")
    return path


def config(mode):
    return TrainConfig(mode=mode, context_length=16, prediction_length=7, max_steps=4,
                       batch_size=2, gradient_accumulation=2, eval_every=2, warmup_steps=1,
                       num_threads=2, lora_rank=2, lora_alpha=4)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_train_export_reload_and_official(model, frame, tmp_path, mode):
    base = base_checkpoint(model, tmp_path)
    cfg = DataConfig(target_columns=["target", "other"], past_covariates=["past"],
                     known_covariates=["known"], categorical_past_covariates=["catpast"],
                     categorical_known_covariates=["catknown"])
    output = tmp_path / mode
    result = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base,
                      data_config=cfg, train_config=config(mode), output_dir=output)
    assert result.metadata["steps"] == 4
    with pytest.raises(ValueError, match="positive"):
        result.predict(frame.iloc[:128], prediction_length=0)
    with pytest.raises(ValueError, match="positive"):
        evaluate(result, frame.iloc[128:], history=frame.iloc[:128], stride=0)
    loaded = FineTunedModel.from_pretrained(output / "model")
    kwargs = dict(history=frame.iloc[:128], future_covariates=frame.iloc[128:135])
    before, after = result.predict(**kwargs), loaded.predict(**kwargs)
    np.testing.assert_allclose(before["median"], after["median"], rtol=1e-4, atol=1e-5)
    assert set(after["target"]) == {"target", "other"}
    future = frame.iloc[128:135].copy()
    future["target"] = "ignored future target"
    future["other"] = np.inf
    ignored = loaded.predict(frame.iloc[:128], future_covariates=future)
    np.testing.assert_allclose(ignored["median"], after["median"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="exactly"):
        loaded.predict(frame.iloc[:128], future_covariates=frame.iloc[129:136])
    future = frame.iloc[128:135].copy()
    future.loc[129, "known"] = np.nan
    assert np.isfinite(loaded.predict(frame.iloc[:128], future_covariates=future)["median"]).all()
    official = TimestralForecaster.from_pretrained(output / "model")
    assert official.forecast(np.arange(16, dtype=float), 7).quantiles.shape == (1, 1, 7, 21)
    eval_result = evaluate(loaded, frame.iloc[128:], history=frame.iloc[:128],
                           output_dir=tmp_path / "evaluation")
    assert np.isfinite(eval_result.metrics["mae"])
    assert (tmp_path / "evaluation" / "predictions.parquet").exists()
    if mode == "lora":
        adapter_path = export(output / "model", tmp_path / "adapter", adapter_only=True)
        adapter = FineTunedModel.from_pretrained(adapter_path)
        np.testing.assert_allclose(adapter.predict(**kwargs)["median"], before["median"],
                                   rtol=1e-4, atol=1e-5)
        weights = base / "model.safetensors"
        weights.write_bytes(weights.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="Base model weights have changed"):
            FineTunedModel.from_pretrained(adapter_path)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_exact_resume(model, frame, tmp_path, mode):
    base = base_checkpoint(model, tmp_path)
    cfg = config(mode)
    if mode == "lora":
        cfg.lora_dropout = 0.1  # Exercise restoration of the global Torch RNG, not just window sampling.
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base,
                      train_config=cfg, output_dir=tmp_path / "direct")
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg,
             output_dir=tmp_path / "resumed", stop_after_steps=1)
    resumed = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg,
                       output_dir=tmp_path / "resumed", resume=tmp_path / "resumed" / "checkpoint.pt")
    for key, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert direct.metadata["best_validation_loss"] == resumed.metadata["best_validation_loss"]


def test_resume_rejects_data_change(model, frame, tmp_path):
    base = base_checkpoint(model, tmp_path)
    cfg = config("full")
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg,
             output_dir=tmp_path / "run", stop_after_steps=1)
    changed = frame.iloc[:96].copy()
    changed.loc[0, "target"] = 200
    with pytest.raises(ValueError, match="fingerprints"):
        finetune(changed, frame.iloc[96:128], model_id=base, train_config=cfg,
                 output_dir=tmp_path / "run", resume=tmp_path / "run" / "checkpoint.pt")


def test_cli_and_tabular_formats(model, frame, tmp_path):
    import yaml

    base = base_checkpoint(model, tmp_path)
    train, val, test = [tmp_path / f"{name}.csv" for name in ["train", "val", "test"]]
    frame.iloc[:96].to_csv(train, index=False)
    frame.iloc[96:128].to_csv(val, index=False)
    frame.iloc[128:].to_csv(test, index=False)
    history = tmp_path / "history.csv"
    frame.iloc[:128].to_csv(history, index=False)
    parquet = tmp_path / "train.parquet"
    frame.iloc[:96].to_parquet(parquet, index=False)
    assert len(read_frame(parquet)) == 96
    payload = {"model_id": str(base), "train_data": str(train), "validation_data": str(val),
               "data": DataConfig().to_dict(), "training": config("full").to_dict(),
               "output_dir": str(tmp_path / "cli")}
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(payload))
    main(["train", "--config", str(yaml_path)])
    main(["evaluate", "--model", str(tmp_path / "cli/model"), "--data", str(test),
          "--history", str(history), "--output", str(tmp_path / "cli-eval")])
    assert json.loads((tmp_path / "cli-eval/metrics.json").read_text())["windows"] > 0
    main(["export", "--model", str(tmp_path / "cli/model"), "--output", str(tmp_path / "export")])
    main(["predict", "--model", str(tmp_path / "cli/model"), "--history", str(train),
          "--output", str(tmp_path / "predictions.csv")])
    assert len(read_frame(tmp_path / "predictions.csv")) == 7


def test_unavailable_devices_rejected(monkeypatch):
    from timestral_ft.compat import check_device
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS.*unavailable"):
        check_device("mps")
    with pytest.raises(ValueError, match="CUDA.*unavailable"):
        check_device("cuda")
