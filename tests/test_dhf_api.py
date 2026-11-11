import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from timestral.forecaster import TimestralForecaster

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, finetune
from timestral_ft.data import DataProcessor, DHFSequenceDataset, WindowDataset


def base_checkpoint(model, tmp_path):
    path = tmp_path / "base"
    path.mkdir()
    model.config.save(path / "config.json")
    save_file(model.state_dict(), path / "model.safetensors")
    return path


def config(mode="full"):
    return TrainConfig(mode=mode, context_length=16, prediction_length=12, max_steps=4,
                       batch_size=2, gradient_accumulation=2, eval_every=2, warmup_steps=1,
                       num_threads=2, lora_rank=2, lora_alpha=4, loss_space="normalized",
                       validation_loss_space="history_scaled", training_strategy="dhf",
                       dhf_max_block_patches=3)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_resume_export_official_parity_and_validation_stays_window(model, frame, tmp_path, mode, monkeypatch):
    import timestral_ft.trainer as trainer

    base = base_checkpoint(model, tmp_path)
    cfg = config(mode)
    if mode == "lora":
        cfg.lora_dropout = .1
    data_cfg = DataConfig(target_columns=["target", "other"],
                          past_covariates=["past"], known_covariates=["known"])
    original = trainer.sample_block_layout
    layouts = []

    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        layouts.append(result.clone())
        return result

    monkeypatch.setattr(trainer, "sample_block_layout", tracked)
    direct = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, data_config=data_cfg,
                      train_config=cfg, output_dir=tmp_path / "direct")
    assert len(layouts) == cfg.max_steps * cfg.gradient_accumulation
    direct_layouts = layouts.copy()
    layouts.clear()
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, data_config=data_cfg,
             train_config=cfg, output_dir=tmp_path / "resume", stop_after_steps=1)
    resumed = finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, data_config=data_cfg,
                       train_config=cfg, output_dir=tmp_path / "resume", resume=tmp_path / "resume/checkpoint.pt")
    assert len(layouts) == len(direct_layouts)
    assert all(torch.equal(a, b) for a, b in zip(layouts, direct_layouts))
    for name, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert direct.metadata["best_validation_loss"] == resumed.metadata["best_validation_loss"]
    assert direct.metadata["dhf"]["sequence_length"] == 28
    assert direct.metadata["dhf"]["potential_supervised_steps_per_target"] == 24
    assert direct.metadata["dhf"]["layout_seed"] == cfg.seed ^ 0x444846
    records = [json.loads(line) for line in (tmp_path / "direct/history.jsonl").read_text().splitlines()]
    assert all(row["supervised_points"] == 24 * 2 * 2 * 2 for row in records)
    assert all(row["forecast_blocks"] == sum(row["block_length_histogram"].values()) for row in records)
    loaded = FineTunedModel.from_pretrained(tmp_path / "direct/model")
    kwargs = dict(history=frame.iloc[:128], future_covariates=frame.iloc[128:140])
    np.testing.assert_allclose(direct.predict(**kwargs)["median"], loaded.predict(**kwargs)["median"],
                               rtol=1e-4, atol=1e-5)
    official = TimestralForecaster.from_pretrained(tmp_path / "direct/model")
    assert official.forecast(np.arange(16, dtype=float), 12).quantiles.shape == (1, 1, 12, 21)
    if mode == "lora":
        direct.export(tmp_path / "adapter", adapter_only=True)
        adapter = FineTunedModel.from_pretrained(tmp_path / "adapter")
        np.testing.assert_allclose(direct.predict(**kwargs)["median"], adapter.predict(**kwargs)["median"],
                                   rtol=1e-4, atol=1e-5)


def test_window_indices_not_changed_by_layout_rng(model, frame, tmp_path):
    import timestral_ft.trainer as trainer

    base = base_checkpoint(model, tmp_path)
    cfg = config()
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg,
             output_dir=tmp_path / "dhf")
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base,
             train_config=replace(cfg, training_strategy="window"), output_dir=tmp_path / "window")
    a = torch.load(tmp_path / "dhf/checkpoint.pt", weights_only=True)["rng"]
    b = torch.load(tmp_path / "window/checkpoint.pt", weights_only=True)["rng"]
    assert torch.equal(a["sampler"], b["sampler"])
    assert a["dhf_layout"] is not None and b["dhf_layout"] is None
    # Validation has no access to either generator and must not draw layouts.
    dataset = WindowDataset(DataProcessor(DataConfig()).fit(frame.iloc[:96]), frame.iloc[96:128],
                            16, 12, history=frame.iloc[:96], stride=12)
    monkeypatch_original = trainer.sample_block_layout
    try:
        trainer.sample_block_layout = lambda *args, **kwargs: pytest.fail("validation sampled a DHF layout")
        assert np.isfinite(trainer._validation_loss(model, dataset, cfg))
    finally:
        trainer.sample_block_layout = monkeypatch_original


def test_dataset_sequence_covariates_and_train_boundary(frame):
    cfg = DataConfig(target_columns=["target", "other"], past_covariates=["past"], known_covariates=["known"])
    processor = DataProcessor(cfg).fit(frame.iloc[:96])
    dataset = DHFSequenceDataset(processor, frame.iloc[:96], 16, 12, min_context_steps=4)
    ordinary = WindowDataset(processor, frame.iloc[:96], 16, 12)
    assert dataset.index == ordinary.index
    assert dataset.fingerprint() != ordinary.fingerprint()
    for index in (0, len(dataset) - 1):
        item = dataset[index]
        assert item["sequence"].shape == (2, 28)
        assert item["past_covariates"].shape == (1, 28)
        assert item["known_covariates"].shape == (1, 28)
    expected = torch.tensor(frame.iloc[68:96][["target", "other"]].to_numpy().T, dtype=torch.float32)
    torch.testing.assert_close(dataset[-1]["sequence"], expected)
    # Earlier within-sequence labels remain eligible even if terminal H is missing.
    modified = frame.iloc[:28].copy()
    modified.loc[16:, ["target", "other"]] = np.nan
    kept = DHFSequenceDataset(processor, modified, 16, 12, min_context_steps=4)
    assert len(kept) == 1
    with pytest.raises(ValueError, match="No eligible windows"):
        WindowDataset(processor, modified, 16, 12)


@pytest.mark.parametrize("changes,match", [
    ({"loss_space": "history_scaled"}, "requires"),
    ({"training_strategy": "bad"}, "training_strategy"),
    ({"dhf_max_block_patches": 0}, "positive integer"),
    ({"dhf_max_block_patches": 1.2}, "positive integer"),
    ({"dhf_min_context_patches": None}, "positive integer"),
    ({"dhf_min_context_patches": True}, "positive integer"),
])
def test_config_rejects_invalid_dhf(changes, match):
    with pytest.raises(ValueError, match=match):
        replace(config(), **changes)


def test_legacy_config_artifact_checkpoint_remain_window(model, frame, tmp_path):
    base = base_checkpoint(model, tmp_path)
    cfg = replace(config(), training_strategy="window", loss_space="history_scaled", dhf_max_block_patches=None)
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg,
             output_dir=tmp_path / "run", stop_after_steps=1)
    checkpoint = tmp_path / "run/checkpoint.pt"
    state = torch.load(checkpoint, weights_only=True)
    for key in ("training_strategy", "dhf_max_block_patches", "dhf_min_context_patches"):
        state["metadata"]["train_config"].pop(key)
    state["rng"].pop("dhf_layout")
    torch.save(state, checkpoint)
    finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base, train_config=cfg,
             output_dir=tmp_path / "run", resume=checkpoint)
    manifest = tmp_path / "run/model/finetune.json"
    metadata = json.loads(manifest.read_text())
    for key in ("training_strategy", "dhf_max_block_patches", "dhf_min_context_patches"):
        metadata["train_config"].pop(key)
    manifest.write_text(json.dumps(metadata))
    loaded = FineTunedModel.from_pretrained(tmp_path / "run/model")
    assert loaded.train_config.training_strategy == "window"


@pytest.mark.parametrize("setting,match", [
    ({"dhf_max_block_patches": 9}, "single decode block"),
    ({"dhf_min_context_patches": 7}, "real target"),
    ({"context_length": 128}, "longest possible history"),
])
def test_model_constraints_rejected(model, frame, tmp_path, setting, match):
    base = base_checkpoint(model, tmp_path)
    cfg = replace(config(), **setting)
    with pytest.raises(ValueError, match=match):
        finetune(frame.iloc[:128], frame.iloc[128:], model_id=base,
                 train_config=cfg, output_dir=tmp_path / "run")


def test_dhf_categories_rejected_but_window_support_preserved(model, frame, tmp_path):
    base = base_checkpoint(model, tmp_path)
    with pytest.raises(ValueError, match="categorical covariates are deferred"):
        finetune(frame.iloc[:96], frame.iloc[96:128], model_id=base,
                 data_config=DataConfig(categorical_known_covariates=["catknown"]),
                 train_config=config(), output_dir=tmp_path / "run")
