"""Real-weight CPU/MPS/CUDA functional acceptance for all Timestral model sizes.

These are deliberately short correctness runs, not large-model quality benchmarks.
Uses a pinned ETTh1 slice, four covariate types, two targets, missing history,
non-patch-aligned lengths, full/LoRA resume, merged and adapter exports.
"""

import argparse
import gc
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timestral.forecaster import TimestralForecaster

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, finetune
from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import check_device, differentiable_forecast, load_model, pinball_loss
from timestral_ft.data import DataProcessor, WindowDataset, collate
from timestral_ft.lora import apply_lora, merge_lora

REVISIONS = {"5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
             "28M": "180698bbf5a50dda3516aa6419798437fb03a430",
             "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
             "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"}


def release(device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def data_fixture(path):
    frame = pd.read_csv(path, parse_dates=["date"]).iloc[:443].copy()
    hour = frame["date"].dt.hour
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["past_category"] = (hour % 3).astype(str)
    frame["known_category"] = frame["date"].dt.dayofweek.astype(str)
    frame.loc[20:23, "OT"] = np.nan
    cfg = DataConfig(time_column="date", target_columns=["OT", "HULL"], past_covariates=["HUFL"],
                     known_covariates=["hour_sin"], categorical_past_covariates=["past_category"],
                     categorical_known_covariates=["known_category"], frequency="h")
    return frame, cfg


def assert_merge_close(actual, expected, context):
    """FP32 distributivity can cancel around zero in original physical units.

    Keep the strict original tolerance for unmodified CPU forecasts. For
    LoRA algebraic reordering only, compare in each history's std units.
    The absolute floor is 0.0001 history std, and raw residuals are reported.
    """
    context = context.detach().cpu().numpy()
    scale = np.nanstd(context, axis=-1, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.)[..., None]
    a, b = actual.detach().cpu().numpy(), expected.detach().cpu().numpy()
    np.testing.assert_allclose(a / scale, b / scale, rtol=1e-4, atol=1e-4)
    return {"max_abs_error": float(np.abs(a - b).max()),
            "max_abs_error_in_history_std": float((np.abs(a - b) / scale).max()),
            "relative_l2_error": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            "tolerance": "rtol1e-4, atol1e-4 after division by target history population std"}


def validate(size, device, args):
    started = time.perf_counter()
    frame, data_config = data_fixture(args.data)
    base_id, revision = f"mistralai/Timestral-{size}", REVISIONS[size]
    root = Path(args.output) / f"{size}_{device.replace(':', '-')}"
    report = Path(args.results) / f"{size}_{device.replace(':', '-')}.json"
    if report.exists():
        saved = json.loads(report.read_text())
        if (saved["revision"] != revision or saved["device"] != device
                or saved["test_data_sha256"] != file_sha256(args.data)):
            raise ValueError("Existing model acceptance uses different weights/device/data; choose new output paths")
        return saved
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = load_model(base_id, revision=revision)
    processor = DataProcessor(data_config).fit(frame.iloc[:320])
    dataset = WindowDataset(processor, frame.iloc[:320], 63, 41)
    cpu_batch = collate([dataset[0]], "cpu")
    labels = cpu_batch.pop("labels")
    expected = TimestralForecaster(model, max_context=63, quantile_real_cap_k=0).forecast(
        **cpu_batch, prediction_length=41, symmetric_averaging=False, clip_nonnegative=False).quantiles.clone()
    count = sum(p.numel() for p in model.parameters())
    model.to(device)
    model.transformer.rope.rebuild_inv_freq()
    batch = collate([dataset[0]], device)
    labels = batch.pop("labels")
    prediction = differentiable_forecast(model, prediction_length=41, **batch)
    rtol, atol = (1e-4, 1e-5) if device == "cpu" else (1e-3, 1e-4)
    torch.testing.assert_close(prediction.cpu(), expected, rtol=rtol, atol=atol)
    parity_error = float((prediction.detach().cpu() - expected).abs().max())
    loss = pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    full_loss = float(loss.detach())
    model.zero_grad(set_to_none=True)
    del prediction, loss
    apply_lora(model)
    adapter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prediction = differentiable_forecast(model, prediction_length=41, **batch)
    torch.testing.assert_close(prediction.cpu(), expected, rtol=rtol, atol=atol)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    pinball_loss(prediction, labels, batch["context"], model.config.quantile_knots).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    before_merge = differentiable_forecast(model, prediction_length=41, **batch).detach()
    merge_lora(model)
    merge_check = assert_merge_close(differentiable_forecast(model, prediction_length=41, **batch),
                                     before_merge, batch["context"])
    del model, optimizer, prediction, before_merge
    release(device)
    modes = {}
    for mode in ["full", "lora"]:
        logging.info("%s %s %s train/resume/export", size, device, mode)
        cfg = TrainConfig(mode=mode, context_length=63, prediction_length=41, max_steps=4,
                          batch_size=1, gradient_accumulation=1, eval_every=2, warmup_steps=1,
                          num_threads=2, device=device, lora_dropout=.1 if mode == "lora" else 0.)
        kwargs = dict(model_id=base_id, revision=revision, data_config=data_config, train_config=cfg)
        direct_path, resume_path = root / mode / "direct", root / mode / "resumed"
        if (direct_path / "model/finetune.json").exists():
            direct = FineTunedModel.from_pretrained(direct_path / "model", device=device,
                                                   prefer_adapter=mode == "lora")
        else:
            direct = finetune(frame.iloc[:320], frame.iloc[320:402], output_dir=direct_path,
                              resume=direct_path / "checkpoint.pt" if (direct_path / "checkpoint.pt").exists() else None,
                              **kwargs)
        checkpoint = resume_path / "checkpoint.pt"
        if not checkpoint.exists():
            finetune(frame.iloc[:320], frame.iloc[320:402], output_dir=resume_path, stop_after_steps=2, **kwargs)
        if (resume_path / "model/finetune.json").exists():
            resumed = FineTunedModel.from_pretrained(resume_path / "model", device=device,
                                                    prefer_adapter=mode == "lora")
        else:
            resumed = finetune(frame.iloc[:320], frame.iloc[320:402], output_dir=resume_path,
                               resume=checkpoint, **kwargs)
        for key, value in direct.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[key],
                                       rtol=0 if device == "cpu" else 1e-5,
                                       atol=0 if device == "cpu" else 1e-6)
        args_predict = {"history": frame.iloc[:402], "future_covariates": frame.iloc[402:]}
        original = direct.predict(**args_predict)["median"].to_numpy()
        loaded = FineTunedModel.from_pretrained(direct_path / "model", device=device)
        np.testing.assert_allclose(original, loaded.predict(**args_predict)["median"], rtol=rtol, atol=atol)
        if mode == "lora":
            adapter_path = root / mode / "adapter"
            if not adapter_path.exists():
                direct.export(adapter_path, adapter_only=True)
            adapter = FineTunedModel.from_pretrained(adapter_path, device=device)
            np.testing.assert_allclose(original, adapter.predict(**args_predict)["median"], rtol=rtol, atol=atol)
            del adapter
        del resumed, loaded
        release(device)
        # Check exported official weights numerically, with the same low-level inputs.
        official = TimestralForecaster.from_pretrained(direct_path / "model", max_context=63,
                                                       quantile_real_cap_k=0)
        official_q = official.forecast(**cpu_batch, prediction_length=41,
                                       symmetric_averaging=False, clip_nonnegative=False).quantiles
        actual_q = differentiable_forecast(direct.model, prediction_length=41, **batch).detach().cpu()
        if mode == "lora":
            export_check = assert_merge_close(actual_q, official_q, batch["context"])
        else:
            torch.testing.assert_close(actual_q, official_q, rtol=rtol, atol=atol)
            export_check = {"max_abs_error": float((actual_q - official_q).abs().max())}
        modes[mode] = {"passed": True, "steps": 4, "trainable_parameters": direct.metadata["trainable_parameters"],
                       "training_seconds": direct.metadata["training_seconds"],
                       "resume": "bitwise identical" if device == "cpu" else "rtol1e-5 atol1e-6",
                       "official_export_max_abs_error": float((actual_q - official_q).abs().max()),
                       "official_export_check": export_check,
                       "merged_reload": True, "adapter_reload": mode == "lora",
                       "weights_sha256": file_sha256(direct_path / "model/model.safetensors")}
        del direct, official, actual_q, official_q
        release(device)
    result = {"size": size, "device": device, "revision": revision, "passed": True,
              "total_parameters": count, "lora_parameters": adapter_count, "base_parity_max_abs_error": parity_error,
              "finite_full_backward_loss": full_loss, "full_backward": True, "lora_zero_delta": True,
              "lora_finite_backward_and_frozen_base": True, "lora_merge_parity": True,
              "lora_merge_check": merge_check,
              "modes": modes, "elapsed_seconds": time.perf_counter() - started,
              "test_data_sha256": file_sha256(args.data), "test_rows": [0, 443],
              "context": 63, "horizon": 41, "targets": data_config.target_columns,
              "four_covariate_types": True, "missing_history": True,
              "scope": "four-step functional acceptance on a small real-data slice, not quality or capacity validation",
              "torch": str(torch.__version__), "platform": platform.platform()}
    write_json(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=list(REVISIONS), default=list(REVISIONS))
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--data", default="data/validation/ETTh1.csv")
    parser.add_argument("--output", default="runs/validation/models")
    parser.add_argument("--results", default="results/validation/models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    Path(args.results).mkdir(parents=True, exist_ok=True)
    for device in args.devices:
        check_device(device)
        for size in args.sizes:
            result = validate(size, device, args)
            print(json.dumps({"size": size, "device": device, "passed": result["passed"],
                              "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
