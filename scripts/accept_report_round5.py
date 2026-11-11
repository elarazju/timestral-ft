"""Limited current-code real-weight larger-model acceptance; not a speed/quality benchmark."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from run_report_round5 import sources
from safetensors.torch import save_file
from timestral import TimestralForecaster
from timestral.loading import resolve_checkpoint

from timestral_ft import InferenceConfig, InferenceRuntime
from timestral_ft.artifacts import file_sha256
from timestral_ft.compat import differentiable_forecast, load_model, pinball_loss
from timestral_ft.inference import synchronize
from timestral_ft.lora import apply_lora, merge_lora

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {"28M": "180698bbf5a50dda3516aa6419798437fb03a430",
             "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
             "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "runs/report-round5/acceptance")
    p.add_argument("--results", type=Path, default=ROOT / "results/report_round5/acceptance")
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    hashes = sources()
    for size, revision in REVISIONS.items():
        checkpoint = resolve_checkpoint(f"mistralai/Timestral-{size}", revision=revision)
        for device in ["cpu", "mps"]:
            for mode in ["full", "lora"]:
                name = f"{size}_{device}_{mode}"
                report = args.results / (name + ".json")
                if report.exists():
                    assert json.loads(report.read_text())["source_hashes"] == hashes
                    continue
                print("START", name, flush=True)
                started = time.perf_counter()
                torch.manual_seed(725)
                x = torch.randn(1, 1, 64).to(device)
                labels = torch.randn(1, 1, 32).to(device)
                model = load_model(checkpoint, device=device)
                if mode == "lora":
                    apply_lora(model)
                model.train()
                out = differentiable_forecast(model, x, 32)
                loss = pinball_loss(out, labels, x, model.config.quantile_knots)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
                gradient_count = sum(p.grad is not None for p in model.parameters())
                value = float(loss.detach().cpu())
                # A small real update verifies current parameter references and merge/export.
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.add_(parameter.grad, alpha=-1e-4)
                model.zero_grad(set_to_none=True)
                del out, loss
                model.eval()
                runtime = InferenceRuntime(model)
                before = runtime.forecast(x, 32).quantiles.cpu()
                low = InferenceRuntime(model, InferenceConfig(precision="bf16"))
                low_prediction = low.forecast(x, 32).quantiles.cpu()
                assert torch.isfinite(low_prediction).all()
                assert (low_prediction.diff(dim=-1) >= 0).all()
                after = runtime.forecast(x, 32).quantiles.cpu()
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                path = args.output / name
                path.mkdir(parents=True, exist_ok=True)
                np.savez(path / "predictions.npz", context=x.cpu().numpy(), labels=labels.cpu().numpy(),
                         fp32=before.numpy(), bf16=low_prediction.numpy())
                receipts = {"fp32": runtime.metadata, "bf16": low.metadata}
                del runtime, low
                model = merge_lora(model).cpu()
                model.config.save(path / "config.json")
                save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()},
                          path / "model.safetensors")
                del model
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
                official = TimestralForecaster.from_pretrained(path)
                exported = official.forecast(x.cpu(), 32).quantiles
                tolerance = (1e-3, 1e-4) if device == "mps" else (1e-4, 1e-5)
                torch.testing.assert_close(before, exported, rtol=tolerance[0], atol=tolerance[1])
                np.save(path / "official.npy", exported.numpy())
                synchronize(device)
                record = {"case": name, "size": size, "device": device, "mode": mode,
                          "shape": [1, 1, 64, 32], "revision": revision, "weights_path": str(checkpoint),
                          "weights_sha256": file_sha256(checkpoint / "model.safetensors"), "source_hashes": hashes,
                          "config_sha256": file_sha256(checkpoint / "config.json"),
                          "protocol_sha256": file_sha256(ROOT / "results/report_round5/protocol.json"),
                          "loss": value, "finite_gradient_tensors": gradient_count,
                          "runtime": receipts, "export_path": str(path),
                          "export_sha256": file_sha256(path / "model.safetensors"),
                          "export_config_sha256": file_sha256(path / "config.json"),
                          "predictions_sha256": file_sha256(path / "predictions.npz"),
                          "official_sha256": file_sha256(path / "official.npy"),
                          "official_export_max_abs_error": float((before - exported).abs().max()),
                          "bf16_relative_l2_error": float(torch.linalg.vector_norm(low_prediction - before) / torch.linalg.vector_norm(before)),
                          "seconds": time.perf_counter() - started,
                          "scope": "one full/LoRA forward-backward-update, default FP32 and BF16 eager forecast; official CPU reload; no large-model compile or long-context quality claim"}
                report.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                print("DONE", name, record["seconds"], flush=True)
                del official, before, after, low_prediction, exported, x, labels
                gc.collect()


if __name__ == "__main__":
    main()
