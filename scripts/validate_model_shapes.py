"""Real-weight multivariate forward/backward checks at representative and boundary shapes."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from validate_models import REVISIONS

from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import check_device, differentiable_forecast, load_model, pinball_loss
from timestral_ft.lora import apply_lora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=list(REVISIONS), default=list(REVISIONS))
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--data", default="data/validation/ETTh1.csv")
    parser.add_argument("--results", default="results/validation/models")
    args = parser.parse_args()
    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data).iloc[:, 1:].to_numpy(dtype=np.float32)
    torch.set_num_threads(2)
    for device in args.devices:
        check_device(device)
        for size in args.sizes:
            path = root / f"{size}_{device.replace(':', '-')}_shapes.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if (saved["revision"] != REVISIONS[size] or saved["device"] != device
                        or saved["data_sha256"] != file_sha256(args.data)):
                    raise ValueError("Existing shape check uses different weights/device/data; choose a new result path")
                continue
            model = load_model(f"mistralai/Timestral-{size}", revision=REVISIONS[size], device=device)
            results = []
            for context, horizon in [(256, 96), (1024, 192), (256, 1024)]:
                x = torch.from_numpy(raw[:context].T.copy()).unsqueeze(0).to(device)
                y = torch.from_numpy(raw[context:context + horizon].T.copy()).unsqueeze(0).to(device)
                started = time.perf_counter()
                prediction = differentiable_forecast(model, x, horizon)
                loss = pinball_loss(prediction, y, x, model.config.quantile_knots)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
                results.append({"context": context, "horizon": horizon, "targets": 7, "batch": 1,
                                "full_finite_backward": True, "loss": float(loss.detach()),
                                "seconds": time.perf_counter() - started})
                model.zero_grad(set_to_none=True)
                del prediction, loss
                print(size, device, context, horizon, "full backward passed", flush=True)
            apply_lora(model)
            x = torch.from_numpy(raw[:256].T.copy()).unsqueeze(0).to(device)
            y = torch.from_numpy(raw[256:352].T.copy()).unsqueeze(0).to(device)
            prediction = differentiable_forecast(model, x, 96)
            loss = pinball_loss(prediction, y, x, model.config.quantile_knots)
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
            write_json(path, {"size": size, "device": device, "revision": REVISIONS[size], "passed": True,
                             "shapes": results, "lora_256_96_backward": True,
                             "data_sha256": file_sha256(args.data),
                             "scope": "single batch finite-gradient shape acceptance, no optimizer updates or quality claims"})
            del model, x, y, prediction, loss
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device.startswith("cuda"):
                torch.cuda.empty_cache()
    print(json.dumps({"completed": True}), flush=True)


if __name__ == "__main__":
    main()
