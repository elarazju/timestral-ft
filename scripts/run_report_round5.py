"""Process-isolated, deterministic runtime matrix (no forecast labels).

Examples: python scripts/run_report_round5.py --output runs/report-round5/preflight --preflight
          python scripts/run_report_round5.py --output runs/report-round5/benchmark
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = {"fp32": {"precision": "fp32"}, "bf16": {"precision": "bf16"},
         "compile": {"compile": True}, "bf16_compile": {"precision": "bf16", "compile": True}}
REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    return sorted(list((ROOT / "src/timestral_ft").glob("*.py"))
                  + [ROOT / "scripts" / name for name in (
                      "run_report_round5.py", "run_report_round5_gift.py",
                      "accept_report_round5.py", "audit_report_round5.py")]
                  + [ROOT / "tests/test_inference.py"]
                  + list((ROOT / "examples/report_round5").glob("*")))


def sources():
    actual = {str(p.relative_to(ROOT)): sha(p) for p in source_paths()}
    protocol = json.loads((ROOT / "results/report_round5/protocol.json").read_text())
    if actual != protocol["source_hashes"]:
        raise ValueError("Sources differ from frozen round-five protocol")
    return actual


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def case(args):
    import numpy as np
    import torch
    from timestral.loading import resolve_checkpoint

    from timestral_ft import InferenceConfig, InferenceRuntime
    from timestral_ft.compat import load_model
    from timestral_ft.inference import synchronize

    device, context, variables, mode = args.case.split(":")
    context, variables = int(context), int(variables)
    path = args.output / args.case.replace(":", "_")
    path.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(725)
    x = (torch.sin(torch.arange(context) / 13)[None, None, :]
         + torch.arange(variables)[None, :, None] / 5
         + torch.randn(1, variables, context) * .1)
    np.save(path / "input.npy", x.numpy())
    record = {"case": args.case, "shape": [1, variables, context, 96],
              "source_hashes": sources(), "input_sha256": sha(path / "input.npy"),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "machine": platform.platform(), "threads": torch.get_num_threads(),
              "requested": InferenceConfig(**MODES[mode]).to_dict(),
              "timing_scope": "input already on device; synchronized forecast only, excludes final prediction.cpu().numpy() transfer and table construction",
              "protocol_sha256": sha(ROOT / "results/report_round5/protocol.json"),
              "cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
              "cold_scope": "fresh process, fresh per-case Inductor directory; OS/file/Metal caches not cleared",
              "memory_scope": "process lifetime ru_maxrss; 10ms sampled MPS driver/current bytes from before load through predictions (may miss brief peaks)",
              "mps_driver_peak_sampled_bytes": 0, "mps_allocated_peak_sampled_bytes": 0}
    stop = threading.Event()

    def sample():
        if device == "mps":
            while not stop.is_set():
                record["mps_driver_peak_sampled_bytes"] = max(record["mps_driver_peak_sampled_bytes"], torch.mps.driver_allocated_memory())
                record["mps_allocated_peak_sampled_bytes"] = max(record["mps_allocated_peak_sampled_bytes"], torch.mps.current_allocated_memory())
                stop.wait(.01)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    runtime = None
    try:
        checkpoint = resolve_checkpoint("mistralai/Timestral-5M", revision=REVISION)
        record.update(weights_path=str(checkpoint), weights_sha256=sha(checkpoint / "model.safetensors"), config_sha256=sha(checkpoint / "config.json"), revision=REVISION)
        t = time.perf_counter()
        model = load_model(checkpoint, device=device)
        synchronize(device)
        record["load_seconds"] = time.perf_counter() - t
        setup_start = time.perf_counter()
        runtime = InferenceRuntime(model, InferenceConfig(**MODES[mode]), max_context=context)
        record["runtime_setup_seconds"] = time.perf_counter() - setup_start
        x = x.to(device)

        def timed():
            synchronize(device)
            start = time.perf_counter()
            prediction = runtime.forecast(context=x, prediction_length=96).quantiles
            synchronize(device)
            return time.perf_counter() - start, prediction.cpu().numpy()

        first, pred = timed()
        record["first_seconds"] = first
        np.save(path / "first.npy", pred)
        record["warmup_seconds"] = [timed()[0] for _ in range(2)]
        durations, predictions = [], []
        for _ in range(5):
            duration, pred = timed()
            durations.append(duration)
            predictions.append(pred)
        np.save(path / "predictions.npy", np.stack(predictions))
        record.update(first_sha256=sha(path / "first.npy"), predictions_sha256=sha(path / "predictions.npy"),
                      stable_seconds=durations, median_seconds=float(np.median(durations)),
                      throughput_forecast_points_per_second=variables * 96 / float(np.median(durations)),
                      finite=bool(np.isfinite(predictions).all()),
                      monotonic=bool((np.diff(predictions, axis=-1) >= 0).all()), status="completed")
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        stop.set()
        monitor.join()
        record["runtime"] = runtime.metadata if runtime else None
        record["rss_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        write(path / "record.json", record)
        print(json.dumps({k: record.get(k) for k in ("case", "status", "first_seconds", "median_seconds", "error")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "runs/report-round5/benchmark")
    p.add_argument("--case")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        case(args)
        return
    contexts = [256] if args.preflight else [256, 2048]
    variables = [1] if args.preflight else [1, 7]
    for device in ["cpu", "mps"]:
        for context in contexts:
            for v in variables:
                for mode in MODES:
                    name = f"{device}:{context}:{v}:{mode}"
                    folder = args.output / name.replace(":", "_")
                    if (folder / "record.json").exists():
                        saved = json.loads((folder / "record.json").read_text())
                        if saved["source_hashes"] != sources():
                            raise ValueError(f"Stale source hashes: {name}; use a new output directory")
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str((folder / "compile-cache").resolve()),
                               TORCHINDUCTOR_COMPILE_THREADS="1")
                    with (folder / "process.log").open("w") as log:
                        start = time.perf_counter()
                        try:
                            result = subprocess.run([sys.executable, __file__, "--output", str(args.output), "--case", name],
                                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                            status = result.returncode
                        except subprocess.TimeoutExpired:
                            status = "timeout_900s"
                        print(name, status, round(time.perf_counter() - start, 2), flush=True)
                    if not (folder / "record.json").exists():
                        write(folder / "record.json", {"case": name, "status": "process_failure", "error": status,
                                                       "source_hashes": sources(),
                                                       "protocol_sha256": sha(ROOT / "results/report_round5/protocol.json")})


if __name__ == "__main__":
    main()
