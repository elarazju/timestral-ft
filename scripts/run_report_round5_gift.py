"""Frozen six-task GIFT runtime comparison; official 5M, no training or test selection.

Run with .cache/gift-probe-env/bin/python. Full official test splits and predictor;
CPU FP32 versus CPU BF16 eager. A deliberately slower precision control is retained.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from run_report_round5 import sources

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".cache/upstream/src"), str(ROOT / ".cache/gift-eval/src")]
TASKS = ["m4_hourly/H/short", "electricity/W/short", "solar/W/short", "hospital/M/short",
         "ett1/H/short", "ett2/H/long"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def official_namespace():
    from timestral_ft.gift import GIFT_REVISION
    repo = ROOT / ".cache/gift-eval"
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == GIFT_REVISION
    notebook = repo / "notebooks/timestral.ipynb"
    assert sha(notebook) == "a04b708eaf44537d5dd10d78bb509caee38f3b27b7fc1245b86d7e066d6d731d"
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {}
    for i in (1, 3, 4, 5):
        exec(compile("".join(cells[i]["source"]), f"{notebook}:cell{i}", "exec"), namespace)
    os.environ["GIFT_EVAL"] = str(ROOT / ".cache/gift-eval-data")
    namespace.update(MAX_ROWS=128, MAX_TOKENS=2**22)
    return namespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "runs/report-round5/gift")
    p.add_argument("--results", type=Path, default=ROOT / "results/report_round5/gift")
    p.add_argument("--task", choices=TASKS)
    args = p.parse_args()
    import numpy as np
    import torch
    from gluonts.model.evaluation import evaluate_forecasts
    from timestral.loading import resolve_checkpoint

    from timestral_ft import InferenceConfig, InferenceRuntime
    from timestral_ft.compat import load_model
    from timestral_ft.gift import BASE_REVISION, DATA_REVISION, GIFT_REVISION

    torch.set_num_threads(4)
    ns = official_namespace()
    properties = json.loads((ROOT / ".cache/gift-eval/notebooks/dataset_properties.json").read_text())
    all_tasks = [t for t in ns["all_configs"](properties) if t[2] in TASKS]
    assert {t[2] for t in all_tasks} == set(TASKS) and len(all_tasks) == 6
    tasks = [t for t in ns["all_configs"](properties) if t[2] in (TASKS if args.task is None else [args.task])]
    source_hashes = sources()
    base = resolve_checkpoint("mistralai/Timestral-5M", revision=BASE_REVISION)
    args.results.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, term, key in tasks:
        ds = ns["Dataset"](name, term=term, to_univariate=False)
        data_files = {str(f.relative_to(ROOT / ".cache/gift-eval-data")): sha(f)
                      for f in sorted((ROOT / ".cache/gift-eval-data" / name).rglob("*")) if f.is_file()}
        manifest = json.loads((ROOT / "results/gift_eval_5m_lora/data_manifest.json").read_text())
        expected_files = {f: info["sha256"] for f, info in manifest["files"].items() if f.startswith(name + "/")}
        assert data_files and data_files == expected_files
        assert manifest["revision"] == DATA_REVISION
        assert all(manifest["files"][f]["sha256"] == digest for f, digest in data_files.items())
        for mode in ["fp32", "bf16"]:
            path = args.output / key.replace("/", "__") / mode
            report = args.results / (key.replace("/", "__") + "." + mode + ".json")
            if report.exists():
                assert json.loads(report.read_text())["source_hashes"] == source_hashes
                continue
            path.mkdir(parents=True, exist_ok=True)
            print("START", key, mode, flush=True)
            model = load_model(base, device="cpu")
            runtime = InferenceRuntime(model, InferenceConfig(precision=mode), max_context=32768)
            predictor = ns["TimestralPredictor"](runtime, ds.prediction_length, bool(ds.past_feat_dynamic_real_dim))
            start = time.perf_counter()
            forecasts = predictor.predict(ds.test_data.input)
            prediction_seconds = time.perf_counter() - start
            arrays = np.stack([f.forecast_array for f in forecasts])
            np.save(path / "forecasts.npy", arrays)
            result = evaluate_forecasts(forecasts, test_data=ds.test_data, metrics=ns["METRICS"], batch_size=512,
                                        axis=None, mask_invalid_label=True, allow_nan_forecast=False,
                                        seasonality=ns["get_seasonality"](ds.freq))
            metrics = {k: float(result[k].iloc[0]) for k in result.columns}
            record = {"task": key, "dataset_name": name, "term": term, "mode": mode, "device": "cpu",
                      "threads": 4, "metrics": metrics, "prediction_seconds": prediction_seconds,
                      "evaluation_seconds": time.perf_counter() - start - prediction_seconds,
                      "test_windows": len(forecasts), "prediction_length": ds.prediction_length,
                      "max_context": 32768, "max_rows": 128, "max_tokens": 2**22,
                      "runtime": runtime.metadata, "source_hashes": source_hashes,
                      "weights_sha256": sha(base / "model.safetensors"), "weights_path": str(base),
                      "config_sha256": sha(base / "config.json"),
                      "protocol_sha256": sha(ROOT / "results/report_round5/protocol.json"),
                      "base_revision": BASE_REVISION, "data_revision": DATA_REVISION,
                      "gift_revision": GIFT_REVISION, "data_files": data_files,
                      "prediction_path": str(path / "forecasts.npy"), "prediction_sha256": sha(path / "forecasts.npy"),
                      "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                      "scope": "complete fixed test split; deterministic runtime comparison, no training/seeds/model generalization claim"}
            write(report, record)
            print("DONE", key, mode, prediction_seconds, metrics, flush=True)
            del runtime, predictor, model, forecasts, arrays


if __name__ == "__main__":
    main()
