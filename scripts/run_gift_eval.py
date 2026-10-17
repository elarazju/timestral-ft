"""Reproduce and fine-tune Timestral-5M on all 97 pinned GIFT-Eval tasks.

Run in the isolated GIFT environment; see examples/gift_eval/README.md.
Task training is independent and test metrics never feed checkpoint selection.
The official pinned notebook supplies the exact predictor and metric protocol.
"""

import argparse
import csv
import fcntl
import gc
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gift-repo", type=Path, default=ROOT / ".cache/gift-eval")
    p.add_argument("--data-dir", type=Path, default=ROOT / ".cache/gift-eval-data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/gift-eval-5m-lora")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results/gift_eval_5m_lora")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-device", help="Optional inference device; training/resume device stays unchanged")
    p.add_argument("--eval-threads", type=int, help="Optional CPU thread budget for inference workers")
    p.add_argument("--wait-training", action="store_true", help="Wait for task training in another process")
    p.add_argument("--fast-seasonal-error", action="store_true",
                   help="Replace equivalent contiguous gathers in the full-history seasonal metric with slices")
    p.add_argument("--skip-locked", action="store_true", help="Claim unoccupied task jobs across workers")
    p.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    p.add_argument("--task", action="append", help="Exact canonical key; omit for all 97")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=2**22)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--validation-limit", type=int, default=512)
    return p.parse_args()


def main():
    args = arguments()
    sys.path[:0] = [str(ROOT / ".cache/upstream/src"), str(args.gift_repo / "src")]
    import numpy as np
    import pandas as pd
    import torch
    from gift_eval.data import Dataset
    from timestral.loading import resolve_checkpoint

    from timestral_ft.artifacts import file_sha256
    from timestral_ft.compat import PortableForecaster
    from timestral_ft.gift import (
        BASE_REVISION,
        DATA_REVISION,
        GIFT_REVISION,
        GiftTrainConfig,
        atomic_json,
        load_gift_adapter,
        seasonal_error_sliced,
        train_gift_task,
    )

    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.set_num_threads(args.threads)
    head = subprocess.check_output(["git", "-C", str(args.gift_repo), "rev-parse", "HEAD"], text=True).strip()
    if head != GIFT_REVISION:
        raise ValueError("GIFT source revision differs from the fixed experiment protocol")
    notebook = args.gift_repo / "notebooks/timestral.ipynb"
    if file_sha256(notebook) != "a04b708eaf44537d5dd10d78bb509caee38f3b27b7fc1245b86d7e066d6d731d":
        raise ValueError("Official notebook differs from the verified protocol")
    properties = json.loads((args.gift_repo / "notebooks/dataset_properties.json").read_text())
    namespace = {}
    cells = json.loads(notebook.read_text())["cells"]
    for i in (1, 3, 4, 5):
        exec(compile("".join(cells[i]["source"]), f"{notebook}:cell{i}", "exec"), namespace)
    os.environ["GIFT_EVAL"] = str(args.data_dir.resolve())
    namespace.update(MAX_ROWS=args.max_rows, MAX_TOKENS=args.max_tokens)
    tasks = namespace["all_configs"](properties)
    if args.task:
        selected = set(args.task)
        if selected - {t[2] for t in tasks}:
            raise ValueError(f"Unknown tasks: {selected}")
        tasks = [t for t in tasks if t[2] in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cfg = GiftTrainConfig(device=args.device, num_threads=args.threads,
                          max_steps=args.max_steps, validation_limit=args.validation_limit)
    protocol = {
        "model": "Timestral-FT-5M-LoRA", "base_model": "mistralai/Timestral-5M",
        "base_revision": BASE_REVISION, "gift_revision": GIFT_REVISION, "data_revision": DATA_REVISION,
        "train_config": asdict(cfg), "eval_max_rows": args.max_rows, "eval_max_tokens": args.max_tokens,
        "eval_context": 32768, "test_selection": "none; checkpoint chosen only on validation loss",
        "scope": "97 independent task-specific adapters", "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
        "validation_objective": "history-scaled masked 21-quantile pinball, up to 512 fixed sampled series",
    }
    override_path = args.output_dir / "train_device_overrides.json"
    overrides = json.loads(override_path.read_text()) if override_path.exists() else {}
    if overrides:
        protocol["train_device_overrides"] = overrides
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a different output directory")
    if not protocol_path.exists():
        atomic_json(protocol_path, protocol)
    result_protocol = args.results_dir / "protocol.json"
    if result_protocol.exists() and json.loads(result_protocol.read_text()) != protocol:
        raise ValueError("Results directory belongs to a different protocol")
    if not result_protocol.exists():
        atomic_json(result_protocol, protocol)
    started = time.perf_counter()
    if args.stage in ("train", "all"):
        for index, (name, term, key) in enumerate(tasks, 1):
            directory = args.output_dir / "tasks" / key.replace("/", "__")
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / ".train.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.skip_locked else 0))
                except BlockingIOError:
                    print(f"BUSY TRAIN {key}", flush=True)
                    continue
                summary_path = directory / "training_summary.json"
                if summary_path.exists() and json.loads(summary_path.read_text())["completed"]:
                    print(f"SKIP TRAIN {index}/{len(tasks)} {key}", flush=True)
                    continue
                print(f"START TRAIN {index}/{len(tasks)} {key}", flush=True)
                dataset = Dataset(name, term=term, to_univariate=False)
                task_cfg = replace(cfg, device=overrides.get(key, cfg.device))
                accelerator_lock = None
                if task_cfg.device.startswith("mps"):
                    accelerator_lock = (args.output_dir / ".mps_train.lock").open("a")
                    try:
                        fcntl.flock(accelerator_lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.skip_locked else 0))
                    except BlockingIOError:
                        accelerator_lock.close()
                        continue
                try:
                    model, summary = train_gift_task(dataset, task_key=key, output_dir=directory, config=task_cfg)
                finally:
                    if accelerator_lock is not None:
                        accelerator_lock.close()
                if not summary["completed"] or summary["steps"] < 1:
                    raise RuntimeError(f"Incomplete task: {key}")
                summary = dict(summary, adapter_sha256=file_sha256(directory / "adapter.safetensors"))
                atomic_json(args.results_dir / f"{key.replace('/', '__')}.training.json", summary)
                base_path = resolve_checkpoint(protocol["base_model"], revision=BASE_REVISION)
                if (base_path / "LICENSE").exists():
                    (directory / "BASE_MODEL_LICENSE").write_text((base_path / "LICENSE").read_text())
                print(f"DONE TRAIN {index}/{len(tasks)} {key} seconds={summary['training_seconds']:.1f}", flush=True)
                del model, dataset
            gc.collect()
            if task_cfg.device.startswith("mps"):
                torch.mps.empty_cache()
    if args.stage in ("evaluate", "all"):
        if args.eval_threads:
            torch.set_num_threads(args.eval_threads)
        if args.fast_seasonal_error:
            import gluonts.model.evaluation as gluonts_evaluation
            gluonts_evaluation.seasonal_error = seasonal_error_sliced
        eval_device = args.eval_device or args.device
        reference = pd.read_csv(args.gift_repo / "results/Timestral-5M/all_results.csv").set_index("dataset")
        baseline = PortableForecaster.from_pretrained(
            protocol["base_model"], revision=BASE_REVISION, device=eval_device, max_context=32768)
        lock_directory = args.output_dir / "eval_locks"
        lock_directory.mkdir(exist_ok=True)
        for index, (name, term, key) in enumerate(tasks, 1):
            directory = args.output_dir / "tasks" / key.replace("/", "__")
            for kind in ("baseline", "lora"):
                result_path = args.results_dir / f"{key.replace('/', '__')}.{kind}.json"
                if result_path.exists():
                    print(f"SKIP EVAL {index}/{len(tasks)} {kind} {key}", flush=True)
                    continue
                if kind == "lora" and args.skip_locked:
                    ready = directory / "training_summary.json"
                    if not ready.exists() or not json.loads(ready.read_text())["completed"]:
                        print(f"PENDING TRAIN {key}", flush=True)
                        continue
                lock = (lock_directory / (result_path.name + ".lock")).open("a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.skip_locked else 0))
                except BlockingIOError:
                    lock.close()
                    print(f"BUSY EVAL {kind} {key}", flush=True)
                    continue
                if result_path.exists():
                    lock.close()
                    continue
                start = time.perf_counter()
                print(f"START EVAL {index}/{len(tasks)} {kind} {key}", flush=True)
                if kind == "lora":
                    if args.wait_training:
                        while not (directory / "training_summary.json").exists():
                            time.sleep(5)
                    summary = json.loads((directory / "training_summary.json").read_text())
                    if not summary["completed"]:
                        raise RuntimeError(f"Cannot evaluate unfinished adapter: {key}")
                    model = load_gift_adapter(directory, device=eval_device)
                    fc = PortableForecaster(model, max_context=32768)
                else:
                    fc = baseline
                metrics = namespace["evaluate_config"](fc, name, term)
                if not all(np.isfinite(v) and v >= 0 for v in metrics.values()):
                    raise FloatingPointError(f"Invalid evaluation metric: {key} {metrics}")
                record = {"task": key, "kind": kind, "metrics": metrics,
                          "seconds": time.perf_counter() - start, "device": eval_device,
                          "threads": torch.get_num_threads(),
                          "seasonal_error_implementation": "equivalent_slices" if args.fast_seasonal_error else "official_gather"}
                if kind == "baseline":
                    record["reference_matches"] = bool(all(
                        np.isclose(value, reference.loc[key, "eval_metrics/" + metric], rtol=1e-3, atol=1e-5)
                        for metric, value in metrics.items()))
                    if not record["reference_matches"]:
                        record["reference_difference"] = {
                            m: float(v - reference.loc[key, "eval_metrics/" + m]) for m, v in metrics.items()}
                else:
                    record["adapter_sha256"] = file_sha256(directory / "adapter.safetensors")
                atomic_json(result_path, record)
                print(f"DONE EVAL {index}/{len(tasks)} {kind} {key} "
                      f"MASE={metrics['MASE[0.5]']:.6g} CRPS={metrics['mean_weighted_sum_quantile_loss']:.6g} "
                      f"seconds={record['seconds']:.1f}", flush=True)
                if kind == "lora":
                    del model
                del fc
                gc.collect()
                if eval_device.startswith("mps"):
                    torch.mps.empty_cache()
                lock.close()
        pending = sum(not (args.results_dir / f"{key.replace('/', '__')}.{kind}.json").exists()
                      for _, _, key in tasks for kind in ("baseline", "lora"))
        if pending:
            print(f"WORKER FINISHED with {pending} evaluation records in other workers; rerun to consolidate", flush=True)
            return
        for kind, model_name in (("baseline", "Timestral-5M-reproduced"), ("lora", protocol["model"])):
            rows = []
            for name, term, key in tasks:
                record = json.loads((args.results_dir / f"{key.replace('/', '__')}.{kind}.json").read_text())
                prop = properties[key.split("/")[0]]
                rows.append({"dataset": key, "model": model_name,
                             **{f"eval_metrics/{k}": record["metrics"][k] for k in namespace["METRIC_COLUMNS"]},
                             "domain": prop["domain"], "num_variates": prop["num_variates"]})
            filename = f"{kind}_all_results.csv" if len(tasks) == 97 else f"partial_{kind}_results.csv"
            with (args.results_dir / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda row: row["dataset"]))
    print(f"FINISHED stage={args.stage} tasks={len(tasks)} wall_seconds={time.perf_counter()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
