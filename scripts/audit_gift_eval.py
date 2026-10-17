"""Audit the full 97-task run and summarize all regressions as well as gains."""

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".cache/upstream/src"))


def gmean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/gift-eval-5m-lora")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/gift_eval_5m_lora")
    parser.add_argument("--gift-repo", type=Path, default=ROOT / ".cache/gift-eval")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from timestral.loading import resolve_checkpoint

    from timestral_ft.artifacts import file_sha256
    from timestral_ft.gift import atomic_json

    def read(path):
        return json.loads(path.read_text())

    with (args.gift_repo / "scripts/dataset_configurations.csv").open() as handle:
        expected = {r["dataset"]: r for r in csv.DictReader(handle)}
    with (args.gift_repo / "results/Seasonal_Naive/all_results.csv").open() as handle:
        seasonal = {r["dataset"]: r for r in csv.DictReader(handle)}
    protocol = read(args.run_dir / "protocol.json")
    base = resolve_checkpoint(protocol["base_model"], revision=protocol["base_revision"])
    base_sha256 = file_sha256(base / "model.safetensors")
    data_manifest = read(args.results_dir / "data_manifest.json")
    for relative, info in data_manifest["files"].items():
        path = ROOT / ".cache/gift-eval-data" / relative
        if path.stat().st_size != info["bytes"] or file_sha256(path) != info["sha256"]:
            raise ValueError(f"Dataset changed during the experiment: {relative}")
    import pyarrow as pa
    import pyarrow.compute as pc
    from datasets import load_from_disk

    inventory = {r["key"]: r for r in read(ROOT / "results/gift_eval_feasibility/task_inventory.json")}
    lengths_cache = {}
    keys = sorted(expected)
    if len(keys) != 97:
        raise ValueError("Official task count changed")
    records, errors = [], []
    for key in keys:
        stem = key.replace("/", "__")
        directory = args.run_dir / "tasks" / stem
        summary = read(directory / "training_summary.json")
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        adapter = load_file(directory / "adapter.safetensors")
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        baseline = read(args.results_dir / f"{stem}.baseline.json")
        tuned = read(args.results_dir / f"{stem}.lora.json")
        validations = [h for h in history if "validation_loss" in h]
        best = min(validations, key=lambda h: h["validation_loss"])
        task_info = inventory[key]
        source_name = task_info["dataset"]
        if source_name not in lengths_cache:
            raw = load_from_disk(str(ROOT / ".cache/gift-eval-data" / source_name))
            column = raw.data.column("target")
            if pa.types.is_list(column.type.value_type) or pa.types.is_large_list(column.type.value_type):
                column = pc.list_flatten(pc.list_slice(column, 0, 1))
            lengths_cache[source_name] = pc.list_value_length(column).to_numpy()
        raw_lengths = lengths_cache[source_name]
        horizon = task_info["horizon"]
        windows = 1 if "m4" in source_name else min(20, max(1, math.ceil(.1*int(raw_lengths.min())/horizon)))
        training_lengths = raw_lengths - horizon * (windows + 1)
        expected_config = dict(protocol["train_config"])
        expected_config["device"] = protocol.get("train_device_overrides", {}).get(key, expected_config["device"])
        checks = {
            "completed": summary["completed"],
            "steps_match": len(history) == summary["steps"] and summary["steps"] > 0,
            "step_order": [h["step"] for h in history] == list(range(1, summary["steps"] + 1)),
            "finite_training": all(math.isfinite(h["train_loss"]) for h in history),
            "validation_selection": best["step"] == summary["best_step"]
            and best["validation_loss"] == summary["validation_loss"],
            "adapter_changed": any(bool(v.count_nonzero()) for k, v in adapter.items() if k.endswith("lora_B")),
            "finite_adapter": all(bool(v.isfinite().all()) for v in adapter.values()),
            "adapter_only": all(k.endswith(("lora_A", "lora_B")) for k in adapter),
            "trainable_parameter_count": sum(v.numel() for v in adapter.values()) == summary["trainable_parameters"],
            "saved_best_checkpoint": set(adapter) == set(checkpoint["best_adapter"])
            and all(torch.equal(v, checkpoint["best_adapter"][k]) for k, v in adapter.items()),
            "checkpoint_complete": checkpoint["step"] == summary["steps"],
            "adapter_matches_evaluation": file_sha256(directory / "adapter.safetensors") == tuned["adapter_sha256"],
            "task_identity": summary["task"] == baseline["task"] == tuned["task"] == key,
            "independent_base": summary["base_revision"] == protocol["base_revision"],
            "base_weights_checksum": summary["base_weights_sha256"] == base_sha256,
            "training_config": summary["train_config"] == expected_config,
            "official_training_cutoffs": summary["data"]["train_length_min"] == int(training_lengths.min())
            and summary["data"]["train_length_max"] == int(training_lengths.max())
            and summary["data"]["series"] == len(raw_lengths)
            and summary["data"]["horizon"] == horizon,
        }
        if not all(checks.values()):
            errors.append({"task": key, "checks": checks})
        row = {"task": key, "domain": expected[key]["domain"], "term": key.split("/")[-1],
               "training_device": summary["train_config"]["device"],
               "steps": summary["steps"], "best_step": summary["best_step"],
               "trainable_parameters": summary["trainable_parameters"],
               "training_seconds": summary["training_seconds"],
               "baseline_reference_matches": baseline["reference_matches"], "checks": checks}
        for label, metric in (("crps", "mean_weighted_sum_quantile_loss"), ("mase", "MASE[0.5]")):
            b, t, s = baseline["metrics"][metric], tuned["metrics"][metric], float(seasonal[key]["eval_metrics/" + metric])
            if not all(math.isfinite(v) and v > 0 for v in (b, t, s)):
                raise ValueError(f"Cannot compute positive geometric mean: {key} {metric}")
            row.update({f"baseline_{label}": b, f"lora_{label}": t,
                        f"baseline_normalized_{label}": b/s, f"lora_normalized_{label}": t/s,
                        f"{label}_ratio": t/b, f"{label}_change_percent": (t/b-1)*100})
        records.append(row)
        # Recover a training receipt if the runner was interrupted just after completion.
        atomic_json(args.results_dir / f"{stem}.training.json",
                    dict(summary, adapter_sha256=file_sha256(directory / "adapter.safetensors")))
    if errors:
        atomic_json(args.results_dir / "audit_errors.json", errors)
        raise RuntimeError(f"{len(errors)} task audit failures")

    def aggregate(rows):
        result = {"tasks": len(rows)}
        for metric in ("crps", "mase"):
            base = gmean([r[f"baseline_normalized_{metric}"] for r in rows])
            tuned = gmean([r[f"lora_normalized_{metric}"] for r in rows])
            result[metric] = {"baseline_normalized_gmean": base, "lora_normalized_gmean": tuned,
                              "relative_change_percent": (tuned/base-1)*100,
                              "improved": sum(r[f"{metric}_ratio"] < 1-1e-8 for r in rows),
                              "regressed": sum(r[f"{metric}_ratio"] > 1+1e-8 for r in rows)}
        return result

    aggregate_results = aggregate(records)
    aggregate_results.update(
        by_domain={d: aggregate([r for r in records if r["domain"] == d]) for d in sorted({r["domain"] for r in records})},
        by_term={d: aggregate([r for r in records if r["term"] == d]) for d in ("short", "medium", "long")},
        training_seconds=sum(r["training_seconds"] for r in records),
        training_devices={d: sum(r["training_device"] == d for r in records)
                          for d in sorted({r["training_device"] for r in records})},
        baseline_reference_matches=sum(r["baseline_reference_matches"] for r in records),
        best_step_counts={str(step): sum(r["best_step"] == step for r in records) for step in (50, 100, 150, 200)},
    )
    atomic_json(args.results_dir / "aggregates.json", aggregate_results)
    atomic_json(args.results_dir / "audit.json", {"tasks": records, "all_checks_pass": True,
                "ranking_note": "Normalized geometric means are scores, not average per-task ranks.",
                "scope": "One seed and fixed 200-step LoRA recipe; does not establish universal adaptation gains."})
    export_rows = [{k: v for k, v in r.items() if k != "checks"} for r in records]
    with (args.results_dir / "task_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(export_rows[0]))
        writer.writeheader()
        writer.writerows(export_rows)

    # A local schema-valid draft only: do not invent public URLs or availability.
    submission = args.results_dir / "submission" / protocol["model"]
    submission.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.results_dir / "lora_all_results.csv", submission / "all_results.csv")
    atomic_json(submission / "config.json", {
        "model": protocol["model"], "model_type": "fine-tuned", "model_dtype": "float32",
        "model_link": None, "code_link": None, "org": None,
        "testdata_leakage": "No", "replication_code_available": "No",
    })
    spec = importlib.util.spec_from_file_location("official_validator", args.gift_repo / "scripts/validate_results.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    validation_errors = validator.validate_submission(submission)
    if validation_errors:
        raise RuntimeError(f"Official schema validation failed: {validation_errors}")
    atomic_json(args.results_dir / "submission_validation.json", {
        "schema_valid": True, "publicly_submitted": False, "public_weights_and_code_available": False,
        "remaining_publication_fields": ["model_link", "code_link", "org", "replication_code_available"],
        "testdata_leakage_note": "Our adaptation uses official training and validation only. The base entry declares no leakage; base pretraining was not independently audited.",
    })
    print(json.dumps(aggregate_results, indent=2))


if __name__ == "__main__":
    main()
