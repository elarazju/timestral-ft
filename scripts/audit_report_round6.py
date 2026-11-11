"""Independent arithmetic and provenance audit; never imports the experiment runner.

Rebuilds raw inputs, injection recipes, 1-NN decisions, six-feature baseline,
calibration thresholds and every reported metric from retained arrays. Does not
recompute neural embeddings; model/API agreement is covered by mechanism tests
and the separately retained CPU/MPS acceptance outputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/report-round6"
LIGHT = ROOT / "results/report_round6"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def nn(bank, queries):
    indices, distances = [], []
    for query in queries:
        d = np.linalg.norm(bank.astype(np.float64) - query.astype(np.float64), axis=1)
        i = int(np.argmin(d))
        indices.append(i)
        distances.append(d[i])
    return np.array(indices), np.array(distances)


def summary(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    prediction = scores > threshold
    tp = sum(int(y == 1 and p) for y, p in zip(labels, prediction))
    fp = sum(int(y == 0 and p) for y, p in zip(labels, prediction))
    fn = int(labels.sum()) - tp
    tn = len(labels) - int(labels.sum()) - fp
    previous_recall, area = 0., 0.
    # Group equal scores: AP is NOT trapezoidal integration or tie-breaking by label.
    for cutoff in sorted(set(scores.tolist()), reverse=True):
        selected = scores >= cutoff
        recall = labels[selected].sum() / labels.sum()
        precision = labels[selected].mean()
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return {"auprc_ap": float(area), "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "fpr": fp / max(1, fp + tn), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(labels), "positive": int(labels.sum())}


def check_tree(expected, actual):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), (set(expected), set(actual))
        for key in expected:
            check_tree(expected[key], actual[key])
    else:
        equal(expected, actual)


def statistical_features(x):
    result = []
    for row in x.astype(np.float64):
        delta = np.abs(row[1:] - row[:-1])
        mean = sum(row) / len(row)
        sd = np.sqrt(sum((row - mean) ** 2) / len(row))
        result.append([mean, sd, min(row), max(row), sum(delta) / len(delta), max(delta)])
    return np.array(result)


def source_audit(protocol):
    paths = set((ROOT / "src/timestral_ft").glob("*.py")) | set((ROOT / "tests").glob("*.py"))
    paths |= {ROOT / "scripts/run_report_round6.py", ROOT / "scripts/audit_report_round6.py", ROOT / "pyproject.toml"}
    paths |= {p for p in (ROOT / "examples/report_round6").glob("*") if p.is_file()}
    expected = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    assert expected == protocol["source_hashes"], "Full source set differs, not just selected keys"
    snapshot = OUT / "source_snapshot"
    for path, sha in expected.items():
        assert digest(snapshot / path) == sha
    import timestral
    upstream = {p.name: digest(p) for p in Path(timestral.__file__).parent.glob("*.py")}
    assert upstream == protocol["upstream_source_hashes"]
    for path, sha in upstream.items():
        assert digest(snapshot / "upstream/timestral" / path) == sha
    for filename in ("protocol.json", "data_manifest.json"):
        assert digest(LIGHT / filename) == digest(OUT / filename) == digest(snapshot / filename)
    assert json.loads((LIGHT / "data_manifest.json").read_text()) == protocol["data"]
    for name, entry in protocol["data"].items():
        assert digest(OUT / "data" / name) == entry["sha256"]
    from timestral.loading import resolve_checkpoint
    model = protocol["model"]
    checkpoint = resolve_checkpoint(model["repo"], revision=model["revision"])
    assert digest(checkpoint / "model.safetensors") == model["weights_sha256"]
    assert digest(checkpoint / "config.json") == model["config_sha256"]
    assert digest(snapshot / "checkpoint_config.json") == model["config_sha256"]


def arrays_audit(result):
    assert set(result["arrays"]) == {"classification.npz", "detection.npz"}
    arrays = {}
    for filename, entry in result["arrays"].items():
        assert digest(OUT / filename) == entry["file_sha256"]
        arrays[filename] = np.load(OUT / filename)
        z = arrays[filename]
        assert set(z.files) == set(entry["arrays"])
        for key in z.files:
            assert np.isfinite(z[key]).all()
            assert list(z[key].shape) == entry["arrays"][key]["shape"]
            assert str(z[key].dtype) == entry["arrays"][key]["dtype"]
            assert hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest() == entry["arrays"][key]["sha256_bytes"]
    assert digest(OUT / "environment-packages.json") == result["environment"]["packages_sha256"]
    return arrays


def classification(z, expected):
    required = {f"{split}_{feature}" for split in ("train", "test") for feature in ("raw", "labels", "mean", "last")}
    required |= {f"{method}_{field}" for method in ("raw", "mean", "last") for field in ("prediction", "neighbor", "distance")}
    assert set(z.files) == required
    for split, count in (("train", 50), ("test", 150)):
        content = (OUT / f"data/GunPoint_{split.upper()}.ts").read_text().splitlines()
        data = False
        rows, labels = [], []
        for line in content:
            if line.lower().strip() == "@data":
                data = True
            elif data and line.strip():
                values, label = line.rsplit(":", 1)
                rows.append(np.fromstring(values, sep=","))
                labels.append(int(label))
        equal(z[f"{split}_raw"], np.array(rows, dtype=np.float32))
        equal(z[f"{split}_labels"], labels)
        assert z[f"{split}_raw"].shape == (count, 150)
        assert set(labels) == {1, 2}
    reports = {}
    for method in ("raw", "mean", "last"):
        neighbor, distance = nn(z[f"train_{method}"], z[f"test_{method}"])
        predictions = z["train_labels"][neighbor]
        equal(z[f"{method}_prediction"], predictions)
        equal(z[f"{method}_neighbor"], neighbor)
        equal(z[f"{method}_distance"], distance)
        confusion = np.zeros((2, 2), dtype=int)
        for truth, predicted in zip(z["test_labels"], predictions):
            confusion[truth - 1, predicted - 1] += 1
        reports[method] = {"accuracy": np.trace(confusion) / 150, "correct": int(np.trace(confusion)), "n": 150,
                           "confusion_matrix_true_rows_pred_columns_1_2": confusion.tolist()}
    for method in ("mean", "last"):
        both = np.stack([z[f"{method}_prediction"] == z["test_labels"], z["raw_prediction"] == z["test_labels"]])
        reports[method]["difference_vs_raw"] = {"accuracy_percentage_points": float(100 * (both[0].mean() - both[1].mean())),
                                               "embedding_only_correct": int(np.count_nonzero(both[0] & ~both[1])),
                                               "raw_only_correct": int(np.count_nonzero(~both[0] & both[1]))}
    accuracies = [reports[method]["accuracy"] for method in ("raw", "mean", "last")]
    reports["accuracy_range_prespecified_methods"] = [min(accuracies), max(accuracies)]
    check_tree(expected, reports)
    return reports


def detection(z, expected):
    required = {"raw_ot", "point_mean", "point_scale", "statistics_center", "statistics_scale"}
    required |= {f"{s}_{f}" for s in ("train", "calibration", "test") for f in ("start", "input")}
    required |= {f"{s}_{m}" for s in ("train", "calibration") for m in ("embedding", "statistics")}
    required |= {f"calibration_{m}_score" for m in ("embedding", "statistics")}
    for seed in (42, 43, 44):
        required |= {f"s{seed}_{f}" for f in ("input", "injection", "embedding", "statistics", "labels")}
        required |= {f"s{seed}_{m}_{f}" for m in ("embedding", "statistics") for f in ("score", "prediction", "neighbor")}
    assert set(z.files) == required
    raw = pd.read_csv(OUT / "data/ETTh2.csv").OT.to_numpy()
    equal(z["raw_ot"], raw)
    center, scale = np.mean(raw[:8640]), np.std(raw[:8640])
    equal([z["point_mean"], z["point_scale"]], [center, scale])
    for name, a, b in (("train", 0, 8640), ("calibration", 8640, 11520), ("test", 11520, 14400)):
        starts = list(range(a, b - 95, 24))
        equal(z[f"{name}_start"], starts)
        x = np.array([(raw[i:i + 96] - center) / scale for i in starts], dtype=np.float32)
        equal(z[f"{name}_input"], x)
        if name != "test":
            equal(z[f"{name}_statistics"], statistical_features(x))
    stats_center = statistical_features(z["train_input"]).mean(0)
    stats_scale = np.maximum(statistical_features(z["train_input"]).std(0), 1e-12)
    equal(z["statistics_center"], stats_center)
    equal(z["statistics_scale"], stats_scale)
    thresholds, banks = {}, {}
    for method in ("embedding", "statistics"):
        bank, calibration = z[f"train_{method}"], z[f"calibration_{method}"]
        if method == "statistics":
            bank, calibration = (bank - stats_center) / stats_scale, (calibration - stats_center) / stats_scale
        banks[method] = bank
        score = nn(bank, calibration)[1]
        equal(z[f"calibration_{method}_score"], score)
        ordered = np.sort(score)
        offset = .99 * (len(ordered) - 1)
        lo = int(np.floor(offset))
        threshold = ordered[lo] + (offset - lo) * (ordered[int(np.ceil(offset))] - ordered[lo])
        thresholds[method] = float(threshold)
    report = {"thresholds": thresholds, "window_counts": {s: len(z[f"{s}_input"]) for s in ("train", "calibration", "test")}, "seeds": {}}
    for seed in (42, 43, 44):
        prefix = f"s{seed}_"
        original = np.array([raw[i:i + 96] for i in z["test_start"]])
        n = len(original)
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, int(.3 * n), replace=False)
        kinds = np.array(([1, 2, 3] * n)[:len(indices)])
        rng.shuffle(kinds)
        injection = np.zeros((n, 4), dtype=np.int64)
        for index, kind in zip(indices, kinds):
            start = int(rng.integers(24, 72))
            sign = int(rng.choice([-1, 1]))
            duration = 1 if kind == 1 else 24
            injection[index] = [kind, start, duration, sign]
            for point in range(start, start + duration):
                if kind == 3:
                    original[index, point] = original[index, start - 1]
                else:
                    original[index, point] += sign * scale * {1: 6, 2: 3}[kind]
        equal(z[prefix + "injection"], injection)
        equal(z[prefix + "input"], ((original - center) / scale).astype(np.float32))
        equal(z[prefix + "statistics"], statistical_features(z[prefix + "input"]))
        labels = injection[:, 0] > 0
        equal(z[prefix + "labels"], labels)
        report["seeds"][str(seed)] = {}
        for method in ("embedding", "statistics"):
            query = z[prefix + method]
            if method == "statistics":
                query = (query - stats_center) / stats_scale
            neighbors, score = nn(banks[method], query)
            equal(z[prefix + method + "_score"], score)
            equal(z[prefix + method + "_neighbor"], neighbors)
            equal(z[prefix + method + "_prediction"], score > thresholds[method])
            metrics = {"overall": summary(labels, score, thresholds[method]), "by_type": {}}
            for kind, name in ((1, "spike"), (2, "level_shift"), (3, "stuck")):
                selected = np.isin(injection[:, 0], [0, kind])
                metrics["by_type"][name] = summary(labels[selected], score[selected], thresholds[method])
            report["seeds"][str(seed)][method] = metrics
    report["seed_summary"] = {}
    for method in ("embedding", "statistics"):
        report["seed_summary"][method] = {}
        for group in ("overall", "spike", "level_shift", "stuck"):
            values = [s[method]["overall"] if group == "overall" else s[method]["by_type"][group] for s in report["seeds"].values()]
            report["seed_summary"][method][group] = {}
            for name in ("auprc_ap", "f1", "fpr", "recall"):
                nums = np.array([v[name] for v in values])
                report["seed_summary"][method][group][name] = {"mean": float(nums.sum() / 3),
                                                               "std_ddof1": float(np.sqrt(((nums - nums.sum() / 3) ** 2).sum() / 2))}
    check_tree(expected, report)
    return report


def acceptance_audit():
    receipt = json.loads((LIGHT / "acceptance.json").read_text())
    assert receipt["protocol_sha256"] == digest(LIGHT / "protocol.json")
    assert digest(OUT / "acceptance_input.npy") == receipt["input_sha256"]
    inputs = np.load(OUT / "acceptance_input.npy")
    assert inputs.shape == (2, 3, 150)
    assert np.isnan(inputs[1, 2]).all()
    assert set(receipt["devices"]) == {"cpu", "mps"}
    outputs = {}
    for device, entry in receipt["devices"].items():
        path = OUT / f"acceptance_{device}.npy"
        assert digest(path) == entry["sha256"]
        outputs[device] = np.load(path)
        assert list(outputs[device].shape) == entry["shape"]
        assert np.isfinite(outputs[device]).all()
        assert np.count_nonzero(outputs[device][1, 2]) == 0
        assert entry["finite"] and entry["forecast_bitwise_unchanged"] and entry["state_dict_bitwise_unchanged"]
    difference = np.abs(outputs["cpu"] - outputs["mps"]).max()
    equal(difference, receipt["devices"]["mps"]["max_abs_vs_cpu"])
    np.testing.assert_allclose(outputs["cpu"], outputs["mps"], rtol=3e-4, atol=3e-4)


def main():
    global ROOT, OUT, LIGHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-root", type=Path, default=OUT)
    parser.add_argument("--results-root", type=Path, default=LIGHT)
    args = parser.parse_args()
    ROOT, OUT, LIGHT = args.source_root.resolve(), args.evidence_root.resolve(), args.results_root.resolve()
    protocol = json.loads((LIGHT / "protocol.json").read_text())
    result = json.loads((LIGHT / "results.json").read_text())
    assert result["protocol_sha256"] == digest(LIGHT / "protocol.json")
    assert result["source_hashes"] == protocol["source_hashes"]
    source_audit(protocol)
    arrays = arrays_audit(result)
    classification(arrays["classification.npz"], result["classification"])
    detection(arrays["detection.npz"], result["detection"])
    acceptance_audit()
    receipt = {"status": "passed", "protocol_sha256": digest(LIGHT / "protocol.json"),
               "results_sha256": digest(LIGHT / "results.json"),
               "checks": ["complete source and array key sets, snapshot and all hashes", "pinned upstream, weights/config and data identity",
                          "official split, all classification predictions/distances/confusions/deltas", "all temporal windows and seed injection reconstruction",
                          "all six baseline features and training-only scaling", "all NN scores and calibration-only thresholds", "all overall/per-type/seed-summary metrics", "CPU/MPS acceptance array hashes, finite/zero/shape and numerical agreement"],
               "limits": "Neural arrays checked by hashes, not recomputed by this auditor; independent hidden-layer reference and device agreement tested separately. No real-fault labels or independent-window inference."}
    (LIGHT / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
