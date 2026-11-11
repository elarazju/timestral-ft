"""Frozen-feature research probe. Freeze sources/protocol BEFORE any test scoring.

python scripts/run_report_round6.py --freeze
python scripts/run_report_round6.py --run
python scripts/run_report_round6.py --accept
python scripts/audit_report_round6.py
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timestral.loading import resolve_checkpoint

from timestral_ft import encode
from timestral_ft.compat import PortableForecaster, load_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/report-round6"
LIGHT = ROOT / "results/report_round6"
REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
AEON = "98c13323251a011722de2d6f68a772554af9b9ea"
DATA_SHA = {"GunPoint_TRAIN.ts": "f842401779fd9800d247d8b33121a1a4643710a19b24917dbdd9a060ca8630d5",
            "GunPoint_TEST.ts": "79332750788a6227b325b96bd0d70130c8eb707b9731f8d7dec62b7a7d36017e",
            "ETTh2.csv": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_paths():
    # Entire local package and test suite, not selected dictionary keys.
    return sorted(list((ROOT / "src/timestral_ft").glob("*.py"))
                  + list((ROOT / "tests").glob("*.py"))
                  + [ROOT / "scripts/run_report_round6.py", ROOT / "scripts/audit_report_round6.py",
                     ROOT / "pyproject.toml"]
                  + list((ROOT / "examples/report_round6").glob("*")))


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in source_paths() if p.is_file()}


def upstream_hashes():
    import timestral
    directory = Path(timestral.__file__).parent
    return {p.name: sha(p) for p in sorted(directory.glob("*.py"))}


def prepare():
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    urls = {f"GunPoint_{split}.ts": f"https://raw.githubusercontent.com/aeon-toolkit/aeon/{AEON}/aeon/datasets/data/GunPoint/GunPoint_{split}.ts"
            for split in ("TRAIN", "TEST")}
    urls["ETTh2.csv"] = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh2.csv"
    for name, url in urls.items():
        p = data / name
        if not p.exists():
            cached = ROOT / "data/validation" / name
            if cached.exists():
                shutil.copy2(cached, p)
            else:
                p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        if sha(p) != DATA_SHA[name]:
            raise ValueError(f"Data checksum mismatch: {name}")
    return {name: {"sha256": DATA_SHA[name], "url": url} for name, url in urls.items()}


def freeze():
    if (LIGHT / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen; do not overwrite completed evidence")
    manifest = prepare()
    checkpoint = resolve_checkpoint("mistralai/Timestral-5M", revision=REVISION)
    protocol = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "amendment": {"prior_protocol_sha256": "931e2a1ad9f19b74ed91cc50b60e738801c1183e45f1e0150a948a05002cc716",
                      "reason": "First execution completed arrays but failed recording pip freeze because uv environment has no pip. No scores were displayed/inspected. Preserve original snapshot/arrays in failed-environment-receipt; replace only environment package inventory with importlib.metadata, extend receipt audits, and rerun unchanged experiment parameters."},
        "purpose": "Exploratory context representations, not official classification or anomaly results",
        "model": {"repo": "mistralai/Timestral-5M", "revision": REVISION,
                  "weights_sha256": sha(checkpoint / "model.safetensors"),
                  "config_sha256": sha(checkpoint / "config.json"), "frozen": True,
                  "device": "cpu", "precision": "fp32", "threads": 4, "batch_size": 16},
        "classification": {"dataset": "GunPoint", "official_description": "https://www.timeseriesclassification.com/description.php?Dataset=GunPoint",
                           "source_note": "Official zip returned HTTP 403; pinned aeon archive mirror, official page confirms dimensions and standard split",
                           "train_n": 50, "test_n": 150, "length": 150, "classes": [1, 2],
                           "features": ["raw", "mean", "last"], "classifier": "1-NN Euclidean, first training index on ties",
                           "normalization": "Keep archive values; no additional external scaling of raw data or embeddings",
                           "selection": "All three prespecified methods reported; test labels used only after prediction for final scoring",
                           "repetitions": "One complete fixed split; no artificial seed repeats"},
        "detection": {"dataset": "ETTh2", "column": "OT", "train": [0, 8640], "calibration": [8640, 11520], "test": [11520, 14400],
                      "window": 96, "stride": 24, "label_unit": "window", "seeds": [42, 43, 44],
                      "normal_assumption": "Unmodified data treated as reference normal; genuine faults are unknown, not adjudicated healthy labels",
                      "injection": "Independent perturbations of overlapping test windows, not a continuous fault timeline",
                      "selection": "Choose floor(0.30*n_test_windows) indices without replacement, shuffle balanced cycling kinds 1/2/3; same input for both methods",
                      "spike": "one point += sign*6*training_raw_std_ddof0; start uniform [24,72), sign uniform +/-1",
                      "level_shift": "24 consecutive points += sign*3*training_raw_std_ddof0; start uniform [24,72)",
                      "stuck": "24 consecutive points set to immediately preceding original value; start uniform [24,72)",
                      "random_order": "default_rng(seed): choice(indices), shuffle(cycling kinds); for each selected index draw start then sign (also for stuck)",
                      "input_scaling": "(raw-training_point_mean)/training_point_std_ddof0, same for both methods",
                      "embedding": "valid-patch mean, flatten variable and hidden axes, no external embedding scaling",
                      "statistics": ["mean(x)", "std(x,ddof=0)", "min(x)", "max(x)", "mean(abs(diff(x)))", "max(abs(diff(x)))"],
                      "statistics_scaling": "each feature centered/scaled using train-window mean/std ddof0; floor scale at 1e-12",
                      "score": "Euclidean distance to closest training reference-normal window, no bank subsampling",
                      "threshold": {"split": "unmodified calibration only", "quantile": .99, "method": "linear", "comparison": ">"},
                      "metrics": "average precision (stepwise PR AUC, ties grouped), fixed threshold F1/FPR/recall; per-kind versus same noninjected windows; seed mean and sample std ddof1",
                      "interpretation": "Offline complete-window detection; overlapping correlated windows; no point adjustment; seeds measure injection variability only, not dataset uncertainty"},
        "data": manifest, "source_hashes": source_hashes(), "upstream_source_hashes": upstream_hashes(),
        "metric_selection": "No label-driven parameter/pooling/threshold/model selection; no training or prior-round winning weights",
    }
    write(LIGHT / "protocol.json", protocol)
    write(LIGHT / "data_manifest.json", manifest)
    snapshot = OUT / "source_snapshot"
    for relative in protocol["source_hashes"]:
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    import timestral
    for p in Path(timestral.__file__).parent.glob("*.py"):
        dest = snapshot / "upstream/timestral" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    for name in ("protocol.json", "data_manifest.json"):
        shutil.copy2(LIGHT / name, snapshot / name)
        shutil.copy2(LIGHT / name, OUT / name)
    shutil.copy2(checkpoint / "config.json", snapshot / "checkpoint_config.json")
    print("Protocol frozen:", sha(LIGHT / "protocol.json"), flush=True)


def verify_protocol():
    p = json.loads((LIGHT / "protocol.json").read_text())
    if p["source_hashes"] != source_hashes() or p["upstream_source_hashes"] != upstream_hashes():
        raise ValueError("Source set or hashes differ from frozen protocol")
    if p["data"] != prepare():
        raise ValueError("Data identity differs from protocol")
    checkpoint = resolve_checkpoint(p["model"]["repo"], revision=p["model"]["revision"])
    for name, key in (("model.safetensors", "weights_sha256"), ("config.json", "config_sha256")):
        if sha(checkpoint / name) != p["model"][key]:
            raise ValueError("Checkpoint identity changed")
    return p, checkpoint


def read_ts(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data")
    rows = [line.strip().rsplit(":", 1) for line in lines[start + 1:] if line.strip()]
    return np.array([[float(v) for v in x.split(",")] for x, _ in rows], dtype=np.float32), np.array([int(y) for _, y in rows])


def features(model, x):
    means, lasts = [], []
    for start in range(0, len(x), 16):
        batch = torch.from_numpy(x[start:start + 16])
        result = encode(model, batch, pooling="mean")
        means.append(result.pooled.cpu().numpy().reshape(len(batch), -1))
        # Avoid recomputing the model: last valid patch, all probe inputs observed.
        lasts.append(result.patches[:, :, -1].cpu().numpy().reshape(len(batch), -1))
    return np.concatenate(means), np.concatenate(lasts)


def nearest(train, query):
    train, query = train.astype(np.float64), query.astype(np.float64)
    distance = ((query[:, None] - train[None]) ** 2).sum(-1)
    index = distance.argmin(-1)
    return index, np.sqrt(distance[np.arange(len(query)), index])


def stats(x):
    difference = np.abs(np.diff(x.astype(np.float64), axis=-1))
    return np.stack([x.mean(-1, dtype=np.float64), x.astype(np.float64).std(-1), x.min(-1), x.max(-1),
                     difference.mean(-1), difference.max(-1)], axis=-1)


def metric(y, score, threshold):
    y = np.asarray(y, dtype=bool)
    pred = score > threshold
    tp, fp, fn, tn = [int(z.sum()) for z in (pred & y, pred & ~y, ~pred & y, ~pred & ~y)]
    order = np.argsort(-score, kind="stable")
    yy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    cumulative = np.cumsum(yy)[ends]
    recall = cumulative / y.sum()
    precision = cumulative / (ends + 1)
    ap = np.sum(np.diff(np.r_[0., recall]) * precision)
    return {"auprc_ap": float(ap), "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "fpr": fp / max(1, fp + tn), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y), "positive": int(y.sum())}


def classify(model):
    train, ytrain = read_ts(OUT / "data/GunPoint_TRAIN.ts")
    test, ytest = read_ts(OUT / "data/GunPoint_TEST.ts")
    assert train.shape == (50, 150) and test.shape == (150, 150)
    assert np.isfinite(train).all() and np.isfinite(test).all()
    train_mean, train_last = features(model, train)
    test_mean, test_last = features(model, test)
    arrays = {"train_raw": train, "test_raw": test, "train_labels": ytrain, "test_labels": ytest,
              "train_mean": train_mean, "test_mean": test_mean, "train_last": train_last, "test_last": test_last}
    result = {}
    for method in ("raw", "mean", "last"):
        indices, distances = nearest(arrays[f"train_{method}"], arrays[f"test_{method}"])
        pred = ytrain[indices]
        arrays[f"{method}_prediction"], arrays[f"{method}_neighbor"], arrays[f"{method}_distance"] = pred, indices, distances
        result[method] = {"accuracy": float((pred == ytest).mean()), "correct": int((pred == ytest).sum()), "n": len(test),
                          "confusion_matrix_true_rows_pred_columns_1_2": [[int(((ytest == a) & (pred == b)).sum()) for b in (1, 2)] for a in (1, 2)]}
    raw_ok = arrays["raw_prediction"] == ytest
    for method in ("mean", "last"):
        ok = arrays[f"{method}_prediction"] == ytest
        result[method]["difference_vs_raw"] = {"accuracy_percentage_points": float(100 * (ok.mean() - raw_ok.mean())),
                                               "embedding_only_correct": int((ok & ~raw_ok).sum()),
                                               "raw_only_correct": int((~ok & raw_ok).sum())}
    result["accuracy_range_prespecified_methods"] = [min(result[m]["accuracy"] for m in ("raw", "mean", "last")),
                                                     max(result[m]["accuracy"] for m in ("raw", "mean", "last"))]
    np.savez_compressed(OUT / "classification.npz", **arrays)
    return result


def detect(model):
    frame = pd.read_csv(OUT / "data/ETTh2.csv")
    raw = frame.OT.to_numpy(dtype=np.float64)
    assert len(raw) >= 14400 and np.isfinite(raw).all()
    dates = pd.to_datetime(frame.date)
    assert dates.is_unique and (dates.diff().dropna() == pd.Timedelta(hours=1)).all()
    mean, scale = raw[:8640].mean(), raw[:8640].std()
    arrays = {"raw_ot": raw, "point_mean": mean, "point_scale": scale}
    windows = {}
    for name, begin, end in (("train", 0, 8640), ("calibration", 8640, 11520), ("test", 11520, 14400)):
        starts = np.arange(begin, end - 96 + 1, 24)
        windows[name] = np.stack([raw[i:i + 96] for i in starts])
        arrays[f"{name}_start"] = starts
        arrays[f"{name}_input"] = ((windows[name] - mean) / scale).astype(np.float32)
    for name in ("train", "calibration"):
        arrays[f"{name}_embedding"] = features(model, arrays[f"{name}_input"])[0]
        arrays[f"{name}_statistics"] = stats(arrays[f"{name}_input"])
    center = arrays["train_statistics"].mean(0)
    spread = np.maximum(arrays["train_statistics"].std(0), 1e-12)
    arrays.update(statistics_center=center, statistics_scale=spread)
    thresholds = {}
    for method in ("embedding", "statistics"):
        bank = arrays[f"train_{method}"]
        query = arrays[f"calibration_{method}"]
        if method == "statistics":
            bank, query = (bank - center) / spread, (query - center) / spread
        score = nearest(bank, query)[1]
        arrays[f"calibration_{method}_score"] = score
        thresholds[method] = float(np.quantile(score, .99, method="linear"))
    result = {"thresholds": thresholds, "window_counts": {name: len(w) for name, w in windows.items()}, "seeds": {}}
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        test = windows["test"].copy()
        n = len(test)
        indices = rng.choice(n, int(.30 * n), replace=False)
        kinds = np.resize(np.array([1, 2, 3]), len(indices))
        rng.shuffle(kinds)
        injection = np.zeros((n, 4), dtype=np.int64)
        for i, kind in zip(indices, kinds):
            start, sign = int(rng.integers(24, 72)), int(rng.choice([-1, 1]))
            duration = 1 if kind == 1 else 24
            injection[i] = [kind, start, duration, sign]
            if kind in (1, 2):
                test[i, start:start + duration] += sign * (6 if kind == 1 else 3) * scale
            else:
                test[i, start:start + duration] = test[i, start - 1]
        x = ((test - mean) / scale).astype(np.float32)
        prefix = f"s{seed}_"
        arrays[prefix + "input"] = x
        arrays[prefix + "injection"] = injection
        arrays[prefix + "embedding"] = features(model, x)[0]
        arrays[prefix + "statistics"] = stats(x)
        y = injection[:, 0] != 0
        arrays[prefix + "labels"] = y
        result["seeds"][str(seed)] = {}
        for method in ("embedding", "statistics"):
            bank, query = arrays[f"train_{method}"], arrays[prefix + method]
            if method == "statistics":
                bank, query = (bank - center) / spread, (query - center) / spread
            neighbors, score = nearest(bank, query)
            arrays[prefix + method + "_score"] = score
            arrays[prefix + method + "_prediction"] = score > thresholds[method]
            arrays[prefix + method + "_neighbor"] = neighbors
            report = {"overall": metric(y, score, thresholds[method]), "by_type": {}}
            for kind, label in ((1, "spike"), (2, "level_shift"), (3, "stuck")):
                take = (injection[:, 0] == 0) | (injection[:, 0] == kind)
                report["by_type"][label] = metric(y[take], score[take], thresholds[method])
            result["seeds"][str(seed)][method] = report
    result["seed_summary"] = {}
    for method in ("embedding", "statistics"):
        result["seed_summary"][method] = {}
        for subset in ("overall", "spike", "level_shift", "stuck"):
            rows = [r[method]["overall"] if subset == "overall" else r[method]["by_type"][subset]
                    for r in result["seeds"].values()]
            result["seed_summary"][method][subset] = {key: {"mean": float(np.mean([r[key] for r in rows])),
                                                                   "std_ddof1": float(np.std([r[key] for r in rows], ddof=1))}
                                                        for key in ("auprc_ap", "f1", "fpr", "recall")}
    np.savez_compressed(OUT / "detection.npz", **arrays)
    return result


def array_manifest():
    result = {}
    for name in ("classification.npz", "detection.npz"):
        with np.load(OUT / name) as z:
            result[name] = {"file_sha256": sha(OUT / name), "arrays": {
                key: {"shape": list(z[key].shape), "dtype": str(z[key].dtype),
                      "sha256_bytes": hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest()}
                for key in sorted(z.files)}}
    return result


def run():
    protocol, checkpoint = verify_protocol()
    if (OUT / "classification.npz").exists() or (OUT / "detection.npz").exists():
        raise FileExistsError("Formal outputs exist; do not overwrite them")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = load_model(checkpoint, device="cpu")
    result = {"classification": classify(model), "detection": detect(model),
              "protocol_sha256": sha(LIGHT / "protocol.json"),
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "numpy": np.__version__, "platform": platform.platform(),
                              "threads": torch.get_num_threads(), "device": "cpu", "precision": "fp32",
                              "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
              "source_hashes": protocol["source_hashes"], "arrays": array_manifest()}
    packages = sorted([{"name": d.metadata["Name"], "version": d.version} for d in importlib.metadata.distributions()], key=lambda d: d["name"].lower())
    write(OUT / "environment-packages.json", packages)
    result["environment"]["packages_sha256"] = sha(OUT / "environment-packages.json")
    write(LIGHT / "results.json", result)
    shutil.copy2(LIGHT / "results.json", OUT / "results.json")
    print(json.dumps(result["classification"], indent=2), flush=True)
    print(json.dumps(result["detection"]["seed_summary"], indent=2), flush=True)


def accept():
    _, checkpoint = verify_protocol()
    torch.set_num_threads(4)
    records, baseline = {}, None
    x = torch.sin(torch.arange(150) / 9).repeat(2, 3, 1)
    x[0, 1, :60] = float("nan")
    x[1, 2] = float("nan")
    np.save(OUT / "acceptance_input.npy", x.numpy())
    for device in ("cpu", "mps"):
        model = load_model(checkpoint, device=device)
        forecast = PortableForecaster(model)
        captured = []
        hook = model.transformer.register_forward_hook(lambda m, a, y: captured.append(y.detach().clone()))
        try:
            before = forecast.forecast(x, 32, symmetric_averaging=False).quantiles
        finally:
            hook.remove()
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        result = encode(model, x)
        reference = captured[0][:, :result.patches.shape[2]].reshape_as(result.patches)
        torch.testing.assert_close(result.patches[result.patch_valid], reference[result.patch_valid], rtol=3e-5, atol=3e-5)
        reference_error = float((result.patches[result.patch_valid] - reference[result.patch_valid]).abs().max())
        after = forecast.forecast(x, 32, symmetric_averaging=False).quantiles
        assert torch.equal(before, after)
        assert all(torch.equal(state[k], v.cpu()) for k, v in model.state_dict().items())
        assert torch.isfinite(result.patches).all() and torch.isfinite(result.pooled).all()
        assert not result.valid[1, 2] and result.pooled[1, 2].count_nonzero() == 0
        h = result.patches.cpu().numpy()
        np.save(OUT / f"acceptance_{device}.npy", h)
        records[device] = {"finite": True, "forecast_bitwise_unchanged": True, "state_dict_bitwise_unchanged": True,
                           "official_context_max_abs_error": reference_error,
                           "shape": list(h.shape), "sha256": sha(OUT / f"acceptance_{device}.npy")}
        if baseline is None:
            baseline = h
        else:
            records[device]["max_abs_vs_cpu"] = float(np.abs(h - baseline).max())
            np.testing.assert_allclose(h, baseline, rtol=3e-4, atol=3e-4)
        del model, forecast, before, after, result
    write(LIGHT / "acceptance.json", {"devices": records, "input_sha256": sha(OUT / "acceptance_input.npy"),
                                      "protocol_sha256": sha(LIGHT / "protocol.json")})
    print(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("freeze", "run", "accept"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        run()
    else:
        accept()


if __name__ == "__main__":
    main()
