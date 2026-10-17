"""Package audited GIFT task adapters, without optimizer states or raw data."""

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / "results/gift_eval_5m_lora"
    manifest = json.loads((results / "adapter_manifest.json").read_text())
    audit = json.loads((results / "training_audit.json").read_text())
    if len(manifest) != 97 or not audit["all_checks_pass"]:
        raise ValueError("All 97 training artifacts must pass acceptance first")
    target = ROOT / "dist/timestral-ft-5m-lora-gift-eval-adapters.zip"
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("README.txt", "97 independent Timestral-5M LoRA adapters.\n"
                         "Weights retain the base model's license; see BASE_MODEL_LICENSE in each task.\n"
                         "The base checkpoint is not bundled. Load each task with timestral_ft.gift.load_gift_adapter.\n"
                         "Extract tasks/<canonical_key_with_double_underscores>/ and pass that directory.\n"
                         "This is a fine-tuned method, with hardware and base revisions recorded per task.\n"
                         "The code's Apache-2.0 license does not relicense model weights.\n")
        for row in manifest:
            directory = (ROOT / row["adapter"]).parent
            for name in ("adapter.safetensors", "adapter_config.json", "training_summary.json",
                         "history.jsonl", "BASE_MODEL_LICENSE"):
                path = directory / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                archive.write(path, f"tasks/{directory.name}/{name}")
        for name in ("protocol.json", "training_audit.json", "aggregates.json", "task_comparison.csv",
                     "README.md", "README.zh-CN.md"):
            path = results / name
            if path.exists():
                archive.write(path, f"experiment/{name}")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupted archive")
        if sum(n.endswith("/adapter.safetensors") for n in archive.namelist()) != 97:
            raise ValueError("Incomplete adapter archive")
    temporary.replace(target)
    print(f"Saved {target} ({target.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
