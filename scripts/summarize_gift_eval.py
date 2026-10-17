"""Write a complete-run report only after the independent 97-task audit passes."""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".cache/upstream/src")]


def main():
    from timestral_ft.artifacts import file_sha256
    from timestral_ft.gift import atomic_json

    output = ROOT / "results/gift_eval_5m_lora"
    run = ROOT / "runs/gift-eval-5m-lora"
    aggregate = json.loads((output / "aggregates.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    validation = json.loads((output / "submission_validation.json").read_text())
    if aggregate["tasks"] != 97 or not audit["all_checks_pass"] or not validation["schema_valid"]:
        raise RuntimeError("A complete successful audit is required before reporting")
    rows, manifest = audit["tasks"], []
    for row in rows:
        artifact = run / "tasks" / row["task"].replace("/", "__") / "adapter.safetensors"
        manifest.append({"task": row["task"], "adapter": artifact.relative_to(ROOT).as_posix(),
                         "sha256": file_sha256(artifact), "bytes": artifact.stat().st_size,
                         "steps": row["steps"], "best_step": row["best_step"],
                         "device": row["training_device"]})
    atomic_json(output / "adapter_manifest.json", manifest)
    files = ["src/timestral_ft/gift.py", "src/timestral_ft/compat.py", "src/timestral_ft/trainer.py",
             "src/timestral_ft/lora.py", "scripts/run_gift_eval.py", "scripts/audit_gift_eval.py",
             "scripts/summarize_gift_eval.py", "tests/test_gift.py", "tests/test_training_path.py",
             "scripts/package_gift_adapters.py",
             ".cache/gift-eval/notebooks/timestral.ipynb", ".cache/gift-eval/src/gift_eval/data.py",
             ".cache/gift-eval/scripts/validate_results.py"]
    atomic_json(output / "source_manifest.json", {p: file_sha256(ROOT / p) for p in files})
    shutil.copyfile(ROOT / "results/gift_eval_feasibility/requirements-probe.txt",
                    output / "requirements-gift.txt")
    comparison = []
    for label, metric in (("CRPS 近似", "crps"), ("MASE", "mase")):
        m = aggregate[metric]
        comparison.append(f"| {label} | {m['baseline_normalized_gmean']:.6f} | "
                          f"{m['lora_normalized_gmean']:.6f} | {m['relative_change_percent']:+.3f}% | "
                          f"{m['improved']} / {m['regressed']} |")
    crps_change = aggregate["crps"]["relative_change_percent"]
    mase_change = aggregate["mase"]["relative_change_percent"]
    if crps_change >= 0 and mase_change >= 0:
        conclusion = "本轮固定预算配方没有带来整体收益：CRPS 和 MASE 的全量聚合均有退化。"
    elif crps_change < 0 and mase_change < 0:
        conclusion = "本轮 CRPS 和 MASE 的全量聚合均有改善，但仍需更多种子和独立数据验证稳定性。"
    else:
        conclusion = "本轮两个聚合指标方向不一致，收益取决于所选指标，不能概括为一致提升。"
    if max(abs(crps_change), abs(mase_change)) < 0.1:
        conclusion += " 两项变化幅度都低于 0.1%；单种子结果不足以判断这种微小差异是否稳定。"
    groups = []
    for key, data in aggregate["by_domain"].items():
        groups.append(f"| {key} | {data['tasks']} | {data['crps']['relative_change_percent']:+.3f}% | "
                      f"{data['mase']['relative_change_percent']:+.3f}% |")
    terms = []
    for key, data in aggregate["by_term"].items():
        terms.append(f"| {key} | {data['tasks']} | {data['crps']['relative_change_percent']:+.3f}% | "
                     f"{data['mase']['relative_change_percent']:+.3f}% |")
    devices = ", ".join(f"{k}: {v}" for k, v in aggregate["training_devices"].items())
    worst = sorted(rows, key=lambda r: r["crps_ratio"], reverse=True)[:5]
    best = sorted(rows, key=lambda r: r["crps_ratio"])[:5]
    details = ["| 任务 | CRPS 变化 | MASE 变化 |", "|---|---:|---:|"]
    for r in best + worst:
        details.append(f"| {r['task']} | {r['crps_change_percent']:+.3f}% | {r['mase_change_percent']:+.3f}% |")
    text = f"""# Timestral-5M LoRA：完整 GIFT-Eval 97 任务

已完成 97 个独立任务微调、97 个原模型复现及 97 个微调模型测试；结果包括全部任务和全部退化。
类型为 `fine-tuned`，本地提交草稿已通过官方格式校验，尚未公开权重／代码或提交榜单。

## 全量指标

每项先除以同任务 Seasonal Naive 指标，再对 97 项取几何均值，越小越好。
CRPS 使用官方九分位数 WQL 近似。变化为 LoRA / 原模型 - 1，负值表示改善。
这是任务等权的几何聚合；任务数更多的数据集／频率不会被重新调整成原始数据集等权。

| 指标 | 原模型复现 | LoRA | 相对变化 | 改善 / 退化任务数 |
|---|---:|---:|---:|---:|
{chr(10).join(comparison)}

{conclusion} 逐任务改善并不意味着所有任务或所有领域都受益。

这是几何均值分数；官方 Rank 汇总逐任务名次，不能由上述分数推算实时排名。

| 领域 | 任务数 | CRPS 变化 | MASE 变化 |
|---|---:|---:|---:|
{chr(10).join(groups)}

| 预测长度类别 | 任务数 | CRPS 变化 | MASE 变化 |
|---|---:|---:|---:|
{chr(10).join(terms)}

CRPS 改善最多与退化最多的各五项如下；完整逐任务结果见 `task_comparison.csv`。

{chr(10).join(details)}

## 固定方法及覆盖

- 每任务从同一固定 Timestral-5M 基础权重独立开始，LoRA rank=8、alpha=16、dropout=0。
- 历史最多 256；短历史左侧 NaN 填充。仅在训练前缀不足 H+1 时使用被掩码的部分标签。
- 种子 42，每任务最多 200 优化步；实际共 {sum(r['steps'] for r in rows):,} 步。
  batch=4、累积=4，AdamW lr=1e-4、weight decay=.01、clip=1，预热 20 后余弦衰减。
- 每 50 步验证，选历史标准差平衡的 21 分位数 masked pinball 最优检查点，patience=3。
  验证最多固定抽取 512 条序列；这是预算约束。测试覆盖全部序列、全部目标和全部官方滚动窗口。
- 训练设备：{devices}。五个尚未开始的重长预测任务依据维度／训练区间计时预检固定为 MPS，
  其余任务保持 CPU；硬件映射保存在 `protocol.json`，每个适配器记录实际设备。
- 适配器在 CPU 统一初始化后移到设备；CPU/MPS 初始矩阵逐位一致，后续 FP32 轨迹可有设备差异。
- 保留官方全部历史协变量，覆盖最高 21 个目标、35 列历史协变量及最长 H=900。
  官方推理读取完整历史至 32768，使用默认后处理、多变量联合预测与全部 11 项指标。
- 官方 Predictor 的 `mean` 键使用中位数；这里沿用官方约定。
- 训练循环累计耗时（包含验证和检查点）：{aggregate['training_seconds']/3600:.3f} 小时。
  各进程并行，这不是墙钟耗时，也不包含所有数据加载、等待和被中断的未提交工作。
- 每个适配器 {rows[0]['trainable_parameters']:,} 个可训练参数；97 份共
  {sum(r['bytes'] for r in manifest)/1e6:.2f} MB，基础权重共享。

## 正确性与边界

原模型 {aggregate['baseline_reference_matches']} / 97 项与官方提交的全部 11 项指标匹配，容差
`rtol=1e-3, atol=1e-5`。原始差异保存在逐任务 baseline JSON，任何不匹配均保留，不隐藏。
独立审计重新核对原始 Arrow 长度对应的官方训练截止边界、完整任务清单、有限损失、步骤顺序、
验证选优、适配器实际更新、基础修订、设备配置，以及评估所用权重哈希。

数值边界修复保留原预测前向值：标准差下限的反向传播在零方差处使用有限导数；有限梯度的
FP32 范数溢出时转 CPU FP64 复核裁剪，真正的 NaN/Inf 仍终止。
`numerical_replay.json` 记录修复前已完成任务的重跑适配器逐位一致证据。
季节误差的等价切片保留完整历史与掩码；四个完整任务的 11 项指标逐位一致。
CPU/MPS 与原版／等价切片评估可混合运行，设备及后期的线程数保存在各条评估记录。

单种子、每任务 200 步和采样验证属于有限预算适配实验，不能证明普遍收益或最优配方。
训练使用了任务自己的训练／验证分区，不属于冻结零样本泛化。基础模型提交声明无泄漏，
本次没有独立审计其完整预训练数据；我们自己的微调和选优仅使用官方训练与验证分区。
本轮没有 CUDA 硬件验证。

## 复现与产物

```sh
python scripts/run_gift_eval.py --stage train
python scripts/run_gift_eval.py --stage evaluate --fast-seasonal-error
python scripts/audit_gift_eval.py
python scripts/summarize_gift_eval.py
```

硬件映射及兼容环境详见 [运行指南](../../examples/gift_eval/README.md)。
重建参考硬件运行时，把 `protocol.json` 中的 `train_device_overrides` 写入运行目录的
`train_device_overrides.json`；否则是新的全 CPU 实验，应使用新的输出目录。
`adapter_manifest.json` 给出本机 97 份适配器路径、哈希、步骤及设备；可通过
`timestral_ft.gift.load_gift_adapter` 加载。完整训练检查点在忽略的 `runs/gift-eval-5m-lora`。
代码、数据修订及依赖分别记录于 `source_manifest.json`、`data_manifest.json`、
`requirements-gift.txt`，训练设置与权重版本在 `protocol.json`。

`submission/Timestral-FT-5M-LoRA` 仅含官方格式的 CSV 与配置。
公开提交前须填写真实模型／代码链接、组织信息并更新复现可用性；这些字段没有虚构。
格式校验通过不代表维护者接受，也不代表已经获得榜单名次。
"""
    (output / "README.zh-CN.md").write_text(text)
    (output / "README.md").write_text(f"""# Timestral-FT-5M-LoRA on GIFT-Eval

Complete 97-task fine-tuning and paired official-protocol evaluation. See the
[full report](README.zh-CN.md), [task comparison](task_comparison.csv),
[independent audit](audit.json) and [reproduction guide](../../examples/gift_eval/README.md).

| Normalized geometric mean | Base | LoRA | Relative change |
|---|---:|---:|---:|
| CRPS approximation | {aggregate['crps']['baseline_normalized_gmean']:.6f} | {aggregate['crps']['lora_normalized_gmean']:.6f} | {aggregate['crps']['relative_change_percent']:+.3f}% |
| MASE | {aggregate['mase']['baseline_normalized_gmean']:.6f} | {aggregate['mase']['lora_normalized_gmean']:.6f} | {aggregate['mase']['relative_change_percent']:+.3f}% |

Lower is better. This is one seed, up to 200 optimizer steps per independent
task adapter, validation capped at 512 preselected series, and **complete official
test coverage**. All gains and regressions are retained. Hardware is recorded
per task; five previously unstarted heavy tasks use MPS and the rest use CPU.
Base replications match the official eleven metrics in {aggregate['baseline_reference_matches']}/97 tasks
within rtol=1e-3 / atol=1e-5. Geometric-mean scores are not official ranks.

The local submission draft passed the official schema validator. It has **not**
been submitted or published; model/code links and organization fields remain
unfilled. Weights retain the base license. Training on GIFT partitions makes
this a **fine-tuned** method, not a zero-shot claim; base pretraining provenance
was not independently audited. CUDA hardware has not been validated.
""")
    print(f"Wrote full report and {len(manifest)} adapter manifests to {output}")


if __name__ == "__main__":
    main()
