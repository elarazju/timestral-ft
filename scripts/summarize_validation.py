"""Render bilingual repository evidence from the completed, audited validation matrix."""

import argparse
import json
from pathlib import Path

import pandas as pd

from timestral_ft.artifacts import file_sha256, write_json


def score_table(frame, index_column, reference=True):
    title = "数据集" if index_column == "dataset" else "迁移目标"
    lines = [f"| {title} | H | 原模型 MSE | 全参数 MSE | LoRA MSE |" + (" DLinear MSE |" if reference else ""),
             "|---|---:|---:|---:|---:|" + ("---:|" if reference else "")]
    for (name, horizon), group in frame.groupby([index_column, "horizon"], sort=True):
        group = group.set_index("mode")

        def value(mode):
            if mode not in group.index:
                return "—"
            row = group.loc[mode]
            return f"{row.mse_mean:.5f}" if pd.isna(row.mse_std) else f"{row.mse_mean:.5f} ± {row.mse_std:.5f}"

        lines.append(f"| {name} | {horizon} | {value('zero-shot')} | {value('full')} | {value('lora')} |"
                     + (f" {value('dlinear')} |" if reference else ""))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/validation")
    args = parser.parse_args()
    root = Path(args.root)
    audit = json.loads((root / "audit.json").read_text())
    assert all(audit[key]["passed"] for key in ["benchmark", "models", "transfer"])
    benchmark = pd.read_csv(root / "benchmark/summary.csv")
    transfer = pd.read_csv(root / "transfer/summary.csv")
    lines = ["# Timestral 微调验证结果", "",
             "本轮完成多数据集适配、冻结权重跨数据集迁移、标准多变量测试协议及四档模型功能验收。",
             "它验证了可复现的功能与选定实验中的效果，不能证明所有任务都有收益或达到行业最优。", "",
             "## 多数据集适配", "",
             "ETTh1、ETTh2、Exchange Rate；历史 96、预测 96／192；种子 42／43／44。",
             "在每个数据集各自训练区间微调，在各自留出的未来测试区间评估；不是跨数据集迁移。",
             "M 任务包含全部七／八个目标，使用训练集逐列均值与总体标准差，测试按步长 1 覆盖所有起点。",
             "下表为标准化 MSE，越小越好；± 为三个种子的样本标准差，不是置信区间。MAE、每种子与逐变量／预测步误差见原始结果。", ""]
    lines += score_table(benchmark, "dataset")
    lines += ["", "以原模型为基准，所有种子均计入：", ""]
    for mode, title in [("full", "全参数"), ("lora", "LoRA")]:
        r = audit["benchmark"]["adaptation_summary"][mode]
        lines.append(f"- {title}：{r['runs']} 次运行中，MAE 改善 {r['mae_wins']} 次，MSE 改善 {r['mse_wins']} 次；"
                     f"MSE 相对改善范围 {r['mse_gain_pct_range'][0]:+.2f}%～{r['mse_gain_pct_range'][1]:+.2f}%。")
    lines += ["", "正改善率表示误差下降，负值表示退化；不筛选最好种子。相关 ETT 数据、重叠窗口、两个预测长度并非独立统计样本。",
              "对照包含持平预测、原模型、两种微调及固定提交的官方 DLinear 网络，共 66 个评估记录、36 次 Timestral 微调和 18 次 DLinear 训练。",
              "Timestral 最多 200 优化步，AdamW、batch4×累积4，验证以 stride H 的 scaled pinball 选优；",
              "DLinear 采用 Adam/MSE、lr1e-4、batch32、最多10轮、type1衰减、patience3，验证为全部起点加权 MSE。",
              "DLinear 网络未改动；本地训练循环包含最终不足一批的样本，验证顺序和聚合与官方循环略有不同。",
              "两者测试标签／标准化／滑窗一致，但预训练、优化目标、验证方式和训练预算不同，因此不是同算力排名或完整论文复现。", "",
              "## 冻结权重迁移", "",
              "直接将 ETTh1 微调导出的网络权重用于 ETTh2 与 Exchange Rate，不在目标数据上更新参数。",
              "目标数据仅用其训练区间统计量进行标准化，并使用其自身的七／八列数组布局；不沿用 ETTh1 表格列配置。",
              "下表同为标准化 MSE，± 为源模型训练种子的样本标准差。DLinear 不参加冻结迁移。", ""]
    lines += score_table(transfer, "target_dataset", reference=False)
    pairs = pd.DataFrame(audit["transfer"]["paired_results"])
    for mode, title in [("full", "全参数"), ("lora", "LoRA")]:
        selected = pairs[pairs["mode"] == mode]
        lines += ["", f"{title}冻结迁移的 {len(selected)} 次运行中，MAE 改善 {(selected.mae_gain_pct > 0).sum()} 次，"
                  f"MSE 改善 {(selected.mse_gain_pct > 0).sum()} 次，MSE 相对改善范围 "
                  f"{selected.mse_gain_pct.min():+.2f}%～{selected.mse_gain_pct.max():+.2f}%。"]
    lines += ["", "冻结迁移结果与分别微调的结果应分别解读；源任务有收益不保证迁移后有收益。", "",
              "## 四档模型与设备", "",
              "| 模型 | CPU | MPS | CUDA |", "|---|---|---|---|"]
    for size in ["5M", "28M", "123M", "209M"]:
        lines.append(f"| {size} | 通过 | 通过 | 无硬件，未验证 |")
    lines += ["", "每个模型／设备组合都使用真实权重和固定版本的 ETTh1 数据，覆盖：", "",
              "- 原模型官方预测一致性、全参数有限梯度、LoRA 零初始增量与有限梯度、冻结基础参数、LoRA 合并。",
              "- 全参数与 LoRA 各四步训练；暂停后恢复；核对最新一步模型、优化器、学习率调度器、Python／NumPy／Torch／采样器／MPS 随机状态。",
              "- 适配器和合并权重重新加载、表格预测、官方接口加载导出权重并进行数值比较。",
              "- 两目标、四类协变量、缺失历史、非整 patch 的 L63/H41；七目标 L256/H96、L1024/H192、L256/H1024 的全参数有限梯度，以及 L256/H96 的 LoRA 有限梯度。", "",
              "CPU 未更新模型官方一致性仍为 rtol1e-4/atol1e-5；跨设备为 rtol1e-3/atol1e-4。",
              "仅真实数据 LoRA 合并／导出按历史标准差换算后采用 rtol1e-4/atol1e-4，记录原始绝对误差与相对 L2 误差。",
              "这是功能及指定形状验收，不能推导大模型充分训练后的预测收益、长时间稳定性、最大上下文／批量容量或吞吐排名。",
              "MPS 网络在 GPU，官方 float64 统计在 CPU；所有本轮大模型检查均为 FP32。", "",
              "## 独立核验与复现", "",
              "独立核验重新读取原始数据，逐一核对 94 个评估记录的全部测试标签、起点及保存预测的 MAE／MSE，",
              "并核对训练集标准化统计量、源文件校验值、预测文件校验值、验证集选优及冻结迁移的源权重校验值。",
              "区间覆盖率由运行时流式计算，本轮未从仅保存中位数的数组独立重算，因此主要结论使用 MAE／MSE。", "",
              "```sh", "python scripts/benchmark_suite.py --jobs 2", "python scripts/transfer_validation.py",
              "python scripts/validate_models.py", "python scripts/validate_model_shapes.py",
              "python scripts/audit_validation.py", "python scripts/summarize_validation.py",
              "TIMESTRAL_INTEGRATION=1 TIMESTRAL_LARGE_MODELS=1 pytest -q", "```", "",
              "脚本复用已完成记录；重新独立运行请指定新的 output/results 目录，并让 audit/summary 的路径参数指向新目录。",
              "三个原始数据文件与 DLinear 源码均固定版本并校验 SHA256。",
              "ETT 使用 8640/2880/2880；Exchange 使用 70/10/20% 划分，日期为仅保序的合成日序号，不构造日历协变量。",
              "本基准的额外全局标准化由实验脚本执行，统计量保存在每个 protocol.json；核心表格 API 不自动套用这些实验统计量。",
              "复用基准权重时须按对应协议处理输入／输出，并保留相同的预测后处理设置。",
              "原模型／全参数／LoRA 三种 Timestral 对照使用相同的中位数／镜像平均；标准化后为有符号数据，关闭非负裁剪。",
              "大文件预测与检查点位于忽略的 runs/validation；小型结果、协议和核验记录位于 results/validation。",
              "并发执行时的耗时不用于 CPU/MPS 或方法间速度排名。", "",
              "## 证据与边界", "",
              "- [全部适配记录](benchmark/runs.csv)及[汇总](benchmark/summary.csv)：保留所有种子与退化。",
              "- [冻结迁移记录](transfer/runs.csv)及[汇总](transfer/summary.csv)。",
              "- [独立核验](audit.json)、[模型检查记录](models)、[验证矩阵](plan.json)。",
              "- [ETDataset 原始来源](https://github.com/zhouhaoyi/ETDataset)、[Exchange 原始来源](https://github.com/laiguokun/multivariate-time-series-data)。",
              "- [固定版本 DLinear](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/DLinear.py)。", "",
              "仅覆盖电力与汇率两个领域、两个预测长度、三个种子、有限适配预算；未验证天气、交通、零售等领域。",
              "是否与预训练数据重叠未知，不能将这些分数称为严格未见数据的零样本泛化成绩。",
              "多种子标准差不替代显著性检验；本轮不对重叠标签做错误的独立样本置信区间。",
              "未进行大模型的多数据集效果实验，也未进行 CUDA 实机验证。", ""]
    (root / "README.zh-CN.md").write_text("\n".join(lines))
    english = ["# Broader Timestral validation", "",
               "Completed: 66 in-dataset benchmark evaluations, 28 frozen-transfer evaluations, and 8 real-weight model/device acceptance cases.",
               "See [the detailed Chinese record](README.zh-CN.md), [benchmark summary](benchmark/README.md), [frozen-transfer scores](transfer/summary.csv), and [independent audit](audit.json).", "",
               "Three datasets (ETTh1/ETTh2/exchange), L96/H96,192, three seeds, M all columns, train-only scaling, test stride1.",
               "Full/LoRA adaptation is compared with base, persistence and the unchanged official DLinear network using an explicit local training loop.",
               "Separate frozen transfer applies ETTh1-adapted network weights to ETTh2/exchange without target-dataset optimization.",
               "All four model sizes pass CPU/MPS short training/resume/export and representative/boundary backward checks; CUDA is unverified.",
               "These checks do not establish universal quality gains, equal-compute rankings, large-model forecast quality, or maximum capacity.", ""]
    (root / "README.md").write_text("\n".join(english))
    write_json(root / "code_provenance.json", {
        "files_sha256": {str(p): file_sha256(p) for parent in ["src/timestral_ft", "scripts", "tests"]
                         for p in sorted(Path(parent).glob("*.py"))},
        "reproduction": "Run scripts from repository root with locked dependencies; results contain pinned model/data revisions",
    })
    print(root / "README.zh-CN.md")


if __name__ == "__main__":
    main()
