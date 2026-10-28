# timestral-ft

**用自己的时间序列数据微调 Timestral。**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)](pyproject.toml)
[![License](https://img.shields.io/badge/Code%20License-Apache--2.0-green)](LICENSE)

[English](README.md) | **简体中文**

`timestral-ft` 是面向 [Mistral AI Timestral](https://github.com/mistralai/timestral)
的社区微调工具包，支持全参数与 LoRA 两种方式。通过统一的 Python API 和
`train`、`evaluate`、`export`、`predict` 四个命令，完成表格数据接入、训练选优、滚动回测和权重导出。

项目复用官方网络，独立提供训练层，并保留实验中的改善与退化结果。
这是独立社区项目，与 Mistral AI 没有隶属关系。

[功能概览](#功能概览) · [快速开始](#快速开始) · [实验结果](#实验结果) · [文档导航](#文档导航)

## 功能概览

- **全参数与 LoRA 微调**：更新全部网络参数，或仅适配时间／变量注意力投影。
  LoRA 默认 rank=8、alpha=16，初始增量为零。
- **适配自己的数据**：接入 DataFrame、CSV、Parquet，支持多序列、多目标联合预测，
  以及历史／未来已知的数值和分类协变量。
- **完整训练流程**：梯度累积与裁剪、预热和余弦学习率、验证选优、早停、断点续训。
- **预测与回测**：输出中位数和全部 21 个分位数，保存逐窗口预测、MAE／MSE／RMSE、
  WQL 和 80% 区间覆盖率。
- **可迁移的训练产物**：保存预处理配置，导出独立 LoRA 适配器，或合并为官方推理格式权重。
- **可复现实验**：提供固定版本 ETTh1 示例、多变量多种子基准、冻结迁移检查，以及完整
  Timestral-5M 与 Timestral-209M 的完整 GIFT-Eval 97 任务 LoRA 微调记录。

## 模型与设备支持

| 基础模型 | CPU 全参数／LoRA | Apple MPS 全参数／LoRA | 预测效果实验 |
|---|---|---|---|
| Timestral-5M | 已验证 | 已验证 | ETTh1、多变量基准、GIFT-Eval |
| Timestral-28M | 已验证 | 已验证 | 尚未开展 |
| Timestral-123M | 已验证 | 已验证 | 尚未开展 |
| Timestral-209M | 已验证 | 已验证 | GIFT-Eval 全部 97 任务 |

“已验证”指使用真实权重完成短程训练、有限梯度、断点恢复、适配器／合并权重加载和官方格式导出检查，
这些功能检查不代表预测效果、最大批量或最大上下文容量已验证。详见[模型验收记录](results/validation/models)。

v0.1 使用**单设备 FP32**，显式选择 `cpu`、`mps`、`cuda` 或 `cuda:1`，默认 CPU。
CUDA 已实现，**尚未进行硬件实测**。MPS 模式下网络在 Apple GPU 上计算，float64 统计在 CPU 上计算；
小规模任务可能在 CPU 上更快。MPS 推理请使用本工具的加载接口。
首版训练预测长度上限为 1,024 步，暂不包含多卡训练和从零预训练。

## 快速开始

### 1. 从源码安装

使用 Git 克隆仓库后安装。参考环境为 Python 3.12，包声明的最低版本为 Python 3.10。

```sh
git clone https://github.com/elarazju/timestral-ft.git
cd timestral-ft
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

使用 NVIDIA GPU 时，先按 [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
安装适合当前 CUDA 环境的 PyTorch。需要固定开发依赖时，可使用
`uv sync --locked --extra dev --python 3.12`。

官方代码依赖固定到提交 `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`。
首次加载模型会从 Hugging Face 下载权重，后续可使用缓存或本地权重目录。
仓库不附带数据集和预训练权重。

### 2. 准备数据并微调

ETTh1 数据准备脚本会校验源文件、按时间划分数据，并生成时间戳特征。
示例预测 `OT`，使用历史 `HUFL`，以及未来已知的日历数值与分类协变量。

```sh
python scripts/run_etth1.py --prepare-only

# LoRA：Timestral-5M，历史 256，预测 96，最多 200 个优化步骤。
timestral-ft train --config examples/etth1.yaml --mode lora --output runs/quickstart/lora

# 全参数微调。
timestral-ft train --config examples/etth1.yaml --mode full --output runs/quickstart/full
```

增加 `--device mps` 或 `--device cuda` 可选择加速器。
新实验需要空的输出目录；中断后使用 `--resume <run>/checkpoint.pt`，并保持数据与配置不变。
详见[暂停和恢复](docs/guide.zh-CN.md#暂停和恢复)。

### 3. 回测与导出

```sh
timestral-ft evaluate --model runs/quickstart/lora/model \
  --data data/etth1/test.parquet --history data/etth1/history.parquet \
  --output runs/quickstart/evaluation

# 导出独立适配器，再合并为官方兼容权重。
timestral-ft export --model runs/quickstart/lora/model \
  --output runs/quickstart/adapter --adapter-only
timestral-ft export --model runs/quickstart/adapter --output runs/quickstart/merged
```

回测目录保存 CSV／Parquet 预测、指标和元数据。合并导出包含 `config.json` 与
`model.safetensors`，可由官方 `TimestralForecaster` 在 CPU／CUDA 上加载。
处理原始表格时需一并保留预处理文件；独立适配器依赖所记录的基础权重及其修订版本。

## Python API

先完成上面的 ETTh1 数据准备，再执行：

```python
import pandas as pd
from timestral_ft import DataConfig, FineTunedModel, TrainConfig, evaluate, finetune

model = finetune(
    "data/etth1/train.parquet",
    "data/etth1/validation.parquet",
    model_id="mistralai/Timestral-5M",
    revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    data_config=DataConfig(
        time_column="date",
        target_columns=["OT"],
        past_covariates=["HUFL"],
        known_covariates=["hour_sin", "hour_cos"],
        categorical_known_covariates=["hour_category", "weekday_category"],
        frequency="h",
    ),
    train_config=TrainConfig(mode="lora", device="cpu"),
    output_dir="runs/python-lora",
)

result = evaluate(
    model, "data/etth1/test.parquet",
    history="data/etth1/history.parquet",
    output_dir="runs/python-lora/evaluation",
)
print(result.metrics)

loaded = FineTunedModel.from_pretrained("runs/python-lora/model", device="cpu")
future = pd.read_parquet("data/etth1/test.parquet").iloc[:96]
forecasts = loaded.predict(
    "data/etth1/history.parquet", 96, future_covariates=future,
)
print(forecasts[["timestamp", "target", "median", "q_0.1", "q_0.9"]].head())
```

预测时会忽略未来表中的目标列，仅使用配置为“未来已知”的协变量。
输出中每行对应一个序列、一个目标和一个预测时间点。
命令行预测参数可通过 `timestral-ft predict --help` 查看。

## 接入自己的数据

每行表示一个序列在一个时间点的观测。请先按时间划分训练、验证和测试集，
再以文件路径或 pandas DataFrame 分别传入。

| `DataConfig` 字段 | 含义 |
|---|---|
| `time_column` | 观测时间戳 |
| `id_column` | 可选序列 ID；单序列时省略 |
| `target_columns` | 一个或多个联合预测的目标列 |
| `past_covariates` | 仅历史可得的数值特征 |
| `known_covariates` | 整个预测区间均已知的数值特征 |
| `categorical_past_covariates` | 仅历史可得的分类特征 |
| `categorical_known_covariates` | 整个预测区间均已知的分类特征 |

窗口标签不会跨越划分边界，验证／测试可以读取此前观测作为历史。
不同序列之间不会通过注意力交换信息。重复时间点会报错；不规则观测需要显式指定
`frequency`，缺少的时间点保留 NaN 掩码，不插值目标。
分类词表仅在训练集建立，随模型保存；未知类别使用保留编码。

默认训练目标是按历史尺度平衡的全部 21 分位数 masked pinball loss。
可通过 `TrainConfig(loss_space="normalized")` 或 YAML 中的同名字段，选择技术报告描述的
逐 patch 归一化空间损失；独立的 `validation_loss_space` 可固定选优准则。
详见[报告对齐记录](docs/report-alignment.zh-CN.md)与[第一轮对照实验](results/report_round1/README.md)。
归一化、分类编码、数值容差和推理兼容层的细节见[使用指南](docs/guide.zh-CN.md)
和[英文技术参考](docs/reference.md)。

## 实验结果

### GIFT-Eval：Timestral-209M 完整 97 任务

**已完成**全部 97 项任务的验证搜索、适配器冻结与官方成对测试。
种子 42，LoRA rank=8、alpha=16，可训练参数 1,105,920。
训练上下文候选上限为 256／2048／8192，各筛选 200 步，验证集优胜方案延长至最多 800 步，
总计执行 113,000 个优化步骤。21 个任务最终按验证结果回退到第 0 步原模型。
测试保留全部官方序列、目标和滚动窗口，推理上下文上限为 32,768。

| 归一化几何均值 ↓ | 原模型 | 选中 LoRA／回退 | 相对变化 | 改善／退化／不变任务数 |
|---|---:|---:|---:|---:|
| CRPS 近似 | 0.446010 | 0.448549 | +0.569% | 31／45／21 |
| MASE | 0.657547 | 0.660880 | +0.507% | 33／43／21 |

**本轮配方未改善测试集整体表现。** 最终选中的检查点在验证集上改善 CRPS／MASE 约
3.74%／4.06%，但收益未迁移到测试集。反复在每条序列的单一验证时间段上选优可能是原因之一，
目前属于诊断假设，尚无因果对照证明；单种子也不足以判断差异的统计显著性。

97 个原模型结果的全部 11 项指标均通过公开参考值一致性检查（rtol=1e-3、atol=1e-5），
检查点与数据完整性审计通过。相对固定公开快照的本地插入模拟名次为 CRPS 6／154、MASE 14／154，
**不是官方榜单名次**。适配器权重尚未公开，尚未正式提交榜单。

[完整报告](results/gift_eval_209m_lora_v2/README.zh-CN.md) ·
[逐任务对比](results/gift_eval_209m_lora_v2/task_comparison.csv) ·
[选中上下文](results/gift_eval_209m_lora_v2/context_lengths.csv) ·
[审计记录](results/gift_eval_209m_lora_v2/audit.json) ·
[复现方案](examples/gift_eval/209m-search.md)

### GIFT-Eval：Timestral-5M 完整 97 任务

**已完成**：97 个独立 Timestral-5M LoRA 适配器、97 项原模型评估和 97 项微调模型评估。
每项任务使用种子 42、训练 200 个优化步骤；验证最多固定抽取 512 条序列，
测试覆盖**全部官方序列、目标和滚动窗口**。92 项任务使用 CPU 训练，5 项使用 MPS。

| 归一化几何均值 ↓ | 基础模型 | LoRA | 相对变化 | 改善／退化任务数 |
|---|---:|---:|---:|---:|
| CRPS 近似 | 0.478864 | 0.479010 | +0.030% | 51／46 |
| MASE | 0.704905 | 0.705210 | +0.043% | 45／52 |

本轮固定预算配方**没有带来整体收益**，变化接近零，单种子不足以判断差异是否稳定。
原模型 97／97 项的全部 11 项指标均在记录的容差内与官方结果匹配。
本实验使用任务的训练／验证分区，属于 `fine-tuned`。
本地提交草稿已通过官方格式校验，**尚未提交榜单**。
几何均值分数与榜单使用的逐任务平均名次不同。

[完整报告](results/gift_eval_5m_lora/README.zh-CN.md) ·
[逐任务对比](results/gift_eval_5m_lora/task_comparison.csv) ·
[审计记录](results/gift_eval_5m_lora/audit.json) ·
[复现指南](examples/gift_eval/README.md)

GIFT-Eval 需要使用记录了 NumPy／pandas 版本的**独立环境**。
请按复现指南完成配置，再运行 `scripts/run_gift_eval.py`。
适配器和优化器检查点由本地运行生成，不附带在源码仓库中。

### 标准多变量基准

Timestral-5M，联合预测 ETT 全部 7 列／汇率全部 8 列目标，历史长度 96，预测长度 96／192，
仅使用训练集统计量标准化，测试滑窗步长为 1。
微调结果为种子 42／43／44 的均值 ± 样本标准差；冻结的基础模型评估一次。

| 数据集 | 预测长度 | 基础模型 MSE ↓ | 全参数 MSE ↓ | LoRA MSE ↓ |
|---|---:|---:|---:|---:|
| ETTh1 | 96 | 0.43385 | 0.41122 ± 0.00098 | 0.41476 ± 0.00096 |
| ETTh1 | 192 | 0.50281 | 0.47749 ± 0.00108 | 0.49542 ± 0.00024 |
| ETTh2 | 96 | 0.30170 | 0.29879 ± 0.00023 | 0.30071 ± 0.00013 |
| ETTh2 | 192 | 0.38884 | 0.38560 ± 0.00011 | 0.38772 ± 0.00001 |
| Exchange Rate | 96 | 0.08178 | 0.07964 ± 0.00015 | 0.08133 ± 0.00019 |
| Exchange Rate | 192 | 0.16748 | 0.16415 ± 0.00063 | 0.16702 ± 0.00037 |

[完整基准报告](results/validation/benchmark/README.md) 包含 MAE、持平预测和官方 DLinear 网络参照及其训练设置。
各方法的预算与训练目标不同，因此不构成同算力排名，也不宣称精确复现论文结果。
预训练数据重叠尚未独立审计。ETTh1 权重冻结迁移到 ETTh2／汇率的结果
[单独报告](results/validation/transfer/summary.csv)。

快速开始中的 [ETTh1 协变量示例](results/etth1_cpu/README.md) 仅预测 `OT`，历史长度 256、
滑窗步长 96，MAE 使用原始单位，与上面的多变量标准化结果不能直接比较。

```sh
# ETTh1 原模型／全参数／LoRA 协变量示例。
python scripts/run_etth1.py

# 多变量基准。
python scripts/benchmark_suite.py --jobs 2
```

冻结迁移、模型验收及独立审计命令见[完整验证流程](docs/reference.md#evaluation-and-verification)。

## 文档导航

| 文档 | 内容 |
|---|---|
| [中文使用指南](docs/guide.zh-CN.md) | 数据准备、配置、训练、续训、预测与导出 |
| [英文技术参考](docs/reference.md) | 训练语义、设备行为、指标与兼容层细节 |
| [报告对齐记录](docs/report-alignment.zh-CN.md) | 官方机制、社区实现选择、第一轮损失及后续任务 |
| [ETTh1 示例配置](examples/etth1.yaml) | 包含数值和分类协变量的可运行配置 |
| [综合验证记录](results/validation/README.zh-CN.md) | 多种子基准、冻结迁移和四档模型验收 |
| [GIFT-Eval 运行指南](examples/gift_eval/README.md) | 环境配置、全任务训练和官方评估 |
| [209M 搜索方案](examples/gift_eval/209m-search.md) | 已完成的验证搜索、冻结后 97 任务测试及 CPU/MPS 执行工具 |
| [209M 设备吞吐实测](results/gift_eval_209m_preflight/README.md) | CPU 任务调度、MPS 进程并发与推理批量对照 |
| [贡献指南](CONTRIBUTING.md) | 开发环境与贡献要求 |

```text
timestral-ft/
├── src/timestral_ft/    # 训练、适配器、表格推理与 GIFT 集成
├── examples/           # YAML 配置与 GIFT-Eval 环境说明
├── scripts/            # 可复现实验、审计与打包
├── tests/              # 小网络测试与可选真实权重检查
├── results/            # 指标、实验协议与验证报告
├── docs/               # 使用指南与技术参考
├── README.md
└── README_zh.md
```

## 开发与测试

```sh
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests scripts
python -m build
```

默认测试使用随机初始化的小网络，不下载模型权重，覆盖官方预测一致性、梯度、序列隔离、
未来目标泄漏、分类输入、断点恢复及适配器／合并导出。真实权重检查需显式开启：

```sh
TIMESTRAL_INTEGRATION=1 TIMESTRAL_LARGE_MODELS=1 pytest -q
```

不可用设备的测试会跳过。需要强制验证某个后端时，使用
`TIMESTRAL_TEST_DEVICE=cuda pytest -q tests/test_devices.py`（或 `mps`）；
此时缺少所需硬件会导致任务失败。大模型功能检查不代表已验证大模型预测收益。

## 致谢与许可证

项目基于 [Timestral](https://github.com/mistralai/timestral)，评估使用
[GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) 和
[GluonTS](https://github.com/awslabs/gluonts)。DLinear 参照网络来自
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)，README 的结构也参考了该项目。

工具代码采用 [Apache-2.0](LICENSE)，上游归属说明见 [NOTICE](NOTICE)。
基础权重及微调衍生权重继续遵循各自的
[Modified MIT 模型许可证](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE)，
包括其中的商业使用条件。导出和合并不改变权重许可。
