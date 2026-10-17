# Timestral 微调工具包使用指南

[项目首页](../README_zh.md) · [English](../README.md)

这是独立社区项目，提供全参数与 LoRA 微调、表格数据接入、回测和官方格式权重导出。
项目与 Mistral AI 没有隶属关系。

## 安装与首次运行

需要 Git 和 Python 3.10 以上；参考环境是 Python 3.12。使用独立虚拟环境：

```sh
uv venv --python 3.12
uv pip install -e '.[dev]'
source .venv/bin/activate
python scripts/run_etth1.py --prepare-only
timestral-ft train --config examples/etth1.yaml
```

依赖的官方代码固定到 `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`。
首次加载模型需要联网下载权重，随后可使用缓存或本地权重目录。仓库不附带模型权重或数据集。

v0.1 支持 CPU、CUDA 与 Apple MPS 的单设备 FP32 训练。CPU、MPS 已进行本机硬件验证；
CUDA 使用相同测试流程，但本机没有 NVIDIA GPU，尚未完成 CUDA 硬件验证。
四档模型均完成 CPU／MPS 真实权重功能验收：全参数与 LoRA 各四步训练、断点续训、适配器与合并权重加载、
官方接口导出一致性，以及七变量的 256→96、1024→192、256→1024 有限梯度检查。
这些短实验验证功能支持，不代表大模型预测收益、最大批量或最大上下文容量已验证。

## 选择训练与预测设备

Python 中设置 `TrainConfig(device="mps")`、`device="cuda"` 或 `device="cuda:1"`。
命令行可直接覆盖配置文件中的设备：

```sh
timestral-ft train --config examples/etth1.yaml --device mps --output runs/my-mps-run
timestral-ft train --config examples/etth1.yaml --device cuda --output runs/my-cuda-run
```

加载时使用 `FineTunedModel.from_pretrained(path, device="mps")`；命令行 `evaluate` 与
`predict` 也支持 `--device mps` 或 `--device cuda`。请求不可用设备会报错，不会静默切到 CPU。

MPS 模式下，网络在 Apple GPU 上计算；float64 归一化统计、分类目标编码及预测尺度更新
在 CPU 上执行，再返回 GPU。尺度更新保留反向传播连接，不把高精度统计直接降为 float32。
CPU/GPU 同步有开销，因此小模型、小窗口场景下 MPS 可能比 CPU 慢。
CUDA 使用设备原生 float64；类别编码在各设备下都先保留为 CPU float64。
断点保存与恢复包含所选加速器的随机状态，包括 MPS dropout。

导出权重格式保持官方兼容；官方原版推理在 MPS 上仍有 float64 限制，MPS 预测请使用本工具加载。
跨设备数值按 `rtol=1e-3, atol=1e-4` 验证，不要求逐位一致。
`python scripts/run_etth1.py --device mps` 可运行 MPS 实验。仓库另附 4 步硬件冒烟测试结果
`results/etth1_mps_smoke`，用于验证链路，不与 200 步 CPU 实验直接比较性能。

## 准备自己的数据

CSV、Parquet 路径与 pandas DataFrame 均可。每行对应一个序列的一个时间点：

| timestamp | series_id | demand | sensor | price | weekday |
|---|---|---:|---:|---:|---|
| 2026-01-01 00:00 | store-a | 10 | 2.3 | 5 | Thu |
| 2026-01-01 01:00 | store-a | 12 | 2.5 | 5 | Thu |

配置列的角色：

```python
from timestral_ft import DataConfig, TrainConfig, finetune

data_config = DataConfig(
    time_column="timestamp",
    id_column="series_id",                 # 单序列时省略
    target_columns=["demand"],             # 多列时联合预测
    past_covariates=["sensor"],            # 仅历史可得
    known_covariates=["price"],            # 历史与预测区间均可得
    categorical_known_covariates=["weekday"],
    frequency="h",                        # 小时采样
)
model = finetune(
    "train.parquet", "validation.parquet",
    data_config=data_config,
    train_config=TrainConfig(mode="lora"),
    output_dir="runs/custom",
)
```

分类历史协变量使用 `categorical_past_covariates`。同一列只能承担一种角色。
“未来已知”表示在实际预测时就能得到的信息，不能把未来才观测到的真实天气等当作已知输入。

先按时间切分训练、验证、测试，再分别传入。训练窗口的标签不得跨越训练边界。
验证和测试可以读取此前历史；测试集只用于最终回测，不参与词表建立、选优或早停。

每个序列的时间点不得重复。不规则时间间隔必须明确 `frequency`，工具将缺少的时间点补成 NaN。
不会双向插值目标，也不会使用未来目标建立归一化统计。数值列的 Inf 和非数值内容会报错。

分类词表只从训练集建立。未出现过的类别使用保留编码 0；缺失保持 NaN。
单目标使用官方基于当前历史窗口的目标编码，多目标沿用官方直接使用类别数值编码的行为。
因此多目标分类编码并非独立的类别 embedding；本项目保持其与官方推理的一致性。

## 两种微调方式

- `mode="full"`：更新全部网络参数，默认学习率 `1e-5`。
- `mode="lora"`：只更新注意力 `wQKV` 和 `wO` 上的适配器，默认学习率 `1e-4`，
  rank=8、alpha=16、dropout=0，初始增量为零。

默认历史长度 256、预测长度 96、batch size 4、梯度累积 4、最多 200 个优化步骤。
使用 AdamW，前 20 步预热后余弦衰减；每 50 步验证，连续三次没有改善则提前停止。
训练损失是按历史标准差平衡的 21 分位数 pinball loss；常数历史使用尺度 1。
损失只统计有效标签，不是对官方未公开训练流程的复现。

训练保留官方逐 patch 更新的预测尺度；镜像平均、非负裁剪和极值裁剪只用于评估预测，
不会应用在训练损失上。训练预测长度最多为单块上限 1024 步。

## 暂停和恢复

```sh
timestral-ft train --config examples/etth1.yaml --stop-after-steps 75
timestral-ft train --config examples/etth1.yaml --resume runs/etth1/full/checkpoint.pt
```

保持原来的 `max_steps`、数据、配置和模型版本不变。恢复会继续最新一步的模型、优化器、
学习率调度器及随机采样状态，不会从最佳模型重新开始。最终导出的是验证集最佳模型。
暂停不会额外触发验证；完成前不会写最终 `model` 导出目录。
非空运行目录不能当作全新实验覆盖，只能恢复或改用新目录。

## 回测与预测

```python
from timestral_ft import evaluate, FineTunedModel

result = evaluate(model, "test.parquet", history="train-and-validation.parquet")
print(result.metrics)
result.save("runs/custom-evaluation")

loaded = FineTunedModel.from_pretrained("runs/custom/model")
predictions = loaded.predict("history.parquet", 96, future_covariates="future.parquet")
```

`history` 只能包含标签区间之前的数据。配置未来已知协变量时，未来表必须恰好覆盖接下来
96 个时间点并包含相应特征列；缺失协变量保留掩码。未来表中即使包含目标列，也不会读取这些目标值。

回测为滚动起点预测，默认步长等于预测长度；所有对照使用同样的官方默认预测后处理。
保存逐窗口 CSV/Parquet、MAE、MSE、RMSE、WQL、80% 区间覆盖率以及配置和数据指纹。
点预测指标使用**传入目标的尺度**：原始表格对应原始单位；评估器不自动进行全数据集标准化。
ETTh1 示例只预测 OT，历史长度 256、滑窗步长 96；不能直接与七变量 M、历史 96、步长 1 的榜单比较。
例如原示例全参数 MAE 1.658971 换算为标准化 OT MAE 约 0.180785，但任务和窗口仍然不同。
`python scripts/verify_etth1_metrics.py --run-standard-control` 可复算单位换算，并运行独立 S/M 数值对照。
WQL 为 `2×分位数损失之和 / (21×真实值绝对值之和)`，分母为零时写 null。
缺失标签不参与统计。80% 区间覆盖率是实际落入 0.1～0.9 分位区间的比例，不保证一定为 80%。

## 导出模型

完成训练后，`runs/custom/model` 内包含官方兼容的 `config.json` 与 `model.safetensors`，
以及本工具需要的列配置、词表和版本元数据。

```sh
timestral-ft export --model runs/custom/model --output runs/adapter --adapter-only
timestral-ft export --model runs/adapter --output runs/merged
```

仅 LoRA 模型可以导出独立适配器；加载适配器需要对应基础权重。合并权重可独立加载。
官方 `TimestralForecaster.from_pretrained("runs/merged")` 接受数组，原始表格请使用本项目
的 `FineTunedModel.predict()`，以复用保存的列角色与类别编码。

## 实验与限制

运行 `python scripts/run_etth1.py` 可复现 ETTh1 原模型、全参数和 LoRA 三组实验。
采用标准 ETT 的 12/4/4 个 30 天数据块切分，固定数据提交、校验值、基础权重版本与种子。
测试集不用于选择学习率、训练步数或检查点。完整结果见 `results/etth1_cpu`。
这是单种子、单数据集的适配示例，ETTh1 可能被预训练使用，不代表零样本泛化成绩。
微调可能退化，结果如实报告。

更广的验证见 [多数据集验证记录](../results/validation/README.zh-CN.md)。实验矩阵为 ETTh1、ETTh2、Exchange Rate，
历史 96、预测 96／192、种子 42／43／44，比较原模型、全参数、LoRA、持平预测和官方 DLinear 网络参照。
采用训练集逐列标准化、全部七／八个目标、测试步长 1。DLinear 使用固定提交的官方网络和单独记录的本地训练循环；
预训练、损失、训练预算与验证聚合规则存在差异，因此不是同算力排名或完整论文复现。

```sh
python scripts/benchmark_suite.py --jobs 2
python scripts/transfer_validation.py
python scripts/validate_models.py
python scripts/validate_model_shapes.py
python scripts/audit_validation.py
python scripts/summarize_validation.py
TIMESTRAL_INTEGRATION=1 TIMESTRAL_LARGE_MODELS=1 pytest -q
```

所有种子及退化结果均保留，报告均值和样本标准差。预测、标签与起点保存于 `runs/validation`，
独立核验会重新读取原始数据，检查完整测试标签与 MAE／MSE 计算。
两类领域、三个种子和有限训练预算仍不足以证明普遍收益；预训练数据是否重叠未知。
另行报告 ETTh1 权重冻结迁移到 ETTh2／汇率的实验：目标数据只提供训练区间标准化统计及数组列布局，
不在目标数据更新权重。分别微调后的收益与冻结迁移的收益不能混为一谈。

未更新参数时的 CPU 官方预测一致性仍采用 `rtol=1e-4, atol=1e-5`。
仅真实数据 LoRA 合并／导出检查按各目标历史标准差换算后使用 `rtol=1e-4, atol=1e-4`，
并保留原始绝对误差及相对 L2 误差。FP32 运算次序变化可能让接近零的分位数超出固定物理单位绝对容差。

工具代码使用 Apache-2.0，基础模型及微调衍生权重仍受各自 Modified MIT 条款约束。
导出不改变模型许可。发布权重前应查看基础模型实际许可证。

## GIFT-Eval

专用的 [GIFT-Eval 运行指南](../examples/gift_eval/README.md) 覆盖 97 个独立任务的 LoRA 微调、
短序列／变长历史、官方训练与验证边界、适配器断点恢复及官方 11 项测试指标。
该接口直接读取 Arrow／GluonTS 序列，不展开为长 DataFrame；推理保留原始多目标与历史协变量角色。
适配器通过 `timestral_ft.gift.load_gift_adapter` 加载，使用官方数组接口预测。

```sh
python scripts/run_gift_eval.py --stage train
python scripts/run_gift_eval.py --stage evaluate --fast-seasonal-error
python scripts/audit_gift_eval.py
python scripts/summarize_gift_eval.py
```

运行在单独的固定依赖环境中。参考运行的五项 MPS 训练硬件映射保存在结果协议中；重建时需按指南
写入运行目录的 `train_device_overrides.json`。独立运行目录和结果目录通过相应命令行选项指定。
验证最多固定抽样 512 条序列，测试使用全部官方序列、目标与滚动窗口。
其九分位数 CRPS 近似与普通表格 API 的 21 分位数 WQL 不同，已有 ETT 指标不能直接用于提交。
完整结果与适配器清单见 [97 任务报告](../results/gift_eval_5m_lora/README.zh-CN.md)。
这是使用任务训练／验证分区的 `fine-tuned` 方法；公开模型及代码链接、实际提交榜单是单独步骤。
