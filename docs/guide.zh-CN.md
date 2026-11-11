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
运行时还会检测 [M1／macOS 26 的线性层偏置问题](https://github.com/pytorch/pytorch/issues/188438)。
受影响的 MPS 环境在 GPU 上分别计算矩阵乘法与偏置相加；其他环境保持原计算路径。
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
默认 `loss_space="history_scaled"`，训练损失是按历史标准差平衡的 21 分位数 pinball loss；
常数历史使用尺度 1。新增 `loss_space="normalized"`，在模型的逐 patch 归一化空间计算损失，
对应技术报告 §3.3／§4.1：目标与预测使用相同推进统计，并应用模型配置的 `asinh` 变换。
后续统计只读取历史与此前预测中位数，不读取未来真实目标。两种方式都只统计有效目标。

`validation_loss_space=None` 默认沿用训练损失；也可以指定 `"history_scaled"` 或 `"normalized"`。
比较两种训练目标时，建议固定共同验证准则，避免同时改变检查点选优方式：

```yaml
training:
  mode: lora
  loss_space: normalized
  validation_loss_space: history_scaled
```

这些字段也可传给 `TrainConfig`，会随模型和检查点保存。不同损失空间的数值不能直接比较。
更换损失空间需要新输出目录；旧配置缺少新增字段时保留历史行为。
默认仍为单预测区间微调；归一化损失或下述 DHF 都不代表复现官方完整预训练流程。
详见[技术报告对齐记录](report-alignment.zh-CN.md)。

训练保留官方逐 patch 更新的预测尺度；镜像平均、非负裁剪和极值裁剪只用于评估预测，
不会应用在训练损失上。训练预测长度最多为单块上限 1024 步。

### 多预测块 DHF 训练

将以下字段放入 YAML 的 `training`，或传给 `TrainConfig`：

```yaml
training:
  mode: lora
  training_strategy: dhf
  loss_space: normalized
  validation_loss_space: history_scaled
  context_length: 256
  prediction_length: 192
  dhf_min_context_patches: 1
  dhf_max_block_patches: 6
```

默认 `training_strategy="window"` 保持现有行为。DHF 中，两种长度之和决定训练序列预算：
上例为 448 点，整条序列均来自训练集。初始 1 个 patch，即已发布模型上的 32 点，只作为上下文；
其余 416 点各被一个预测块覆盖。默认窗口方案只监督末尾 192 点。
DHF 实际历史取决于块起点，上例可为 32–416 点；不要将配置的 256 误读为每个块的历史长度。

每个样本单独采样布局，同一序列的所有变量共享布局；块长均匀采样自 1 到最大值，
末块截断。这是本项目的采样选择，官方报告未披露其分布。
`dhf_max_block_patches=None` 默认取 `ceil(prediction_length / patch_size)`，
不得超过所选模型单块上限；初始上下文必须留下至少一个真实监督位置。
最长可能历史也不得超过模型上下文上限。不整除 patch 的序列在右侧用缺失值补齐，
补齐部分不会成为标签，输出裁回原长度。

DHF 要求 `loss_space="normalized"`，可训练全部参数或 LoRA。
数值型历史协变量和未来已知协变量覆盖完整训练序列，按固定角色处理；
仅未来已知变量允许全窗归一化和双向读取，角色掩码阻止未来目标／历史协变量泄漏。
分类协变量使用逐样本、逐起点的安全参考路径，见下述说明；不会共享数值路径的一次前向加速。

验证始终使用普通单起点预测，上例读取 256 点历史预测 192 点；不会采样 DHF 布局。
比较训练策略时固定相同的 `validation_loss_space` 和验证长度。
每步记录实际监督点数、预测块数及块长分布；块数按样本计算，同一布局不会按目标数重复计数。
窗口与布局使用独立随机生成器，二者状态随检查点恢复；验证不消耗二者的随机流。

对照实验、相同步数与近似监督点数控制的边界见[第二轮实验记录](../results/report_round2/README.md)，
机制、真实权重和设备覆盖见[第二轮验收报告](../results/report_round2/acceptance.zh-CN.md)。

### 连续缺口与角色采样

两种增强默认关闭，可独立启用。将下列字段加到 `training`：

```yaml
training:
  missing_probability: 0.5
  missing_max_span: 64
  dynamic_role_sampling: true
  role_probabilities: [0.5, 0.25, 0.25]
```

`missing_probability` 默认 0，取值 0–1；每个数值行按此概率添加一段连续缺口。
`missing_max_span` 默认 64，必须为正整数，实际不超过序列长度；长度与合法起点均匀采样。
输入副本与原始标签独立：人工隐藏的有限值仍可监督，原始 NaN 不计分；目标在某块之前没有有效历史时，该块不计分。
分类代码不做数值缺口增强。窗口训练中，原本不可见的未来目标槽被隐藏不会额外改变前向；
DHF 则还可以隐藏后续块本来能够读取的上下文副本。

`dynamic_role_sampling` 默认 false，只对原 `target_columns` 采样角色。
概率顺序为目标／历史协变量／未来已知协变量，默认 [.5,.25,.25]，总和为 1，目标概率必须大于 0。
若抽样后没有目标，会随机选回一个；固定声明的协变量保持原角色。角色按样本固定，同一样本的 DHF 块共用。
非目标行不参与损失，角色通过官方注意力权限和归一化方式生效；这些分布属于项目选择，不是官方公开配方。

窗口、布局、缺口、角色分别使用独立且可恢复的随机状态；验证不采样增强。
缺口／角色种子为 `seed ^ 0x4D495353`／`seed ^ 0x524F4C45`。
日志增加原始／输入观测点、人工隐藏点、隐藏且监督点、三个角色行数；改增强配置须建立新实验。
全参数／LoRA、窗口两种损失和 DHF 都支持这些配置，旧配置保留原行为。

DHF 分类输入对每个起点重新调用单起点接口：历史截在起点，已知未来截在当前块末尾，
不使用未来目标建立类别表示。固定上游仅对当前单目标样本做目标编码，多目标沿用数值代码，
动态角色可能改变分支。表格词表紧凑且只拟合训练数据；底层任意大类别代码在多目标 FP32 回退时不保证无损。
类别编码前保持 CPU float64，离散编码本身不提供输入梯度，但支持网络／LoRA 参数训练。
该参考路径需要多次前向，不等同于纯数值路径的全窗口已知协变量归一化范围或共享加速。

可运行 `python examples/report_round3/smoke.py --device cpu` 检查接入流程。
四组增强、完整／固定缺口测试及全部指标见[第三轮实验](../results/report_round3/README.md)，
字段与限制见[技术参考](reference.md#missing-observations-and-dynamic-roles)。本轮测试仍是全目标预测，
没有未来已知输入；机制通过不等于条件预测质量收益已经验证。


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
