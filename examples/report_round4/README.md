# 合成数据与分阶段继续训练

这是社区定义的合成先验和继续训练接口，不是 Timestral 未公开的生成器或从零预训练配方。
从固定已发布 5M 权重起步，按样本混合真实窗口与在线生成序列。默认不启用混合或阶段。

## 最小示例

```sh
python examples/report_round4/smoke.py
# 或先准备训练/验证表，再直接使用嵌套 YAML：
python examples/report_round4/smoke.py --prepare-only
timestral-ft train --config examples/report_round4/smoke.yaml
```

新运行使用空输出目录。示例为四步功能演示；正式对照使用下述独立脚本。
Python API 与 YAML 对应：

```python
from timestral_ft import SyntheticConfig, TrainingStage, TrainConfig, finetune

cfg = TrainConfig(
    mode="lora", training_strategy="dhf", loss_space="normalized",
    validation_loss_space="history_scaled", context_length=256, prediction_length=192,
    max_steps=200, synthetic=SyntheticConfig(),
    stages=[TrainingStage(100, 128, 1.0), TrainingStage(100, 256, .5)],
)
# train/validation tables and DataConfig must declare your actual target columns.
# finetune(train, validation, data_config=..., train_config=cfg,
#          revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f", output_dir="runs/my-curriculum")
assert TrainConfig(**cfg.to_dict()).to_dict() == cfg.to_dict()
```

`TrainingStage(steps, context_length, synthetic_probability)` 的步数之和必须等于 `max_steps`。
顶层 `context_length` 固定验证和默认预测历史；阶段上下文只影响训练。
阶段在完整优化步后切换，学习率计划和优化器持续，不重新预热或清零。
`stop_after_steps` 可以停在边界前、边界上或边界后；续训须使用相同完整配置。
原有`patience`早停仍然适用，可能提前结束后续阶段；应查看实际步数/阶段进度，而不是只读取计划。

只有出现正合成概率时才限制为纯目标列 schema；声明数值或分类协变量会在启动时明确拒绝。
纯真实阶段仍支持原有协变量。原目标的动态角色和缺口增强仍可与合成混用。
纯合成阶段不要求存在该长度的真实窗口，但仍保存真实训练表的身份和预处理配置。
验证始终使用真实验证表，绝不进入训练生成或窗口抽样。

## 生成分布 v1

每个变量的潜在轨迹包含线性／二次趋势、一个随机周期及两个谐波、AR(1)高斯噪声；
再独立抽取水平／斜率变化与局部高斯事件。后续变量以配置强度依赖前一变量的滞后值，
使用固定 `/3` 尺度；线性或 `2*tanh` 变换以配置概率选择。计数变量用裁剪对数率的Poisson观测，
其他变量乘随机正尺度并加随机偏置。具体均匀范围见 `src/timestral_ft/synthetic.py`；
全部概率与幅度见 `SyntheticConfig`。它没有使用真实数据拟合，周期按步而非日历单位解释。

源选择流为 `seed ^ 0x4D495854`，合成生成流为 `seed ^ 0x53594E54`，均为独立CPU Torch Generator。
窗口、DHF布局、缺口、角色和模型dropout随机状态继续独立保存。
日志逐步记录阶段、来源概率、实际两种来源样本与监督点；checkpoint还保存累计曝光和下一阶段位置。
合成概率为零的整套计划不创建或消耗混合／生成随机流，保留原有抽样顺序。

## 正式实验与独立审计

```sh
python scripts/run_report_round4.py --phase preflight --device cpu
python scripts/run_report_round4.py --phase run --device cpu
python scripts/audit_report_round4.py
python scripts/accept_report_round4.py --output runs/report-round4-acceptance-final
ruff check src tests scripts
pytest -q
```

三组共享前半C128、后半C256，总200步；真实／混合／先合成后混合概率分别为0→0、.5→.5、1→.5。
三个种子42/43/44，七变量、H192、DHF、batch4×accum4、LoRA r8/alpha16、lr1e-4。
仅ETTh1训练和共同`history_scaled`验证选优，所有九个适配器校验值冻结后才评估H96/H192。
ETTh2只做同一权重冻结迁移；其训练段提供预处理均值/标准差和MASE分母，不参与权重训练或选优。
测试非重叠起点，H96有30个，H192有15个；不是stride1榜单。
结果、来源快照和负结果见[第四轮记录](../../results/report_round4/README.md)。
