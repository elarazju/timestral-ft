# 第三轮验收记录

## 实现与行为

输入副本、原始标签、原始观测掩码和监督掩码独立。连续缺口可跨 patch，人工隐藏值仍可计分，原始 NaN 不计分；前缀无有效观测的目标／块不计分。角色采样限于原目标数值列，至少保留一个目标，固定声明协变量保持角色。数值路径继承官方角色注意力与归一化，分类路径按样本／起点安全重建输入。

数值 DHF 仍共享前向；带分类协变量时逐块运行，未来已知输入截到当前块末尾，不能声称与全窗口数值路径具有相同条件信息或相同计算成本。官方目标编码只适用于当前单目标样本，多目标沿用数值类别代码。类别代码和编码输入不提供梯度，但模型／LoRA 参数梯度支持训练。

## 检查

- `.venv/bin/ruff check src tests scripts examples/report_round3`：通过。
- `.venv/bin/python -m pytest -q`：**208 passed, 50 skipped**，执行 35.96 秒；父 agent 独立复跑为同样结果（29.10 秒）。50 个跳过包含未启用的真实权重集成测试与无 CUDA 硬件条件。
- `TIMESTRAL_INTEGRATION=1 .venv/bin/python -m pytest tests/test_augmentation_acceptance.py -q`：真实 5M 的数值／分类路径、全参数／LoRA、CPU／MPS **8 passed**，CUDA **4 skipped**。
- `.venv/bin/python examples/report_round3/smoke.py --output runs/report-round3-example-smoke`：完成四步示例训练，成功导出模型与 `forecast.csv`。
- `.venv/bin/python scripts/run_report_round3.py --phase all`：完整 24 模型矩阵、测试和独立审计通过；200 步，无降档、无省略数据集／种子。
- `.venv/bin/python scripts/audit_report_round3.py`：独立入口可重验同一冻结证据；源文件哈希、随机状态及数值审计结果见 `integrity_audit.json`。
- `git diff --check`：通过；博客目录没有跟踪文件，且被本地 exclude 及构建排除规则隔离。


测试覆盖：连续缺口与原标签不变、自然缺失和全缺前缀、人工隐藏值计分、至少一个目标、固定协变量角色不变、未来目标和历史协变量隔离、已知未来影响预测、类别 past/known 的起点范围、多样本隔离、有限参数梯度、四条随机流独立恢复、缺失随机状态拒绝恢复、旧配置默认值、全参数／LoRA、窗口两种损失、DHF、合并权重和适配器导出。

父 agent 另行以混合角色、不同目标数、人工缺口和原始 NaN 运行了独立前向与梯度／扰动验收，数值和分类两条路径通过。原始回执保存在 `runs/report-round3-root-acceptance/independent-mechanisms.json`。

父 agent 还跨轮核对了 ETTh1／ETTh2 × 3 种子的全部六个无增强组：第三轮选中模型的 `model.safetensors` SHA256 与第二轮对应 DHF 模型逐个精确相同，选优步也一致。该证据验证默认关闭增强的真实训练兼容性，不是新增效果结论；回执见 `runs/report-round3-root-acceptance/disabled-augmentation-reproducibility.json`。

正式实验先冻结运行时源码和协议，再训练并共同验证选优所有模型，随后冻结选中权重并开始测试。结果审计验证 24 个模型、104 个结果条目、208 个划分分数、1456 个独立指标标量以及完整汇总表；最终实际状态以 `integrity_audit.json` 为准。

CUDA 无可用硬件，明确跳过；不将 CPU/MPS 证据外推到 CUDA，也不将短序列或 5M 验收外推到所有模型尺寸与最长上下文。此次测试任务没有已知未来输入，不把角色掩码机制通过等同于条件预测精度提升。
