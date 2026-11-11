# 第四轮验收：可恢复混采与阶段计划

固定已发布Timestral-5M revision `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`；CPU/MPS实机，CUDA不可用。
全部12项使用与正式实验一致的最终源码；每项保存source SHA和验收脚本SHA，复用时必须完全匹配。

## 功能与兼容性

- 完整测试：229 passed、50 skipped；`ruff check src tests scripts`通过。跳过项包含未启用集成/大模型及不可用CUDA。
- 21项新增测试覆盖生成复现与组件差异、计数约束、滞后依赖未来隔离、嵌套JSON/YAML、阶段总步数约束和明确拒绝混合协变量。
- full/LoRA × window/DHF × 边界前/边界/边界后暂停，保留dropout、缺口和动态角色；权重、随机流、逐步日志与验证选优精确一致。
- 纯合成长阶段无需同长度真实窗口；后续短混合阶段可以继续。纯真实阶段保留数值/分类协变量。禁用混合时与旧路径权重及随机流一致。
- 真实5M短阶段检查：前2步C32纯合成→后2步C64混合、H32、B1/V1、DHF normalizedloss。共同验证固定C64，断点恰在边界；CPU/MPS full/LoRA均逐字节恢复。
- 四个短阶段case均验证合并权重加载与官方CPU预测，最大绝对差均为0；LoRA还检查适配器单独导出和重载。
- 已运行 `python examples/report_round4/smoke.py`：公开嵌套YAML从真实5M权重完成ETTh1四步示例，不作为效果结论。

## 逐级容量实测

以下长序列是单起点归一化损失前向/反向及一次优化更新，B1/V1/H32；不是长DHF训练，也不是官方长上下文预训练。
各case独立子进程，内存峰值为进程累计RSS高水位；MPS额外记录100ms采样的allocator/driver最大值，可能漏掉短峰值。
正式9模型训练在全部容量作业结束后才开始，未与容量压力测试并行。

|设备|模式|历史长度|通过|耗时(s)|进程峰值RSS(MiB)|MPS driver采样峰值(MiB)|
|---|---|---:|---|---:|---:|---:|
|cpu|full|64|是|2.363|610.1|—|
|cpu|lora|64|是|1.566|506.8|—|
|cpu|full|8192|是|0.553|449.6|—|
|cpu|lora|8192|是|0.519|401.8|—|
|cpu|full|32768|是|0.645|576.1|—|
|cpu|lora|32768|是|0.615|535.6|—|
|mps|full|64|是|3.893|790.2|139.7|
|mps|lora|64|是|2.999|750.1|83.7|
|mps|full|8192|是|0.892|545.4|139.7|
|mps|lora|8192|是|0.773|524.5|106.7|
|mps|full|32768|是|0.857|548.9|1195.7|
|mps|lora|32768|是|0.805|528.7|1179.7|

完整逐case信息见[acceptance.json](acceptance.json)，本地原始日志在`runs/report-round4-acceptance-final`。
短case耗时包含多次训练/恢复/重载；长case仅一次前后向/更新，不能直接将这两类行的秒数作性能对比。
未测试大batch/多变量长上下文、CUDA或长期质量；本轮不改变默认训练配方。

## 复核命令

```sh
ruff check src tests scripts
pytest -q
python scripts/accept_report_round4.py --output runs/report-round4-acceptance-final
python scripts/audit_report_round4.py
```

未来轮次改变源码时，旧结果应用本轮`runs/report-round4/source_snapshot`中的代码或对应提交审计；不能将新源码套入旧协议。
