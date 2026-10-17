# Broader Timestral validation

Completed: 66 in-dataset benchmark evaluations, 28 frozen-transfer evaluations, and 8 real-weight model/device acceptance cases.
See [the detailed Chinese record](README.zh-CN.md), [benchmark summary](benchmark/README.md), [frozen-transfer scores](transfer/summary.csv), and [independent audit](audit.json).

Three datasets (ETTh1/ETTh2/exchange), L96/H96,192, three seeds, M all columns, train-only scaling, test stride1.
Full/LoRA adaptation is compared with base, persistence and the unchanged official DLinear network using an explicit local training loop.
Separate frozen transfer applies ETTh1-adapted network weights to ETTh2/exchange without target-dataset optimization.
All four model sizes pass CPU/MPS short training/resume/export and representative/boundary backward checks; CUDA is unverified.
These checks do not establish universal quality gains, equal-compute rankings, large-model forecast quality, or maximum capacity.
