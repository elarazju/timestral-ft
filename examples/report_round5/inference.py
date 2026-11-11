"""Run from the repository root with the installed environment."""

import torch

from timestral_ft import InferenceConfig, InferenceRuntime
from timestral_ft.compat import load_model

model = load_model("mistralai/Timestral-5M", revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f")
runtime = InferenceRuntime(model, InferenceConfig(precision="bf16", warmup_steps=2))
context = torch.sin(torch.arange(256) / 13)[None, None, :]
forecast = runtime.forecast(context, 96)
print(forecast.quantiles.shape)
print(runtime.metadata)
