"""Small, explicit LoRA implementation for upstream attention projections."""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.multiplier = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        delta = torch.nn.functional.linear(self.dropout(inputs), self.lora_A)
        return self.base(inputs) + torch.nn.functional.linear(delta, self.lora_B) * self.multiplier

    def merged(self):
        result = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None, device=self.base.weight.device,
                           dtype=self.base.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(self.base.weight + (self.lora_B @ self.lora_A) * self.multiplier)
            if result.bias is not None:
                result.bias.copy_(self.base.bias)
        return result


def apply_lora(model, rank=8, alpha=16.0, dropout=0.0):
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("Model already contains LoRA modules")
    model.requires_grad_(False)
    matched = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith((".wQKV", ".wO")):
            parent_name, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attribute, LoRALinear(module, rank, alpha, dropout))
            matched += 1
    if not matched:
        raise ValueError("No compatible attention projections found in this model")
    return model


def merge_lora(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attribute, module.merged())
    return model


def adapter_state(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
            if k.endswith(("lora_A", "lora_B"))}
