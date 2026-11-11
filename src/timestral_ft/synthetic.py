"""Versioned synthetic prior and sample-level source mixing.

CPU torch RNG only: generation never consumes window, layout, augmentation or
model/dropout streams. Parameters describe a project prior, not Timestral's
unpublished pretraining distribution. No real labels enter this generator.
"""
import math

import torch

from .config import SyntheticConfig

SYNTHETIC_SEED_XOR = 0x53594E54
MIXTURE_SEED_XOR = 0x4D495854


def lagged_parent(parent, lag):
    """Fixed scale preserves causal dependence; never normalize with future values."""
    return torch.cat([parent[:1].expand(lag), parent[:-lag]]) / 3.0


def generate_sequence(length: int, variables: int, config: SyntheticConfig, generator: torch.Generator):
    """Draw [variables, time] FP32 series, with causal cross-variable dependence.

    Each independent latent has linear/quadratic trend, two periodic harmonics,
    AR(1) Gaussian innovations, optional level/slope shift and Gaussian event.
    Later variables mix their own latent with a lagged previous variable, linearly
    or through tanh. Count observations are Poisson draws from clipped log rates.
    """
    if length < 2 or variables < 1:
        raise ValueError("Synthetic length >= 2 and variables >= 1 required")
    c = config
    def uniform(low=0., high=1.):
        return low + (high - low) * float(torch.rand((), generator=generator))
    t = torch.arange(length, dtype=torch.float64)
    u = t / max(1, length - 1)
    rows = []
    for _ in range(variables):
        trend = c.trend_scale * (uniform(-2, 2) * u + uniform(-1, 1) * u.square())
        period, phase = uniform(c.min_period, c.max_period), uniform(0, 2 * math.pi)
        seasonal = c.seasonal_scale * uniform(0.2, 2) * (
            torch.sin(2 * math.pi * t / period + phase)
            + uniform(-0.5, 0.5) * torch.cos(4 * math.pi * t / period + phase))
        noise = torch.randn(length, dtype=torch.float64, generator=generator) * c.noise_scale
        rho = uniform(-0.5, 0.8)
        noise_array = noise.numpy()  # CPU view; same dedicated Torch draws, no NumPy RNG.
        for j in range(1, length):
            noise_array[j] += rho * noise_array[j - 1]
        x = trend + seasonal + noise
        if uniform() < c.shift_probability:
            start = int(uniform(0.2, 0.8) * length)
            x[start:] += uniform(-3, 3) + uniform(-2, 2) * (u[start:] - u[start])
        if uniform() < c.event_probability:
            center, width = uniform(0, length - 1), uniform(1, max(2, length / 12))
            x += uniform(-4, 4) * torch.exp(-0.5 * ((t - center) / width).square())
        if rows:
            lag = min(length - 1, int(uniform(1, 9)))
            parent = lagged_parent(rows[-1], lag)
            if uniform() < c.nonlinear_probability:
                parent = torch.tanh(parent) * 2
            x = (1 - c.dependency_strength) * x + c.dependency_strength * parent
        if uniform() < c.count_probability:
            x = torch.poisson(torch.exp(x.clamp(-3, 3) + uniform(0, 1)), generator=generator)
        else:
            x = x * math.exp(uniform(-1, 1)) + uniform(-2, 2)
        rows.append(x)
    return torch.stack(rows).float()


def synthetic_item(context_length, prediction_length, variables, config, generator, strategy):
    sequence = generate_sequence(context_length + prediction_length, variables, config, generator)
    if strategy == "dhf":
        return {"sequence": sequence}
    return {"context": sequence[:, :context_length], "labels": sequence[:, context_length:]}


def stage_at(stages, completed_steps):
    """Return stage index and completed steps within it; terminal state is explicit."""
    offset = 0
    for index, stage in enumerate(stages):
        if completed_steps < offset + stage.steps:
            return index, completed_steps - offset
        offset += stage.steps
    return len(stages), 0
