"""Reusable opt-in inference execution, separate from training and weight artifacts.

Only the normalized Transformer/head core is autocast or compiled. Scaling,
category encoding, refinement and postprocessing retain the portable FP32/FP64
path. A shallow model view shares parameters without registering a compiled module
in the original model. Runtime instances are not thread safe; use one per worker.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .compat import PortableForecaster


@dataclass(frozen=True)
class InferenceConfig:
    precision: str = "fp32"
    compile: bool = False
    warmup_steps: int = 0
    fallback: str = "error"

    def __post_init__(self):
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("Inference precision must be fp32 or bf16")
        if type(self.compile) is not bool:
            raise ValueError("compile must be a boolean")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.fallback not in {"error", "fp32_eager"}:
            raise ValueError("fallback must be error or fp32_eager")

    def to_dict(self):
        return asdict(self)


def synchronize(device):
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class InferenceRuntime:
    """Reusable forecast executor. Warmup runs once per distinct input signature.

    Compilation requires the entire normalized core to compile (fullgraph=True).
    Metadata counts successfully returned compiled graphs and their executions,
    not just creation of a lazy torch.compile wrapper. No global Dynamo flags are
    changed. A requested unsupported mode raises unless fallback is explicit.
    """

    def __init__(self, model, config=None, **forecaster_options):
        self.config = config or InferenceConfig()
        self.original_model = model
        self.device = next(model.parameters()).device
        self._check_weights()
        self._view = copy.copy(model)
        self._view._modules = model._modules.copy()
        self._view._parameters = model._parameters.copy()
        self._view._buffers = model._buffers.copy()
        self._eager = model._run_two_stream
        self._core = self._eager
        self._warmed = set()
        self._metadata = {
            "requested": self.config.to_dict(), "effective": None,
            "device": str(self.device), "status": "pending", "failure_reason": None,
            "compiled_graphs": 0, "compiled_executions": 0,
            "compile_fullgraph": True, "compile_scope": "normalized transformer and quantile head",
            "graph_breaks": 0, "warmup_calls": 0, "forecast_calls": 0,
            "core_output_dtypes": [], "autocast_probe_dtype": None,
            "successful_calls_by_mode": {}, "metadata_scope": "cumulative lifetime of this runtime, including warmup",
            "precision_scope": "BF16 autocast eligible core operations; FP32 patch projection, residuals, quantile reconstruction; unchanged statistics", "statistics": "unchanged portable FP64 moments / FP32 frames",
        }
        self._precision = self.config.precision
        if self.config.compile:
            # Custom backend observes execution of actual Inductor output. Keep
            # this callable out of Module registration and exported state_dict.
            self._core = torch.compile(self._eager, backend=self._backend, fullgraph=True)
        self._view._run_two_stream = self._run_core
        self.forecaster = PortableForecaster(self._view, **forecaster_options)

    def _check_weights(self):
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in self.original_model.parameters()):
            raise ValueError("InferenceRuntime requires FP32 model parameters; rebuild from FP32 weights")

    def _backend(self, graph, example_inputs):
        compiled = torch._inductor.compile(graph, example_inputs)
        self._metadata["compiled_graphs"] += 1

        def execute(*args):
            result = compiled(*args)
            self._metadata["compiled_executions"] += 1
            return result

        return execute

    def _run_core(self, *args, **kwargs):
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self._precision == "bf16"):
            if self._precision == "bf16" and self._metadata["autocast_probe_dtype"] is None:
                probe = torch.ones(2, 2, device=self.device)
                dtype = torch.nn.functional.linear(probe, probe).dtype
                self._metadata["autocast_probe_dtype"] = str(dtype)
                if dtype != torch.bfloat16:
                    raise RuntimeError(f"BF16 autocast requested but linear probe returned {dtype}")
            result = self._core(*args, **kwargs)
        dtype = str(result.dtype)
        if dtype not in self._metadata["core_output_dtypes"]:
            self._metadata["core_output_dtypes"].append(dtype)
        # Affine/sinh inversion and the advancing statistical frame must not
        # inherit the head's low precision.
        return result.float()

    @property
    def metadata(self):
        result = copy.deepcopy(self._metadata)
        result["compiled_executed"] = result["compiled_executions"] > 0
        result["warmed_signatures"] = len(self._warmed)
        result["mixed_execution"] = len(result["successful_calls_by_mode"]) > 1
        return result

    def _signature(self, kwargs):
        def signature(value):
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            if hasattr(value, "shape"):
                return (tuple(value.shape), str(value.dtype), str(getattr(value, "device", "cpu")))
            return repr(value)
        return tuple((key, signature(value)) for key, value in sorted(kwargs.items()))

    def _forecast(self, kwargs):
        try:
            result = self.forecaster.forecast(**kwargs)
            if self.config.compile and self._core is not self._eager and not self._metadata["compiled_executions"]:
                raise RuntimeError("Compilation requested but no compiled graph executed")
            self._metadata.update(status="fallback" if self._metadata["failure_reason"] else "supported",
                                  effective={"precision": self._precision, "compile": self._core is not self._eager})
            mode = self._precision + ("_compile" if self._core is not self._eager else "_eager")
            counts = self._metadata["successful_calls_by_mode"]
            counts[mode] = counts.get(mode, 0) + 1
            return result
        except Exception as exc:
            self._metadata.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}", effective=None)
            if self.config.fallback == "error" or (self._precision == "fp32" and self._core is self._eager):
                raise
            self._precision, self._core = "fp32", self._eager
            return self._forecast(kwargs)

    @property
    def quantile_levels(self):
        return self.forecaster.quantile_levels

    @property
    def max_context(self):
        return self.forecaster.max_context

    def forecast(self, context=None, prediction_length=None, **kwargs):
        kwargs.update(context=context, prediction_length=prediction_length)
        self._check_weights()
        if self.original_model.training:
            raise ValueError("InferenceRuntime requires model.eval(); it does not alter training state")
        if next(self.original_model.parameters()).device != self.device:
            raise ValueError("Model device changed; create a new InferenceRuntime")
        signature = self._signature(kwargs)
        if signature not in self._warmed:
            for _ in range(self.config.warmup_steps):
                self._forecast(kwargs)
                self._metadata["warmup_calls"] += 1
            self._warmed.add(signature)
        result = self._forecast(kwargs)
        self._metadata["forecast_calls"] += 1
        return result
