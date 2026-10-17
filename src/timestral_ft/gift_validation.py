"""Validation-only GIFT selection with the official forecast and metric semantics.

GluonTS is imported lazily, and remains an optional experiment dependency.
No test iterator, published test metric, or leaderboard result enters selection.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .compat import PortableForecaster
from .gift import _padded, atomic_json, seasonal_error_sliced, validation_loss

LEVELS = tuple(i / 10 for i in range(1, 10))


def selection_score(crps, mase, baseline_crps, baseline_mase):
    """Pareto safeguard on validation, then equal-weight geometric ratios.

    Step zero has score one. Accept a checkpoint only when neither validation
    metric regresses. This is a validation safeguard, never a test guarantee.
    """
    values = crps, mase, baseline_crps, baseline_mase
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise FloatingPointError("Selection needs positive finite validation metrics")
    ratios = crps / baseline_crps, mase / baseline_mase
    return math.sqrt(math.prod(ratios)) if max(ratios) <= 1 else max(ratios)


@dataclass(frozen=True)
class GiftValidationConfig:
    context_length: int = 32768
    max_rows: int = 16
    max_tokens: int = 2**17
    seasonality: int = 1
    record_legacy_objective: bool = True

    def __post_init__(self):
        if min(self.context_length, self.max_rows, self.max_tokens, self.seasonality) < 1:
            raise ValueError("Validation budgets and seasonality must be positive")


class GiftValidation:
    """Reuse fixed training/validation prefixes, with full inference history.

    MASE denominators use the complete training prefix, as in official
    GluonTS evaluation. The network sees at most context_length timestamps.
    """

    def __init__(self, reference_path, config=None, base_identity=None):
        self.path = Path(reference_path)
        self.config = config or GiftValidationConfig()
        self.base_identity = base_identity
        self.last = None

    @property
    def metadata(self):
        return {
            "name": "official-nine-quantile-crps-mase-pareto-v1",
            "score": "sqrt(CRPS/base_CRPS * MASE/base_MASE) if both ratios <= 1; otherwise max ratio",
            "baseline_candidate": "step zero, selected if no checkpoint has score < 1",
            "config": asdict(self.config),
            "base_identity": self.base_identity,
            "data": "fixed seeded official validation suffix; no test access",
        }

    @torch.no_grad()
    def measure(self, model, windows, cfg):
        from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

        model.eval()
        fc = PortableForecaster(model, max_context=self.config.context_length)
        indices = [next(i for i, q in enumerate(fc.quantile_levels) if abs(q - x) < 1e-9)
                   for x in LEVELS]
        evaluators = [MASE()(axis=None), MeanWeightedSumQuantileLoss(LEVELS)(axis=None)]
        order = sorted(range(len(windows.validation)),
                       key=lambda i: windows.train[windows.validation_indices[i]][0].shape[-1])
        start = 0
        while start < len(order):
            first = windows.train[windows.validation_indices[order[start]]]
            rows = first[0].shape[0] + (0 if first[1] is None else first[1].shape[0])
            end = start + 1
            while end < len(order):
                length = min(self.config.context_length,
                             windows.train[windows.validation_indices[order[end]]][0].shape[-1])
                count = end + 1 - start
                if count * rows > self.config.max_rows or count * rows * (length + windows.horizon) > self.config.max_tokens:
                    break
                end += 1
            subset = order[start:end]
            entries = [windows.train[windows.validation_indices[i]] for i in subset]
            length = min(self.config.context_length, max(x.shape[-1] for x, _ in entries))
            context = np.stack([_padded(x, length) for x, _ in entries])
            past = None if first[1] is None else np.stack([_padded(c, length) for _, c in entries])
            predictions = fc.forecast(context, windows.horizon, past_covariates=past).quantiles
            predictions = predictions[..., indices].float().cpu().numpy()
            if not np.isfinite(predictions).all():
                raise FloatingPointError("Non-finite validation forecast")
            labels = np.stack([windows.validation[i]["labels"] for i in subset])
            # np.array matches GluonTS _get_data_batch's handling of masked
            # seasonal errors; the metric objects supply its exact reductions.
            scales = np.array([seasonal_error_sliced(np.ma.masked_invalid(x),
                              self.config.seasonality, time_axis=-1) for x, _ in entries])
            data = {"label": np.ma.masked_invalid(labels), "seasonal_error": scales,
                    **{str(q): predictions[..., i] for i, q in enumerate(LEVELS)}}
            for evaluator in evaluators:
                evaluator.update(data)
            start = end
        values = {e.name: float(e.get()) for e in evaluators}
        result = {"crps": values["mean_weighted_sum_quantile_loss"], "mase": values["MASE[0.5]"]}
        if self.config.record_legacy_objective:
            result["scaled_pinball"] = validation_loss(model, windows, cfg)
        if not all(math.isfinite(v) for v in result.values()):
            raise FloatingPointError("Non-finite official validation metric")
        return result

    def __call__(self, model, windows, cfg):
        result = None
        identity = {"data_fingerprint": windows.fingerprint, "validation": self.metadata,
                    "base_model_config": asdict(model.config)}
        identity = json.loads(json.dumps(identity))
        if self.path.exists():
            reference = json.loads(self.path.read_text())
            if reference["identity"] != identity:
                raise ValueError("Validation reference belongs to a different protocol")
        else:
            # The trainer calls this at step zero, before any optimizer update.
            # Fail if someone attempts to establish a baseline using an adapter
            # that has already changed the model.
            from .lora import adapter_state
            if any(bool(v.count_nonzero()) for k, v in adapter_state(model).items() if k.endswith("lora_B")):
                raise ValueError("Baseline reference requires zero initialized LoRA deltas")
            metrics = self.measure(model, windows, cfg)
            result = dict(metrics)
            reference = {"identity": identity, "metrics": metrics}
            atomic_json(self.path, reference)
        if result is None:
            result = self.measure(model, windows, cfg)
        baseline = reference["metrics"]
        result.update(crps_ratio=result["crps"] / baseline["crps"],
                      mase_ratio=result["mase"] / baseline["mase"])
        result["score"] = selection_score(result["crps"], result["mase"], baseline["crps"], baseline["mase"])
        self.last = result
        return result
