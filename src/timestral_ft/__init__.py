"""Community fine-tuning for Timestral. Independent of Mistral AI."""

from .artifacts import FineTunedModel, export
from .config import DataConfig, TrainConfig
from .evaluation import EvaluationResult, evaluate
from .trainer import finetune

__version__ = "0.1.0"
__all__ = ["DataConfig", "TrainConfig", "FineTunedModel", "EvaluationResult", "finetune", "evaluate", "export"]
