"""Serializable, shared configuration for the Python and command-line interfaces."""

from dataclasses import asdict, dataclass, field

UPSTREAM_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"


@dataclass
class DataConfig:
    time_column: str = "timestamp"
    id_column: str | None = None
    target_columns: list[str] = field(default_factory=lambda: ["target"])
    past_covariates: list[str] = field(default_factory=list)
    known_covariates: list[str] = field(default_factory=list)
    categorical_past_covariates: list[str] = field(default_factory=list)
    categorical_known_covariates: list[str] = field(default_factory=list)
    frequency: str | None = None

    def __post_init__(self):
        if not self.target_columns:
            raise ValueError("At least one target column is required")
        roles = self.target_columns + self.covariate_columns
        if len(roles) != len(set(roles)):
            raise ValueError("Each feature must have exactly one role")
        if self.time_column in roles or self.id_column in roles:
            raise ValueError("Time/ID columns cannot also be features")
        if self.id_column == self.time_column:
            raise ValueError("Time and ID columns must differ")

    @property
    def covariate_columns(self):
        return (self.past_covariates + self.known_covariates
                + self.categorical_past_covariates + self.categorical_known_covariates)

    @property
    def categorical_columns(self):
        return self.categorical_past_covariates + self.categorical_known_covariates

    def to_dict(self):
        return asdict(self)


@dataclass
class SyntheticConfig:
    """Project-defined prior; all ranges use uniform draws, not an official recipe."""
    version: int = 1
    trend_scale: float = 1.0
    seasonal_scale: float = 1.0
    noise_scale: float = 0.2
    shift_probability: float = 0.5
    event_probability: float = 0.5
    count_probability: float = 0.2
    nonlinear_probability: float = 0.5
    dependency_strength: float = 0.5
    min_period: float = 8.0
    max_period: float = 128.0

    def __post_init__(self):
        import math
        if self.version != 1:
            raise ValueError("Unsupported synthetic generator version")
        for name in ("trend_scale", "seasonal_scale", "noise_scale", "min_period", "max_period"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 2 <= self.min_period <= self.max_period:
            raise ValueError("Synthetic periods require 2 <= min_period <= max_period")
        for name in ("shift_probability", "event_probability", "count_probability",
                     "nonlinear_probability", "dependency_strength"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class TrainingStage:
    steps: int
    context_length: int
    synthetic_probability: float = 0.0

    def __post_init__(self):
        for name in ("steps", "context_length"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Stage {name} must be a positive integer")
        if not 0 <= self.synthetic_probability <= 1:
            raise ValueError("synthetic_probability must be in [0, 1]")


@dataclass
class TrainConfig:
    mode: str = "full"
    context_length: int = 256
    prediction_length: int = 96
    max_steps: int = 200
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_steps: int = 20
    gradient_clip: float = 1.0
    eval_every: int = 50
    patience: int = 3
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    loss_space: str = "history_scaled"
    validation_loss_space: str | None = None
    training_strategy: str = "window"
    dhf_max_block_patches: int | None = None
    dhf_min_context_patches: int = 1

    missing_probability: float = 0.0
    missing_max_span: int = 64
    dynamic_role_sampling: bool = False
    role_probabilities: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.25])

    synthetic: SyntheticConfig | None = None
    stages: list[TrainingStage] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.synthetic, dict):
            self.synthetic = SyntheticConfig(**self.synthetic)
        if self.synthetic is not None and not isinstance(self.synthetic, SyntheticConfig):
            raise ValueError("synthetic must be a SyntheticConfig or mapping")
        self.stages = [TrainingStage(**s) if isinstance(s, dict) else s for s in self.stages]
        if any(not isinstance(s, TrainingStage) for s in self.stages):
            raise ValueError("stages must contain TrainingStage values or mappings")
        if self.stages and sum(s.steps for s in self.stages) != self.max_steps:
            raise ValueError("Stage steps must sum to max_steps")
        if any(s.synthetic_probability for s in self.stages) and self.synthetic is None:
            raise ValueError("Synthetic stages require an explicit synthetic configuration")
        if not 0 <= self.missing_probability <= 1:
            raise ValueError("missing_probability must be in [0, 1]")
        if isinstance(self.missing_max_span, bool) or not isinstance(self.missing_max_span, int) or self.missing_max_span < 1:
            raise ValueError("missing_max_span must be a positive integer")
        if not isinstance(self.dynamic_role_sampling, bool):
            raise ValueError("dynamic_role_sampling must be boolean")
        self.role_probabilities = list(self.role_probabilities)
        if (len(self.role_probabilities) != 3 or any(not 0 <= p <= 1 for p in self.role_probabilities)
                or abs(sum(self.role_probabilities) - 1) > 1e-8 or self.role_probabilities[0] <= 0):
            raise ValueError("role_probabilities must be three probabilities summing to one with positive target probability")
        if self.mode not in {"full", "lora"}:
            raise ValueError("mode must be full or lora")
        if self.loss_space not in {"history_scaled", "normalized"}:
            raise ValueError("loss_space must be history_scaled or normalized")
        if self.validation_loss_space not in {None, "history_scaled", "normalized"}:
            raise ValueError("validation_loss_space must be None, history_scaled or normalized")
        if self.training_strategy not in {"window", "dhf"}:
            raise ValueError("training_strategy must be window or dhf")
        if self.training_strategy == "dhf" and self.loss_space != "normalized":
            raise ValueError("DHF training requires loss_space='normalized'")
        if self.dhf_min_context_patches is None:
            raise ValueError("dhf_min_context_patches must be a positive integer")
        for name in ("dhf_min_context_patches", "dhf_max_block_patches"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("context_length", "prediction_length", "max_steps", "batch_size",
                     "gradient_accumulation", "eval_every", "patience", "num_threads", "lora_rank"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.prediction_length > 1024:
            raise ValueError("v0.1 training supports at most 1024 forecast steps")
        if self.warmup_steps < 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer/scheduler settings")
        if self.learning_rate is None:
            self.learning_rate = 1e-4 if self.mode == "lora" else 1e-5
        if self.learning_rate <= 0 or not 0 <= self.lora_dropout < 1 or self.lora_alpha <= 0:
            raise ValueError("Invalid learning rate or LoRA settings")

    def to_dict(self):
        return asdict(self)
