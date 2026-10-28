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

    def __post_init__(self):
        if self.mode not in {"full", "lora"}:
            raise ValueError("mode must be full or lora")
        if self.loss_space not in {"history_scaled", "normalized"}:
            raise ValueError("loss_space must be history_scaled or normalized")
        if self.validation_loss_space not in {None, "history_scaled", "normalized"}:
            raise ValueError("validation_loss_space must be None, history_scaled or normalized")
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
