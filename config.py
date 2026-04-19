"""
Central configuration for the ModernBERT fine-tuning pipeline.
All hyperparameters, paths, and constants are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal


# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
VIZ_DIR = OUTPUTS_DIR / "visualizations"
REPORTS_DIR = OUTPUTS_DIR / "reports"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

for _d in [DATA_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, VIZ_DIR, REPORTS_DIR, EXPERIMENTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# Model defaults
DEFAULT_MODEL_NAME: str = "answerdotai/ModernBERT-base"
NUM_LABELS: int = 9
MAX_SEQ_LENGTH: int = 2048

# Data defaults
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1
RANDOM_SEED: int = 42

# Training defaults
DEFAULT_LEARNING_RATE: float = 2e-5
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_NUM_EPOCHS: int = 5
DEFAULT_WARMUP_RATIO: float = 0.1
GRADIENT_ACCUMULATION_STEPS: int = 1
EARLY_STOPPING_PATIENCE: int = 3
LABEL_SMOOTHING: float = 0.0

# Class weight strategies
WeightStrategy = Literal["sqrt_inverse", "inverse", "effective_samples", "none"]
DEFAULT_WEIGHT_STRATEGY: WeightStrategy = "sqrt_inverse"

# Hyperparameter search grid
HYPERPARAM_GRID = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "per_device_train_batch_size": [8, 16],
    "weight_decay": [0.0, 0.01],
    "num_train_epochs": [3, 5],
    "warmup_ratio": [0.06, 0.1],
}

# Logging
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class TrainingConfig:
    """
    Unified training configuration. Can be overridden per experiment.
    """
    model_name: str = DEFAULT_MODEL_NAME
    num_labels: int = NUM_LABELS
    max_seq_length: int = MAX_SEQ_LENGTH

    # Data
    csv_path: str = str(DATA_DIR / "subset.csv")
    text_column: str = "Haber Gövdesi"
    label_column: str = "Sınıf"
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO

    # Training
    learning_rate: float = DEFAULT_LEARNING_RATE
    per_device_train_batch_size: int = DEFAULT_BATCH_SIZE
    per_device_eval_batch_size: int = DEFAULT_BATCH_SIZE * 2
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    num_train_epochs: int = DEFAULT_NUM_EPOCHS
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    label_smoothing_factor: float = LABEL_SMOOTHING

    # Class weights
    weight_strategy: WeightStrategy = DEFAULT_WEIGHT_STRATEGY

    # Precision
    fp16: bool = False
    bf16: bool = False

    # Misc
    seed: int = RANDOM_SEED
    output_dir: str = str(CHECKPOINTS_DIR)
    logging_dir: str = str(LOGS_DIR)
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1_macro"
    greater_is_better: bool = True

    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

    # W&B / TensorBoard
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Text preprocessing
    lowercase: bool = False
    remove_punctuation: bool = True
