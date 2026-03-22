"""
Configuration dataclass and YAML serialization for training.

All hyperparameters, paths, and settings are centralized in the ``Config``
dataclass.  Configurations can be loaded from and saved to YAML files.
"""

import logging
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration for the WLASL recognition system.

    All fields have sensible defaults so that a minimal YAML file only
    needs to override the values that differ from the defaults.
    """

    # --- Paths ---
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # --- Dataset ---
    wlasl_variant: int = 100
    T: int = 64
    image_size: int = 224
    num_workers: int = 4

    # --- Model ---
    approach: str = "stgcn_ce"  # stgcn_ce or stgcn_proto
    backbone: str = "r2plus1d_18"
    num_keypoints: int = 543
    num_classes: int = 100
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.5
    embedding_dim: int = 128  # Final embedding dimension for ST-GCN encoder
    gcn_channels: list[int] = field(default_factory=lambda: [64, 128, 128])  # ST-GCN channel widths
    use_attention_pool: bool = False  # Use attention-weighted temporal pooling
    drop_path_rate: float = 0.0  # Stochastic depth drop rate (0 = disabled)
    use_cross_attention: bool = False  # Cross-branch attention fusion
    aux_loss_weight: float = 0.0  # Auxiliary branch loss weight (0 = disabled)

    # Fusion-specific
    fusion: str = "concat"  # concat or attention
    fusion_dim: int = 256

    # --- Features ---
    use_motion: bool = True  # Concatenate velocity (frame differences) with position
    normalize_embeddings: bool = True  # L2-norm embeddings (True for proto, False for CE)

    # --- Training ---
    epochs: int = 100
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 10
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    fp16: bool = True
    weighted_sampling: bool = False
    early_stopping_patience: int = 15
    mixup_alpha: float = 0.2  # Mixup interpolation parameter (0 = disabled)
    head_dropout: float = 0.3  # Dropout for classification head (stgcn_ce approach)
    class_weighted_loss: bool = True  # Inverse-frequency class weights in CE loss

    # --- Scheduler ---
    scheduler: str = "onecycle"  # onecycle or cosine

    # --- Logging ---
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "wlasl-recognition"
    wandb_run_name: Optional[str] = None
    log_interval: int = 10  # steps between logging

    # --- Evaluation ---
    use_tta: bool = False  # Test-time augmentation (horizontal flip averaging)

    # --- Inference ---
    confidence_threshold: float = 0.6
    smoothing_window: int = 1
    buffer_size: int = 64
    fps_display: bool = True

    # --- Inference (sign detection) ---
    # Thresholds are in normalized-coordinates per second (FPS-independent).
    # MotionDetector measures actual frame timing to convert displacement/frame
    # to velocity/second, so these work identically on 30fps and 60fps cameras.
    min_buffer_frames: int = 30  # Minimum real frames before prediction
    prediction_cooldown: float = 1.0  # Seconds to wait after a confident prediction
    motion_start_threshold: float = 0.30  # Hand velocity (norm-coords/sec) to start — requires deliberate movement
    motion_end_threshold: float = 0.10  # Hand velocity (norm-coords/sec) to detect sign end
    motion_settle_time: float = 0.27  # Seconds of low velocity to confirm sign end
    max_sign_duration: float = 3.0  # Seconds before forcing sign completion
    inference_poll_interval: float = 0.1  # Seconds between inference loop checks
    pre_sign_duration: float = 0.5  # Seconds of pre-sign context to retain on IDLE->SIGNING

    # --- Resume ---
    resume_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        """Derive num_classes and model architecture from wlasl_variant.

        Smaller subsets scale down model size AND dropout to keep the
        regularisation proportional to model capacity.
        """
        variant_to_classes = {100: 100, 300: 300, 1000: 1000, 2000: 2000}
        if self.wlasl_variant in variant_to_classes:
            self.num_classes = variant_to_classes[self.wlasl_variant]

        variant_to_arch = {
            100:  {"d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.1},
            300:  {"d_model": 192, "nhead": 6, "num_layers": 4, "dropout": 0.3},
            1000: {"d_model": 256, "nhead": 8, "num_layers": 5, "dropout": 0.4},
            2000: {"d_model": 384, "nhead": 8, "num_layers": 6, "dropout": 0.5},
        }
        if self.wlasl_variant in variant_to_arch:
            arch = variant_to_arch[self.wlasl_variant]
            self.d_model = arch["d_model"]
            self.nhead = arch["nhead"]
            self.num_layers = arch["num_layers"]
            self.dropout = arch["dropout"]


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file.

    Keys in the YAML file that match Config field names will override
    the defaults.  Unknown keys are logged as warnings and ignored.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    Config
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    valid_fields = {f.name for f in fields(Config)}
    kwargs = {}
    for key, value in raw.items():
        if key in valid_fields:
            kwargs[key] = value
        else:
            logger.warning("Unknown config key '%s' in %s (ignored)", key, path)

    cfg = Config(**kwargs)
    logger.info("Loaded config from %s", path)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Save a Config to a YAML file.

    Parameters
    ----------
    cfg : Config
        The configuration to save.
    path : str or Path
        Destination YAML file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(cfg)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    logger.info("Saved config to %s", path)
