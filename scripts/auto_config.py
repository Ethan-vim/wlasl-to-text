"""
Auto-detect hardware and generate an optimized YAML config.

Probes your GPU (CUDA), Apple Silicon (MPS), or CPU, determines a
performance tier, and writes a ready-to-train config for the selected
approach and WLASL variant.

Usage:
    python scripts/auto_config.py --approach stgcn_ce
    python scripts/auto_config.py --approach stgcn_ce --variant 100
    python scripts/auto_config.py --approach stgcn_proto
    python scripts/auto_config.py --approach stgcn_proto --variant 300
    python scripts/auto_config.py --approach stgcn_ce --dry-run
    python scripts/auto_config.py --approach stgcn_ce --device cpu
    python scripts/auto_config.py --approach stgcn_proto --backup
"""

import argparse
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

VALID_APPROACHES = ("stgcn_ce", "stgcn_proto")
VALID_VARIANTS = (100, 300, 1000, 2000)

APPROACH_TO_FILE = {
    "stgcn_ce": "stgcn_ce.yaml",
    "stgcn_proto": "stgcn_proto.yaml",
}

APPROACH_TO_NAME = {
    "stgcn_ce": "stgcn_ce",
    "stgcn_proto": "stgcn_proto",
}


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


@dataclass
class HardwareInfo:
    """Detected hardware information."""

    device: str  # "cuda", "mps", "cpu"
    device_name: str  # e.g. "NVIDIA RTX 4090", "Apple M2", "CPU"
    vram_gb: float  # 0.0 for CPU/MPS
    cuda_version: str  # e.g. "12.1", "" for non-CUDA
    cpu_cores: int
    platform_name: str
    torch_version: str
    gpu_count: int  # number of CUDA GPUs


def detect_hardware(device_override: str | None = None) -> HardwareInfo:
    """Auto-detect the available hardware.

    Parameters
    ----------
    device_override : str or None
        Force a specific device ("cuda", "mps", or "cpu").

    Returns
    -------
    HardwareInfo
    """
    cpu_cores = os.cpu_count() or 1
    plat = platform.platform()

    try:
        import torch
    except ImportError:
        return HardwareInfo(
            device="cpu",
            device_name=platform.processor() or "Unknown CPU",
            vram_gb=0.0,
            cuda_version="",
            cpu_cores=cpu_cores,
            platform_name=plat,
            torch_version="(not installed)",
            gpu_count=0,
        )

    torch_ver = torch.__version__

    # Determine device
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Gather device-specific info
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        vram_gb = props.total_memory / (1024**3)
        cuda_version = torch.version.cuda or ""
        gpu_count = torch.cuda.device_count()
    elif device == "mps":
        proc = platform.processor()
        device_name = f"Apple {proc}" if proc else "Apple Silicon"
        vram_gb = 0.0
        cuda_version = ""
        gpu_count = 0
    else:
        device_name = platform.processor() or "CPU"
        vram_gb = 0.0
        cuda_version = ""
        gpu_count = 0

    return HardwareInfo(
        device=device,
        device_name=device_name,
        vram_gb=round(vram_gb, 1),
        cuda_version=cuda_version,
        cpu_cores=cpu_cores,
        platform_name=plat,
        torch_version=torch_ver,
        gpu_count=gpu_count,
    )


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def determine_tier(hw: HardwareInfo) -> str:
    """Classify hardware into a performance tier.

    Returns
    -------
    str
        One of "high", "mid", "low", "cpu".
    """
    if hw.device == "cuda":
        if hw.vram_gb >= 16:
            return "high"
        elif hw.vram_gb >= 8:
            return "mid"
        else:
            return "low"
    return "cpu"  # MPS and CPU both get cpu tier


# ---------------------------------------------------------------------------
# Config value generation
# ---------------------------------------------------------------------------


def build_config_values(
    approach: str,
    variant: int,
    tier: str,
    hw: HardwareInfo,
) -> dict:
    """Build a dict of all config values for the given combination.

    Parameters
    ----------
    approach : str
        "stgcn_ce", "pose", "video", or "fusion".
    variant : int
        100, 300, 1000, or 2000.
    tier : str
        "high", "mid", "low", or "cpu".
    hw : HardwareInfo
        Detected hardware info.

    Returns
    -------
    dict
        All config key-value pairs.
    """
    # --- Base values per approach ---
    if approach == "stgcn_ce":
        cfg = {
            "approach": "stgcn_ce",
            "wlasl_variant": variant,
            "num_keypoints": 75,
            "T": 64,
            "use_motion": True,
            "d_model": 128,
            "gcn_channels": [64, 128, 128],
            "num_layers": 3,
            "dropout": 0.1,
            "embedding_dim": 128,
            "normalize_embeddings": False,
            "use_attention_pool": False,
            "drop_path_rate": 0.0,
            "use_cross_attention": False,
            "aux_loss_weight": 0.0,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2,
            "head_dropout": 0.2,
            "class_weighted_loss": True,
            "num_workers": 4,
            "batch_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 10,
            "grad_clip": 1.0,
            "fp16": False,
            "weighted_sampling": True,
            "early_stopping_patience": 30,
            "scheduler": "onecycle",
            "epochs": 200,
            "use_tta": True,
        }
    else:  # stgcn_proto
        cfg = {
            "approach": "stgcn_proto",
            "wlasl_variant": variant,
            "num_keypoints": 75,
            "T": 64,
            "use_motion": True,
            "d_model": 128,
            "gcn_channels": [64, 128, 128],
            "num_layers": 3,
            "dropout": 0.1,
            "normalize_embeddings": True,
            "use_attention_pool": False,
            "drop_path_rate": 0.0,
            "use_cross_attention": False,
            "aux_loss_weight": 0.0,
            "n_way": 10,
            "k_shot": 3,
            "q_query": 2,
            "num_episodes": 200,
            "num_workers": 4,
            "batch_size": 16,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 10,
            "grad_clip": 1.0,
            "fp16": False,
            "weighted_sampling": False,
            "early_stopping_patience": 30,
            "scheduler": "cosine",
            "epochs": 200,
            "use_tta": True,
        }

    # --- Tier-specific overrides (hardware-dependent) ---
    tier_overrides = _get_tier_overrides(tier, hw)
    cfg.update(tier_overrides)

    # Common settings across all configs
    cfg.update({
        "use_wandb": False,
        "use_tensorboard": True,
        "log_interval": 10,
        "confidence_threshold": 0.6,
        "smoothing_window": 5,
        "fps_display": True,
        "min_buffer_frames": 30,
        "prediction_cooldown": 1.0,
        "motion_start_threshold": 0.30,
        "motion_end_threshold": 0.10,
        "motion_settle_time": 0.27,
        "max_sign_duration": 3.0,
        "inference_poll_interval": 0.1,
        "pre_sign_duration": 0.5,
        "data_dir": "data",
        "output_dir": "outputs",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
    })

    # buffer_size matches T
    cfg["buffer_size"] = cfg["T"]

    return cfg


def _get_tier_overrides(tier: str, hw: HardwareInfo) -> dict:
    """Get hardware-dependent parameter overrides for a tier."""
    overrides: dict = {}

    # Both stgcn_ce and stgcn_proto use the same batch size tiers
    batch_map = {"high": 64, "mid": 32, "low": 16, "cpu": 8}
    overrides["batch_size"] = batch_map[tier]

    # FP16 only on CUDA
    overrides["fp16"] = hw.device == "cuda"

    # num_workers based on device and cores
    if hw.device == "cuda":
        overrides["num_workers"] = min(8, hw.cpu_cores)
    elif hw.device == "mps":
        overrides["num_workers"] = 0
    else:
        overrides["num_workers"] = min(2, hw.cpu_cores)

    return overrides


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


def render_yaml(approach: str, values: dict, hw: HardwareInfo, tier: str) -> str:
    """Render a formatted YAML config string with comments.

    Parameters
    ----------
    approach : str
        "stgcn_ce" or "stgcn_proto".
    values : dict
        Config values from ``build_config_values``.
    hw : HardwareInfo
        Hardware info for the header comment.
    tier : str
        Tier name for the header comment.

    Returns
    -------
    str
        YAML content ready to write to file.
    """
    # Header
    vram_str = f"{hw.vram_gb} GB VRAM, " if hw.vram_gb > 0 else ""
    cuda_str = f"CUDA {hw.cuda_version}" if hw.cuda_version else hw.device.upper()
    header = (
        f"## Auto-generated by scripts/auto_config.py\n"
        f"## Hardware: {hw.device_name} ({vram_str}{cuda_str})\n"
        f"## Tier: {tier} | Approach: {values['approach']} | "
        f"Variant: WLASL{values['wlasl_variant']}\n"
        f"## Re-run to regenerate, or edit manually.\n"
    )

    def _bool(v: bool) -> str:
        return "true" if v else "false"

    def _lr(v: float) -> str:
        return f"{v:.1e}"

    if approach == "stgcn_ce":
        gcn_str = str(values["gcn_channels"])
        body = f"""\
approach: {values['approach']}
wlasl_variant: {values['wlasl_variant']}
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)
num_keypoints: {values['num_keypoints']}
T: {values['T']}
use_motion: {_bool(values['use_motion'])}

# Model (ST-GCN encoder)
d_model: {values['d_model']}
gcn_channels: {gcn_str}
num_layers: {values['num_layers']}
dropout: {values['dropout']}
embedding_dim: {values['embedding_dim']}
normalize_embeddings: {_bool(values['normalize_embeddings'])}
use_attention_pool: {_bool(values['use_attention_pool'])}
drop_path_rate: {values['drop_path_rate']}
use_cross_attention: {_bool(values['use_cross_attention'])}
aux_loss_weight: {values['aux_loss_weight']}

# Cross-entropy training
label_smoothing: {values['label_smoothing']}
mixup_alpha: {values['mixup_alpha']}
head_dropout: {values['head_dropout']}
class_weighted_loss: {_bool(values['class_weighted_loss'])}

# Training
epochs: {values['epochs']}
batch_size: {values['batch_size']}
lr: {_lr(values['lr'])}
weight_decay: {_lr(values['weight_decay'])}
warmup_epochs: {values['warmup_epochs']}
grad_clip: {values['grad_clip']}
fp16: {_bool(values['fp16'])}
weighted_sampling: {_bool(values['weighted_sampling'])}
early_stopping_patience: {values['early_stopping_patience']}
scheduler: {values['scheduler']}
num_workers: {values['num_workers']}

# Evaluation
use_tta: {_bool(values['use_tta'])}

# Inference
confidence_threshold: {values['confidence_threshold']}
smoothing_window: {values['smoothing_window']}
buffer_size: {values['buffer_size']}
fps_display: {_bool(values['fps_display'])}

# Sign detection (thresholds in normalized-coords/second, FPS-independent)
min_buffer_frames: {values['min_buffer_frames']}
prediction_cooldown: {values['prediction_cooldown']}
motion_start_threshold: {values['motion_start_threshold']}
motion_end_threshold: {values['motion_end_threshold']}
motion_settle_time: {values['motion_settle_time']}
max_sign_duration: {values['max_sign_duration']}
inference_poll_interval: {values['inference_poll_interval']}
pre_sign_duration: {values['pre_sign_duration']}

# Paths
data_dir: {values['data_dir']}
checkpoint_dir: {values['checkpoint_dir']}
"""

    else:  # stgcn_proto
        gcn_str = str(values["gcn_channels"])
        body = f"""\
approach: {values['approach']}
wlasl_variant: {values['wlasl_variant']}
# num_classes is auto-derived from wlasl_variant (100 -> 100, 300 -> 300, etc.)
num_keypoints: {values['num_keypoints']}
T: {values['T']}
use_motion: {_bool(values['use_motion'])}

# Model (ST-GCN encoder)
d_model: {values['d_model']}
gcn_channels: {gcn_str}
num_layers: {values['num_layers']}
dropout: {values['dropout']}
normalize_embeddings: {_bool(values['normalize_embeddings'])}
use_attention_pool: {_bool(values['use_attention_pool'])}
drop_path_rate: {values['drop_path_rate']}
use_cross_attention: {_bool(values['use_cross_attention'])}
aux_loss_weight: {values['aux_loss_weight']}

# Prototypical training
n_way: {values['n_way']}
k_shot: {values['k_shot']}
q_query: {values['q_query']}
num_episodes: {values['num_episodes']}

# Training
epochs: {values['epochs']}
batch_size: {values['batch_size']}
lr: {_lr(values['lr'])}
weight_decay: {_lr(values['weight_decay'])}
warmup_epochs: {values['warmup_epochs']}
grad_clip: {values['grad_clip']}
fp16: {_bool(values['fp16'])}
early_stopping_patience: {values['early_stopping_patience']}
scheduler: {values['scheduler']}
num_workers: {values['num_workers']}

# Evaluation
use_tta: {_bool(values['use_tta'])}

# Logging
use_wandb: {_bool(values['use_wandb'])}
use_tensorboard: {_bool(values['use_tensorboard'])}
log_interval: {values['log_interval']}

# Inference
confidence_threshold: {values['confidence_threshold']}
smoothing_window: {values['smoothing_window']}
buffer_size: {values['buffer_size']}
fps_display: {_bool(values['fps_display'])}

# Sign detection (thresholds in normalized-coords/second, FPS-independent)
min_buffer_frames: {values['min_buffer_frames']}
prediction_cooldown: {values['prediction_cooldown']}
motion_start_threshold: {values['motion_start_threshold']}
motion_end_threshold: {values['motion_end_threshold']}
motion_settle_time: {values['motion_settle_time']}
max_sign_duration: {values['max_sign_duration']}
inference_poll_interval: {values['inference_poll_interval']}
pre_sign_duration: {values['pre_sign_duration']}

# Paths
data_dir: {values['data_dir']}
output_dir: {values['output_dir']}
checkpoint_dir: {values['checkpoint_dir']}
log_dir: {values['log_dir']}
"""

    return header + "\n" + body


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(
    hw: HardwareInfo,
    tier: str,
    approach: str,
    variant: int,
    values: dict,
    output_path: Path,
) -> None:
    """Print a human-readable hardware detection and config summary."""
    print()
    print("=" * 60)
    print("  WLASL Auto-Config")
    print("=" * 60)
    print()
    print("  Hardware")
    print("  " + "-" * 40)
    print(f"  Device          : {hw.device_name}")
    if hw.vram_gb > 0:
        print(f"  VRAM            : {hw.vram_gb} GB")
    if hw.cuda_version:
        print(f"  CUDA            : {hw.cuda_version}")
    if hw.gpu_count > 1:
        print(f"  GPU count       : {hw.gpu_count}")
    print(f"  CPU cores       : {hw.cpu_cores}")
    print(f"  PyTorch         : {hw.torch_version}")
    print(f"  Platform        : {hw.platform_name}")
    print()
    print("  Configuration")
    print("  " + "-" * 40)
    print(f"  Tier            : {tier}")
    print(f"  Approach        : {values['approach']}")
    print(f"  WLASL variant   : {variant}")
    print(f"  Output file     : {output_path}")
    print()
    print("  Key Parameters (hardware-optimized)")
    print("  " + "-" * 40)
    print(f"  batch_size      : {values['batch_size']}")
    print(f"  T               : {values['T']}")
    print(f"  fp16            : {values['fp16']}")
    print(f"  num_workers     : {values['num_workers']}")
    if "image_size" in values:
        print(f"  image_size      : {values['image_size']}")
    print(f"  lr              : {values['lr']}")
    print(f"  epochs          : {values['epochs']}")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect hardware and generate an optimized YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/auto_config.py --approach stgcn_ce
  python scripts/auto_config.py --approach stgcn_ce --variant 100
  python scripts/auto_config.py --approach stgcn_proto
  python scripts/auto_config.py --approach stgcn_proto --variant 300
  python scripts/auto_config.py --approach stgcn_ce --dry-run
  python scripts/auto_config.py --approach stgcn_ce --device cpu
  python scripts/auto_config.py --approach stgcn_proto --backup
""",
    )
    parser.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=list(VALID_APPROACHES),
        help="Model approach: stgcn_ce (recommended) or stgcn_proto",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=100,
        choices=list(VALID_VARIANTS),
        help="WLASL variant (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Override detected device (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the generated config without writing to disk",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Back up existing config to {filename}.bak before overwriting",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path (default: configs/{approach}.yaml)",
    )
    args = parser.parse_args()

    # Detect hardware
    hw = detect_hardware(args.device)
    tier = determine_tier(hw)

    # Build config
    values = build_config_values(args.approach, args.variant, tier, hw)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = CONFIGS_DIR / APPROACH_TO_FILE[args.approach]

    # Render YAML
    yaml_content = render_yaml(args.approach, values, hw, tier)

    # Print summary
    print_summary(hw, tier, args.approach, args.variant, values, output_path)

    if args.dry_run:
        print("  [DRY RUN] Generated config:\n")
        print(yaml_content)
        return

    # Backup if requested
    if args.backup and output_path.exists():
        bak_path = output_path.with_suffix(".yaml.bak")
        shutil.copy2(output_path, bak_path)
        print(f"  Backed up {output_path} -> {bak_path}")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content, encoding="utf-8")
    print(f"  Wrote {output_path}")

    # Check if data exists
    splits_dir = PROJECT_ROOT / "data" / "splits" / f"WLASL{args.variant}"
    if not splits_dir.exists():
        print()
        print(f"  Note: {splits_dir.relative_to(PROJECT_ROOT)} not found.")
        print(f"  Run preprocessing first:")
        print(f"    python -m src.data.preprocess --data-dir data --subset WLASL{args.variant}")

    print()


if __name__ == "__main__":
    main()
