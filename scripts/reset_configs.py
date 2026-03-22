"""
Reset all YAML configs in configs/ to the recommended defaults from README.md.

This script overwrites:
    - configs/stgcn_ce.yaml     (ST-GCN + Cross-Entropy — recommended default)
    - configs/stgcn_proto.yaml  (ST-GCN + Prototypical Network — few-shot)

Usage:
    python scripts/reset_configs.py
    python scripts/reset_configs.py --dry-run          # preview without writing
    python scripts/reset_configs.py --only stgcn_ce    # reset only stgcn_ce.yaml
    python scripts/reset_configs.py --only stgcn_proto # reset only stgcn_proto.yaml
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ---------------------------------------------------------------------------
# Default configs — these match the README.md "Recommended Configurations"
# section combined with Config dataclass defaults from src/training/config.py.
# ---------------------------------------------------------------------------

STGCN_CE_YAML = """\
## ST-GCN + Cross-Entropy (recommended default — best model)
## Optimized configuration (WLASL100).

approach: stgcn_ce
wlasl_variant: 100
num_keypoints: 75
T: 64
use_motion: true
use_augmentation: true

# Model (ST-GCN encoder)
d_model: 128
gcn_channels: [64, 128, 128]
num_layers: 3
dropout: 0.1
embedding_dim: 128
use_attention_pool: false
drop_path_rate: 0.0
use_cross_attention: false
aux_loss_weight: 0.0
normalize_embeddings: false

# Cross-entropy training
label_smoothing: 0.1
mixup_alpha: 0.2
head_dropout: 0.2
class_weighted_loss: true

# Training
epochs: 200
batch_size: 32
lr: 1.0e-3
weight_decay: 1.0e-4
warmup_epochs: 10
grad_clip: 1.0
fp16: false
weighted_sampling: true
early_stopping_patience: 30
scheduler: onecycle
num_workers: 4

# Evaluation
use_tta: true

# Logging
use_wandb: false
use_tensorboard: true
log_interval: 10

# Inference
confidence_threshold: 0.6
smoothing_window: 5
buffer_size: 64
fps_display: true

# Sign detection
min_buffer_frames: 30
prediction_cooldown: 1.0
motion_start_threshold: 0.005
motion_end_threshold: 0.003
motion_settle_frames: 8
max_sign_duration: 90
static_sign_timeout: 45
inference_poll_interval: 0.1

# Paths
data_dir: data
output_dir: outputs
checkpoint_dir: checkpoints
log_dir: logs
"""

STGCN_PROTO_YAML = """\
## ST-GCN + Prototypical Network (few-shot)
## Episodic metric learning for low-sample scenarios.
## Optimized configuration (WLASL100).

approach: stgcn_proto
wlasl_variant: 100
num_keypoints: 75
T: 64
use_motion: true

# Model (ST-GCN encoder)
d_model: 128
gcn_channels: [64, 128, 128]
num_layers: 3
dropout: 0.1
use_attention_pool: false
drop_path_rate: 0.0
normalize_embeddings: true

# Prototypical training
n_way: 10
k_shot: 3
q_query: 2
num_episodes: 200

# Training
epochs: 200
batch_size: 16
lr: 1.0e-3
weight_decay: 1.0e-4
warmup_epochs: 10
grad_clip: 1.0
fp16: false
early_stopping_patience: 30
scheduler: cosine
num_workers: 4

# Evaluation
use_tta: true

# Logging
use_wandb: false
use_tensorboard: true
log_interval: 10

# Inference
confidence_threshold: 0.6
smoothing_window: 5
buffer_size: 64
fps_display: true

# Sign detection
min_buffer_frames: 30
prediction_cooldown: 1.0
motion_start_threshold: 0.005
motion_end_threshold: 0.003
motion_settle_frames: 8
max_sign_duration: 90
static_sign_timeout: 45
inference_poll_interval: 0.1

# Paths
data_dir: data
output_dir: outputs
checkpoint_dir: checkpoints
log_dir: logs
"""

CONFIGS = {
    "stgcn_ce": ("stgcn_ce.yaml", STGCN_CE_YAML),
    "stgcn_proto": ("stgcn_proto.yaml", STGCN_PROTO_YAML),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset configs/ to README.md recommended defaults"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying files",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=list(CONFIGS),
        help="Reset only one config file (stgcn_ce or stgcn_proto)",
    )
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.only: CONFIGS[args.only]} if args.only else CONFIGS

    for key, (filename, content) in targets.items():
        path = CONFIGS_DIR / filename

        if args.dry_run:
            print(f"[DRY RUN] Would write {path}")
            print(content)
            print("=" * 60)
            continue

        path.write_text(content, encoding="utf-8")
        print(f"  Reset {path}")

    if not args.dry_run:
        print(f"\n  {len(targets)} config(s) reset to README.md recommended defaults.")


if __name__ == "__main__":
    main()
