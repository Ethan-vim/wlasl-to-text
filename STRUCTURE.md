# Project Structure & Workflow

This document maps the entire pipeline from data download to live inference, showing which files call which and how data flows through the system.

---

## End-to-End Workflow

```
 PHASE 1: DATA                PHASE 2: TRAINING              PHASE 3: USAGE
 ─────────────                ──────────────────              ──────────────

 download_wlasl.py ─┐
                    ├─> data/raw/*.mp4
 download_kaggle.py ┘         │
                              v
              validate_videos.py
                              │
                              v
                      preprocess.py
                     ┌────────┴────────┐
                     v                 v
            data/processed/    data/splits/WLASL{N}/
            *.npy (keypoints)  train.csv, val.csv, test.csv
                     │                 │
                     └────────┬────────┘
                              v
                    ┌─── train.py ───┐         evaluate.py
                    │   (loads data, │              │
                    │    builds      │              v
                    │    model,      │        eval_results/
                    │    trains)     │        confusion_matrix.png
                    │               │
                    v               v
              checkpoints/    logs/
              best_model.pt   tensorboard/
                    │
         ┌──────────┼──────────┐
         v          v          v
    predict.py  live_demo.py  export_onnx.py
    (single      (webcam       (ONNX
     video)       real-time)    export)
```

---

## File Dependency Graph

Shows which project files each module imports from (`src.*` imports only).

```
                         ┌──────────────────┐
                         │  config.py       │
                         │  (Config,        │
                         │   load_config,   │
                         │   save_config)   │
                         └───────┬──────────┘
                   ┌─────────────┼─────────────────────────────┐
                   │             │             │               │
                   v             v             v               v
            ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐
            │ train.py │  │evaluate. │  │ predict.  │  │ live_demo. │
            │          │  │  py      │  │   py      │  │    py      │
            └──┬───┬───┘  └──┬───┬──┘  └──┬──┬──┬──┘  └──┬──┬─────┘
               │   │         │   │        │  │  │        │  │
          ┌────┘   │    ┌────┘   │        │  │  │        │  │
          v        v    v        v        │  │  │        │  │
    ┌──────────┐ ┌──────────┐             │  │  │        │  │
    │augment.py│ │dataset.py│             │  │  │        │  │
    └──────────┘ └──────────┘             │  │  │        │  │
                                          │  │  │        │  │
               ┌──────────────────────────┘  │  │        │  │
               v                             v  │        v  │
    ┌────────────────┐            ┌─────────────────┐      │
    │ preprocess.py  │            │  stgcn.py       │      │
    │ (normalize,    │            │ (STGCNEncoder,  │      │
    │  keypoints)    │            │  body/hand      │      │
    └────────────────┘            │  graph branches)│      │
                                  └────────┬────────┘      │
                                  ┌────────┼───────────────┘
                                  v        v
                        ┌──────────────┐ ┌──────────────┐
                        │classifier.py │ │prototypical. │
                        │(STGCNClassi- │ │  py          │
                        │ fier + head) │ │(PrototypicalN│
                        └──────────────┘ │ etwork)      │
                                         └──────────────┘
```

---

## Module Details

### Scripts (Entry Points)

| Script | Purpose | Imports From | Output |
|--------|---------|--------------|--------|
| `scripts/download_wlasl.py` | Download annotations + print video instructions | `src.data.preprocess` | `data/annotations/`, `data/raw/` |
| `scripts/download_kaggle.py` | Download full video archive from Kaggle | `src.data.preprocess` | `data/raw/*.mp4` |
| `scripts/validate_videos.py` | Detect and remove fake HTML video files | (none) | Cleaned `data/raw/` |
| `scripts/reset_configs.py` | Reset YAML configs to README defaults | (none) | `configs/*.yaml` |
| `scripts/check_mediapipe.py` | Verify MediaPipe installation, diagnose `solutions` import issues | (none) | Diagnostic output to stdout |
| `scripts/auto_config.py` | Auto-detect hardware (CUDA/MPS/CPU) and generate optimized configs (CE and proto both include `use_cross_attention`, `aux_loss_weight`) | (none) | `configs/*.yaml` |

### Data Pipeline (`src/data/`)

| Module | Key Functions / Classes | Used By |
|--------|------------------------|---------|
| `preprocess.py` | `download_wlasl_annotations()`, `parse_wlasl_annotations()`, `extract_keypoints_mediapipe()`, `normalize_keypoints()`, `preprocess_dataset()`, `create_splits()` | `download_wlasl.py`, `download_kaggle.py`, `predict.py`, `live_demo.py` |
| `augment.py` | `TemporalCrop`, `TemporalSpeedPerturb`, `KeypointHorizontalFlip`, `KeypointYawRotation`, `KeypointRotation`, `KeypointTranslation`, `KeypointDropout`, `KeypointNoise`, `KeypointScale`, `Compose`, `get_train_transforms()`, `get_ce_train_transforms()`, `get_val_transforms()` | `train_ce.py`, `train_prototypical.py`, `evaluate.py`, `predict.py` |
| `dataset.py` | `WLASLKeypointDataset`, `get_dataloader()` | `train_ce.py`, `train_prototypical.py`, `evaluate.py` |

### Models (`src/models/`)

| Module | Key Classes | Build Function | Approaches |
|--------|-------------|----------------|------------|
| `__init__.py` | — | `build_model(cfg)` | Unified factory dispatching on `cfg.approach` |
| `stgcn.py` | `STGCNEncoder`, `DropPath`, `AttentionPool`, `CrossBranchAttention` | `build_stgcn_encoder(cfg)`, `forward_with_branches()` | ST-GCN encoder with body/hand graph branches, dilated TCN, joint importance weighting, optional attention pooling, stochastic depth, and cross-branch attention |
| `classifier.py` | `STGCNClassifier` | `build_classifier(cfg)`, `forward_with_aux()` | ST-GCN encoder + two-layer head + optional auxiliary branch heads (stgcn_ce) |
| `prototypical.py` | `PrototypicalNetwork` | — | Prototypical network wrapper for few-shot (stgcn_proto) |

The unified `build_model(cfg)` factory in `__init__.py` dispatches on `cfg.approach` to the correct model builder. All `build_*_model()` functions take a `Config` object and return an `nn.Module`.

### Training (`src/training/`)

| Module | Key Functions | Imports From |
|--------|---------------|--------------|
| `config.py` | `Config` (dataclass with `embedding_dim`, `gcn_channels`, etc.), `load_config()`, `save_config()` | (none — leaf dependency) |
| `train.py` | `main()` — CLI dispatcher | `config`, `train_ce`, `train_prototypical` |
| `train_ce.py` | `train_one_epoch()`, `validate()`, `main()` — cross-entropy training with label smoothing, mixup, OneCycleLR/cosine scheduler, and optional auxiliary branch losses | `config`, `augment`, `dataset`, `models` |
| `train_prototypical.py` | `train_prototypical()` — episodic prototypical training loop | `config`, `augment`, `dataset`, `episode_sampler`, `models` |
| `evaluate.py` | `compute_metrics()`, `plot_confusion_matrix()`, `find_hard_negatives()`, `evaluate_latency()`, `main()` | `config`, `augment`, `dataset`, `models` |

### Inference (`src/inference/`)

| Module | Key Classes / Functions | Imports From |
|--------|------------------------|--------------|
| `predict.py` | `SignPredictor`, `_load_class_names()` | `config`, `augment`, `preprocess`, `models` |
| `live_demo.py` | `FrameBuffer`, `MotionDetector`, `LivePredictor`, `ASLDisplay`, `run_demo()` — FPS-independent motion detection via time-based velocity (displacement/dt), pre-trigger buffering (retains sign onset frames on IDLE→SIGNING), pose quality gate (skips inference if <30% valid shoulder frames), and "No body detected" display warning | `config`, `preprocess`, `models` |
| `export_onnx.py` | `export_to_onnx()`, `verify_onnx()`, `benchmark_onnx()` | `config`, `models` |

---

## Data Flow Diagrams

### Training Data Flow

```
data/raw/*.mp4
       │
       v
  ┌─────────────────────────────────┐
  │  preprocess.py                  │
  │  extract_keypoints_mediapipe()  │
  │  normalize_keypoints()          │  ──> data/processed/*.npy  (T, 543, 3)
  │    1. shoulder-center + scale   │
  │    2. face-center relative      │
  │    3. depth normalization       │
  │    4. hand-relative to wrist    │
  │  create_splits()                │  ──> data/splits/WLASL{N}/train.csv
  └─────────────────────────────────┘

data/processed/*.npy + data/splits/WLASL{N}/train.csv
       │
       v
  ┌─────────────────────────────────┐
  │  dataset.py                     │
  │  WLASLKeypointDataset           │
  │    __getitem__():               │
  │      load .npy                  │
  │      slice to 75 keypoints     │  (drop face landmarks)
  │      pad/crop to T frames      │  (reflection padding)
  │      compute velocity (motion)  │  ──> (T, 75*6) when use_motion=True
  │      apply augmentations        │  (incl. KeypointYawRotation for 3D viewpoint simulation)
  │
  │      flatten to (T, input_dim)  │
  └─────────────────────────────────┘
       │
       v
  ┌─────────────────────────────────┐
  │  train.py                       │
  │  train_one_epoch():             │
  │    mixup (if enabled)           │
  │    forward pass through model   │
  │    loss + backprop              │
  │  validate():                    │
  │    forward pass (no augment)    │
  │    compute top-1 / top-5 acc    │  ──> checkpoints/best_model.pt
  │    early stopping check         │  ──> logs/ (TensorBoard)
  └─────────────────────────────────┘
```

### Inference Data Flow

```
Single Video (predict.py):

  video.mp4 ──> MediaPipe ──> normalize (shoulder+hand-relative) ──> velocity ──> model ──> top-5 predictions
       │                                                   ^
       │         OR                                        │
  keypoints.npy ──────────────> velocity ──────────────────┘


Live Demo (live_demo.py):

  Webcam ──> MediaPipe ──> MotionDetector ──> FrameBuffer(max_sign_duration * camera_fps + 10)
    │                           │                      │
    │                   FPS-independent:               │
    │                   velocity = displacement/dt     │
    │                   (dt via time.monotonic())       │
    │                           │                      │
    │                           │  IDLE→SIGNING: trim buffer to pre_sign_frames (keep sign onset)
    │                           │              state: IDLE/SIGNING/COMPLETED
    │                           │                      │
    │                           │              inference only on COMPLETED:
    │                           │              ├─ settle_time elapsed ──> COMPLETED
    │                           │              └─ max_sign_duration   ──> COMPLETED
    │                           │                      │
    │                           v                      v
    │                     pose quality gate: skip if <30% of buffer
    │                     frames have valid shoulder landmarks
    │                           │ (pass)
    │                           v
    │                     normalize ──> TemporalCrop(T) ──> model ──> prediction
    │                     (full sign frames: TemporalCrop uniformly
    │                      samples to T, matching training pipeline)
    │                                                           │
    v                                                           v
  Display <───── overlay predicted gloss + confidence + motion state
                 (high-conf: full cooldown, low-conf: 30% cooldown)
                 Shows "No body detected" warning when pose_landmarks is None


ONNX Export (export_onnx.py):

  checkpoint ──> load model ──> torch.onnx.export() ──> model.onnx
                                     │
                              verify (optional) ──> ONNX Runtime forward pass
                              benchmark (optional) ──> avg latency over 100 runs
```

### Model Architecture Flow (ST-GCN)

```
Input: (batch, T, input_dim)     input_dim = 75*3 or 75*6 (with motion)
              │
              v
    ┌──────────────────────────┐
    │  Reshape to graph        │  (B, C, T, V) where V=num_keypoints
    └─────────┬────────────────┘
              │
    ┌─────────┼─────────┐
    v         v         v
  ┌──────┐ ┌──────┐ ┌──────┐
  │ Body │ │ Left │ │Right │     Separate graph convolution branches
  │ GCN  │ │ Hand │ │ Hand │     with dilated TCN, DropPath, joint importance
  │(33kp)│ │(21kp)│ │(21kp)│
  └──┬───┘ └──┬───┘ └──┬───┘
     └─────────┼─────────┘
               v
    ┌─────────────────────┐
    │  Avg Pool or        │  Pool over time+joints per branch
    │  AttentionPool      │  (optional attention-weighted temporal pooling)
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  CrossBranchAttn?   │  Optional cross-branch attention fusion
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  Concat + Project   │  Fuse branch outputs
    └─────────┬───────────┘
              v
    ┌─────────────────────┐
    │  L2 Normalize?      │  Only when normalize_embeddings=True (proto)
    └─────────┬───────────┘
              v
    ┌─────────────────────────────┐
    │  Classification head (CE)   │  Linear→LayerNorm→ReLU→Dropout→Linear
    │  OR Prototypical distance   │  Distance to class prototypes
    └─────────┬───────────────────┘
              v
    Output: (batch, num_classes) logits or distances
```

---

## Configuration Flow

```
configs/stgcn_ce.yaml (default)       ──> load_config() ──> Config dataclass
configs/stgcn_proto.yaml                                        │
                                                    ┌───────────┼───────────┐
                                                    v           v           v
                                              train.py    evaluate.py  predict.py
                                                                        live_demo.py
                                                                        export_onnx.py

Config.__post_init__() auto-derives (scales with variant size):
    wlasl_variant: 100  ──>  num_classes: 100,  d_model: 128, nhead: 4, num_layers: 2, dropout: 0.1
    wlasl_variant: 300  ──>  num_classes: 300,  d_model: 192, nhead: 6, num_layers: 4, dropout: 0.3
    wlasl_variant: 1000 ──>  num_classes: 1000, d_model: 256, nhead: 8, num_layers: 5, dropout: 0.4
    wlasl_variant: 2000 ──>  num_classes: 2000, d_model: 384, nhead: 8, num_layers: 6, dropout: 0.5
    Note: d_model, nhead, num_layers, dropout are auto-scaled per variant.

YAML-configurable model fields (read by load_config into Config dataclass):
    embedding_dim: 128          # Final embedding dimension for ST-GCN encoder
    gcn_channels: [64, 128, 128]  # Channel widths per ST-GCN block
```

---

## Tests (`tests/`)

Each test file maps to one or more source modules:

| Test File | Tests For |
|-----------|-----------|
| `test_augment.py` | `src/data/augment.py` — all transform classes and pipeline presets |
| `test_config.py` | `src/training/config.py` — defaults, load/save, YAML roundtrip |
| `test_dataset.py` | `src/data/dataset.py` — Dataset, DataLoader, pad/crop, motion features |
| `test_evaluate.py` | `src/training/evaluate.py` — metrics, TTA, hard negatives, latency |
| `test_export_onnx.py` | `src/inference/export_onnx.py` — ONNX export and verification |
| `test_live_demo.py` | `src/inference/live_demo.py` — FrameBuffer, MotionDetector (FPS-independent, time-based velocity), prediction smoothing |
| `test_models.py` | `src/models/` — STGCNEncoder, STGCNClassifier, PrototypicalNetwork, normalize_embeddings, DropPath, AttentionPool, CrossBranchAttention, auxiliary heads, 75-keypoint models |
| `test_predict.py` | `src/inference/predict.py` — SignPredictor inference paths |
| `test_preprocess.py` | `src/data/preprocess.py` — normalization, annotation parsing, splits |
| `test_train.py` | `src/training/train.py` — accuracy, mixup helpers |
| `test_dependencies.py` | All `requirements.txt` libraries — version checks, feature compatibility, src module imports (110 tests) |

All tests use `conftest.py` shared fixtures (tmp datasets, keypoint generators) and are fully isolated (no project data or checkpoints needed).

---

## CLI Entry Points

| Command | Module | What It Does |
|---------|--------|-------------|
| `python scripts/download_wlasl.py` | `scripts/download_wlasl.py` | Download annotations, print video download instructions |
| `python scripts/download_kaggle.py` | `scripts/download_kaggle.py` | Download all videos from Kaggle |
| `python scripts/validate_videos.py` | `scripts/validate_videos.py` | Scan and clean fake video files |
| `python scripts/reset_configs.py` | `scripts/reset_configs.py` | Reset configs to recommended defaults |
| `python scripts/check_mediapipe.py` | `scripts/check_mediapipe.py` | Verify MediaPipe installation |
| `python scripts/auto_config.py` | `scripts/auto_config.py` | Auto-detect hardware and generate optimized config |
| `python -m src.data.preprocess` | `src/data/preprocess.py` | Extract keypoints from videos, create splits |
| `python -m src.training.train --config ... [--device cpu\|cuda\|mps]` | `src/training/train.py` | Train a model (auto-detects device, or force with --device) |
| `python -m src.training.evaluate` | `src/training/evaluate.py` | Evaluate a trained model |
| `python -m src.inference.predict` | `src/inference/predict.py` | Predict on a single video |
| `python -m src.inference.live_demo` | `src/inference/live_demo.py` | Run real-time webcam demo |
| `python -m src.inference.export_onnx` | `src/inference/export_onnx.py` | Export model to ONNX format |
