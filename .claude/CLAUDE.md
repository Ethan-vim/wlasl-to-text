# CLAUDE.md

> **STRICT RULE: Read and follow `EXPECTATIONS.md` at all times. Every response must comply with the rules defined there. This is non-negotiable.**

## Project Overview

**Live American Sign Language Recognition using WLASL** — a real-time ASL word-level recognition system that captures webcam video, extracts hand/body keypoints via MediaPipe Holistic (543 landmarks per frame), and predicts the signed word using deep learning models trained on the [WLASL dataset](https://github.com/dxli94/WLASL).

Uses **ST-GCN + Prototypical Network** — a spatiotemporal graph convolutional encoder with episodic metric learning for few-shot classification (~3–8 samples/class).

| Module | Purpose |
|--------|---------|
| `src/models/stgcn.py` | ST-GCN encoder with body/hand graph branches, conditional L2 normalization |
| `src/models/prototypical.py` | Prototypical network wrapper + `build_model()` |
| `src/models/classifier.py` | STGCNClassifier: ST-GCN encoder + two-layer classification head (CE) |
| `src/models/__init__.py` | Unified `build_model()` factory dispatching on `cfg.approach` |

## Environment

- **Python 3.12** (virtual env in `.venv`)
- Activate: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (Windows)
- Install: `pip install -r requirements.txt`
- Tests: `python -m pytest` (317 tests across 12 test files, fully isolated via `tmp_path`)

## Key Files to Read Before Editing

- `README.md` — Full user guide, configuration reference, troubleshooting, recommended configs
- `STRUCTURE.md` — File dependency graph, data flow diagrams, every CLI entry point
- `EXPECTATIONS.md` — **Mandatory behavioral rules. Always follow.**
- `src/training/config.py` — `Config` dataclass. `num_classes` is always auto-derived from `wlasl_variant`. Never set `num_classes` manually.

## Project Structure

```
configs/                  YAML configs (stgcn_ce.yaml, stgcn_proto.yaml)
data/
  raw/                    Downloaded .mp4 video files (shared across variants)
  processed/              Extracted .npy keypoints (shared across variants)
  annotations/            WLASL_v0.3.json
  splits/WLASL{N}/        train.csv, val.csv, test.csv per variant
src/
  data/
    preprocess.py         Download annotations, extract keypoints, normalize, create splits
    augment.py            Temporal + spatial augmentations (rotation, dropout, flip, noise, etc.)
    dataset.py            PyTorch Dataset (WLASLKeypointDataset) + motion feature computation
    episode_sampler.py    EpisodicBatchSampler for N-way K-shot episodes
  models/
    __init__.py           Unified build_model() factory dispatching on cfg.approach
    stgcn.py              STGCNEncoder with body/hand graph branches
    classifier.py         STGCNClassifier: ST-GCN encoder + two-layer classification head
    prototypical.py       PrototypicalNetwork wrapper, build_model()
  training/
    config.py             Config dataclass + YAML load/save
    train.py              CLI dispatcher -> train_ce or train_prototypical
    train_ce.py           Standard cross-entropy training loop (CE-specific augmentation)
    train_prototypical.py Episodic prototypical training loop
    evaluate.py           Metrics, TTA, confusion matrix, hard negatives, latency benchmark
  inference/
    predict.py            Single-video prediction (SignPredictor)
    live_demo.py          Real-time webcam demo (threaded capture + inference + display)
    export_onnx.py        ONNX export, verification, benchmarking
scripts/
    download_wlasl.py     Download annotations from GitHub
    download_kaggle.py    Download full video archive from Kaggle (~12K videos)
    validate_videos.py    Detect/remove HTML files masquerading as .mp4
    auto_config.py        Auto-detect hardware, generate optimized YAML config
    reset_configs.py      Reset configs/ to README defaults
    check_mediapipe.py    Diagnose MediaPipe installation issues
tests/                    317 tests across 12 test files + conftest.py (all isolated)
```

## Pipeline Summary

```
download (Kaggle/URL) -> validate_videos -> preprocess (keypoints/frames)
    -> train -> evaluate -> predict / live_demo / export_onnx
```

## Important Implementation Details

- **MediaPipe import** has a fallback path (`mediapipe.python.solutions`) for Windows/Python 3.12 compatibility — see `_import_mediapipe_holistic()` in `preprocess.py`
- **Normalization** centers on shoulder midpoint (not nose) and scales by shoulder width — see `normalize_keypoints()` in `preprocess.py`
- **Motion features** (`use_motion: true`) concatenate frame-to-frame velocity with position, producing 6 features per keypoint — computed in `dataset.py`
- **CE augmentation** uses milder `get_ce_train_transforms()` (no TemporalFlip, narrower ranges) vs proto's `get_train_transforms()`
- **normalize_embeddings** — `False` for CE (unconstrained logits), `True` for proto (distance-based classification)
- **Episodic training** samples N-way K-shot episodes from the training set — see `train_prototypical.py`
- **TTA** (`use_tta: true`) averages predictions over original + horizontally flipped input — see `_flip_keypoints_tensor()` in `evaluate.py`
- **Multiprocessing** uses `spawn` context (not `fork`) to avoid MediaPipe crashes on macOS
- All split CSVs live under `data/splits/WLASL{N}/` — multiple variants coexist without conflict
- **Face landmark dropping**: Dataset slices 543 keypoints to 75 at load time (33 body + 21 left hand + 21 right hand), dropping unused face landmarks
- **Reflection padding**: Short sequences are padded by mirroring backwards (not repeating last frame), preserving non-zero velocity in padded region
- **Dilated temporal convolutions**: ST-GCN blocks use exponentially increasing dilation [1, 2, 4] for wider receptive field (~57 frames)
- **Joint importance**: Learnable per-joint parameter in each ST-GCN branch, applied before pooling
- **Optional features** (disabled by default, enable in config): `use_attention_pool` (attention-weighted temporal pooling), `drop_path_rate` (stochastic depth), `use_cross_attention` (cross-branch attention fusion), `aux_loss_weight` (auxiliary per-branch classification heads)
