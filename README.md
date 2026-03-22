<div align="center">
  <h1>Vox</h1>
  <p><strong>Real-time American Sign Language recognition -- bridging the communication gap between deaf and hearing communities.</strong></p>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
    <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"/></a>
  </p>
</div>

---

**Over 500,000 deaf and hard-of-hearing people in the US use American Sign Language as their primary language, yet most hearing people cannot understand it.** This project bridges that gap. Point a webcam at someone signing and get real-time word-level predictions -- no special hardware, no cloud dependency, just a laptop and a camera. Built on the [WLASL dataset](https://github.com/dxli94/WLASL) with spatiotemporal graph convolutions (ST-GCN) that read hand and body skeleton movements frame-by-frame, it runs locally on CPU, CUDA, or Apple Silicon.

---

## Table of Contents

- **[Architecture](#architecture)** — Line 61
- **[Quick Start](#quick-start)** — Line 93
  - [1. Environment Setup](#1-environment-setup) — Line 95
    - [Installing PyTorch with CUDA support](#installing-pytorch-with-cuda-support) — Line 133
  - [2. Download the Dataset](#2-download-the-dataset) — Line 169
    - [Option A: Kaggle (Recommended)](#option-a-kaggle-recommended--fastest) — Line 171
    - [Option B: Official WLASL scripts](#option-b-official-wlasl-scripts-url-based) — Line 236
    - [Validate downloaded videos](#validate-downloaded-videos) — Line 247
    - [End-to-end quick start with Kaggle](#end-to-end-quick-start-with-kaggle) — Line 264
    - [Choosing a wlasl_variant with Kaggle](#choosing-a-wlasl_variant-with-kaggle) — Line 305
    - [Device-specific configuration after Kaggle download](#device-specific-configuration-after-kaggle-download) — Line 320
  - [3. Preprocess Data](#3-preprocess-data) — Line 383
    - [Working with multiple variants](#working-with-multiple-variants) — Line 409
  - [4. Train a Model](#4-train-a-model) — Line 464
  - [5. Evaluate](#5-evaluate) — Line 499
  - [6. Run the Live Demo](#6-run-the-live-demo) — Line 525
  - [7. Single Video Prediction](#7-single-video-prediction) — Line 555
  - [8. Export to ONNX](#8-export-to-onnx) — Line 607
  - [9. Run Tests](#9-run-tests) — Line 635
- **[Project Structure](#project-structure)** — Line 678
- **[Configuration Guide](#configuration-guide)** — Line 745
  - [Auto-Configure for Your Hardware](#auto-configure-for-your-hardware) — Line 759
- **[Approach Details](#approach-details)** — Line 881
  - [ST-GCN Encoder (Shared)](#stgcn-encoder-shared)
  - [Cross-Entropy (stgcn_ce)](#cross-entropy-stgcn_ce)
  - [Prototypical (stgcn_proto)](#prototypical-stgcn_proto)
- **[Troubleshooting](#troubleshooting)** — Line 919
  - [HTML files masquerading as videos](#html-files-masquerading-as-videos) — Line 921
  - [MediaPipe installation issues](#mediapipe-installation-issues) — Line 924
  - [CUDA out of memory](#cuda-out-of-memory) — Line 936
  - [Webcam not detected](#webcam-not-detected) — Line 942
  - ["No body detected" warning in live demo](#no-body-detected-warning-in-live-demo)
  - [Low accuracy](#low-accuracy) — Line 948
  - [Diagnosing partial data](#diagnosing-partial-data-most-common-issue) — Line 953
  - [wlasl_variant / num_classes mismatch](#wlasl_variant--num_classes-mismatch) — Line 1000
- **[Recommended Configurations](#recommended-configurations)** — Line 1006
  - [WLASL100 (recommended starting point)](#wlasl100-recommended-starting-point) — Line 1010
  - [WLASL300](#wlasl300) — Line 1043
  - [WLASL1000](#wlasl1000) — Line 1090
  - [WLASL2000](#wlasl2000) — Line 1113
- **[Tips & Best Practices](#tips--best-practices)** — Line 1124
  - [Hardware-Specific Setup](#hardware-specific-setup) — Line 1126
  - [Training with Limited Data](#training-with-limited-data) — Line 1149
  - [Improving Accuracy](#improving-accuracy) — Line 1161
  - [What to Expect](#what-to-expect) — Line 1172
  - [Common Pitfalls](#common-pitfalls) — Line 1184
- **[Recommended Library & CUDA Versions](#recommended-library--cuda-versions)** — Line 1195
  - [PyTorch ↔ CUDA Compatibility](#pytorch--cuda-compatibility) — Line 1199
  - [MediaPipe](#mediapipe) — Line 1218
  - [Other Key Libraries](#other-key-libraries) — Line 1231
- **[Citation](#citation)** — Line 1248
- **[License](#license)** — Line 1260

---

## Architecture

```
                    ST-GCN Keypoint Pipeline
                    +-------------------------------+
                    |  MediaPipe     ST-GCN         |
Webcam  --> Frame --+  Holistic  --> Encoder   -----+--> Predicted
Feed        Buffer  |  Keypoints    (body/hand      |    Gloss +
(OpenCV)    (T=64)  |  + Velocity    graph branches)|    Confidence
                    +-------------------------------+
```

**Two training approaches are available — `stgcn_ce` is the recommended default:**

| Approach | Config | Model | Description | WLASL-100 Top-1 |
| -------- | ------ | ----- | ----------- | --------------- |
| **`stgcn_ce` (recommended)** | `configs/stgcn_ce.yaml` | ST-GCN + two-layer head | Standard cross-entropy classification. **Best overall model** for closed-set classification with known classes. | 45–55% |
| `stgcn_proto` | `configs/stgcn_proto.yaml` | ST-GCN + prototypical net | Episodic metric learning. Best for few-shot scenarios or adding new signs without retraining. | ~100% (few-shot) |

> **Start here:** Use `stgcn_ce` unless you specifically need few-shot classification. It is our best-performing model for standard training.

Both approaches share the same ST-GCN encoder (`src/models/stgcn.py`) with body/hand graph branches. The key difference is that `stgcn_ce` disables L2 embedding normalization (for unconstrained logits) while `stgcn_proto` enables it (for distance-based classification).

To switch between approaches, change the `approach` field in your YAML config file (see [Switching Between Approaches](#switching-between-approaches)).


---

## Quick Start

### 1. Environment Setup

**Linux/macOS:**

```bash
git clone <this-repo-url>
cd "Live American Sign Language Recognition using WLASL"

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python scripts/auto_config.py --approach stgcn_ce
```

**Windows (PowerShell):**

```powershell
git clone <this-repo-url>
cd "Live American Sign Language Recognition using WLASL"

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/auto_config.py --approach stgcn_ce
```

**Windows (Command Prompt):**

```cmd
git clone <this-repo-url>
cd "Live American Sign Language Recognition using WLASL"

python -m venv .venv
.venv\Scripts\activate.bat

pip install -r requirements.txt
python scripts/auto_config.py --approach stgcn_ce
```

#### Installing PyTorch with CUDA support

The default `pip install -r requirements.txt` installs the CPU-only version of PyTorch. If you have an NVIDIA GPU, install the CUDA-enabled version **before** running `pip install -r requirements.txt` (or after, to overwrite the CPU version):

```bash
# CUDA 12.4 (recommended for modern GPUs — RTX 30xx/40xx/50xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older GPUs or driver versions)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (default — no flag needed, but explicit if you want to be sure)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

To check which CUDA version your driver supports:

```bash
nvidia-smi    # look for "CUDA Version" in the top-right corner
```

After installing, verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, version: {torch.version.cuda}')"
```

These commands work identically on Linux, macOS, and Windows.

> **Windows note:** All `python` commands in this README work on both platforms. When you see `\` at the end of a line (bash line continuation), replace it with ``` (backtick) in PowerShell or `^` in Command Prompt. Platform-specific shell commands (file operations, venv activation) show both variants where they differ.

---

### 2. Download the Dataset

#### Option A: Kaggle (Recommended — fastest)

The full WLASL video archive (~12,000 videos, ~5 GB) is available on [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed). This is the fastest way to get the data since it downloads as a single archive.

**One-time Kaggle API setup:**

**Linux/macOS:**

```bash
# kaggle is already included in requirements.txt, so if you ran
# pip install -r requirements.txt, it's already installed.
# Otherwise: pip install kaggle

# Get your API token from https://www.kaggle.com/settings → "Create New Token"
# Move the downloaded kaggle.json to ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Windows (PowerShell):**

```powershell
# Get your API token from https://www.kaggle.com/settings → "Create New Token"
# Move the downloaded kaggle.json to %USERPROFILE%\.kaggle\
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
# No chmod needed on Windows
```

**Windows (Command Prompt):**

```cmd
mkdir "%USERPROFILE%\.kaggle"
move "%USERPROFILE%\Downloads\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json"
```

**Download:**

```bash
python scripts/download_kaggle.py --subset WLASL100
```

This downloads **all** ~12K videos to `data/raw/` (the full archive, regardless of `--subset`), fetches the annotation JSON, and prints a summary of how many videos match the chosen subset. The `--subset` flag only controls the summary output — the download itself always fetches the full dataset.

You can also use the Kaggle CLI directly:

**Linux/macOS:**

```bash
kaggle datasets download -d risangbaskoro/wlasl-processed -p data/_kaggle_download --unzip
mv data/_kaggle_download/videos/*.mp4 data/raw/
rm -rf data/_kaggle_download
```

**Windows (PowerShell):**

```powershell
kaggle datasets download -d risangbaskoro/wlasl-processed -p data\_kaggle_download --unzip
Move-Item data\_kaggle_download\videos\*.mp4 data\raw\
Remove-Item -Recurse -Force data\_kaggle_download
```

> **Note:** The Kaggle archive contains all ~12K videos for all WLASL variants (100–2000). You only download once — the preprocessing step (Step 3) filters to your chosen subset.

#### Option B: Official WLASL scripts (URL-based)

```bash
# Download the WLASL annotation JSON and print video download instructions
python scripts/download_wlasl.py --subset WLASL100
```

This creates `data/annotations/WLASL_v0.3.json` and the directory structure under `data/`. Follow the printed instructions to download the actual video files from the [official WLASL repo](https://github.com/dxli94/WLASL) or a community mirror, then place them in `data/raw/`.

> **Note:** Many original WLASL URLs have expired. The Kaggle option (Option A) typically provides significantly more videos.

#### Validate downloaded videos

Many WLASL URLs have expired. When a URL is dead, servers often return an HTML redirect page (saved as `.mp4`) instead of a 404. This applies to both Kaggle and URL-based downloads. Run the validator before preprocessing to remove these fake files:

```bash
# Report how many invalid files exist
python scripts/validate_videos.py --video-dir data/raw

# Delete invalid files (HTML redirects and empty files)
python scripts/validate_videos.py --video-dir data/raw --delete

# Delete and save a list of valid video IDs for reference
python scripts/validate_videos.py --video-dir data/raw --delete --save-valid data/valid_ids.txt
```

The validator checks the first 256 bytes of each file for HTML signatures (`<!DOCTYPE html>`, `<html>`, etc.) and reports counts of valid / HTML / empty files. The preprocessing pipeline also skips unreadable files automatically, but cleaning them up first saves processing time.

#### End-to-end quick start with Kaggle

If you want the fastest path from zero to training, run these commands in order. This uses **`stgcn_ce` (recommended default — our best model)**:

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Auto-detect hardware and generate optimized stgcn_ce config (recommended)
python scripts/auto_config.py --approach stgcn_ce

# 3. Configure Kaggle API (one-time — see "One-time Kaggle API setup" above)

# 4. Download all videos from Kaggle (~5 GB, ~12K videos)
#    --subset only controls the annotation summary printed after download;
#    the full archive is always downloaded regardless of the variant chosen.
python scripts/download_kaggle.py --subset WLASL100

# 5. Validate and clean up bad files
python scripts/validate_videos.py --video-dir data/raw --delete

# 6. Extract keypoints
python -m src.data.preprocess --data-dir data --subset WLASL100 --mode keypoints

# 7. Train (see device-specific configs below)
python -m src.training.train --config configs/stgcn_ce.yaml

# 8. Evaluate
#    Linux/macOS uses \ for line continuation; Windows PowerShell uses `
python -m src.training.evaluate \
    --config configs/stgcn_ce.yaml \
    --checkpoint checkpoints/best_model.pt \
    --split val --output-dir eval_results

On Windows PowerShell, the evaluate command (step 8) becomes:

```powershell
python -m src.training.evaluate `
    --config configs/stgcn_ce.yaml `
    --checkpoint checkpoints/best_model.pt `
    --split val --output-dir eval_results
```

#### Choosing a `wlasl_variant` with Kaggle

The Kaggle archive contains all ~12K videos covering every WLASL variant. After downloading, you choose which variant to train on by setting `wlasl_variant` in your YAML config (and matching the `--subset` flag during preprocessing). The variant controls the number of sign classes — `num_classes` is auto-derived, so you never set it manually.


| Variant               | Classes | Approx. Training Samples | Difficulty | Recommended For                             |
| --------------------- | ------- | ------------------------ | ---------- | ------------------------------------------- |
| `wlasl_variant: 100`  | 100     | 800–1,200                | Easiest    | First-time setup, prototyping, CPU training |
| `wlasl_variant: 300`  | 300     | 2,000–3,500              | Moderate   | Better vocabulary coverage with a GPU       |
| `wlasl_variant: 1000` | 1,000   | 5,000–8,000              | Hard       | Research, large-GPU setups                  |
| `wlasl_variant: 2000` | 2,000   | 8,000–12,000             | Hardest    | Full dataset, consider fusion (Approach C)  |


**Start with `wlasl_variant: 100`** — it has the most samples per class, trains fastest, and gives the highest per-class accuracy. Scale up once your pipeline is working.

Larger variants need more model capacity. See the [Recommended Configurations](#recommended-configurations) section for variant-specific hyperparameters (deeper layers, wider `d_model`, adjusted LR/dropout).

#### Device-specific configuration after Kaggle download

After downloading the Kaggle dataset, adjust `configs/stgcn_ce.yaml` (recommended default) for your hardware before training (step 6). Set `wlasl_variant` to match the subset you preprocessed in step 5.

**CPU-only (no GPU):**

```yaml
approach: stgcn_ce
wlasl_variant: 100          # match your preprocessed subset (100, 300, 1000, or 2000)
normalize_embeddings: false  # required for CE
fp16: false                  # FP16 only works on CUDA
batch_size: 8                # smaller batches for CPU
num_workers: 2
T: 64
lr: 1.0e-3
weight_decay: 1.0e-4
scheduler: cosine
warmup_epochs: 10
weighted_sampling: true
early_stopping_patience: 30
epochs: 200
```

Use `--device cpu` for inference and live demo. Stick to keypoint approaches (`stgcn_ce` or `stgcn_proto`) — they are lightweight and run well on CPU.

**GPU / CUDA:**

```yaml
approach: stgcn_ce
wlasl_variant: 100          # match your preprocessed subset (100, 300, 1000, or 2000)
normalize_embeddings: false  # required for CE
fp16: true                   # faster training, lower memory
batch_size: 32               # increase to 64 for large GPUs
num_workers: 4
T: 64
lr: 1.0e-3
weight_decay: 1.0e-4
scheduler: cosine
warmup_epochs: 10
weighted_sampling: true
early_stopping_patience: 30
epochs: 200
```

Monitor GPU memory with `nvidia-smi`. If you run out of memory, reduce `batch_size` first, then `T`.

**Apple Silicon (M1/M2/M3/M4):**

```yaml
approach: stgcn_ce
wlasl_variant: 100          # match your preprocessed subset (100, 300, 1000, or 2000)
normalize_embeddings: false  # required for CE
fp16: false                  # MPS does not support FP16 reliably
batch_size: 16
num_workers: 0               # MPS requires 0 workers
T: 64
lr: 1.0e-3
weight_decay: 1.0e-4
scheduler: cosine
warmup_epochs: 10
weighted_sampling: true
early_stopping_patience: 30
epochs: 200
```

Use `--device cpu` for the live demo to avoid MPS overhead. Install MediaPipe with `pip install mediapipe-silicon` if the standard package fails.

---

### 3. Preprocess Data

The preprocessing pipeline is **source-agnostic** — it reads mp4 files from `data/raw/` regardless of whether they were downloaded via Kaggle (Option A) or URL-based scripts (Option B). No extra flags or options are needed.

Extract MediaPipe Holistic keypoints (543 landmarks per frame) from all valid videos. The extraction uses `model_complexity=2` with lower detection/tracking confidence thresholds (0.3) to maximize the number of frames with valid keypoints:

```bash
python -m src.data.preprocess --data-dir data --subset WLASL100 --mode keypoints
```

> **Note:** If you previously preprocessed data with older settings (model_complexity=1, confidence thresholds=0.5), regenerate your keypoints by deleting `data/processed/` and re-running the command above. The new settings produce higher quality keypoints and capture more frames.

This creates:

```
data/processed/<video_id>.npy       # keypoints, shape (T, 543, 3) — shared across variants
data/splits/WLASL100/train.csv      # variant-specific split files
data/splits/WLASL100/val.csv
data/splits/WLASL100/test.csv
```

To extract raw frames instead (needed for Approach B/C training):

```bash
python -m src.data.preprocess --data-dir data --subset WLASL100 --mode frames
```

Use `--max-workers N` to control parallel extraction (default: 4).

#### Working with multiple variants

WLASL100 ⊂ WLASL300 ⊂ WLASL1000 ⊂ WLASL2000 — each larger variant is a superset of the smaller ones. You can preprocess multiple variants without conflicts:

- `**data/raw/**` and `**data/processed/**` are shared — keypoints are stored by `video_id` and reused across variants. Already-extracted `.npy` files are skipped automatically.
- `**data/splits/WLASL{N}/**` is variant-specific — each variant gets its own `train/val/test.csv` files that never overwrite each other.

**With Kaggle (recommended):** Since Kaggle downloads all ~12K videos at once, you already have all the data. Just preprocess each variant:

```bash
# Download once (all variants included)
python scripts/download_kaggle.py --subset WLASL100

# Preprocess WLASL100
python -m src.data.preprocess --data-dir data --subset WLASL100

# Scale up — only new videos are extracted; WLASL100 splits are untouched
python -m src.data.preprocess --data-dir data --subset WLASL300
```

**With URL-based download:** Download annotations per variant, then add videos:

```bash
# First variant
python scripts/download_wlasl.py --subset WLASL100
# ...download videos to data/raw/...
python -m src.data.preprocess --data-dir data --subset WLASL100

# Scale up — only new videos are extracted; WLASL100 splits are untouched
python scripts/download_wlasl.py --subset WLASL300
# ...download the additional WLASL300 videos to data/raw/...
python -m src.data.preprocess --data-dir data --subset WLASL300
```

After two variants, your `data/` tree looks like:

```
data/
├── raw/                          # All videos (shared)
├── processed/                    # .npy keypoints by video_id (shared)
├── annotations/
│   └── WLASL_v0.3.json
└── splits/
    ├── WLASL100/                 # 100-class splits
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── WLASL300/                 # 300-class splits — coexist safely
        ├── train.csv
        ├── val.csv
        └── test.csv
```

---

### 4. Train a Model

Set `wlasl_variant` in your config to match the subset you preprocessed. All scripts automatically resolve split files from `data/splits/WLASL{N}/`.

```bash
# ST-GCN + Cross-Entropy (recommended — best model for standard training)
python -m src.training.train --config configs/stgcn_ce.yaml

# ST-GCN + Prototypical Network (alternative — for few-shot scenarios only)
python -m src.training.train --config configs/stgcn_proto.yaml

# Force a specific device (auto-detects by default: CUDA > MPS > CPU)
python -m src.training.train --config configs/stgcn_ce.yaml --device cpu
```

Training checkpoints are saved to `checkpoints/` and logs to `logs/`.

```bash
# Monitor training in real time
tensorboard --logdir logs/
```

To train on a different variant, either edit `wlasl_variant` in an existing config or use a separate config file:

```bash
# Copy and modify — only change wlasl_variant (num_classes is auto-derived)
cp configs/stgcn_ce.yaml configs/stgcn_ce_wlasl300.yaml          # Linux/macOS
# copy configs\stgcn_ce.yaml configs\stgcn_ce_wlasl300.yaml    # Windows

# Edit wlasl_variant: 300 in the new file
python -m src.training.train --config configs/stgcn_ce_wlasl300.yaml
```

---

### 5. Evaluate

**Linux/macOS:**

```bash
python -m src.training.evaluate \
    --config configs/stgcn_ce.yaml \
    --checkpoint checkpoints/best_model.pt \
    --split val \
    --output-dir eval_results
```

**Windows (PowerShell):**

```powershell
python -m src.training.evaluate `
    --config configs/stgcn_ce.yaml `
    --checkpoint checkpoints/best_model.pt `
    --split val `
    --output-dir eval_results
```

This prints top-1/top-5 accuracy, per-class breakdown, and saves a confusion matrix heatmap to `eval_results/`.

---

### 6. Run the Live Demo

**Linux/macOS:**

```bash
python -m src.inference.live_demo \
    --config configs/stgcn_ce.yaml \
    --checkpoint checkpoints/best_model.pt \
    --camera 0 \
    --device cpu
```

**Windows (PowerShell):**

```powershell
python -m src.inference.live_demo `
    --config configs/stgcn_ce.yaml `
    --checkpoint checkpoints/best_model.pt `
    --camera 0 `
    --device cpu
```

**Controls:**

- `q` — quit
- `s` — save the current prediction to a log file

The demo runs three threads: a capture thread reads webcam frames continuously, an inference thread runs the model when a sign is detected as complete, and the main thread renders the overlay. Predictions are smoothed over the last 5 inference windows and only displayed when confidence exceeds the configured threshold (default: 0.6).

**Motion-aware sign detection:** The demo uses a `MotionDetector` that tracks hand keypoint velocity to detect when a sign starts and ends. The detector is **fully FPS-independent**: it measures real time between frames via `time.monotonic()` and converts displacement into velocity in normalized-coordinates per second, so the same thresholds work identically on 30fps, 60fps, or any other camera rate. The status bar shows the current state: `IDLE` (waiting for motion), `SIGNING` (motion in progress), or `COMPLETED` (sign finished, running inference).

**Inference only triggers on COMPLETED state** (the full `IDLE` -> `SIGNING` -> `COMPLETED` transition). The detector enters `SIGNING` when hand velocity exceeds `motion_start_threshold` (0.30 norm-coords/sec), then transitions to `COMPLETED` either when velocity drops below `motion_end_threshold` (0.10 norm-coords/sec) for `motion_settle_time` (0.27s) or when `max_sign_duration` (3.0s) elapses. High-confidence predictions commit to the display and enter a full cooldown. Low-confidence results still display but use a shorter cooldown (30% of `prediction_cooldown`) so the user can retry sooner.

**Buffer management:** The buffer is sized dynamically from the camera FPS (`max_sign_duration * camera_fps + 10`) rather than a fixed `buffer_size` value, ensuring the buffer can hold the full sign duration regardless of camera frame rate. When the motion detector transitions from `IDLE` to `SIGNING`, the buffer is trimmed to keep only the most recent `pre_sign_duration` seconds of frames (default 0.5s). This "pre-trigger buffer" preserves the sign's preparation phase (the hand movement ramp-up that precedes the velocity threshold) while discarding old idle frames. The training data includes full video clips with preparation frames, so preserving sign onset in the live buffer keeps inference data consistent with what the model was trained on.

**Pose quality gate:** Before running the model, `predict_buffer` checks that at least 30% of buffered frames have valid shoulder landmarks. If fewer than 30% pass, inference is skipped entirely and returns `None`, preventing garbage predictions from low-quality pose data. A red "No body detected" warning is shown on the display when MediaPipe cannot detect pose landmarks in the current frame.

**Confidence scaling:** Predictions from partially-filled buffers have their confidence scaled by the buffer fill ratio (`real_frames / T`), so incomplete signs naturally fall below the confidence threshold.

---

### 7. Single Video Prediction

**Linux/macOS:**

```bash
# From a video file
python -m src.inference.predict \
    --video path/to/video.mp4 \
    --config configs/stgcn_ce.yaml \
    --checkpoint checkpoints/best_model.pt

# From a pre-extracted keypoint .npy file
python -m src.inference.predict \
    --keypoints data/processed/12345.npy \
    --config configs/stgcn_ce.yaml \
    --checkpoint checkpoints/best_model.pt

# Specify device (auto, cpu, cuda, mps)
python -m src.inference.predict \
    --video path/to/video.mp4 \
    --config configs/stgcn_ce.yaml \
    --checkpoint checkpoints/best_model.pt \
    --device cpu
```

**Windows (PowerShell):**

```powershell
# From a video file
python -m src.inference.predict `
    --video path\to\video.mp4 `
    --config configs\stgcn_ce.yaml `
    --checkpoint checkpoints\best_model.pt

# From a pre-extracted keypoint .npy file
python -m src.inference.predict `
    --keypoints data\processed\12345.npy `
    --config configs\stgcn_ce.yaml `
    --checkpoint checkpoints\best_model.pt

# Specify device (auto, cpu, cuda, mps)
python -m src.inference.predict `
    --video path\to\video.mp4 `
    --config configs\stgcn_ce.yaml `
    --checkpoint checkpoints\best_model.pt `
    --device cpu
```

Returns the predicted gloss, confidence score, and top-5 alternatives.

---

### 8. Export to ONNX

**Linux/macOS:**

```bash
python -m src.inference.export_onnx \
    --config configs/stgcn_ce.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output model.onnx \
    --verify \
    --benchmark
```

**Windows (PowerShell):**

```powershell
python -m src.inference.export_onnx `
    --config configs/stgcn_ce.yaml `
    --checkpoint checkpoints/best_model.pt `
    --output model.onnx `
    --verify `
    --benchmark
```

`--verify` runs a forward pass through ONNX Runtime to confirm output shapes match. `--benchmark` measures average inference latency over 100 runs. Use `--opset N` to set the ONNX opset version (default: 17).

---

### 9. Run Tests

**Linux/macOS:**

```bash
source .venv/bin/activate
python -m pytest                          # full test suite (317 tests)
python -m pytest tests/test_augment.py    # specific test file
python -m pytest tests/test_dependencies.py  # dependency compatibility tests
python -m pytest -q                       # quiet output
```

Or without activating the venv:

```bash
.venv/bin/python -m pytest
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
python -m pytest                          # full test suite (317 tests)
python -m pytest tests\test_augment.py    # specific test file
python -m pytest -q                       # quiet output
```

Or without activating the venv:

```powershell
.venv\Scripts\python -m pytest
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
python -m pytest
```

Tests are fully isolated — they use pytest's `tmp_path` fixture for all file I/O and never touch project data, configs, or checkpoints. The `pyproject.toml` configures test discovery, verbose output, and warning suppression.

**Note:** `pytest` must be installed in the venv (`pip install pytest`). It is not listed in `requirements.txt` because it is a dev-only dependency.

The `test_dependencies.py` file (110 tests) verifies that every third-party library used by the `src/` code is importable, meets the minimum version from `requirements.txt`, and that the specific features relied upon (e.g. `batch_first` Transformers, `label_smoothing` in CrossEntropyLoss, seaborn heatmaps, ONNX Runtime sessions) work correctly with the installed versions.

---

## Project Structure

```
.
├── configs/
│   ├── stgcn_ce.yaml            # Default config for cross-entropy training
│   └── stgcn_proto.yaml         # Config for prototypical few-shot training
├── data/
│   ├── raw/                     # Downloaded video files — Kaggle or URL-based (shared)
│   ├── processed/               # Extracted keypoints as .npy (shared across variants)
│   ├── annotations/             # WLASL JSON annotation file
│   └── splits/
│       ├── WLASL100/            # train/val/test CSVs for 100-class variant
│       ├── WLASL300/            # train/val/test CSVs for 300-class variant
│       └── ...                  # one subdirectory per variant
├── src/
│   ├── data/
│   │   ├── preprocess.py        # Download, parse, extract & normalize keypoints
│   │   ├── augment.py           # Temporal & spatial augmentations
│   │   └── dataset.py           # PyTorch Dataset + motion feature computation
│   ├── models/
│   │   ├── __init__.py          # Unified build_model() factory
│   │   ├── stgcn.py             # ST-GCN encoder with body/hand graph branches
│   │   ├── classifier.py        # STGCNClassifier: ST-GCN encoder + two-layer head (CE)
│   │   └── prototypical.py      # Prototypical network wrapper for few-shot
│   ├── training/
│   │   ├── config.py            # Config dataclass + YAML serialization
│   │   ├── train.py             # CLI dispatcher for training
│   │   ├── train_ce.py          # Cross-entropy training loop
│   │   ├── train_prototypical.py # Episodic prototypical training loop
│   │   └── evaluate.py          # Metrics, TTA, confusion matrix
│   └── inference/
│       ├── predict.py           # Single-video prediction
│       ├── export_onnx.py       # ONNX export & latency benchmark
│       └── live_demo.py         # Real-time webcam demo
├── tests/
│   ├── conftest.py              # Shared fixtures (tmp datasets, keypoint helpers)
│   ├── test_augment.py          # Augmentation classes & pipeline presets
│   ├── test_config.py           # Config defaults, load/save, YAML roundtrip
│   ├── test_dataset.py          # Dataset, DataLoader, pad/crop, motion features
│   ├── test_evaluate.py         # Metrics, TTA flip, hard negatives, latency
│   ├── test_export_onnx.py      # ONNX export & verification
│   ├── test_live_demo.py        # FrameBuffer, prediction smoothing
│   ├── test_models.py           # STGCNEncoder, STGCNClassifier, PrototypicalNetwork
│   ├── test_predict.py          # SignPredictor inference paths
│   ├── test_preprocess.py       # Normalization, annotation parsing, splits
│   ├── test_train.py            # Accuracy, mixup helpers
│   └── test_dependencies.py    # Library version & feature compatibility (110 tests)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_keypoint_visualization.ipynb
│   └── 03_error_analysis.ipynb
├── scripts/
│   ├── download_wlasl.py        # Download annotations, print video instructions
│   ├── download_kaggle.py       # Download videos from Kaggle (fast alternative)
│   ├── validate_videos.py       # Detect and remove HTML-disguised video files
│   ├── reset_configs.py         # Reset all configs/ to README.md recommended defaults
│   ├── check_mediapipe.py       # Verify MediaPipe installation and diagnose issues
│   └── auto_config.py           # Auto-detect hardware and generate optimized configs
├── checkpoints/                 # Saved model weights
├── logs/                        # TensorBoard training logs
├── pyproject.toml               # Pytest configuration
├── requirements.txt
├── CONTRIBUTING.md              # Contribution guide, commit standards, signing setup
└── STRUCTURE.md                 # Full workflow & dependency graph (which file calls which)
```

For a detailed breakdown of the entire pipeline — data flow diagrams, file dependency graphs, model architecture flow, and all CLI entry points — see `[STRUCTURE.md](STRUCTURE.md)`.

---

## Configuration Guide

All hyperparameters live in YAML files under `configs/`. The table below shows **all** settings and their defaults from the `Config` dataclass (`src/training/config.py`). You only need to override values that differ from the defaults.

To reset all config files back to the recommended defaults:

```bash
python scripts/reset_configs.py                     # reset both configs
python scripts/reset_configs.py --only stgcn_ce     # reset only stgcn_ce.yaml
python scripts/reset_configs.py --only stgcn_proto  # reset only stgcn_proto.yaml
python scripts/reset_configs.py --dry-run            # preview without writing
```

### Auto-Configure for Your Hardware

Detect your GPU/CPU and generate an optimized config automatically. The script probes CUDA VRAM, Apple Silicon MPS, or CPU, classifies your hardware into a performance tier, and writes a ready-to-train config:

```bash
# Cross-entropy approach (recommended — best model)
python scripts/auto_config.py --approach stgcn_ce

# Prototypical approach (alternative — few-shot scenarios only)
python scripts/auto_config.py --approach stgcn_proto --variant 100

# Preview without writing
python scripts/auto_config.py --approach stgcn_ce --dry-run

# Force CPU mode (e.g. no GPU or Apple Silicon)
python scripts/auto_config.py --approach stgcn_ce --device cpu

# Back up existing config before overwriting
python scripts/auto_config.py --approach stgcn_ce --backup
```

**Hardware tiers** (auto-detected from CUDA VRAM):


| Tier     | VRAM      | `batch_size` | `fp16` |
| -------- | --------- | ------------ | ------ |
| **high** | >= 16 GB  | 64           | true   |
| **mid**  | >= 8 GB   | 32           | true   |
| **low**  | >= 4 GB   | 16           | true   |
| **cpu**  | MPS / CPU | 8            | false  |


Windows (PowerShell / Command Prompt): the commands are identical — just run them in your terminal.

**Paths:**


| Parameter        | Description               | Default       |
| ---------------- | ------------------------- | ------------- |
| `data_dir`       | Root data directory       | `data`        |
| `output_dir`     | Output directory          | `outputs`     |
| `checkpoint_dir` | Checkpoint save directory | `checkpoints` |
| `log_dir`        | Training log directory    | `logs`        |


**Dataset:**


| Parameter       | Description                                             | Default |
| --------------- | ------------------------------------------------------- | ------- |
| `wlasl_variant` | Dataset size: `100`, `300`, `1000`, `2000`              | `100`   |
| `num_classes`   | Auto-derived from `wlasl_variant` — do not set manually | `100`   |
| `T`             | Temporal sequence length in frames                      | `64`    |
| `image_size`    | Spatial resolution for video models (Approach B/C)      | `224`   |
| `num_workers`   | DataLoader worker processes                             | `4`     |


**Model:**


| Parameter       | Description                                                                                           | Default            |
| --------------- | ----------------------------------------------------------------------------------------------------- | ------------------ |
| `approach`              | `stgcn_ce` or `stgcn_proto`                                                                    | `stgcn_ce`         |
| `num_keypoints`         | Number of landmarks per frame after face drop (33 pose + 21 left hand + 21 right hand)         | `75`               |
| `d_model`               | ST-GCN embedding dimension (auto-scaled per variant: 128/192/256/384 for 100/300/1000/2000)    | `128`              |
| `embedding_dim`         | Final embedding dimension output by the ST-GCN encoder                                         | `128`              |
| `gcn_channels`          | Channel widths for each ST-GCN block (list of ints)                                            | `[64, 128, 128]`   |
| `nhead`                 | Number of attention heads (auto-scaled per variant: 4/6/8/8 for 100/300/1000/2000)             | `4`                |
| `num_layers`            | Number of encoder layers (auto-scaled per variant: 2/4/5/6 for 100/300/1000/2000)              | `2`                |
| `dropout`               | Dropout rate (auto-scaled per variant: 0.1/0.3/0.4/0.5 for 100/300/1000/2000)                 | `0.5`              |
| `use_motion`            | Concatenate velocity (frame differences) with position features                                 | `true`             |
| `normalize_embeddings`  | L2-normalize encoder embeddings (`true` for proto distance-based, `false` for CE logit-based)  | `true`             |
| `use_attention_pool`    | Attention-weighted temporal pooling (replaces global average pool)                              | `false`            |
| `drop_path_rate`        | Stochastic depth drop rate per ST-GCN block (0 = disabled)                                     | `0.0`              |
| `use_cross_attention`   | Cross-branch attention fusion between body/hand branches                                        | `false`            |


**Training:**


| Parameter                 | Description                                                      | Default    |
| ------------------------- | ---------------------------------------------------------------- | ---------- |
| `epochs`                  | Maximum training epochs                                          | `250`      |
| `batch_size`              | Training batch size                                              | `32`       |
| `lr`                      | Learning rate                                                    | `3e-4`     |
| `weight_decay`            | AdamW weight decay                                               | `5e-4`     |
| `warmup_epochs`           | Linear warmup epochs before scheduler takes over                 | `15`       |
| `label_smoothing`         | Label smoothing for cross-entropy loss (0 = disabled)            | `0.1`      |
| `grad_clip`               | Max gradient norm for clipping                                   | `1.0`      |
| `fp16`                    | Mixed-precision (FP16) training                                  | `true`     |
| `weighted_sampling`       | Weighted sampler to counter class imbalance                      | `false`    |
| `early_stopping_patience` | Epochs without val improvement before stopping                   | `50`       |
| `mixup_alpha`             | Mixup interpolation strength (0 = disabled)                      | `0.2`      |
| `head_dropout`            | Dropout before classification head (CE approach)                 | `0.2`      |
| `class_weighted_loss`     | Inverse-frequency class weights in CE loss (prevents class collapse) | `true`  |
| `scheduler`               | LR scheduler: `onecycle` (per-batch) or `cosine` (per-epoch warmup + cosine annealing) | `onecycle` |
| `aux_loss_weight`         | Weight for auxiliary branch classification losses (0 = disabled) | `0.0`      |


**Evaluation:**


| Parameter | Description                                          | Default |
| --------- | ---------------------------------------------------- | ------- |
| `use_tta` | Test-time augmentation via horizontal flip averaging | `false` |


**Inference / Live Demo:**


| Parameter              | Description                                            | Default |
| ---------------------- | ------------------------------------------------------ | ------- |
| `confidence_threshold` | Minimum confidence for live display                    | `0.6`   |
| `smoothing_window`     | Number of inference windows to smooth predictions over | `5`     |
| `fps_display`          | Show FPS counter on live demo overlay                  | `true`  |

**Sign Detection (Live Demo):**

All motion thresholds are in **normalized-coordinates per second**, making them fully FPS-independent. The `MotionDetector` measures real time between frames via `time.monotonic()` and converts displacement to velocity, so the same config works on any camera frame rate.

| Parameter                 | Description                                                    | Default |
| ------------------------- | -------------------------------------------------------------- | ------- |
| `min_buffer_frames`       | Minimum real frames before prediction (~1s at 30fps)           | `30`    |
| `prediction_cooldown`     | Seconds to wait after a confident prediction before next (low-confidence uses 30% of this value) | `1.0`   |
| `motion_start_threshold`  | Hand velocity to detect sign start (norm-coords/sec) -- requires deliberate hand movement | `0.30`  |
| `motion_end_threshold`    | Hand velocity to detect sign end (norm-coords/sec) -- detects when signing stops | `0.10`  |
| `motion_settle_time`      | Seconds of low velocity to confirm sign end                    | `0.27`  |
| `max_sign_duration`       | Maximum sign length in seconds before forcing completion       | `3.0`   |
| `inference_poll_interval` | Seconds between inference loop state checks                    | `0.1`   |
| `pre_sign_duration`       | Seconds of pre-sign frames to retain when signing starts (preserves sign onset for model accuracy) | `0.5`   |


**Logging:**


| Parameter         | Description                              | Default             |
| ----------------- | ---------------------------------------- | ------------------- |
| `use_wandb`       | Enable Weights & Biases logging          | `false`             |
| `use_tensorboard` | Enable TensorBoard logging               | `true`              |
| `wandb_project`   | W&B project name                         | `wlasl-recognition` |
| `wandb_run_name`  | W&B run name (auto-generated if not set) | `null`              |
| `log_interval`    | Steps between log entries                | `10`                |


**Resume:**


| Parameter           | Description                              | Default |
| ------------------- | ---------------------------------------- | ------- |
| `resume_checkpoint` | Path to checkpoint for resuming training | `null`  |


`num_classes` is **always** auto-derived from `wlasl_variant` (100 → 100, 300 → 300, etc.). Any explicit `num_classes` in the YAML is silently overridden — do not set it manually. Similarly, `d_model`, `nhead`, `num_layers`, and `dropout` are all auto-scaled per variant in `__post_init__`:

| Variant   | `d_model` | `nhead` | `num_layers` | `dropout` |
|-----------|-----------|---------|--------------|-----------|
| WLASL100  | 128       | 4       | 2            | 0.1       |
| WLASL300  | 192       | 6       | 4            | 0.3       |
| WLASL1000 | 256       | 8       | 5            | 0.4       |
| WLASL2000 | 384       | 8       | 6            | 0.5       |

Dropout is intentionally low for smaller variants: a tiny model (d_model=128, 2 layers) paired with high dropout (≥0.5) applies so much regularisation that the gradient signal is crushed and the loss never decreases. The classification head uses a two-layer architecture: `Linear → LayerNorm → ReLU → Dropout → Linear` for a nonlinear decision surface. LayerNorm is used instead of BatchNorm to avoid mode collapse with small per-class batch representation.

---

## Approach Details

### ST-GCN Encoder (Shared)

MediaPipe Holistic extracts 543 landmarks per frame (33 pose + 21 left hand + 21 right hand + 468 face), centered on the shoulder midpoint and scaled by shoulder width. At load time, face landmarks are dropped, keeping only the first 75 keypoints (pose + hands). Hand landmarks are further normalized relative to their respective wrist, reducing noise from absolute wrist movement. When `use_motion: true` (default), frame-to-frame velocity is concatenated with position, producing 6 features per keypoint `(x, y, z, dx, dy, dz)`.

The ST-GCN encoder processes keypoints through separate body (33 landmarks) and hand (21+21 landmarks) graph convolution branches, then fuses them into a single embedding vector.

**Advantages:** Lightweight, fast inference, background-invariant.

**Pipeline:** Video → MediaPipe → Shoulder-Centered Normalization → Hand-Relative Normalization → Motion Features → Augment → ST-GCN → Classification

### Cross-Entropy (`stgcn_ce`)

Standard classification with an enhanced two-layer head (`Linear → LayerNorm → ReLU → Dropout → Linear`). Uses `normalize_embeddings: false` so encoder outputs are unconstrained (important — L2 normalization caps logit magnitudes and causes loss plateaus).

**CE-specific augmentation** (milder than proto to avoid over-regularization with small datasets):

- Temporal speed perturbation (0.85x–1.15x)
- Random temporal crop to T frames
- Keypoint horizontal flip with landmark swapping (p=0.5)
- Keypoint yaw rotation (up to 15 degrees, p=0.5)
- Keypoint rotation (up to 10 degrees, p=0.5)
- Keypoint translation (up to 0.05 shift, p=0.5)
- Keypoint noise (sigma=0.01)
- Random scaling (0.95x–1.05x)
- Keypoint dropout (frame=0.05, landmark=0.03, p=0.3)

**Note:** No TemporalFlip (reversing a sign changes its meaning), no label smoothing, no mixup — these are counterproductive with ~8 samples/class.

### Prototypical (`stgcn_proto`)

Episodic N-way K-shot metric learning. Uses `normalize_embeddings: true` to project embeddings onto the unit sphere for distance-based classification.

**Proto augmentation** (stronger regularization, includes TemporalFlip):

- Temporal speed perturbation (0.7x–1.4x)
- Random temporal crop to T frames
- Temporal flip (p=0.3)
- Keypoint horizontal flip with landmark swapping
- Keypoint yaw rotation (up to 30 degrees, p=0.5)
- Keypoint rotation (up to 15 degrees)
- Keypoint translation (up to 0.1 shift)
- Keypoint noise (sigma=0.02)
- Random scaling (0.9x–1.1x)
- Keypoint dropout (frame=0.15, landmark=0.1, p=0.7)

---

## Switching Between Approaches

The `approach` field in your YAML config controls which model architecture and training loop are used. **`stgcn_ce` is recommended for most users.**

1. **Change the config file:** Use `configs/stgcn_ce.yaml` (recommended) for cross-entropy or `configs/stgcn_proto.yaml` for prototypical training.
2. **Or edit the `approach` field** in your existing config:
   - `approach: stgcn_ce` **(recommended)** — Uses `STGCNClassifier` (ST-GCN encoder + two-layer head) trained with standard cross-entropy loss. Uses `normalize_embeddings: false` and milder augmentation. Best overall model when all target classes are known at training time.
   - `approach: stgcn_proto` — Uses `PrototypicalNetwork` (ST-GCN encoder + episodic metric learning) trained with N-way K-shot episodes. Uses `normalize_embeddings: true` and stronger augmentation. Use only for few-shot classification or when you need to add new sign classes without retraining.

Both approaches share the same ST-GCN encoder (`src/models/stgcn.py`) and data pipeline. The unified `build_model()` factory in `src/models/__init__.py` dispatches to the correct model based on `cfg.approach`.

---

## Troubleshooting

### HTML files masquerading as videos

Expired WLASL URLs often return an HTML lander page (saved as `.mp4`) rather than a 404. Run `scripts/validate_videos.py --delete` before preprocessing to clean these up (see [Step 2](#2-download-the-dataset)).

### MediaPipe installation issues

- Compatible with Python 3.9–3.12.
- On macOS with Apple Silicon: `pip install mediapipe-silicon`
- **Run the diagnostic script** to verify your installation:
  ```bash
  python scripts/check_mediapipe.py
  ```
  This prints your mediapipe version, available modules, and whether the `Holistic` model is accessible.
- If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, your MediaPipe version is incompatible. This commonly happens on **Windows with Python 3.12** and newer mediapipe builds. Fix with: `pip install --force-reinstall mediapipe==0.10.11` (or `mediapipe-silicon` on Apple Silicon). The code has a fallback import path (`mediapipe.python.solutions`) that resolves this for some versions automatically.
- Zero-padded keypoints for frames where detection fails are handled automatically.
- Preprocessing uses `spawn` multiprocessing context (not `fork`) to avoid MediaPipe crashes on macOS.

### FP16 / AMP not supported on Apple Silicon (MPS)

PyTorch's Automatic Mixed Precision (`torch.amp.autocast`) does **not** support `device_type='mps'`. If you run on Apple Silicon and set `fp16: true`, training will crash with:

```
RuntimeError: User specified an unsupported autocast device_type 'mps'
```

**Fix:** always set `fp16: false` when training on MPS (the default configs and `auto_config.py` already do this). FP16 training is only available on CUDA (NVIDIA) GPUs.

### CUDA out of memory

- Reduce `batch_size` (try 8 or 4).
- Enable `fp16: true`.
- Reduce `T` (e.g., 32 instead of 64).
- For video models, set `image_size: 112`.

### Webcam not detected

- Try `--camera 1` or `--camera 2`.
- Linux: check `ls -la /dev/video`*.
- macOS: grant camera access in System Settings → Privacy & Security → Camera.
- Windows: check Device Manager → Cameras. Grant camera access in Settings → Privacy & security → Camera.

### "No body detected" warning in live demo

The live demo displays a red "No body detected - adjust position/lighting" warning when MediaPipe cannot detect pose landmarks in the current frame. Additionally, `predict_buffer` applies a pose detection quality gate: if fewer than 30% of buffered frames have valid shoulder landmarks, inference is skipped entirely and returns `None`. This prevents garbage predictions from low-quality pose data. To resolve:

- Improve lighting -- avoid strong backlighting or very dim environments.
- Move closer to the camera so your upper body is clearly visible.
- Reduce occlusion -- ensure your torso and shoulders are not blocked.

### Low accuracy

- Check split CSV row counts to ensure enough training videos were downloaded.
- Enable `weighted_sampling: true` for class-imbalanced subsets.
- Run the error analysis notebook (`notebooks/03_error_analysis.ipynb`) to find confused class pairs.

### Diagnosing partial data (most common issue)

Many WLASL URLs have expired, so you will likely end up with far fewer usable videos than the annotation file lists. This is the single biggest factor in accuracy. Check your effective dataset size:

**Linux/macOS:**

```bash
# Row counts in split CSVs (includes videos you may not have)
wc -l data/splits/WLASL100/*.csv

# How many .npy keypoint files were actually produced
ls data/processed/*.npy | wc -l
```

**Windows (PowerShell):**

```powershell
# Row counts in split CSVs
Get-ChildItem data\splits\WLASL100\*.csv | ForEach-Object { Write-Host "$($_.Name): $((Get-Content $_).Count) lines" }

# How many .npy keypoint files were actually produced
(Get-ChildItem data\processed\*.npy).Count
```

**Cross-platform (Python):**

```bash
# Effective training samples (rows in CSV that have matching .npy files)
python -c "
import pandas as pd; from pathlib import Path
train = pd.read_csv('data/splits/WLASL100/train.csv')
npy = Path('data/processed')
eff = train[train['video_id'].apply(lambda v: (npy/f'{v}.npy').exists())]
counts = eff['label_idx'].value_counts()
print(f'Effective train: {len(eff)} samples, {eff[\"label_idx\"].nunique()} classes')
print(f'Samples/class: min={counts.min()}, mean={counts.mean():.1f}, max={counts.max()}')
print(f'Classes with <=2 samples: {(counts<=2).sum()}')
"
```

**If effective samples < 500:** Training will be very challenging. The default `configs/stgcn_ce.yaml` is tuned for this scenario (milder augmentation, no label smoothing/mixup, weighted sampling). Expect 30–50% top-1 accuracy with CE, or try `stgcn_proto` for better few-shot performance.

**To get more data:**

1. Use the Kaggle download script for the full ~12K video archive: `python scripts/download_kaggle.py`
2. Re-run preprocessing after adding new videos — already-processed `.npy` files are skipped automatically.
3. Try `--subset WLASL300` to include more glosses (you may have videos for classes outside WLASL100).

### `wlasl_variant` / `num_classes` mismatch

`num_classes` is always auto-derived from `wlasl_variant`. If your YAML says `wlasl_variant: 300` but you only preprocessed WLASL100, training will fail because `data/splits/WLASL300/train.csv` does not exist. Make sure `wlasl_variant` in your config matches the subset you preprocessed.

---

## Recommended Configurations

These are tuned starting points for each dataset variant. Copy the base config and modify:

### WLASL100 (recommended starting point)

~2,000 annotations, ~100 glosses. Expect 400–1,200 usable training samples depending on download availability. Uses **`stgcn_ce` (recommended — our best model)** by default. Architecture is auto-scaled by `Config.__post_init__`.

```yaml
approach: stgcn_ce
wlasl_variant: 100
T: 64
use_motion: true               # velocity features (position + frame differences)
normalize_embeddings: false    # CRITICAL for CE — do not enable
label_smoothing: 0.1           # mild smoothing for regularization
mixup_alpha: 0.2               # mixup augmentation for regularization
head_dropout: 0.2              # dropout in two-layer classification head
batch_size: 32
lr: 1.0e-3
scheduler: cosine
warmup_epochs: 10
weighted_sampling: true        # important — classes are imbalanced
early_stopping_patience: 30
epochs: 200
```

**Alternative: Prototypical training** (best for few-shot scenarios with ~3-8 samples/class):

```yaml
approach: stgcn_proto
wlasl_variant: 100
T: 64
use_motion: true
normalize_embeddings: true     # required for distance-based classification
batch_size: 16
lr: 1.0e-3
scheduler: cosine
warmup_epochs: 10
early_stopping_patience: 30
epochs: 200
```

With very few training videos (<500 usable), reduce batch size:

```yaml
batch_size: 16
lr: 1.0e-4
```

### WLASL300

~5,000 annotations, 300 glosses. More data per class on average. Model scales up automatically (`d_model=192`, `nhead=6`, `num_layers=4`) via `Config.__post_init__`.

```yaml
approach: stgcn_ce
wlasl_variant: 300
T: 64
normalize_embeddings: false
batch_size: 32
lr: 1.0e-3
scheduler: cosine
warmup_epochs: 10
weighted_sampling: true
early_stopping_patience: 30
epochs: 300
```

### WLASL1000

Much larger class count. Needs more model capacity and training time. Model scales up automatically (`d_model=256`, `nhead=8`, `num_layers=5`) via `Config.__post_init__`.

```yaml
approach: stgcn_ce
wlasl_variant: 1000
T: 64
normalize_embeddings: false
batch_size: 64
lr: 1.0e-3
scheduler: cosine
warmup_epochs: 10
weighted_sampling: true
early_stopping_patience: 30
epochs: 350
```

### WLASL2000

Largest variant. Model scales up to `d_model=384`, `nhead=8`, `num_layers=6` via `Config.__post_init__`.

```yaml
approach: stgcn_ce
wlasl_variant: 2000
T: 64
normalize_embeddings: false
batch_size: 64
lr: 1.0e-3
scheduler: cosine
warmup_epochs: 10
weighted_sampling: true
early_stopping_patience: 30
epochs: 400
```

---

## Tips & Best Practices

### Hardware-Specific Setup

**CPU-only (no GPU)**

```yaml
fp16: false                # FP16 only works on CUDA
batch_size: 8              # smaller batches to avoid memory pressure
num_workers: 2
```

Stick to keypoint-based approaches (`stgcn_ce` or `stgcn_proto`). They are lightweight and run well on CPU.

**GPU / CUDA**

```yaml
fp16: true
batch_size: 32             # increase to fill GPU memory (or 64 for large GPUs)
num_workers: 4
```

Monitor GPU memory with `nvidia-smi`. If you run out of memory, reduce `batch_size` first, then `T`.

**Apple Silicon (M1/M2/M3)**

- Install MediaPipe: `pip install mediapipe-silicon`
- MPS backend is supported by PyTorch but gains over CPU are inconsistent for these model sizes. Use `--device cpu` for the live demo to avoid MPS overhead.
- Preprocessing already uses `spawn` multiprocessing context to avoid macOS fork crashes.

### Training with Limited Data

WLASL's expired URLs mean you may only get 30–60% of the annotated videos. When your training set is small (<500 samples for 100 classes):

1. **Enable weighted sampling** (`weighted_sampling: true`) — ensures every class is seen equally despite imbalance.
2. **Use smaller batch sizes** (8–16) so the model sees more update steps per epoch.
3. **Try prototypical training** (`approach: stgcn_proto`) — designed for few-shot scenarios with ~3-8 samples/class.
4. **Consider reducing label smoothing and mixup** (`label_smoothing: 0.0`, `mixup_alpha: 0.0`) if your dataset is very small (<100 samples).
5. **Download from Kaggle** (`python scripts/download_kaggle.py`) — the full ~12K video archive is available as a single download.

### Improving Accuracy

- **Start with `stgcn_ce` (recommended)** — our best model. It trains fastest and is easiest to debug.
- **Enable motion features** (`use_motion: true`) — velocity information captures signing dynamics and typically adds 5–8% accuracy.
- **Reduce or disable label smoothing and mixup** for very small datasets — they can be counterproductive with fewer than ~100 total samples.
- **Set `normalize_embeddings: false` for CE** — L2 normalization caps logit magnitudes and causes loss plateaus.
- **Enable TTA for evaluation** (`use_tta: true`) — averages predictions over original + horizontally flipped input for 2–4% evaluation boost.
- **Use the error analysis notebook** (`notebooks/03_error_analysis.ipynb`) to find which classes are confused, then inspect those videos manually.
- **Use cosine scheduler** (`scheduler: cosine`) — cosine annealing with warm-up converges better than onecycle for long training runs.
- **Increase sequence length** (`T: 96` or `T: 128`) if signs in your dataset are long.

### What to Expect

Realistic accuracy ranges depend heavily on how many videos you have:


| Dataset  | Training Samples | stgcn_ce (expected) | stgcn_proto (expected) |
| -------- | ---------------- | ------------------- | ---------------------- |
| WLASL100 | 400–800          | 45–55% top-1        | ~100% (few-shot)       |
| WLASL100 | 800–1,500        | 55–65% top-1        | ~100% (few-shot)       |
| WLASL300 | 2,000–4,000      | 40–50% top-1        | varies by N-way        |


If your val loss is around `ln(num_classes)` (e.g., 4.6 for 100 classes), the model is near random — check that enough training samples are being loaded.

### Common Pitfalls

1. **HTML videos**: Run `scripts/validate_videos.py --delete` before preprocessing. Expired WLASL URLs return HTML pages saved as `.mp4` files.
2. **Corrupt videos**: Some downloads are truncated (OpenCV reports "moov atom not found"). The preprocessing pipeline skips these automatically, but they inflate your file count.
3. **Wrong video count**: The official WLASL download scripts fetch ALL ~21,000 videos (all 2,000 glosses), not just your target variant. The preprocessing pipeline filters to the correct subset.
4. **MediaPipe warnings**: `inference_feedback_manager.cc` warnings are harmless TFLite logs. Suppress with `GLOG_minloglevel=2 python ...` (Linux/macOS) or `$env:GLOG_minloglevel=2; python ...` (Windows PowerShell) or `set GLOG_minloglevel=2 && python ...` (Windows cmd).
5. **Empty val set**: If the val split CSV has very few rows, some classes may have zero val samples. This makes early stopping and accuracy metrics unreliable — check row counts after preprocessing: `wc -l data/splits/WLASL100/*.csv` (Linux/macOS) or `Get-ChildItem data\splits\WLASL100\*.csv | ForEach-Object { Write-Host "$($_.Name): $((Get-Content $_).Count)" }` (Windows PowerShell).
6. **OneCycleLR NaN loss**: If you resume training from a checkpoint with a different total step count, the scheduler can go out of range. Start fresh or use `scheduler: cosine` for resumed runs.

---

## Recommended Library & CUDA Versions

This project requires `torch>=2.1.0,<2.5.0`. The table below shows which CUDA toolkit versions are compatible with each PyTorch release, and the matching torchvision / torchaudio versions.

### PyTorch ↔ CUDA Compatibility


| PyTorch | torchvision | torchaudio | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | Install command                                                                                                     |
| ------- | ----------- | ---------- | --------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------- |
| 2.4.1   | 0.19.1      | 2.4.1      | Yes       | Yes       | Yes       | `pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124` |
| 2.3.1   | 0.18.1      | 2.3.1      | Yes       | Yes       | No        | `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121` |
| 2.2.2   | 0.17.2      | 2.2.2      | Yes       | Yes       | No        | `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121` |
| 2.1.2   | 0.16.2      | 2.1.2      | Yes       | Yes       | No        | `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121` |


**How to choose:**

- **CUDA 12.4** — Use PyTorch 2.4.x. Best for RTX 30xx/40xx/50xx with recent drivers.
- **CUDA 12.1** — Supported by all versions above. Safe default for most setups.
- **CUDA 11.8** — Supported by all versions above. Use if your driver is older or you're on a shared cluster.
- **CPU only** — Append `--index-url https://download.pytorch.org/whl/cpu` to any install command.

Check your CUDA version with `nvidia-smi` (top-right corner shows the maximum CUDA version your driver supports).

> **Note:** The `--index-url` flag must match your CUDA version, not your PyTorch version. If you install the wrong CUDA variant, `torch.cuda.is_available()` will return `False`.

### MediaPipe


| Platform              | Package             | Version Range        | Install command                                        |
| --------------------- | ------------------- | -------------------- | ------------------------------------------------------ |
| Linux / Windows       | `mediapipe`         | `>=0.10.7,<=0.10.14` | `pip install mediapipe` (included in requirements.txt) |
| macOS (Apple Silicon) | `mediapipe-silicon` | `>=0.10.7`           | `pip install mediapipe-silicon`                        |
| macOS (Intel)         | `mediapipe`         | `>=0.10.7,<=0.10.14` | `pip install mediapipe`                                |


- MediaPipe is compatible with **Python 3.9–3.12**.
- On Apple Silicon, the standard `mediapipe` package may fail to install. Use `mediapipe-silicon` instead — it provides the same API.
- On **Windows with Python 3.12**, some mediapipe versions expose `solutions` under `mediapipe.python.solutions` instead of `mediapipe.solutions`. If you hit this issue, pin to `mediapipe==0.10.11`.
- Both packages provide `mediapipe.solutions.holistic` used by the preprocessing pipeline.

### Other Key Libraries


| Library          | Required Version   | Notes                                                                                                   |
| ---------------- | ------------------ | ------------------------------------------------------------------------------------------------------- |
| `opencv-python`  | `>=4.8.0,<4.11.0`  | Video I/O and frame capture. Webcam access requires system camera permissions.                          |
| `numpy`          | `>=1.24.0,<2.1.0`  | NumPy 2.x introduced breaking changes — stay below 2.1 for compatibility with all dependencies.         |
| `onnxruntime`    | `>=1.16.0,<1.20.0` | For ONNX export verification. Use `onnxruntime-gpu` instead if you want GPU-accelerated ONNX inference. |
| `pytorchvideo`   | `>=0.1.5,<0.2.0`   | SlowFast and X3D backbones (Approach B). Only needed if using video models.                             |
| `albumentations` | `>=1.3.1,<1.5.0`   | Image augmentations for video frame preprocessing (Approach B/C).                                       |
| `kaggle`         | `>=1.6.0,<1.8.0`   | Kaggle API for dataset download. Only needed if using `scripts/download_kaggle.py`.                     |


For the full list of dependencies with version ranges, see `[requirements.txt](requirements.txt)`.

Source: [PyTorch CUDA Compatibility Matrix](https://github.com/eminsafa/pytorch-cuda-compatibility) | [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

---

## Citation

```bibtex
@inproceedings{li2020word,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1459--1469},
  year={2020}
}
```

## License

This project is for educational and research purposes. The WLASL dataset has its own licensing terms — check the [official repository](https://github.com/dxli94/WLASL) before commercial use.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
