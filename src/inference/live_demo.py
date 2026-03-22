"""
Real-time webcam inference for ASL sign language recognition.

This is the main deliverable of the project.  It opens a webcam feed,
runs MediaPipe Holistic on each frame to extract keypoints, buffers a
rolling window of T frames, runs the trained model every inference
interval, and displays the predicted sign with confidence overlaid on
the video feed.

Architecture:
    - Capture thread: reads webcam frames at full speed
    - Inference thread: runs model on the buffered keypoints periodically
    - Main thread: renders the display overlay and handles user input

Press 'q' to quit.  Press 's' to save the current prediction to a log.
"""

import argparse
import collections
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.augment import KeypointHorizontalFlip, get_val_transforms
from src.data.preprocess import (
    NUM_KEYPOINTS,
    _import_mediapipe_drawing,
    _import_mediapipe_holistic,
    normalize_keypoints,
)
from src.models import build_model
from src.training.config import Config, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Thread-safe fixed-size deque for storing recent keypoint frames.

    Parameters
    ----------
    max_size : int
        Maximum number of frames to retain.
    """

    def __init__(self, max_size: int = 64) -> None:
        self.max_size = max_size
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        """Add a keypoint frame to the buffer (thread-safe)."""
        with self._lock:
            self._buffer.append(frame)

    def get_all(self) -> np.ndarray:
        """Return all buffered frames as a single NumPy array.

        Returns
        -------
        np.ndarray
            Shape ``(N, NUM_KEYPOINTS, 3)`` where N <= max_size.
            Returns an empty array if buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
            return np.stack(list(self._buffer), axis=0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Remove all frames from the buffer."""
        with self._lock:
            self._buffer.clear()

    def trim_to(self, n: int) -> None:
        """Keep only the most recent *n* frames, discarding older ones.

        Used for pre-trigger buffering: when signing starts, retain a few
        seconds of pre-sign context (sign onset) instead of clearing
        everything. If the buffer has fewer than *n* frames, nothing
        is removed.

        Parameters
        ----------
        n : int
            Maximum number of frames to retain.
        """
        with self._lock:
            while len(self._buffer) > n:
                self._buffer.popleft()


# ---------------------------------------------------------------------------
# Motion detector
# ---------------------------------------------------------------------------


class MotionDetector:
    """Detects sign start/end by tracking hand keypoint velocity.

    Uses a state machine (IDLE -> SIGNING -> COMPLETED) to determine
    when a sign has been completed before triggering inference.

    All velocity comparisons are in **normalized-coordinates per second**,
    making them independent of camera FPS.  The detector measures the
    actual time delta between ``update()`` calls and divides displacement
    by dt to get a consistent velocity regardless of whether the camera
    runs at 30fps, 60fps, or anything in between.

    Parameters
    ----------
    cfg : Config
        Configuration with motion detection thresholds (per-second units).
    """

    # Hand keypoint indices in the 543-keypoint array (left: 33-53, right: 54-74)
    HAND_START = 33
    HAND_END = 75

    def __init__(self, cfg: "Config") -> None:
        self.start_threshold = cfg.motion_start_threshold  # norm-coords/sec
        self.end_threshold = cfg.motion_end_threshold  # norm-coords/sec
        self.settle_duration = getattr(cfg, "motion_settle_time", 0.27)  # seconds
        self.max_duration = cfg.max_sign_duration  # seconds

        self._state = "IDLE"
        self._prev_hand_kps: Optional[np.ndarray] = None
        self._last_update_time: Optional[float] = None
        self._signing_elapsed: float = 0.0
        self._settle_elapsed: float = 0.0

    @property
    def state(self) -> str:
        """Current state: IDLE, SIGNING, or COMPLETED."""
        return self._state

    def _hand_displacement(self, keypoints: np.ndarray) -> float:
        """Compute mean L2 displacement of hand keypoints from previous frame.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape ``(NUM_KEYPOINTS, 3)`` for a single frame.

        Returns
        -------
        float
            Mean L2 displacement across the 42 hand keypoints
            (normalized-coordinate units, NOT per second).
        """
        hand_kps = keypoints[self.HAND_START:self.HAND_END]

        # Skip frames where hands weren't detected (all zeros) to avoid
        # false velocity spikes that incorrectly trigger SIGNING state.
        hand_norm = np.linalg.norm(hand_kps, axis=1).sum()
        if hand_norm < 1e-6:
            self._prev_hand_kps = None
            return 0.0

        if self._prev_hand_kps is None:
            self._prev_hand_kps = hand_kps.copy()
            return 0.0
        displacement = np.linalg.norm(hand_kps - self._prev_hand_kps, axis=1)
        self._prev_hand_kps = hand_kps.copy()
        return float(np.mean(displacement))

    def update(self, keypoints: np.ndarray, dt: Optional[float] = None) -> str:
        """Ingest a new frame and return the current state.

        Measures wall-clock time between calls to convert displacement
        into velocity (norm-coords/second), so thresholds behave
        identically on any camera FPS.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape ``(NUM_KEYPOINTS, 3)`` for the latest frame.
        dt : float or None
            Override time delta in seconds (for testing).  When None,
            wall-clock time is measured automatically.

        Returns
        -------
        str
            Current state after processing: IDLE, SIGNING, or COMPLETED.
        """
        if dt is None:
            now = time.monotonic()
            if self._last_update_time is not None:
                dt = now - self._last_update_time
            else:
                dt = 0.0
            self._last_update_time = now

        # Clamp dt to avoid spikes from pauses or first-frame artifacts
        dt = max(min(dt, 0.5), 0.0)

        displacement = self._hand_displacement(keypoints)

        # Convert displacement-per-frame to velocity-per-second
        vel = displacement / dt if dt > 0 else 0.0

        if self._state == "IDLE":
            if vel >= self.start_threshold:
                self._state = "SIGNING"
                self._signing_elapsed = 0.0
                self._settle_elapsed = 0.0

        elif self._state == "SIGNING":
            self._signing_elapsed += dt
            if vel < self.end_threshold:
                self._settle_elapsed += dt
            else:
                self._settle_elapsed = 0.0

            if self._settle_elapsed >= self.settle_duration:
                self._state = "COMPLETED"
            elif self._signing_elapsed >= self.max_duration:
                self._state = "COMPLETED"

        return self._state

    def reset(self) -> None:
        """Reset to IDLE state and clear all history."""
        self._state = "IDLE"
        self._prev_hand_kps = None
        self._signing_elapsed = 0.0
        self._settle_elapsed = 0.0


# ---------------------------------------------------------------------------
# Live predictor
# ---------------------------------------------------------------------------


class LivePredictor:
    """Manages the model and performs inference on buffered keypoint sequences.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the model checkpoint.
    cfg : Config
        Configuration.
    device : str
        Device for inference.
    class_names : list[str] or None
        Gloss names indexed by label.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: Config,
        device: str = "cpu",
        class_names: Optional[list[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.class_names = class_names or [str(i) for i in range(cfg.num_classes)]

        # Build model based on configured approach
        self.model = build_model(cfg)
        ckpt = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )
        state_dict = ckpt["model_state_dict"]
        state_dict.pop("prototypes", None)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Compute prototypes for prototypical models
        if hasattr(self.model, 'compute_prototypes'):
            self._load_prototypes(cfg)

        self._use_classify = hasattr(self.model, 'classify')
        self.transform = get_val_transforms(T=cfg.T)
        self._use_tta = getattr(cfg, "use_tta", False)
        self._hflip = KeypointHorizontalFlip(p=1.0, centered=True)

        # MediaPipe — uses shared helper that handles Windows/Python 3.12 fallback
        self._mp_holistic = _import_mediapipe_holistic()
        holistic_mod, drawing_mod, styles_mod = _import_mediapipe_drawing()
        self._mp_drawing = drawing_mod
        self._mp_drawing_styles = styles_mod
        self.holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        logger.info("LivePredictor initialized (device=%s)", self.device)

    def _load_prototypes(self, cfg: Config) -> None:
        """Compute prototypes from the training set for prototypical models."""
        from src.data.augment import get_val_transforms
        from src.data.dataset import WLASLKeypointDataset, get_dataloader

        data_dir = Path(cfg.data_dir)
        train_csv = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / "train.csv"
        if not train_csv.exists():
            logger.warning("Training CSV not found at %s; skipping prototype computation", train_csv)
            return
        transform = get_val_transforms(T=cfg.T)
        train_dataset = WLASLKeypointDataset(
            split_csv=train_csv,
            keypoint_dir=data_dir / "processed",
            transform=transform,
            T=cfg.T,
            use_motion=getattr(cfg, "use_motion", False),
        )
        proto_loader = get_dataloader(
            train_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=getattr(cfg, "num_workers", 0),
        )
        self.model.compute_prototypes(proto_loader)
        logger.info("Prototypes computed from training set")

    def preprocess_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, object]:
        """Extract MediaPipe keypoints from a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR frame from OpenCV.

        Returns
        -------
        tuple[np.ndarray, object]
            Keypoint array of shape ``(NUM_KEYPOINTS, 3)`` and the
            MediaPipe results object (for drawing landmarks).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        kps = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        offset = 0

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += 33

        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += 21

        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]
        offset += 21

        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                kps[offset + i] = [lm.x, lm.y, lm.z]

        return kps, results

    def predict_buffer(self, buffer: FrameBuffer) -> Optional[dict]:
        """Run model inference on the current buffer contents.

        Parameters
        ----------
        buffer : FrameBuffer
            The rolling keypoint buffer.

        Returns
        -------
        dict or None
            Prediction result or None if buffer is too short.
        """
        keypoints = buffer.get_all()  # (N, NUM_KEYPOINTS, 3)

        min_frames = getattr(self.cfg, "min_buffer_frames", 30)
        if keypoints.shape[0] < min_frames:
            logger.debug("Buffer too short: %d < %d", keypoints.shape[0], min_frames)
            return None

        # Check pose detection quality: shoulders (landmarks 11, 12) must be
        # detected for normalize_keypoints to produce valid output.  Without
        # shoulders, centering/scaling is undefined and the model gets garbage.
        shoulder_kps = keypoints[:, [11, 12], :]  # (N, 2, 3)
        shoulder_norms = np.linalg.norm(shoulder_kps.reshape(keypoints.shape[0], -1), axis=1)
        pose_detected_frac = float(np.mean(shoulder_norms > 1e-6))
        if pose_detected_frac < 0.3:
            logger.debug(
                "Poor pose detection: %.0f%% of frames have shoulders — skipping inference",
                pose_detected_frac * 100,
            )
            return None

        # Check hand detection quality
        hand_kps = keypoints[:, 33:75, :]  # left + right hand
        hand_norms = np.linalg.norm(hand_kps.reshape(keypoints.shape[0], -1), axis=1)
        hand_detected_frac = float(np.mean(hand_norms > 1e-6))
        logger.debug(
            "Buffer: %d frames, pose: %.0f%%, hands: %.0f%%",
            keypoints.shape[0], pose_detected_frac * 100, hand_detected_frac * 100,
        )

        # Normalize
        keypoints = normalize_keypoints(keypoints)

        # Drop face landmarks: keep only pose (33) + left hand (21) + right hand (21) = 75
        keypoints = keypoints[:, :75, :]

        # Apply val transforms (temporal crop/pad to T frames) — matches eval pipeline
        keypoints = self.transform(keypoints)

        # Ensure 3D: (T, K, 3)
        if keypoints.ndim == 2:
            T_actual = keypoints.shape[0]
            keypoints = keypoints.reshape(T_actual, -1, 3)

        # Compute velocity if use_motion is enabled
        if getattr(self.cfg, "use_motion", False):
            velocity = np.zeros_like(keypoints)
            velocity[1:] = keypoints[1:] - keypoints[:-1]
            keypoints = np.concatenate([keypoints, velocity], axis=-1)  # (T, 75, 6)

        # Flatten and convert to tensor
        T = keypoints.shape[0]
        keypoints_flat = keypoints.reshape(T, -1)  # (T, 75*C)
        tensor = torch.from_numpy(keypoints_flat).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self._use_classify:
                logits = self.model.classify(tensor)
            else:
                logits = self.model(tensor)

            # Test-Time Augmentation: average logits with horizontal flip
            if self._use_tta and keypoints.ndim == 3:
                C_feat = keypoints.shape[2]
                if C_feat == 6:
                    # Split position/velocity, flip each separately (match evaluate.py)
                    pos = keypoints[:, :, :3].copy()
                    vel = keypoints[:, :, 3:].copy()
                    pos_flip = self._hflip(pos)
                    vel_flip = self._hflip(vel)
                    kps_flipped = np.concatenate([pos_flip, vel_flip], axis=-1)
                else:
                    kps_flipped = self._hflip(keypoints.copy())
                flipped_flat = kps_flipped.reshape(T, -1)
                flipped_tensor = torch.from_numpy(flipped_flat).float().unsqueeze(0).to(self.device)
                if self._use_classify:
                    logits_flip = self.model.classify(flipped_tensor)
                else:
                    logits_flip = self.model(flipped_tensor)
                logits = (logits + logits_flip) / 2.0

            probs = F.softmax(logits, dim=1).squeeze(0)

        top5_probs, top5_indices = probs.topk(5)
        top5_probs = top5_probs.cpu().numpy()
        top5_indices = top5_indices.cpu().numpy()

        pred_idx = int(top5_indices[0])
        confidence = float(top5_probs[0])
        gloss = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
        top5 = [
            (
                self.class_names[int(i)] if int(i) < len(self.class_names) else str(i),
                float(p),
            )
            for i, p in zip(top5_indices, top5_probs)
        ]

        return {
            "gloss": gloss,
            "confidence": confidence,
            "label_idx": pred_idx,
            "top5": top5,
        }

    @staticmethod
    def smooth_predictions(
        recent_preds: list[dict],
        mode: str = "avg",
    ) -> Optional[dict]:
        """Smooth recent predictions to reduce flickering.

        Parameters
        ----------
        recent_preds : list[dict]
            List of recent prediction dicts.
        mode : str
            ``'avg'`` for probability averaging, ``'majority'`` for
            majority vote.

        Returns
        -------
        dict or None
            Smoothed prediction.
        """
        if not recent_preds:
            return None

        if mode == "majority":
            # Majority vote on the top-1 prediction
            votes = [p["gloss"] for p in recent_preds]
            counter = collections.Counter(votes)
            winner, count = counter.most_common(1)[0]
            # Find the most recent prediction with this gloss for full info
            for p in reversed(recent_preds):
                if p["gloss"] == winner:
                    return {**p, "confidence": count / len(votes)}
            return recent_preds[-1]

        else:  # avg
            # Average the top-5 probabilities across all prediction windows.
            # Glosses absent from a window's top-5 are treated as 0 probability,
            # so we divide by the total number of windows (not just appearances).
            n_windows = len(recent_preds)
            gloss_probs: dict[str, float] = {}
            for p in recent_preds:
                for gloss, prob in p["top5"]:
                    gloss_probs[gloss] = gloss_probs.get(gloss, 0.0) + prob

            # Compute mean probability for each gloss
            avg_probs = {g: total / n_windows for g, total in gloss_probs.items()}
            sorted_glosses = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)

            top5 = sorted_glosses[:5]
            winner = top5[0][0]
            confidence = top5[0][1]

            # Find label_idx from the most recent prediction
            label_idx = recent_preds[-1].get("label_idx", 0)
            for p in recent_preds:
                if p["gloss"] == winner:
                    label_idx = p["label_idx"]
                    break

            return {
                "gloss": winner,
                "confidence": float(confidence),
                "label_idx": label_idx,
                "top5": [(g, float(p)) for g, p in top5],
            }


# ---------------------------------------------------------------------------
# Display overlay
# ---------------------------------------------------------------------------


class ASLDisplay:
    """Handles drawing prediction overlays on the webcam feed."""

    # Colors (BGR)
    BG_COLOR = (40, 40, 40)
    TEXT_COLOR = (255, 255, 255)
    ACCENT_COLOR = (0, 200, 100)
    CONF_BAR_BG = (80, 80, 80)
    CONF_BAR_FG = (0, 200, 100)
    LOW_CONF_FG = (0, 100, 200)

    @staticmethod
    def draw_overlay(
        frame: np.ndarray,
        prediction: Optional[dict],
        confidence: float,
        top5: Optional[list[tuple[str, float]]],
        mp_results: Optional[object] = None,
    ) -> np.ndarray:
        """Draw the prediction overlay on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on (modified in place).
        prediction : dict or None
            Current prediction dict.
        confidence : float
            Confidence score for display.
        top5 : list or None
            Top-5 predictions.
        mp_results : object or None
            MediaPipe results for drawing landmarks.

        Returns
        -------
        np.ndarray
            The annotated frame.
        """
        h, w = frame.shape[:2]

        # Draw MediaPipe landmarks if available
        if mp_results is not None:
            mp_holistic, mp_drawing, _ = _import_mediapipe_drawing()
            if mp_drawing is None:
                return frame  # Skip landmark drawing if mediapipe unavailable
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            if mp_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    mp_results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(100, 200, 100), thickness=1
                    ),
                )
            if mp_results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    mp_results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(200, 100, 100), thickness=2
                    ),
                )
            if mp_results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    mp_results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(100, 100, 200), thickness=2
                    ),
                )

        # Semi-transparent overlay panel at the top
        overlay = frame.copy()
        panel_h = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_h), ASLDisplay.BG_COLOR, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        if prediction is not None and confidence > 0:
            gloss = prediction.get("gloss", "---")

            # Main prediction text
            cv2.putText(
                frame,
                gloss.upper(),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                ASLDisplay.ACCENT_COLOR,
                3,
                cv2.LINE_AA,
            )

            # Confidence bar
            bar_x, bar_y = 20, 70
            bar_w, bar_h = 200, 20
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_w, bar_y + bar_h),
                ASLDisplay.CONF_BAR_BG,
                -1,
            )
            fill_w = int(bar_w * confidence)
            bar_color = ASLDisplay.CONF_BAR_FG if confidence > 0.6 else ASLDisplay.LOW_CONF_FG
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_w, bar_y + bar_h),
                bar_color,
                -1,
            )
            cv2.putText(
                frame,
                f"{confidence:.0%}",
                (bar_x + bar_w + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ASLDisplay.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            # Top-5 predictions (right side)
            if top5 is not None:
                x_start = w - 280
                for i, (g, p) in enumerate(top5):
                    y_pos = 25 + i * 20
                    text = f"{i + 1}. {g}: {p:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (x_start, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        ASLDisplay.TEXT_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
        else:
            cv2.putText(
                frame,
                "Waiting for sign...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (150, 150, 150),
                2,
                cv2.LINE_AA,
            )

        return frame


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------


def run_demo(
    cfg: Config,
    checkpoint_path: str | Path,
    camera_id: int = 0,
    device: str = "cpu",
) -> None:
    """Run the real-time ASL recognition demo.

    Opens a webcam, extracts keypoints per frame, buffers them,
    and periodically runs inference with smooth display updates.

    Parameters
    ----------
    cfg : Config
        Configuration.
    checkpoint_path : str or Path
        Path to the model checkpoint.
    camera_id : int
        OpenCV camera device ID.
    device : str
        Inference device.
    """
    # Load class names
    import pandas as pd

    class_names = [str(i) for i in range(cfg.num_classes)]
    data_dir = Path(cfg.data_dir)
    for split in ["train", "val"]:
        csv_path = data_dir / "splits" / f"WLASL{cfg.wlasl_variant}" / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                idx = int(row["label_idx"])
                if idx < cfg.num_classes:
                    class_names[idx] = row["gloss"]
            break

    # Open webcam early so we can query its FPS for buffer sizing
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", camera_id)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    if camera_fps <= 0:
        camera_fps = 30.0  # fallback if driver doesn't report FPS
    logger.info("Camera FPS: %.1f", camera_fps)

    pre_sign_dur = getattr(cfg, "pre_sign_duration", 0.5)
    pre_sign_frames = int(pre_sign_dur * camera_fps)
    logger.info("Pre-sign buffer: %d frames (%.1fs)", pre_sign_frames, pre_sign_dur)

    # Initialize components
    predictor = LivePredictor(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        device=device,
        class_names=class_names,
    )
    # Buffer holds max_sign_duration seconds of frames.  TemporalCrop then
    # uniformly samples down to T, matching the training pipeline.
    max_sign_sec = getattr(cfg, "max_sign_duration", 3.0)
    buffer_frames = int(max_sign_sec * camera_fps) + 10  # small margin
    buffer = FrameBuffer(max_size=buffer_frames)
    display = ASLDisplay()

    # State
    current_prediction: Optional[dict] = None
    current_confidence: float = 0.0
    current_top5: Optional[list] = None
    current_motion_state: str = "IDLE"
    recent_predictions: list[dict] = []
    last_mp_results: Optional[object] = None
    inference_lock = threading.Lock()
    motion_lock = threading.Lock()
    running = True
    saved_predictions: list[str] = []

    # Motion detector
    motion_detector = MotionDetector(cfg)

    # FPS tracking
    frame_times: collections.deque[float] = collections.deque(maxlen=30)

    # Prediction display expiry
    prediction_time: float = 0.0
    prediction_display_timeout: float = 3.0  # seconds before clearing stale prediction

    # Inference thread — motion-aware with cooldown
    def inference_loop() -> None:
        nonlocal current_prediction, current_confidence, current_top5, recent_predictions
        nonlocal prediction_time
        cooldown_until = 0.0
        poll_interval = getattr(cfg, "inference_poll_interval", 0.1)

        while running:
            time.sleep(poll_interval)
            if not running:
                break

            now = time.time()

            # Respect cooldown after a confident prediction
            if now < cooldown_until:
                continue

            # Read motion state (set by main thread)
            with motion_lock:
                state = current_motion_state

            # Only infer after the user has actually signed: the state
            # machine must go IDLE → SIGNING → COMPLETED before we run
            # the model.  This prevents spurious predictions on idle frames.
            if state != "COMPLETED":
                continue

            result = predictor.predict_buffer(buffer)
            if result is None:
                continue

            with inference_lock:
                recent_predictions.append(result)
                if len(recent_predictions) > cfg.smoothing_window:
                    recent_predictions = recent_predictions[-cfg.smoothing_window:]

                smoothed = predictor.smooth_predictions(recent_predictions, mode="avg")

                if smoothed is not None and smoothed["confidence"] >= cfg.confidence_threshold:
                    # High confidence — commit prediction and reset for next sign
                    current_prediction = smoothed
                    current_confidence = smoothed["confidence"]
                    current_top5 = smoothed.get("top5")
                    prediction_time = now

                    buffer.clear()
                    with motion_lock:
                        motion_detector.reset()
                    recent_predictions.clear()
                    cooldown_until = now + getattr(cfg, "prediction_cooldown", 1.0)

                else:
                    # Low confidence on a completed/idle sign — show result
                    # but use a shorter cooldown so the user can retry sooner.
                    if smoothed is not None:
                        current_prediction = smoothed
                        current_confidence = smoothed["confidence"]
                        current_top5 = smoothed.get("top5")
                        prediction_time = now
                    buffer.clear()
                    with motion_lock:
                        motion_detector.reset()
                    recent_predictions.clear()
                    cooldown_until = now + getattr(cfg, "prediction_cooldown", 1.0) * 0.3

    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    inference_thread.start()

    print("ASL Recognition Demo started. Press 'q' to quit, 's' to save prediction.")

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            t0 = time.perf_counter()

            # Extract keypoints and feed motion detector
            kps, mp_results = predictor.preprocess_frame(frame)
            with motion_lock:
                prev_state = motion_detector.state
                m_state = motion_detector.update(kps)
                current_motion_state = m_state
                # Trim idle frames when signing begins — keep only
                # pre_sign_frames of context so the model sees sign onset
                # (preparation phase). Training data includes full videos
                # with prep frames; discarding them all creates a mismatch.
                if prev_state == "IDLE" and m_state == "SIGNING":
                    buffer.trim_to(pre_sign_frames)
            buffer.push(kps)
            last_mp_results = mp_results

            # Get current prediction (thread-safe read)
            with inference_lock:
                # Expire stale predictions so display reverts to "Waiting..."
                if (current_prediction is not None
                        and prediction_time > 0
                        and (time.time() - prediction_time) > prediction_display_timeout):
                    current_prediction = None
                    current_confidence = 0.0
                    current_top5 = None
                pred_copy = current_prediction
                conf_copy = current_confidence
                top5_copy = current_top5

            # Draw overlay
            frame = display.draw_overlay(
                frame,
                pred_copy,
                conf_copy,
                top5_copy,
                last_mp_results,
            )

            # FPS counter
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            if cfg.fps_display and len(frame_times) > 1:
                avg_time = np.mean(list(frame_times))
                fps = 1.0 / max(avg_time, 1e-6)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (frame.shape[1] - 120, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Body detection warning
            if mp_results is not None and not mp_results.pose_landmarks:
                cv2.putText(
                    frame,
                    "No body detected - adjust position/lighting",
                    (20, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Buffer status and motion state
            buf_len = len(buffer)
            state_colors = {
                "IDLE": (150, 150, 150),
                "SIGNING": (0, 200, 255),
                "COMPLETED": (0, 255, 0),
            }
            state_color = state_colors.get(m_state, (150, 150, 150))
            cv2.putText(
                frame,
                f"Buffer: {buf_len}/{cfg.buffer_size}  [{m_state}]",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                state_color,
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if pred_copy is not None:
                    entry = f"{pred_copy['gloss']} ({conf_copy:.2f})"
                    saved_predictions.append(entry)
                    print(f"Saved: {entry}")

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        predictor.holistic.close()

        if saved_predictions:
            print("\nSaved predictions:")
            for entry in saved_predictions:
                print(f"  - {entry}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live ASL Recognition Demo")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    run_demo(cfg, args.checkpoint, camera_id=args.camera, device=args.device)
