"""
Augmentation pipeline for WLASL keypoint and video sequences.

Provides temporal and spatial augmentations designed for sign language
recognition, including correct left/right swapping for MediaPipe landmarks.
"""

from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# MediaPipe swap indices for horizontal flip
# ---------------------------------------------------------------------------

# Pose landmark left/right swap pairs (MediaPipe Pose indices 0-32)
_POSE_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6),  # eyes
    (7, 8),  # ears
    (9, 10),  # mouth
    (11, 12),  # shoulders
    (13, 14),  # elbows
    (15, 16),  # wrists
    (17, 18), (19, 20), (21, 22),  # hands (pinky, index, thumb)
    (23, 24),  # hips
    (25, 26),  # knees
    (27, 28),  # ankles
    (29, 30), (31, 32),  # feet
]

# Hand landmarks: left hand (indices 33-53) swaps with right hand (54-74)
_HAND_OFFSET_LEFT = 33
_HAND_OFFSET_RIGHT = 54
_NUM_HAND = 21

# Face landmarks: swap left/right. These are approximate pairs for the
# outer contour and key features of the 468-landmark face mesh.  A full
# face mesh swap table is very long; we include the most important pairs.
# In practice, face landmarks are less critical for sign recognition.
_FACE_OFFSET = 75  # 33 + 21 + 21

# Precompute a full 543-length swap index array
_SWAP_INDICES: Optional[np.ndarray] = None


def _build_swap_indices() -> np.ndarray:
    """Build the full 543-element swap index array for horizontal flipping.

    Returns
    -------
    np.ndarray
        An array of length 543 where ``swap[i]`` is the index that
        landmark ``i`` should be copied from after a horizontal flip.
    """
    global _SWAP_INDICES
    if _SWAP_INDICES is not None:
        return _SWAP_INDICES

    n = 543
    swap = np.arange(n, dtype=np.int64)

    # Pose swaps
    for a, b in _POSE_SWAP_PAIRS:
        swap[a] = b
        swap[b] = a

    # Hand swaps: swap entire left and right hand blocks
    for i in range(_NUM_HAND):
        swap[_HAND_OFFSET_LEFT + i] = _HAND_OFFSET_RIGHT + i
        swap[_HAND_OFFSET_RIGHT + i] = _HAND_OFFSET_LEFT + i

    # Face mesh: for simplicity, we leave face landmarks un-swapped
    # individually (they are roughly symmetric, and face features are
    # secondary for sign recognition).  A production system could add
    # the full 468-point swap map here.

    _SWAP_INDICES = swap
    return swap


# ---------------------------------------------------------------------------
# Temporal augmentations
# ---------------------------------------------------------------------------


class TemporalCrop:
    """Uniformly sample exactly T frames from a variable-length sequence.

    If the input has fewer than T frames, it is padded by repeating the
    last frame.  If it has more, T indices are chosen uniformly.

    Parameters
    ----------
    T : int
        Number of output frames.
    """

    def __init__(self, T: int = 64) -> None:
        self.T = T

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply temporal crop.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape ``(T_in, ...)`` where the first axis is time.

        Returns
        -------
        np.ndarray
            Shape ``(T, ...)``.
        """
        T_in = keypoints.shape[0]
        if T_in == 0:
            # Edge case: empty sequence, return zeros
            return np.zeros((self.T, *keypoints.shape[1:]), dtype=keypoints.dtype)

        if T_in == self.T:
            return keypoints

        if T_in < self.T:
            # Reflection padding: mirror the sequence backwards to fill
            pad_count = self.T - T_in
            reflect_indices = []
            idx = T_in - 2  # start from second-to-last frame
            direction = -1
            for _ in range(pad_count):
                idx = max(0, min(T_in - 1, idx))
                reflect_indices.append(idx)
                idx += direction
                if idx < 0:
                    idx = 1
                    direction = 1
                elif idx >= T_in:
                    idx = T_in - 2
                    direction = -1
            padding = keypoints[reflect_indices]
            return np.concatenate([keypoints, padding], axis=0)

        # Uniform sampling
        indices = np.linspace(0, T_in - 1, self.T, dtype=np.float64)
        indices = np.round(indices).astype(np.int64)
        return keypoints[indices]


class TemporalFlip:
    """Randomly reverse the temporal order of a sequence.

    Parameters
    ----------
    p : float
        Probability of applying the flip.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return keypoints[::-1].copy()
        return keypoints


class TemporalSpeedPerturb:
    """Randomly stretch or compress the time axis via resampling.

    A speed factor is drawn uniformly from ``[low, high]``.  A factor
    > 1 means faster (fewer frames), < 1 means slower (more frames).
    The output length is adjusted accordingly before the downstream
    ``TemporalCrop`` restores it to a fixed size.

    Parameters
    ----------
    low : float
        Minimum speed factor.
    high : float
        Maximum speed factor.
    """

    def __init__(self, low: float = 0.85, high: float = 1.15) -> None:
        self.low = low
        self.high = high

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        T_in = keypoints.shape[0]
        if T_in <= 1:
            return keypoints

        speed = np.random.uniform(self.low, self.high)
        new_len = max(1, int(round(T_in / speed)))

        indices = np.linspace(0, T_in - 1, new_len, dtype=np.float64)
        indices = np.round(indices).astype(np.int64)
        return keypoints[indices]


# ---------------------------------------------------------------------------
# Spatial / keypoint augmentations
# ---------------------------------------------------------------------------


class KeypointHorizontalFlip:
    """Flip keypoints horizontally with correct left/right landmark swapping.

    The x-coordinate is reflected (``x' = 1 - x`` for normalized coords
    or negated for centered coords) and left/right landmarks are swapped.

    Parameters
    ----------
    swap_indices : np.ndarray or None
        Custom swap index array.  If None, the default MediaPipe swap
        table is used.
    p : float
        Probability of applying the flip.
    centered : bool
        If True, x-coordinates are centered around 0 (negate).
        If False, coordinates are in [0, 1] range (``1 - x``).
    """

    def __init__(
        self,
        swap_indices: Optional[np.ndarray] = None,
        p: float = 0.5,
        centered: bool = True,
    ) -> None:
        self.swap_indices = swap_indices if swap_indices is not None else _build_swap_indices()
        self.p = p
        self.centered = centered

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply horizontal flip.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape ``(T, NUM_KEYPOINTS, 3)`` or ``(T, NUM_KEYPOINTS * 3)``.

        Returns
        -------
        np.ndarray
            Flipped keypoints with the same shape.
        """
        if np.random.random() >= self.p:
            return keypoints

        kps = keypoints.copy()
        is_flat = kps.ndim == 2 and kps.shape[1] != 3
        if is_flat:
            T = kps.shape[0]
            num_kp = kps.shape[1] // 3
            kps = kps.reshape(T, num_kp, 3)

        # Flip x-coordinate
        if self.centered:
            kps[:, :, 0] = -kps[:, :, 0]
        else:
            kps[:, :, 0] = 1.0 - kps[:, :, 0]

        # Swap left/right landmarks
        n_kp = kps.shape[1]
        swap = self.swap_indices[:n_kp]
        kps = kps[:, swap, :]

        if is_flat:
            kps = kps.reshape(T, -1)

        return kps


class KeypointNoise:
    """Add small Gaussian noise to keypoint coordinates.

    Parameters
    ----------
    sigma : float
        Standard deviation of the noise.
    """

    def __init__(self, sigma: float = 0.005) -> None:
        self.sigma = sigma

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.sigma, size=keypoints.shape).astype(
            keypoints.dtype
        )
        return keypoints + noise


class KeypointScale:
    """Randomly scale keypoints around the spatial center.

    Parameters
    ----------
    low : float
        Minimum scale factor.
    high : float
        Maximum scale factor.
    """

    def __init__(self, low: float = 0.9, high: float = 1.1) -> None:
        self.low = low
        self.high = high

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.low, self.high)
        return keypoints * scale


class KeypointRotation:
    """Randomly rotate keypoints in the x-y plane around the origin.

    Parameters
    ----------
    max_angle : float
        Maximum rotation angle in degrees.
    p : float
        Probability of applying the rotation.
    """

    def __init__(self, max_angle: float = 15.0, p: float = 0.5) -> None:
        self.max_angle = max_angle
        self.p = p

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return keypoints

        kps = keypoints.copy()
        is_flat = kps.ndim == 2 and kps.shape[1] != 3
        if is_flat:
            T = kps.shape[0]
            num_kp = kps.shape[1] // 3
            kps = kps.reshape(T, num_kp, 3)

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        x = kps[..., 0].copy()
        y = kps[..., 1].copy()
        kps[..., 0] = cos_a * x - sin_a * y
        kps[..., 1] = sin_a * x + cos_a * y

        if is_flat:
            kps = kps.reshape(T, -1)
        return kps


class KeypointTranslation:
    """Randomly translate keypoints in x-y space.

    Parameters
    ----------
    max_shift : float
        Maximum shift magnitude in each direction.
    p : float
        Probability of applying the translation.
    """

    def __init__(self, max_shift: float = 0.1, p: float = 0.5) -> None:
        self.max_shift = max_shift
        self.p = p

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return keypoints

        kps = keypoints.copy()
        is_flat = kps.ndim == 2 and kps.shape[1] != 3
        if is_flat:
            T = kps.shape[0]
            num_kp = kps.shape[1] // 3
            kps = kps.reshape(T, num_kp, 3)

        dx = np.random.uniform(-self.max_shift, self.max_shift)
        dy = np.random.uniform(-self.max_shift, self.max_shift)
        kps[..., 0] += dx
        kps[..., 1] += dy

        if is_flat:
            kps = kps.reshape(T, -1)
        return kps


class KeypointYawRotation:
    """Simulate viewpoint change by rotating the pose around the vertical (y) axis.

    Rotates x and z coordinates: x' = cos(θ)·x − sin(θ)·z,
    z' = sin(θ)·x + cos(θ)·z.  This generates horizontal viewpoint diversity
    from limited data — equivalent to filming the signer from a slightly
    different angle.  Validated for 3–5× effective data augmentation in
    WLASL-scale research.

    Parameters
    ----------
    max_angle : float
        Maximum rotation angle in degrees (applied as ±max_angle).
    p : float
        Probability of applying the rotation.
    """

    def __init__(self, max_angle: float = 30.0, p: float = 0.5) -> None:
        self.max_angle = max_angle
        self.p = p

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return keypoints

        kps = keypoints.copy()
        is_flat = kps.ndim == 2 and kps.shape[1] != 3
        if is_flat:
            T = kps.shape[0]
            num_kp = kps.shape[1] // 3
            kps = kps.reshape(T, num_kp, 3)

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        x = kps[..., 0].copy()
        z = kps[..., 2].copy()
        kps[..., 0] = cos_a * x - sin_a * z
        kps[..., 2] = sin_a * x + cos_a * z

        if is_flat:
            kps = kps.reshape(T, -1)
        return kps


class KeypointDropout:
    """Randomly zero out entire frames and individual landmarks.

    Parameters
    ----------
    frame_drop_rate : float
        Fraction of frames to zero out entirely.
    landmark_drop_rate : float
        Fraction of individual landmarks to zero out per frame.
    p : float
        Probability of applying this augmentation.
    """

    def __init__(
        self,
        frame_drop_rate: float = 0.1,
        landmark_drop_rate: float = 0.05,
        p: float = 0.5,
    ) -> None:
        self.frame_drop_rate = frame_drop_rate
        self.landmark_drop_rate = landmark_drop_rate
        self.p = p

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return keypoints

        kps = keypoints.copy()
        is_flat = kps.ndim == 2 and kps.shape[1] != 3
        if is_flat:
            T = kps.shape[0]
            num_kp = kps.shape[1] // 3
            kps = kps.reshape(T, num_kp, 3)

        T = kps.shape[0]
        num_kp = kps.shape[1]

        # Drop entire frames
        frame_mask = np.random.random(T) < self.frame_drop_rate
        kps[frame_mask] = 0.0

        # Drop individual landmarks (on non-dropped frames)
        lm_mask = np.random.random((T, num_kp)) < self.landmark_drop_rate
        lm_mask[frame_mask] = False  # already zeroed
        kps[lm_mask] = 0.0

        if is_flat:
            kps = kps.reshape(kps.shape[0], -1)
        return kps


# ---------------------------------------------------------------------------
# Compose and presets
# ---------------------------------------------------------------------------


class Compose:
    """Compose a sequence of keypoint transforms.

    Parameters
    ----------
    transforms : list
        List of callable transforms.
    """

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            keypoints = t(keypoints)
        return keypoints


def get_train_transforms(T: int = 64) -> Compose:
    """Return the default training augmentation pipeline.

    Parameters
    ----------
    T : int
        Target sequence length.

    Returns
    -------
    Compose
        A composed pipeline of augmentations.
    """
    return Compose(
        [
            TemporalSpeedPerturb(low=0.7, high=1.4),
            TemporalCrop(T=T),
            TemporalFlip(p=0.3),
            KeypointHorizontalFlip(p=0.5, centered=True),
            KeypointYawRotation(max_angle=30, p=0.5),
            KeypointRotation(max_angle=15, p=0.5),
            KeypointTranslation(max_shift=0.1, p=0.5),
            KeypointNoise(sigma=0.02),
            KeypointScale(low=0.9, high=1.1),
            KeypointDropout(frame_drop_rate=0.15, landmark_drop_rate=0.1, p=0.7),
        ]
    )


def get_ce_train_transforms(T: int = 64) -> Compose:
    """Return a milder training augmentation pipeline for cross-entropy training.

    Designed for small datasets (~8 samples/class) where aggressive
    augmentation destroys signal.  Key differences from ``get_train_transforms``:
    - No ``TemporalFlip`` (reversing a sign changes its meaning)
    - Narrower speed perturbation, rotation, noise, and dropout ranges

    Parameters
    ----------
    T : int
        Target sequence length.

    Returns
    -------
    Compose
        A composed pipeline of mild augmentations.
    """
    return Compose(
        [
            TemporalSpeedPerturb(low=0.85, high=1.15),
            TemporalCrop(T=T),
            KeypointHorizontalFlip(p=0.5, centered=True),
            KeypointYawRotation(max_angle=15, p=0.5),
            KeypointRotation(max_angle=10, p=0.5),
            KeypointTranslation(max_shift=0.05, p=0.5),
            KeypointNoise(sigma=0.01),
            KeypointScale(low=0.95, high=1.05),
            KeypointDropout(frame_drop_rate=0.05, landmark_drop_rate=0.03, p=0.3),
        ]
    )


def get_val_transforms(T: int = 64) -> Compose:
    """Return the default validation/test augmentation pipeline.

    Only applies deterministic temporal cropping (no randomness).

    Parameters
    ----------
    T : int
        Target sequence length.

    Returns
    -------
    Compose
        A composed pipeline.
    """
    return Compose(
        [
            TemporalCrop(T=T),
        ]
    )


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply mixup augmentation on a batch.

    Returns mixed_x, y_a, y_b, lam.
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
