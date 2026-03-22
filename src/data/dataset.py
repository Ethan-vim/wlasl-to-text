"""
PyTorch Dataset classes for WLASL keypoint data.

Provides ``WLASLKeypointDataset`` for loading precomputed keypoint files,
along with a factory function for creating DataLoaders with optional
class-balanced sampling.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WLASLKeypointDataset(Dataset):
    """Dataset that loads precomputed ``.npy`` keypoint files.

    Each sample is a tuple ``(keypoints_tensor, label)`` where
    ``keypoints_tensor`` has shape ``(T, num_features)`` with features
    being the flattened ``(x, y, z)`` per landmark, optionally
    concatenated with velocity features when ``use_motion=True``.

    Parameters
    ----------
    split_csv : str or Path
        Path to a CSV file with columns ``video_id``, ``label_idx``, ``gloss``.
    keypoint_dir : str or Path
        Directory containing ``{video_id}.npy`` files.
    transform : callable or None
        Augmentation pipeline (operates on shape ``(T_var, num_keypoints, 3)``).
    T : int
        Target sequence length.  If the transform does not include a
        ``TemporalCrop``, this dataset will pad/crop to ``T``.
    use_motion : bool
        If True, append velocity (frame differences) to each frame's
        features, doubling the per-landmark feature count from 3 to 6.
    """

    def __init__(
        self,
        split_csv: str | Path,
        keypoint_dir: str | Path,
        transform: Optional[Callable] = None,
        T: int = 64,
        use_motion: bool = False,
    ) -> None:
        self.keypoint_dir = Path(keypoint_dir)
        self.transform = transform
        self.T = T
        self.use_motion = use_motion

        df = pd.read_csv(split_csv)
        # Filter to only rows whose .npy file exists
        valid_mask = df["video_id"].apply(
            lambda vid: (self.keypoint_dir / f"{vid}.npy").exists()
        )
        self.df = df[valid_mask].reset_index(drop=True)
        n_missing = len(df) - len(self.df)
        if n_missing > 0:
            logger.warning(
                "Filtered %d / %d samples (missing .npy files). "
                "Run preprocessing or download more videos to increase coverage.",
                n_missing,
                len(df),
            )

        self.labels = self.df["label_idx"].values
        self.video_ids = self.df["video_id"].values
        self.glosses = self.df["gloss"].values

        # Build gloss-to-label mapping
        self.gloss_to_label: dict[str, int] = {}
        for _, row in self.df.iterrows():
            self.gloss_to_label[row["gloss"]] = int(row["label_idx"])

        self.num_classes = self.df["label_idx"].nunique()

        # Warn about sparse data
        total_classes_in_csv = df["label_idx"].nunique()
        if self.num_classes < total_classes_in_csv:
            logger.warning(
                "%d / %d classes in the split have no usable data (missing .npy files).",
                total_classes_in_csv - self.num_classes,
                total_classes_in_csv,
            )

        logger.info(
            "WLASLKeypointDataset: %d samples, %d classes", len(self.df), self.num_classes
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_id = self.video_ids[idx]
        label = int(self.labels[idx])

        npy_path = self.keypoint_dir / f"{video_id}.npy"
        keypoints = np.load(str(npy_path))  # (T_var, 543, 3)

        # Drop face landmarks: keep only pose (33) + left hand (21) + right hand (21) = 75
        keypoints = keypoints[:, :75, :]

        if self.transform is not None:
            keypoints = self.transform(keypoints)

        # If after transform the temporal length does not match T, fix it
        if keypoints.shape[0] != self.T:
            keypoints = self._pad_or_crop(keypoints)

        # Ensure 3D shape: (T, K, 3)
        if keypoints.ndim == 2:
            T = keypoints.shape[0]
            keypoints = keypoints.reshape(T, -1, 3)

        if self.use_motion:
            # Compute velocity (frame differences); first frame velocity is zero
            velocity = np.zeros_like(keypoints)
            velocity[1:] = keypoints[1:] - keypoints[:-1]
            # Concatenate: (T, K, 3) + (T, K, 3) -> (T, K, 6)
            keypoints = np.concatenate([keypoints, velocity], axis=-1)

        # Flatten spatial dims: (T, K, C) -> (T, K*C)
        T, K, C = keypoints.shape
        keypoints = keypoints.reshape(T, K * C)

        tensor = torch.from_numpy(keypoints).float()
        return tensor, label

    def _pad_or_crop(self, keypoints: np.ndarray) -> np.ndarray:
        """Ensure the sequence has exactly ``self.T`` frames."""
        T_in = keypoints.shape[0]
        if T_in == 0:
            return np.zeros((self.T, *keypoints.shape[1:]), dtype=np.float32)
        if T_in >= self.T:
            indices = np.linspace(0, T_in - 1, self.T, dtype=np.int64)
            return keypoints[indices]
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


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    weighted_sampling: bool = False,
) -> DataLoader:
    """Create a DataLoader with optional class-balanced weighted sampling.

    When ``weighted_sampling`` is True, each sample's weight is the
    inverse of its class frequency, ensuring balanced mini-batches.
    ``shuffle`` is automatically disabled when a sampler is used.

    Parameters
    ----------
    dataset : Dataset
        A WLASL dataset instance.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle (ignored if ``weighted_sampling`` is True).
    num_workers : int
        Number of data loading workers.
    weighted_sampling : bool
        If True, use ``WeightedRandomSampler`` for class imbalance.

    Returns
    -------
    DataLoader
    """
    sampler = None
    if weighted_sampling and hasattr(dataset, "labels"):
        labels = dataset.labels
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )

    # MPS backend deadlocks with multiprocessing workers — force 0
    if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        sampler=sampler,
        persistent_workers=num_workers > 0,
    )
