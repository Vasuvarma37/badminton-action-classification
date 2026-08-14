"""
utils/dataset.py
────────────────
PyTorch Dataset and DataLoader factory for the badminton keypoint data.

Each sample is a (sequence, label) pair where:
    sequence : torch.FloatTensor  shape (FRAMES_PER_VIDEO, INPUT_SIZE)
               INPUT_SIZE = 26 (13 common joints × 2 coords)
    label    : torch.LongTensor   scalar  ∈ {0…5}

The raw MediaPipe .npy files (shape T×66) are projected to the 13-joint
common skeleton on-the-fly via utils/joint_mapping.mediapipe_to_common().

Split strategy
──────────────
Stratified 80/10/10 within each class so that no single class
dominates any particular split.
"""

import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KEYPOINTS_DIR, ACTION_CLASSES, FRAMES_PER_VIDEO, FEATURES_PER_FRAME,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, CLASS_WEIGHTS,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
)
from utils.joint_mapping import mediapipe_to_common, resample_sequence

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BadmintonDataset(Dataset):
    """
    Loads pre-extracted .npy keypoint files (or in-memory arrays) and
    serves them as tensors.

    Parameters
    ----------
    samples : list of (path_or_array, label_idx)
        Each item is either:
          - (str path, int)  → .npy file on disk  (MediaPipe 66-dim format)
          - (np.ndarray, int) → already-loaded 26-dim common-joint array
    augment : bool
        Apply lightweight augmentation (noise, flip, scale) when True.
    """

    def __init__(
        self,
        samples: list[tuple],
        augment: bool = False,
    ):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        src, label = self.samples[idx]

        if isinstance(src, str):
            # Load raw MediaPipe .npy (T, 66) and project to common (T, 26)
            raw = np.load(src).astype(np.float32)
            seq = mediapipe_to_common(raw)              # (T, 26)
        else:
            # Already a pre-processed common-joint array (from BST adapter)
            seq = np.asarray(src, dtype=np.float32)    # (T, 26)

        # Resample / pad to exact length
        seq = resample_sequence(seq, FRAMES_PER_VIDEO)  # (FRAMES_PER_VIDEO, 26)

        if self.augment:
            seq = self._augment(seq)

        return (
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.long),
        )

    # ── Augmentation ──────────────────────────────────────────────────────────
    @staticmethod
    def _augment(seq: np.ndarray) -> np.ndarray:
        """
        Lightweight augmentations that preserve temporal structure:
          1. Gaussian coordinate noise
          2. Random horizontal flip  (mirrors the court side)
          3. Random uniform scaling  [0.9 – 1.1]
          4. Random temporal shift   (roll by ≤ 2 frames, wrap-around)
        """
        # 1. Noise
        seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)

        # 2. Horizontal flip: x-coords are at even column indices
        if np.random.rand() < 0.5:
            seq[:, 0::2] = 1.0 - seq[:, 0::2]

        # 3. Scale
        seq = seq * np.random.uniform(0.9, 1.1)

        # 4. Temporal shift
        shift = np.random.randint(-2, 3)
        if shift != 0:
            seq = np.roll(seq, shift, axis=0)

        return np.clip(seq, -2.0, 2.0)   # BST values can exceed [0,1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_disk_samples(
    keypoints_dir: str = KEYPOINTS_DIR,
    action_classes: list[str] = ACTION_CLASSES,
) -> list[tuple[str, int]]:
    """Scan keypoints_dir and return list of (npy_path, class_index)."""
    samples = []
    for label_idx, action in enumerate(action_classes):
        action_dir = os.path.join(keypoints_dir, action)
        if not os.path.isdir(action_dir):
            logger.warning("Keypoint folder missing: %s", action_dir)
            continue
        for fname in sorted(os.listdir(action_dir)):
            if fname.endswith(".npy"):
                samples.append((os.path.join(action_dir, fname), label_idx))
    logger.info("MediaPipe samples on disk: %d", len(samples))
    return samples


def split_samples(
    samples: list,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list, list, list]:
    """
    Stratified split into train / val / test.
    Returns three lists of (path_or_array, label) pairs.
    """
    labels = [s[1] for s in samples]

    # First split: train vs (val + test)
    train, temp, _, temp_labels = train_test_split(
        samples, labels,
        test_size=1.0 - train_ratio,
        stratify=labels,
        random_state=seed,
    )
    # Second split: val vs test (equal halves of the remainder)
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed,
    )

    logger.info("Split → train: %d | val: %d | test: %d",
                len(train), len(val), len(test))
    return train, val, test


def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples under-represented classes.

    Uses CLASS_WEIGHTS from config to apply per-class importance factors,
    then further adjusts by inverse class frequency.
    """
    from config import CLASS_WEIGHTS as CW
    n_classes = len(CW)
    # Count samples per class
    class_counts = [0] * n_classes
    for _, lbl in samples:
        class_counts[lbl] += 1

    # Weight per sample = config_weight / class_count
    sample_weights = []
    for _, lbl in samples:
        count = max(class_counts[lbl], 1)
        w = CW[lbl] / count
        sample_weights.append(w)

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(samples),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    keypoints_dir: str = KEYPOINTS_DIR,
    action_classes: list[str] = ACTION_CLASSES,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
    extra_train_samples: list | None = None,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader).

    Parameters
    ----------
    keypoints_dir        : str  — MediaPipe .npy root (may be empty if BST only)
    action_classes       : list[str]
    batch_size           : int
    augment_train        : bool — augment training set
    extra_train_samples  : list — pre-loaded (array, label) tuples from BST
                           to add to the training split only
    use_weighted_sampler : bool — use WeightedRandomSampler for training
    """
    disk_samples = _collect_disk_samples(keypoints_dir, action_classes)

    if not disk_samples and not extra_train_samples:
        raise RuntimeError(
            f"No .npy files found under '{keypoints_dir}' and no BST samples provided.\n"
            "Run  python main.py --mode extract  OR  python download_bst.py"
        )

    if disk_samples:
        train_s, val_s, test_s = split_samples(disk_samples)
    else:
        train_s, val_s, test_s = [], [], []

    # Merge BST samples into training split only
    if extra_train_samples:
        train_s = train_s + extra_train_samples
        logger.info(
            "After BST merge → train: %d | val: %d | test: %d",
            len(train_s), len(val_s), len(test_s),
        )

    def _loader(split_samples, augment, shuffle, weighted=False):
        ds = BadmintonDataset(split_samples, augment=augment)
        sampler = None
        if weighted and len(split_samples) > 0:
            sampler = make_weighted_sampler(split_samples)
            shuffle = False   # mutually exclusive with sampler
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=False,
        )

    train_loader = _loader(
        train_s, augment=augment_train,
        shuffle=not use_weighted_sampler,
        weighted=use_weighted_sampler,
    )
    val_loader  = _loader(val_s,  augment=False, shuffle=False)
    test_loader = _loader(test_s, augment=False, shuffle=False)

    return train_loader, val_loader, test_loader
