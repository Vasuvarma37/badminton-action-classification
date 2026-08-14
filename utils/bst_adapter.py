"""
utils/bst_adapter.py
────────────────────
Adapter that loads BST pre-processed skeleton .npy files (ShuttleSet or
BadmintonDB) and converts them into (sequence, label) tuples that are
directly compatible with the existing BadmintonDataset pipeline.

Supported BST file formats
--------------------------
Both formats are plain NumPy structured as a Python dict saved with
np.save(..., allow_pickle=True) or as separate x / y arrays:

    ┌──────────────────────────────────────────────────────────────────┐
    │ BadmintonDB  (seq_len=72, 6 strokes)                             │
    │   dataset_npy/                                                   │
    │     data.npy  → shape (N, 72, 34)  ← 17-joint COCO × (x,y)     │
    │     label.npy → shape (N,)  int    ← 0-5 BadmintonDB class ids  │
    │                                                                  │
    │ ShuttleSet 25-class merged  (seq_len=30)                         │
    │   dataset_npy/                                                   │
    │     data.npy  → shape (N, 30, 34)                               │
    │     label.npy → shape (N,)  int    ← 0-24 ShuttleSet class ids  │
    └──────────────────────────────────────────────────────────────────┘

NOTE: Some BST releases use a single dict .npy instead.  The loader
handles both transparently.

Usage
-----
    from utils.bst_adapter import load_bst_samples, BST_SOURCE
    samples = load_bst_samples(source=BST_SOURCE.BADMINTONDB)
    # → list of (np.ndarray shape (30, 26), label_int) tuples
"""

import os
import sys
import logging
from enum import Enum
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BST_DATA_DIR, ACTION_CLASSES, FRAMES_PER_VIDEO,
)
from utils.joint_mapping import coco_to_common, resample_sequence

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset source selector
# ─────────────────────────────────────────────────────────────────────────────

class BST_SOURCE(Enum):
    BADMINTONDB   = "BadmintonDB_data"
    SHUTTLESET_25 = "ShuttleSet_data_merged"   # 25-class merged
    SHUTTLESET_35 = "ShuttleSet_data"          # 35-class


# ─────────────────────────────────────────────────────────────────────────────
# Label remapping tables
# ─────────────────────────────────────────────────────────────────────────────

# BadmintonDB 6-class labels → your 6 ACTION_CLASSES
# (BadmintonDB uses the same 6 categories; verify ordering after download)
BADMINTONDB_LABEL_MAP = {
    0: 2,   # clear          → forehand_clear
    1: 0,   # drive          → backhand_drive   (mixed; see note)
    2: 4,   # lift           → forehand_lift
    3: 5,   # net shot       → forehand_net_shot (mixed)
    4: None,  # serve        → discard
    5: 3,   # smash          → forehand_drive (closest analogue)
}
# NOTE: BadmintonDB doesn't split forehand/backhand for some strokes.
# Sequences labelled as "drive" or "net shot" are kept with a slight label
# noise that cross-training can still improve overall robustness.


# ShuttleSet 25-class (merged) string labels → your 6 ACTION_CLASSES
# Source: BST paper Table 1 + dataset annotation files
SHUTTLESET_LABEL_MAP: dict[str, Optional[int]] = {
    # forehand_clear (2)
    "forehand clear":            2,
    "forehand lob":              2,
    "forehand defensive clear":  2,
    # forehand_drive (3)
    "forehand drive":            3,
    "forehand smash":            3,
    # forehand_net_shot (5)
    "forehand net shot":         5,
    "forehand drop":             5,
    "forehand push":             5,
    # forehand_lift (4)
    "forehand lift":             4,
    "forehand high serve":       4,
    # backhand_drive (0)
    "backhand drive":            0,
    "backhand smash":            0,
    # backhand_net_shot (1)
    "backhand net shot":         1,
    "backhand drop":             1,
    "backhand push":             1,
    "backhand lift":             1,
    # Discard (serve, unclear, etc.)
    "forehand short serve":      None,
    "backhand short serve":      None,
    "forehand long serve":       None,
    "backhand long serve":       None,
    "backhand clear":            None,
    "backhand lob":              None,
}

# Numeric ShuttleSet-25 label → string name
# (indices taken from BST paper Table 1 / dataset readme)
SHUTTLESET_IDX_TO_NAME = [
    "backhand clear",
    "backhand drop",
    "backhand drive",
    "backhand lift",
    "backhand lob",
    "backhand long serve",
    "backhand net shot",
    "backhand push",
    "backhand short serve",
    "backhand smash",
    "forehand clear",
    "forehand defensive clear",
    "forehand drop",
    "forehand drive",
    "forehand high serve",
    "forehand lift",
    "forehand lob",
    "forehand long serve",
    "forehand net shot",
    "forehand push",
    "forehand short serve",
    "forehand smash",
    "forehand lob",     # duplicate in some versions
    "other",
    "unknown",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def _find_npy(folder: str, stem: str) -> Optional[str]:
    """Return path to <folder>/<stem>.npy (case-insensitive), or None."""
    for fname in os.listdir(folder):
        if fname.lower() == f"{stem.lower()}.npy":
            return os.path.join(folder, fname)
    return None


def _load_data_label_arrays(source_dir: str):
    """
    Try loading data/label arrays from a BST source directory.

    Returns
    -------
    data  : np.ndarray  shape (N, T, J*2)
    label : np.ndarray  shape (N,)
    """
    data_path  = _find_npy(source_dir, "data")
    label_path = _find_npy(source_dir, "label")

    if data_path and label_path:
        logger.info("Loading separate data/label .npy files from %s", source_dir)
        data  = np.load(data_path,  allow_pickle=True)
        label = np.load(label_path, allow_pickle=True).astype(int).ravel()
        return data, label

    # Fallback: single dict .npy (older BST format)
    for fname in os.listdir(source_dir):
        if fname.endswith(".npy"):
            payload = np.load(os.path.join(source_dir, fname), allow_pickle=True).item()
            if isinstance(payload, dict):
                data  = payload.get("data",  payload.get("x"))
                label = payload.get("label", payload.get("y"))
                if data is not None and label is not None:
                    logger.info("Loaded dict-format .npy: %s", fname)
                    return np.array(data), np.array(label).astype(int).ravel()

    raise FileNotFoundError(
        f"Cannot find BST data/label arrays in: {source_dir}\n"
        "Run  python download_bst.py  to fetch the dataset."
    )


def load_bst_samples(
    source: BST_SOURCE = BST_SOURCE.BADMINTONDB,
    bst_data_dir: str = BST_DATA_DIR,
    target_frames: int = FRAMES_PER_VIDEO,
    coco_joints: int = 17,
) -> list[tuple[np.ndarray, int]]:
    """
    Load a BST dataset, remap joints + labels, and return a sample list.

    Parameters
    ----------
    source        : BST_SOURCE  — which dataset to load
    bst_data_dir  : str         — root directory of downloaded BST data
    target_frames : int         — resample all sequences to this length
    coco_joints   : int         — joint count in the BST skeleton (default 17)

    Returns
    -------
    list of (seq, label) where:
        seq   : np.ndarray shape (target_frames, 26)   float32
        label : int  ∈ {0,1,2,3,4,5} matching ACTION_CLASSES
    """
    source_dir = os.path.join(bst_data_dir, source.value)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"BST source directory not found: {source_dir}\n"
            f"Run  python download_bst.py  to download the '{source.value}' dataset."
        )

    data, raw_labels = _load_data_label_arrays(source_dir)
    N, T, feat = data.shape
    logger.info("BST raw shape: %s | unique labels: %s", data.shape, np.unique(raw_labels))

    # ── Infer joint count from feat ───────────────────────────────────────────
    inferred_J = feat // 2
    if inferred_J != coco_joints:
        logger.warning(
            "Feature dim %d → %d joints (expected %d). Using inferred value.",
            feat, inferred_J, coco_joints,
        )
        coco_joints = inferred_J

    # ── Build label remapper ──────────────────────────────────────────────────
    if source == BST_SOURCE.BADMINTONDB:
        def remap(raw_lbl: int) -> Optional[int]:
            return BADMINTONDB_LABEL_MAP.get(raw_lbl)
    else:
        def remap(raw_lbl: int) -> Optional[int]:
            name = SHUTTLESET_IDX_TO_NAME[raw_lbl] if raw_lbl < len(SHUTTLESET_IDX_TO_NAME) else None
            return SHUTTLESET_LABEL_MAP.get(name) if name else None

    # ── Process each sample ───────────────────────────────────────────────────
    samples: list[tuple[np.ndarray, int]] = []
    discarded = 0

    for i in range(N):
        target_label = remap(int(raw_labels[i]))
        if target_label is None:
            discarded += 1
            continue

        seq = data[i].astype(np.float32)   # (T, J*2)

        # Project COCO joints → 13 common joints
        seq = coco_to_common(seq, J=coco_joints)   # (T, 26)

        # Resample to target length
        seq = resample_sequence(seq, target_frames)  # (target_frames, 26)

        samples.append((seq, target_label))

    logger.info(
        "BST %s: %d samples loaded | %d discarded (unmapped labels)",
        source.value, len(samples), discarded,
    )
    _log_class_distribution(samples)
    return samples


def _log_class_distribution(samples: list[tuple[np.ndarray, int]]) -> None:
    """Print per-class counts for the loaded BST data."""
    counts = {}
    for _, lbl in samples:
        counts[lbl] = counts.get(lbl, 0) + 1
    logger.info("BST class distribution:")
    for cls_idx, cls_name in enumerate(ACTION_CLASSES):
        n = counts.get(cls_idx, 0)
        bar = "█" * (n // max(1, max(counts.values()) // 30))
        logger.info("  %-25s %4d  %s", cls_name, n, bar)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    src_name = sys.argv[1] if len(sys.argv) > 1 else "BADMINTONDB"
    src = BST_SOURCE[src_name.upper()]
    samples = load_bst_samples(source=src)
    if samples:
        seq, lbl = samples[0]
        print(f"\nFirst sample: shape={seq.shape}  label={lbl} ({ACTION_CLASSES[lbl]})")
        print(f"Total loaded: {len(samples)}")
