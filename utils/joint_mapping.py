"""
utils/joint_mapping.py
──────────────────────
Defines the canonical 13-joint skeleton that both MediaPipe-33 and
BST/COCO-17 skeletons can be projected onto, enabling the two datasets
to be mixed without a feature-dimension mismatch.

Common joint order (index 0-12):
    0  L-shoulder      1  R-shoulder
    2  L-elbow         3  R-elbow
    4  L-wrist         5  R-wrist
    6  L-hip           7  R-hip
    8  L-knee          9  R-knee
    10 L-ankle         11 R-ankle
    12 nose / head

MediaPipe landmark indices used:
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

COCO 17-joint indices used by BST/MMPose:
    0  nose     1  L-eye   2  R-eye  3  L-ear  4  R-ear
    5  L-sho    6  R-sho   7  L-elb  8  R-elb  9  L-wri  10 R-wri
    11 L-hip   12  R-hip  13  L-kne 14  R-kne  15  L-ank 16  R-ank
"""

import numpy as np

# ── MediaPipe-33 → common-13 index table ──────────────────────────────────────
# Each element is a MediaPipe landmark index.
# Nose (idx 0 in MediaPipe) maps to common joint 12.
_MP_TO_COMMON = [
    11,  # 0  L-shoulder
    12,  # 1  R-shoulder
    13,  # 2  L-elbow
    14,  # 3  R-elbow
    15,  # 4  L-wrist
    16,  # 5  R-wrist
    23,  # 6  L-hip
    24,  # 7  R-hip
    25,  # 8  L-knee
    26,  # 9  R-knee
    27,  # 10 L-ankle
    28,  # 11 R-ankle
    0,   # 12 nose
]

# ── COCO-17 → common-13 index table ───────────────────────────────────────────
_COCO_TO_COMMON = [
    5,   # 0  L-shoulder
    6,   # 1  R-shoulder
    7,   # 2  L-elbow
    8,   # 3  R-elbow
    9,   # 4  L-wrist
    10,  # 5  R-wrist
    11,  # 6  L-hip
    12,  # 7  R-hip
    13,  # 8  L-knee
    14,  # 9  R-knee
    15,  # 10 L-ankle
    16,  # 11 R-ankle
    0,   # 12 nose
]

NUM_COMMON_JOINTS = 13


def mediapipe_to_common(seq: np.ndarray) -> np.ndarray:
    """
    Project a MediaPipe-33 sequence to the common 13-joint space.

    Parameters
    ----------
    seq : np.ndarray, shape (T, 66)
        Flattened [x0,y0, x1,y1, …, x32,y32] MediaPipe output.

    Returns
    -------
    np.ndarray, shape (T, 26)   [x,y] for each of the 13 common joints.
    """
    T = seq.shape[0]
    # Reshape to (T, 33, 2)
    joints = seq.reshape(T, 33, 2)
    # Select the 13 joints and flatten back
    common = joints[:, _MP_TO_COMMON, :]          # (T, 13, 2)
    return common.reshape(T, NUM_COMMON_JOINTS * 2).astype(np.float32)


def coco_to_common(seq: np.ndarray, J: int = 17) -> np.ndarray:
    """
    Project a COCO-17 sequence to the common 13-joint space.

    Parameters
    ----------
    seq : np.ndarray, shape (T, J*2) or (T, J, 2)
        BST/MMPose output — either flat or with explicit joint dimension.
    J   : int
        Number of joints in the source skeleton (default 17 for COCO).

    Returns
    -------
    np.ndarray, shape (T, 26)
    """
    T = seq.shape[0]
    if seq.ndim == 2:
        joints = seq.reshape(T, J, 2)
    else:                                           # already (T, J, 2)
        joints = seq
    common = joints[:, _COCO_TO_COMMON, :]          # (T, 13, 2)
    return common.reshape(T, NUM_COMMON_JOINTS * 2).astype(np.float32)


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a sequence to `target_len` frames using linear interpolation.

    Parameters
    ----------
    seq        : np.ndarray, shape (T, F)
    target_len : int

    Returns
    -------
    np.ndarray, shape (target_len, F)
    """
    T = seq.shape[0]
    if T == target_len:
        return seq
    src_idx = np.linspace(0, T - 1, target_len)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (src_idx - lo)[:, None]          # (target_len, 1)
    return ((1 - frac) * seq[lo] + frac * seq[hi]).astype(np.float32)
