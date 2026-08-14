"""
mediapipe_extractor.py
─────────────────────
Extracts skeleton keypoints from badminton videos using MediaPipe Pose.

Replaces AlphaPose from the original paper with a lightweight, pip-installable
alternative. MediaPipe returns 33 landmarks (vs AlphaPose's 17), giving richer
spatial information while being orders of magnitude simpler to set up.

Output per video:
    numpy array of shape (FRAMES_PER_VIDEO, NUM_LANDMARKS * 2)
    where the last dim is [x0, y0, x1, y1, ..., x32, y32] (normalised 0-1).

Usage:
    python -m keypoint_extraction.mediapipe_extractor   # batch extraction
    OR called via main.py --mode extract
"""

import os
import sys
import logging
import numpy as np
import cv2

# MediaPipe is imported with a graceful error message
try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)

# Add project root to path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, KEYPOINTS_DIR, ACTION_CLASSES,
    FRAMES_PER_VIDEO, NUM_LANDMARKS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────
_mp_pose = mp.solutions.pose


def _sample_frames_uniform(cap: cv2.VideoCapture, n: int) -> list[np.ndarray]:
    """Return n uniformly-spaced frames from an already-opened VideoCapture."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def extract_keypoints_from_video(
    video_path: str,
    n_frames: int = FRAMES_PER_VIDEO,
    min_detection_confidence: float = 0.5,
) -> np.ndarray | None:
    """
    Extract skeleton keypoints from a single video file.

    Returns
    -------
    np.ndarray of shape (n_frames, NUM_LANDMARKS * 2), dtype float32
        Flattened [x, y] for each of the 33 MediaPipe landmarks per frame.
        Missing / undetected frames are filled with zeros.
    None
        If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    frames = _sample_frames_uniform(cap, n_frames)
    cap.release()

    if not frames:
        logger.warning("No frames extracted from: %s", video_path)
        return None

    keypoint_sequence = np.zeros((n_frames, NUM_LANDMARKS * 2), dtype=np.float32)

    with _mp_pose.Pose(
        static_image_mode=True,          # treat each frame independently
        model_complexity=1,              # 0=lite, 1=full, 2=heavy → balanced
        min_detection_confidence=min_detection_confidence,
        enable_segmentation=False,
    ) as pose:
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                coords = []
                for lm in result.pose_landmarks.landmark:
                    coords.extend([lm.x, lm.y])  # normalised [0, 1]
                keypoint_sequence[i] = coords
            # else: leave row as zeros (pose not detected)

    return keypoint_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction(
    data_dir: str = DATA_DIR,
    keypoints_dir: str = KEYPOINTS_DIR,
    action_classes: list[str] = ACTION_CLASSES,
    n_frames: int = FRAMES_PER_VIDEO,
    overwrite: bool = False,
) -> None:
    """
    Iterate over every action class folder, process each video, and save
    the resulting keypoint array as a .npy file.

    Directory layout expected:
        data_dir/
            backhand_drive/       *.mp4 / *.avi / ...
            backhand_net_shot/
            forehand_clear/
            forehand_drive/

    Output:
        keypoints_dir/
            backhand_drive/       <video_stem>.npy
            backhand_net_shot/
            forehand_clear/
            forehand_drive/
    """
    os.makedirs(keypoints_dir, exist_ok=True)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    total_saved = 0
    total_failed = 0

    for action in action_classes:
        src_dir = os.path.join(data_dir, action)
        dst_dir = os.path.join(keypoints_dir, action)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            logger.warning("Action folder not found: %s — skipping.", src_dir)
            continue

        video_files = [
            f for f in os.listdir(src_dir)
            if os.path.splitext(f)[1].lower() in video_exts
        ]

        logger.info("Processing [%s] — %d videos found.", action, len(video_files))

        for vf in video_files:
            stem = os.path.splitext(vf)[0]
            out_path = os.path.join(dst_dir, f"{stem}.npy")

            if not overwrite and os.path.exists(out_path):
                continue  # skip already processed

            kp = extract_keypoints_from_video(
                os.path.join(src_dir, vf), n_frames=n_frames
            )
            if kp is not None:
                np.save(out_path, kp)
                total_saved += 1
                logger.debug("  Saved → %s", out_path)
            else:
                total_failed += 1
                logger.warning("  Failed → %s", vf)

    logger.info(
        "Extraction complete. Saved: %d | Failed: %d", total_saved, total_failed
    )


# ─────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_extraction()
