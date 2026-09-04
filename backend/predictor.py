"""
backend/predictor.py

Loads the trained BadmintonBiGRU checkpoint once at startup and
exposes a predict_from_video() function for use by the FastAPI routes.
"""

import os
import sys
import tempfile
import logging
import numpy as np
import torch

# Allow imports from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import (
    MODEL_SAVE_DIR, ACTION_CLASSES,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FC_HIDDEN, NUM_CLASSES,
    FRAMES_PER_VIDEO,
)
from model.bigru_attention import BadmintonBiGRU
from keypoint_extraction.mediapipe_extractor import extract_keypoints_from_video
from utils.joint_mapping import mediapipe_to_common, resample_sequence

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Singleton model (loaded once at startup)
# ──────────────────────────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        checkpoint_name = os.environ.get("CHECKPOINT_NAME", "best_bigru.pth")
        ckpt_path = os.path.join(MODEL_SAVE_DIR, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {ckpt_path}\n"
                "Set CHECKPOINT_NAME or MODEL_SAVE_DIR environment variables."
            )
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        m = BadmintonBiGRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            fc_hidden=FC_HIDDEN,
        ).to(DEVICE)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        _model = m
        logger.info(
            "Model loaded from %s (epoch %s, val_acc %.2f%%)",
            ckpt_path,
            ckpt.get("epoch", "?"),
            ckpt.get("val_acc", 0.0) * 100,
        )
    return _model


def warmup():
    """Call at startup to pre-load the model into memory."""
    _get_model()
    logger.info("Model warm-up complete on device: %s", DEVICE)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict_from_video(video_bytes: bytes) -> dict:
    """
    Full pipeline: video bytes -> keypoints -> model -> result dict.

    Returns
    -------
    {
        "class":        str,           # predicted action class name
        "confidence":   float,         # 0-1 confidence for top class
        "scores":       list[float],   # softmax scores for all classes
        "classes":      list[str],     # class names in score order
        "attention":    list[float],   # per-frame attention weights (30 values)
        "keypoints":    list[list],    # (FRAMES_PER_VIDEO, 26) -- for skeleton viz
        "error":        str or None,
    }
    """
    # 1. Write bytes to temp file so OpenCV can open it
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        # 2. Extract raw MediaPipe keypoints (T, 66)
        raw_kp = extract_keypoints_from_video(tmp_path, n_frames=FRAMES_PER_VIDEO)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if raw_kp is None:
        return {
            "class": None,
            "confidence": 0.0,
            "scores": [],
            "classes": ACTION_CLASSES,
            "attention": [],
            "keypoints": [],
            "error": "Could not extract pose from video. Ensure the player is clearly visible.",
        }

    # 3. Project to 13-joint common skeleton (T, 26)
    common_kp = mediapipe_to_common(raw_kp)
    common_kp = resample_sequence(common_kp, FRAMES_PER_VIDEO)

    # 4. Tensor shape (1, T, 26)
    x = torch.from_numpy(common_kp).unsqueeze(0).to(DEVICE)

    # 5. Run inference
    model = _get_model()
    with torch.no_grad():
        logits, attn_weights = model(x, return_attention=True)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        attn  = attn_weights.squeeze(0).cpu().tolist()

    pred_idx   = int(np.argmax(probs))
    pred_class = ACTION_CLASSES[pred_idx]
    confidence = probs[pred_idx]

    return {
        "class":      pred_class,
        "confidence": round(confidence, 4),
        "scores":     [round(p, 4) for p in probs],
        "classes":    ACTION_CLASSES,
        "attention":  [round(a, 4) for a in attn],
        "keypoints":  common_kp.tolist(),   # 30 frames x 26 values (normalized 0-1)
        "error":      None,
    }
