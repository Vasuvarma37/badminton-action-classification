"""
backend/main.py

FastAPI application entry point.

Endpoints:
  GET  /health        -- liveness probe for Render / Docker
  GET  /classes       -- list of action class names
  POST /predict       -- upload video -> classification result
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predictor import warmup, predict_from_video
from config import ACTION_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Lifespan: warm-up model on startup
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    try:
        warmup()
        logger.info("Model ready.")
    except Exception as e:
        logger.error("Model load failed: %s", e)
    yield
    logger.info("Shutting down.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Badminton Action Classifier API",
    description="BiGRU + Attention model for classifying badminton shots from video.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React frontend (and any origin in dev)
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:80",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production via env var if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Liveness probe used by Render and Docker health checks."""
    return {"status": "ok", "model": "BadmintonBiGRU"}


@app.get("/classes", tags=["Model"])
async def get_classes():
    """Return the list of supported action class names."""
    return {"classes": ACTION_CLASSES}


@app.post("/predict", tags=["Inference"])
async def predict(video: UploadFile = File(...)):
    """
    Classify a badminton shot from an uploaded video file.

    - Accepts: mp4, avi, mov, mkv (max ~100 MB)
    - Returns: predicted class, confidence, per-class scores, attention weights,
               skeleton keypoints for visualization
    """
    # Basic validation
    allowed_types = {"video/mp4", "video/avi", "video/quicktime",
                     "video/x-msvideo", "video/x-matroska", "video/webm",
                     "application/octet-stream"}
    content_type = video.content_type or ""
    filename = video.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower()

    if content_type not in allowed_types and ext not in {"mp4", "avi", "mov", "mkv", "webm"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Please upload a video.",
        )

    # Read bytes
    video_bytes = await video.read()
    if len(video_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    max_size = 150 * 1024 * 1024   # 150 MB
    if len(video_bytes) > max_size:
        raise HTTPException(status_code=413, detail="Video too large. Max size is 150 MB.")

    logger.info(
        "Received video: %s (%s bytes, type=%s)",
        filename, len(video_bytes), content_type,
    )

    # Run prediction
    try:
        result = predict_from_video(video_bytes)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if result.get("error"):
        return JSONResponse(
            status_code=422,
            content={"detail": result["error"], **result},
        )

    return result
