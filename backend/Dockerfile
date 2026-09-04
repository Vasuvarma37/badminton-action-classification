# ──────────────────────────────────────────────────────────────────
# Backend Dockerfile — FastAPI + PyTorch + MediaPipe
# ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System libs needed by OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer caching)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY config.py            ./config.py
COPY model/               ./model/
COPY keypoint_extraction/ ./keypoint_extraction/
COPY utils/               ./utils/
COPY backend/             ./backend/

# Expose port
EXPOSE 8000

# Health-check so Docker/Render can verify the API is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run from the backend sub-directory so relative imports work
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
