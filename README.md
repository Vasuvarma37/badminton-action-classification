<<<<<<< HEAD
# 🏸 Badminton Action Classifier

An AI-powered web app that classifies badminton shot types from video using a **BiGRU + Attention** neural network and **MediaPipe** pose estimation.

**Live Demo:** _Deploy to Render using the steps below_

## Shot Types Recognized

| Shot | Description |
|------|-------------|
| Backhand Drive | Fast flat stroke from the backhand side |
| Backhand Net Shot | Delicate touch shot near the net |
| Forehand Clear | High overhead defensive/attacking clear |
| Forehand Drive | Aggressive flat drive with the forehand |

## Architecture

- **Model:** 3-layer Bidirectional GRU + Bahdanau Attention (PyTorch)
- **Input:** 30 frames × 13 joints × 2 coords = (30, 26)
- **Keypoints:** MediaPipe Pose → 13-joint common skeleton
- **Backend:** FastAPI (Python)
- **Frontend:** React + Vite (dark theme, glassmorphism UI)
- **Deployment:** Docker + Render

---

## 🚀 Quick Start (Docker — Recommended)

### Prerequisites
- Docker + Docker Compose installed

### Run locally (production build)
```bash
docker compose --profile prod up --build
```
- **Frontend:** http://localhost:80
- **API:** http://localhost:8000

### Run locally (dev with hot-reload)
```bash
# Terminal 1 — Start backend
docker compose up backend --build

# Terminal 2 — Start frontend dev server
docker compose --profile dev up frontend-dev
```
- **Frontend:** http://localhost:5173
- **API:** http://localhost:8000

---

## 🖥️ Local Dev (without Docker)

### Backend (FastAPI)
```bash
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

---

## ☁️ Deploy to Render (via GitHub)

1. **Push** this repo to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Set the `VITE_API_URL` in the frontend service to your backend URL (e.g. `https://badminton-api.onrender.com`)
5. Click **Apply** — both services deploy automatically

> **Note:** The `model/saved_models/best_bigru.pth` checkpoint (~50MB) must be committed to the repo. Render's backend Docker image will load it at startup.

---

## Project Structure

```
├── backend/
│   ├── main.py          # FastAPI app
│   ├── predictor.py     # Model inference wrapper
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/       # Home, Classify, About
│   │   ├── components/  # Navbar, Footer, SkeletonViz, ...
│   │   └── index.css    # Dark-theme design system
│   ├── nginx.conf
│   └── Dockerfile
├── model/
│   ├── bigru_attention.py
│   └── saved_models/best_bigru.pth
├── keypoint_extraction/
│   └── mediapipe_extractor.py
├── utils/
│   ├── dataset.py
│   └── joint_mapping.py
├── config.py
├── docker-compose.yml
└── render.yaml
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/classes` | List action classes |
| POST | `/predict` | Upload video → classification |

### POST /predict
```
Content-Type: multipart/form-data
Body: video=<file>

Response:
{
  "class": "forehand_clear",
  "confidence": 0.9214,
  "scores": [0.03, 0.02, 0.92, 0.03],
  "classes": ["backhand_drive", "backhand_net_shot", "forehand_clear", "forehand_drive"],
  "attention": [...30 values...],
  "keypoints": [...30 frames × 26 values...]
}
```
=======
# badminton-action-classification
these project helps in predict the shot type in badminton
>>>>>>> 09b8e9425f254632e6eb59992f565fb67ac0662e
