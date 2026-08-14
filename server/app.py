import os
import sys
import tempfile
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add the parent directory to the path so we can import from config, model, etc.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import ACTION_CLASSES, FRAMES_PER_VIDEO
from keypoint_extraction.mediapipe_extractor import extract_keypoints_from_video
from utils.joint_mapping import mediapipe_to_common, resample_sequence
from model.evaluate import load_best_model

app = FastAPI(title="Badminton Action Classification API")

# Load model globally when app starts
print("Loading PyTorch model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model, _ = load_best_model("best_bilstm.pth", device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Serve the static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/frontend", StaticFiles(directory=static_dir, html=True), name="static")

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"error": "Model failed to load."})

    # Save uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to save video: {str(e)}"})

    try:
        # 1. Extract 33 MediaPipe keypoints (shape: N x 66)
        kp_33 = extract_keypoints_from_video(temp_video_path, n_frames=FRAMES_PER_VIDEO)
        
        if kp_33 is None:
             return JSONResponse(status_code=400, content={"error": "Could not extract keypoints from video."})

        # 2. Project to 13 common joints (shape: 30 x 26)
        kp_13 = mediapipe_to_common(kp_33)

        # 3. Resample to exactly 30 frames (just in case)
        kp_13_resampled = resample_sequence(kp_13, FRAMES_PER_VIDEO)

        # 4. Prepare tensor for model (shape: 1 x 30 x 26)
        seq_tensor = torch.tensor(kp_13_resampled, dtype=torch.float32).unsqueeze(0).to(device)

        # 5. Inference
        with torch.no_grad():
            logits, attn = model(seq_tensor, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            
            confidence = probs[0][pred_idx].item()
            action_name = ACTION_CLASSES[pred_idx]

        return {
            "prediction": action_name,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Inference failed: {str(e)}"})
    finally:
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/frontend/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
