import os
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
try:
    from backend.utils import load_gesture_classifier, normalize_raw_landmarks
except ImportError:
    from utils import load_gesture_classifier, normalize_raw_landmarks

app = FastAPI(title="Anvaya Sign Language Detection API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and mapping globally
model, classes_list, model_type = load_gesture_classifier()

# Absolute paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

@app.get("/api/status")
async def get_status():
    """Returns the model status and available classes."""
    global model, classes_list, model_type
    
    # Reload model if it wasn't loaded successfully
    if model is None:
        model, classes_list, model_type = load_gesture_classifier()
        
    return {
        "status": "ready" if model is not None else "no_model_found",
        "model_type": model_type,
        "classes": classes_list or []
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")
    
    global model, classes_list, model_type
    if model is None:
        # Try reloading the model in case it was trained since server started
        model, classes_list, model_type = load_gesture_classifier()
        
    try:
        while True:
            # Expecting raw coordinates from frontend as a JSON list of floats
            data = await websocket.receive_text()
            
            if model is None:
                await websocket.send_json({
                    "error": "Model not loaded on server. Please train a model first.",
                    "label": "ERROR",
                    "confidence": 0.0
                })
                continue
                
            try:
                raw_landmarks = json.loads(data)
                
                # Check for correct coordinates count (21 landmarks * 3 coords = 63 values)
                if len(raw_landmarks) != 63:
                    await websocket.send_json({
                        "error": f"Invalid landmark dimensions. Expected 63, got {len(raw_landmarks)}",
                        "label": "ERROR",
                        "confidence": 0.0
                    })
                    continue
                
                # Normalize raw landmarks
                normalized = normalize_raw_landmarks(raw_landmarks)
                
                # Run prediction
                features = normalized.reshape(1, -1)
                
                if model_type == "sklearn":
                    pred_idx = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(probabilities[pred_idx])
                    label = classes_list[pred_idx]
                elif model_type == "keras":
                    predictions = model.predict(features, verbose=0)[0]
                    pred_idx = int(np.argmax(predictions))
                    confidence = float(predictions[pred_idx])
                    label = classes_list[pred_idx]
                else:
                    await websocket.send_json({
                        "error": "Unsupported model type",
                        "label": "ERROR",
                        "confidence": 0.0
                    })
                    continue
                
                # Return prediction result
                await websocket.send_json({
                    "label": label,
                    "confidence": confidence
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                await websocket.send_json({"error": f"Inference error: {str(e)}"})
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")

# Mount frontend directory for serving static index.html and assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {FRONTEND_DIR}")
