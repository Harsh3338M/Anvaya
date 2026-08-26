# Anvaya 
**Zero-GPU Real-Time Hand Gesture & Sign Translation System with Temporal Debouncing & Native TTS**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe%20Hands-orange.svg)](https://developers.google.com/mediapipe)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)

---

## 📌 Overview & Research Gap

Traditional vision-based sign language translation systems typically rely on heavy deep learning architectures (e.g., 3D-CNNs, Spatial-Temporal Graph Convolutional Networks, or Deep Transformers) that require dedicated cloud GPUs, introduce significant network latency, and fail on consumer-grade hardware. Conversely, existing lightweight coordinate models suffer from **movement epenthesis** (transition noise/flicker between signs), distance sensitivity, and lack of integrated real-time verbal feedback.

**Anvaya** bridges these critical research gaps by introducing:
1. **Mathematical Coordinate Invariance:** Translation of 21 3D hand keypoints into an origin-relative (wrist-anchored) and Euclidean-max normalized $1 \times 63$ feature tensor, rendering the system invariant to camera distance and spatial positioning.
2. **Temporal Debounce Stabilization:** A majority-voting temporal sliding window ($w=7$ frames) with confidence thresholding ($\tau \ge 0.85$) that filters intermediate transition noise between signs.
3. **Zero-GPU, Low-Latency Web Architecture:** A sub-10ms CPU ONNX inference pipeline communicating over full-duplex WebSockets, coupled with zero-latency client-side speech synthesis via the native browser Web Speech API.

---

## 🛠️ System Architecture


```

[ Web Camera Feed (640x480 @ 30 FPS) ]
│
▼
[ Landmark Extraction (MediaPipe Hands: 21 Keypoints) ]
│
▼
[ Mathematical Normalization Layer ]
• Translation:  ΔP_i = P_i - P_wrist
• Scale Norm:   P̂_i = ΔP_i / max(||ΔP_i||) ──> (1 x 63) Tensor
│
▼
[ Inference Engine (ONNX Runtime / Quantized MLP) ]
│
▼
[ Temporal State Machine & Debounce Filter ]
• Sliding Window (w = 7) Majority Voting
• Epenthesis Suppression (Threshold >= 0.85)
│
┌────────┴────────┐
▼                 ▼
[ Real-Time Text Stream ] [ Native Web Speech API (Client TTS) ]

```

---

## 🧰 Tech Stack

| Component | Technology | Role & Justification |
| :--- | :--- | :--- |
| **Landmark Engine** | MediaPipe Hands, OpenCV | Isolates 21 3D keypoints without pixel-matrix compute overhead |
| **Feature Engineering** | NumPy, SciPy | Vectorized Euclidean distance scaling & spatial shifting |
| **Model & Inference** | TensorFlow/Keras, ONNX Runtime | CPU-optimized tabular classification with sub-5ms forward passes |
| **Backend Service** | FastAPI, Uvicorn, WebSockets | Async, full-duplex bidirectional stream handling at 30+ FPS |
| **Frontend UI** | HTML5, CSS3, Vanilla JavaScript | Lightweight canvas rendering and video frame acquisition |
| **Audio Engine** | Web Speech API (`SpeechSynthesis`) | Client-side native TTS execution with zero server latency |

---

## 📂 Project Structure

```text
anvaya/
├── model_training/
│   ├── dataset/              # Raw & normalized keypoint datasets (.csv / .npy)
│   ├── data_collection.py    # Landmark extraction & coordinate recording script
│   ├── train_model.py        # Model architecture, training & validation pipeline
│   ├── export_onnx.py        # Keras-to-ONNX conversion script for CPU speedup
│   └── gesture_model.onnx    # Optimized serialized model weights
├── backend/
│   ├── app.py                # FastAPI WebSocket server & inference router
│   ├── normalizer.py         # Origin shift & Euclidean scaling logic
│   ├── debounce.py           # Temporal rolling-window stabilization filter
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html            # Main web UI
│   ├── style.css             # Interface styling
│   └── script.js             # WebSocket streaming & Web Speech API integration
└── README.md

```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Modern Web Browser (Chrome / Edge / Firefox)
* Standard Web Camera

### 1. Clone & Setup Environment

```bash
git clone [https://github.com/your-username/anvaya.git](https://github.com/your-username/anvaya.git)
cd anvaya

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt

```

### 3. Model Training & Export (Optional if using pre-trained `.onnx`)

```bash
cd model_training
python data_collection.py    # Record gesture coordinates
python train_model.py        # Train the MLP classifier
python export_onnx.py        # Export to optimized ONNX format
cd ..

```

### 4. Run the Application

* **Start the FastAPI WebSocket Server:**
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

```


* **Launch the Frontend Client:**
Open `frontend/index.html` directly in your browser or run a simple static server:
```bash
cd ../frontend
python -m http.server 3000

```


Navigate to `http://localhost:3000` and grant camera permissions.

---

## 📊 Key Performance Targets

* **Frame Rate:** $\ge 30$ FPS sustained on 4-core consumer CPUs.
* **Inference Latency:** $< 10$ ms per frame.
* **Classification Accuracy:** $\ge 95\%$ categorical validation accuracy.
* **Jitter Mitigation:** $\ge 90\%$ reduction in frame transition false positives via debounce filtering.

---

## 👥 Contributors

* **Minor Project Group (B.Tech Semester V)** – Department of Computer Science & Engineering
