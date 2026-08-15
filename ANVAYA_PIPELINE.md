# Anvaya: System Pipeline & Engineering Architecture

An end-to-end technical roadmap, execution pipeline, and hardware-optimized integration guide for **Anvaya** (Hand Gesture and Sign Language Detection $\to$ Text $\to$ Speech).

---

## 1. End-to-End System Architecture

```
[ Web Camera Feed (Frontend) ]
             │  (640x480 @ 30 FPS / WebRTC or WebSocket)
             ▼
[ Stream Ingestion & Preprocessing (Backend / MediaPipe) ]
             │  Extract 21 (x, y, z) Hand Keypoints
             ▼
[ Landmark Normalization Pipeline ]
             │  Origin Shift (Wrist = 0,0,0) + Scale Invariance
             │  Vector Shape: (1, 63) for Static / (30, 63) for Dynamic
             ▼
[ Inference Engine (Trained ML/DL Model) ]
             │  Softmax Probability > Threshold (e.g., 0.80)
             ▼
[ Gesture Stabilization & Text Buffer ]
             │  Debounce / Rolling Window Filter -> Sentence Assembler
             ▼
[ Output Layer ]
     ├── 1. Text UI Output (Real-time display)
     └── 2. Speech Synthesis (Web Speech API / TTS Engine)
```

---

## 2. Component Breakdown & Tech Stack

| Module | Responsibility | Recommended Tech Stack | Performance Rationale |
|---|---|---|---|
| **Data Collection** | Capturing frame landmark vectors per gesture class | Python 3.10+, OpenCV, MediaPipe Hands, NumPy | Lightweight coordinate logging directly into `.csv` / `.npy` files. |
| **Model Training** | Neural network / classifier training on tabular keypoint data | Scikit-Learn, TensorFlow / Keras, Pandas | Fast CPU training (< 1 min for MLP on tabular coordinate datasets). |
| **Model Export** | Serialization for ultra-low latency CPU inference | H5, ONNX Runtime (`onnxruntime`), Joblib | Sub-5ms CPU inference without GPU dependencies. |
| **Backend & Ingestion** | Real-time frame streaming and model serving | FastAPI / Flask, WebSockets (`websockets`), Uvicorn | Async WebSocket handles 30 FPS bidirectional communication with low overhead. |
| **Frontend UI** | Video capture, landmark visualizer, output stream | HTML5 Canvas, Vanilla JS / React, WebRTC / MediaDevices API | Native browser camera access with zero third-party client weight. |
| **Speech Engine** | Text-to-Speech (TTS) conversion | Web Speech API (`window.speechSynthesis`) or gTTS / pyttsx3 | Client-side native browser TTS requires 0 server compute or latency. |

---

## 3. Detailed Execution Pipeline (Phase-by-Phase)

### Phase 1: Landmark Extraction & Dataset Engineering
1. **Camera Ingestion:** Initialize OpenCV `cv2.VideoCapture(0)` locked to $640 \times 480$ resolution.
2. **Keypoint Tracking:** Pass frames to MediaPipe Hands with `max_num_hands=1` (or 2), `min_detection_confidence=0.7`, and `min_tracking_confidence=0.5`.
3. **Mathematical Normalization:**
   * Extract $(x_i, y_i, z_i)$ coordinates for all 21 keypoints ($i = 0 \dots 20$).
   * Set the wrist coordinate $(x_0, y_0, z_0)$ as the origin:
     $$\Delta x_i = x_i - x_0, \quad \Delta y_i = y_i - y_0, \quad \Delta z_i = z_i - z_0$$
   * Normalize by max Euclidean distance to make scale invariant to camera distance:
     $$d_{\max} = \max_{i} \sqrt{\Delta x_i^2 + \Delta y_i^2 + \Delta z_i^2}$$
     $$\hat{x}_i = \frac{\Delta x_i}{d_{\max}}, \quad \hat{y}_i = \frac{\Delta y_i}{d_{\max}}, \quad \hat{z}_i = \frac{\Delta z_i}{d_{\max}}$$
4. **Storage:** Flatten into a 63-element feature vector and append with a label to `dataset/landmarks_data.csv`. Target 300–500 samples per gesture.

---

### Phase 2: Model Design & Training

#### Option A: Static Signs (Alphabets, Discrete Commands)
* **Architecture:** Multi-Layer Perceptron (MLP)
  * `Input Layer`: (63,)
  * `Dense Layer 1`: 128 units, ReLU, Dropout(0.2)
  * `Dense Layer 2`: 64 units, ReLU, Dropout(0.2)
  * `Dense Layer 3`: 32 units, ReLU
  * `Output Layer`: N classes, Softmax
* **Optimizer:** Adam (lr=0.001), Loss: `sparse_categorical_crossentropy` / `categorical_crossentropy`.

#### Option B: Dynamic Gestures (Words, Full Motions)
* **Architecture:** LSTM / GRU Network
  * `Input Layer`: (30 frames, 63 keypoints)
  * `LSTM Layer 1`: 64 units, return_sequences=True
  * `LSTM Layer 2`: 32 units
  * `Dense Layer`: 32 units, ReLU
  * `Output Layer`: N classes, Softmax

---

### Phase 3: Web & Backend Integration

1. **Option 1 (Client-Side MediaPipe - Lowest Latency):**
   * Run `@mediapipe/camera_utils` and `@mediapipe/hands` directly in the browser via JavaScript CDN.
   * Send the 63 normalized coordinates over WebSocket to FastAPI.
   * FastAPI runs `model.predict(vector)` and returns `{ "label": "A", "confidence": 0.94 }`.
   * **Bandwidth:** < 1 KB/s per frame (zero video data sent over network).

2. **Option 2 (Server-Side OpenCV Processing):**
   * Browser grabs frame from `<video>` element, converts to base64 JPEG, and streams over WebSocket.
   * FastAPI backend decodes JPEG, passes through MediaPipe Python, runs inference, and responds with the label.

---

### Phase 4: Output Synthesis (Text & Voice)

1. **Debounce / Smoothing Filter:**
   * Maintain a rolling buffer of the last 5–10 predictions.
   * Emit the character/word only if the same label appears consecutively for $\ge 7$ frames to eliminate jitter.
2. **Sentence Buffer:**
   * Combine characters into words; trigger an end-of-gesture delimiter (e.g., hand drop or specific gesture) to finalize words.
3. **Audio Synthesis:**
   * Invoke client-side browser API:
     ```javascript
     function speak(text) {
       const utterance = new SpeechSynthesisUtterance(text);
       utterance.rate = 1.0;
       window.speechSynthesis.speak(utterance);
     }
     ```

---

## 4. Hardware Optimization & Deployment Precautions

* **Resolution Caps:** Keep input webcam capture at $640 \times 480$ (30 FPS). Higher resolutions only add latency with zero gain in landmark accuracy.
* **Vectorized Preprocessing:** Always use NumPy array slicing rather than Python loops when subtracting wrist offsets.
* **Format Conversion:** For CPU deployment (e.g. Ryzen 5 7520U), convert `.h5` model to ONNX runtime format for instantaneous forward passes.
* **Thread Decoupling:** Keep video capture and audio playback on separate browser threads to ensure continuous frame processing without UI freezing.
