# Anvaya: Research-Driven System Pipeline & Phase-Wise Engineering Specification

A comprehensive technical architecture, mathematical normalization framework, and execution pipeline for **Anvaya** (Real-Time Hand Gesture and Sign Language Recognition with Temporal Debouncing & Native TTS).

---

## 1. System Engineering Architecture & Data Flow


```

[ Web Camera (640x480 @ 30 FPS) ]
│
▼
[ MediaPipe Hands (21 3D Keypoints) ]
│  Raw Coordinates: P_i = (x_i, y_i, z_i), i ∈ [0, 20]
▼
[ Geometric Normalization Engine ] ──► (Addresses Scale & Position Invariance Gap)
│  1. Origin Shift: ΔP_i = P_i - P_wrist
│  2. Scale Invariance: P̂_i = ΔP_i / max(||ΔP_i||)
│  Feature Tensor Shape: (1, 63)
▼
[ ONNX Quantized Inference Engine ] ──► (Addresses Zero-GPU Latency Gap)
│  Multi-Layer Perceptron (Dense 128 -> 64 -> 32 -> Softmax)
│  Sub-5ms Forward Pass (CPU Bound)
▼
[ Temporal State Machine & Debounce Filter ] ──► (Addresses Movement Epenthesis Gap)
│  Sliding Window (w = 7 frames) Majority Voting
│  Confidence Threshold: τ ≥ 0.85
▼
[ Output Layer ]
├── 1. Text UI Output Stream (Asynchronous UI DOM Update)
└── 2. Client-Side Speech Engine (Native Web Speech API)

```

---

## 2. Research Gaps & Engineering Solutions

| Identified Research Gap | Technical Limitation | Anvaya Engineering Solution |
| :--- | :--- | :--- |
| **1. Movement Epenthesis & Transition Flutter** | Intermediate movements between discrete signs generate transient noise, causing erratic classification spikes. | Implementation of a **Rolling-Window Majority Voting State Machine** ($w=7$ frames) with confidence gating ($\tau \ge 0.85$) to enforce stable token emission. |
| **2. Scale & Monocular Spatial Variance** | User distance from the web camera and positional shifts distort raw pixel coordinates. | **Vectorized Euclidean Max Normalization** centering landmarks relative to the wrist origin ($\Delta P_i = P_i - P_0$) and scaled by maximum Euclidean radius. |
| **3. High GPU Overhead vs. Browser Accessibility** | Deep 3D-CNNs and Graph Convolutions require dedicated CUDA hardware and introduce cloud streaming latency. | Lightweight tabular MLP exported to **ONNX Runtime**, coupled with client-side **Web Speech API** for zero-latency, zero-cost edge translation. |

---

## 3. Mathematical Normalization Engine

For each video frame containing a detected hand, the 21 landmarks are represented as:
$$P_i = (x_i, y_i, z_i), \quad \forall i \in \{0, 1, \dots, 20\}$$

### Step 1: Wrist-Centric Translation
Landmark $0$ (the anatomical wrist) is assigned as the coordinate system origin $(0, 0, 0)$:
$$\Delta x_i = x_i - x_0, \quad \Delta y_i = y_i - y_0, \quad \Delta z_i = z_i - z_0$$

### Step 2: Euclidean Maximum Distance Scaling
To ensure the feature representation remains scale-invariant across varying user-to-lens distances:
$$d_{\max} = \max_{i \in \{0, \dots, 20\}} \sqrt{\Delta x_i^2 + \Delta y_i^2 + \Delta z_i^2}$$
$$\hat{x}_i = \frac{\Delta x_i}{d_{\max}}, \quad \hat{y}_i = \frac{\Delta y_i}{d_{\max}}, \quad \hat{z}_i = \frac{\Delta z_i}{d_{\max}}$$

### Step 3: Feature Tensor Assembly
The normalized scalar coordinates are flattened into a 1D tabular vector:
$$\mathbf{V} = \left[ \hat{x}_0, \hat{y}_0, \hat{z}_0, \hat{x}_1, \hat{y}_1, \hat{z}_1, \dots, \hat{x}_{20}, \hat{y}_{20}, \hat{z}_{20} \right]^T \in \mathbb{R}^{63}$$

---

## 4. Phase-Wise Project Execution Plan

### Phase I: Dataset Engineering & Mathematical Preprocessing
* **Step 1.1 (Data Acquisition Script):** Write `data_collection.py` using OpenCV and MediaPipe Hands to capture 300–500 samples per class at $640 \times 480$ resolution.
* **Step 1.2 (Vector Normalization Pipeline):** Embed origin-shifting and Euclidean scaling into a standalone module (`normalizer.py`) to process coordinates in real time.
* **Step 1.3 (Dataset Serialization):** Store feature vectors into partitioned `.csv` / `.npy` files with corresponding class integer labels.

### Phase II: Model Architecture, Training & ONNX Optimization
* **Step 2.1 (Neural Classifier Architecture):** Construct a Multi-Layer Perceptron (MLP) in TensorFlow/Keras:
  * `Input`: Shape $(63,)$
  * `Dense Layer 1`: 128 units, ReLU, Batch Normalization, Dropout(0.2)
  * `Dense Layer 2`: 64 units, ReLU, Dropout(0.2)
  * `Dense Layer 3`: 32 units, ReLU
  * `Output Layer`: $N$ Classes, Softmax activation
* **Step 2.2 (Training & Validation):** Train using Adam optimizer ($\eta = 0.001$) and categorical cross-entropy loss with an 80/20 train-test split.
* **Step 2.3 (ONNX Serialization):** Convert the `.h5` model to `gesture_model.onnx` using `tf2onnx` to optimize CPU forward pass speed ($< 5$ ms).

### Phase III: Backend Streaming & Temporal Debounce Logic
* **Step 3.1 (FastAPI WebSocket Service):** Establish full-duplex WebSocket endpoints in `app.py` capable of ingesting landmark vectors and broadcasting inference results asynchronously.
* **Step 3.2 (Temporal Debounce Algorithm):** Implement a rolling queue of capacity $w = 7$. A token is emitted only if:
  $$\text{Count}(\text{Predicted Class}) \ge 5 \quad \text{and} \quad \bar{p}_{\text{confidence}} \ge 0.85$$
  This prevents intermittent misclassifications during hand entry, exit, and gesture transitions.

### Phase IV: Frontend Web UI & Speech Integration
* **Step 4.1 (Webcam Ingestion Interface):** Configure HTML5 `<video>` and `<canvas>` stream using `navigator.mediaDevices.getUserMedia()`.
* **Step 4.2 (Prediction Display & Sentence Assembler):** Render real-time confidence meters and a persistent dynamic sentence buffer on screen.
* **Step 4.3 (Zero-Latency Speech Engine):** Hook output emissions into the browser's native `window.speechSynthesis` API:
  ```javascript
  function emitSpeech(word) {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(word);
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }

### Phase V: Verification, Benchmarking & Testing

* **Benchmark 1:** Frame Rate (Target: $\ge 30$ FPS on commodity quad-core CPUs).
* **Benchmark 2:** End-to-End Latency (Target: $< 30$ ms from camera capture to UI/audio output).
* **Benchmark 3:** Validation Accuracy & Jitter Reduction (Target: $\ge 95\%$ accuracy, $\ge 90\%$ false positive transition suppression).

```

```