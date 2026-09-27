# Anvaya v2: Two-Hand, Motion-Aware, Fully In-Browser Sign Recognition Pipeline

This supersedes the v1 pipeline (single-hand MLP + Python backend). Kept in this repo for
historical reference is the rationale for *why* each change was made — see the table in the
root `README.md`.

---

## 1. System Architecture & Data Flow

```
[ Webcam Feed (in-browser, getUserMedia) ]
        │
        ▼
[ MediaPipe Hands — JS/WASM build, num_hands = 2 ]
        │  Per detected hand: 21 3D keypoints P_i = (x_i, y_i, z_i), i in [0,20]
        │  Plus a "handedness" label (Left / Right) per detected hand
        ▼
[ Per-Hand Geometric Normalization ]  ──► (Scale & position invariance, per hand independently)
        │  1. Origin shift per hand: ΔP_i = P_i − P_wrist(that hand)
        │  2. Scale invariance per hand: P̂_i = ΔP_i / max(‖ΔP_i‖)
        │  3. Missing hand → 63 zero values + presence flag = 0
        │  Per-frame Feature Vector Shape: (1, 128)
        ▼
[ Rolling Sequence Buffer ]
        │  Holds the last T = 24 frames → shape (24, 128)
        ▼
[ 1D-CNN Inference Engine — TensorFlow.js, in-browser ]  ──► (Zero-GPU, zero-server latency)
        │  Conv1D stack over the time axis, trained offline, loaded frozen (no runtime learning)
        ▼
[ Temporal Debounce / Majority-Vote Stabilizer ]  ──► (Suppresses flicker between predictions)
        │  Rolling window of recent predictions, majority vote + confidence gate
        ▼
[ Output Layer ]
├── 1. Text UI Output Stream (DOM update)
└── 2. Client-Side Speech Engine (native window.speechSynthesis)
```

**Key architectural rule:** nothing after the webcam capture step touches a network. All boxes
in the diagram above run inside the browser tab. The only network dependency is the initial
page load (HTML/CSS/JS/model files) and, optionally, the CDN-hosted MediaPipe/TensorFlow.js
libraries if you choose not to self-host them.

---

## 2. Research Gaps & Engineering Solutions (updated)

| Research Gap | Technical Limitation | Anvaya v2 Solution | Known Residual Limitation |
| :--- | :--- | :--- | :--- |
| **Movement Epenthesis & Transition Flutter** | Intermediate movements between discrete signs generate transient noise. | Rolling-window majority-vote debounce over CNN predictions, plus confidence gating. | Very fast sign-to-sign transitions can still be missed — acceptable per project scope ("miss some signs is fine"). |
| **Signs Defined by Motion, Not Static Pose** | A single-frame classifier structurally cannot distinguish a moving sign from a static one. | Replaced single-frame MLP with a **1D-CNN over a 24-frame sequence window**, so the model sees a short trajectory, not one snapshot. | Signs that take meaningfully longer than ~0.8s to articulate may need a larger T in future work. |
| **Two-Hand Signs & Single-Hand Fallback** | Real sign language regularly uses both hands; a fixed one-hand-only input shape cannot represent this. | 128-value per-frame vector (64 per hand: 63 coords + 1 presence flag). Missing hand is zero-filled with flag = 0, not dropped or treated as an error. | Hand identity swap (left/right) confusion is possible if MediaPipe's handedness label flips mid-sequence — mitigated but not fully eliminated. |
| **Scale & Monocular Spatial Variance** | User distance from camera and positional shifts distort raw pixel coordinates. | Per-hand wrist-centered origin shift + Euclidean-max scaling (unchanged from v1, just applied independently per hand). | Still 2D-camera-limited; true depth ambiguity is not resolved by this normalization. |
| **Occlusion (closed fists / hidden fingers)** | MediaPipe estimates hidden landmarks rather than observing them; accuracy drops when fingers are curled or blocked by the other hand. | None — this is a stated, physical limitation of monocular tracking. | **Explicitly out of scope.** Class list should favor signs where both hands stay reasonably visible to the camera. |
| **GPU Overhead vs. Browser Accessibility** | Deep 3D-CNNs / raw-pixel video models need dedicated GPU hardware. | Small 1D-CNN operating on **landmark coordinates**, not raw pixel frames — kept intentionally lightweight, runs via TensorFlow.js on CPU/WebGL in-browser. | N/A — this is the design goal being satisfied. |
| **Network Latency from Server Round-Trips** | v1's WebSocket-to-Python-backend design added per-frame network latency and required same-network deployment. | **Backend removed entirely.** MediaPipe (JS) + TensorFlow.js run inference fully client-side. | Internet is still required once, to load the page and libraries (or self-host them for fully offline use — see `docs/GUIDE.md`). |

---

## 3. Mathematical Normalization Engine (per hand)

Applied **independently to each detected hand** (up to 2 per frame). For a detected hand, the
21 landmarks are:
$$P_i = (x_i, y_i, z_i), \quad \forall i \in \{0, 1, \dots, 20\}$$

### Step 1: Wrist-Centric Translation
Landmark 0 (the wrist) is the origin for *that hand*:
$$\Delta x_i = x_i - x_0, \quad \Delta y_i = y_i - y_0, \quad \Delta z_i = z_i - z_0$$

### Step 2: Euclidean Maximum Distance Scaling
$$d_{\max} = \max_{i \in \{0, \dots, 20\}} \sqrt{\Delta x_i^2 + \Delta y_i^2 + \Delta z_i^2}$$
$$\hat{x}_i = \frac{\Delta x_i}{d_{\max}}, \quad \hat{y}_i = \frac{\Delta y_i}{d_{\max}}, \quad \hat{z}_i = \frac{\Delta z_i}{d_{\max}}$$

### Step 3: Per-Hand Vector Assembly (64 values)
$$\mathbf{H} = \left[ \hat{x}_0, \hat{y}_0, \hat{z}_0, \dots, \hat{x}_{20}, \hat{y}_{20}, \hat{z}_{20}, f \right]^T \in \mathbb{R}^{64}$$
where $f \in \{0, 1\}$ is the **presence flag**: $f = 1$ if this hand was detected in the
frame, $f = 0$ if not (in which case all 63 coordinate values are also set to 0).

### Step 4: Two-Hand Frame Vector (128 values)
Hands are assigned to a fixed slot — **Left** and **Right**, per MediaPipe's own handedness
classification — so the model always sees hand identity in a consistent position, regardless
of detection order:
$$\mathbf{V}_{\text{frame}} = \left[ \mathbf{H}_{\text{left}} \;\Vert\; \mathbf{H}_{\text{right}} \right] \in \mathbb{R}^{128}$$

### Step 5: Sequence Tensor Assembly (the CNN input)
The last $T = 24$ frame vectors are stacked in time order:
$$\mathbf{S} = \left[ \mathbf{V}_{\text{frame}}^{(t-23)}, \dots, \mathbf{V}_{\text{frame}}^{(t)} \right] \in \mathbb{R}^{24 \times 128}$$

This is the tensor fed into the 1D-CNN.

---

## 4. Phase-Wise Project Execution Plan (v2)

### Phase I: Sequence Dataset Engineering (dev only, Python)
* **Step 1.1:** `data_collection/capture_sequences.py` — OpenCV + MediaPipe Hands (`num_hands=2`),
  records a `(24, 128)` sequence per sample, for a chosen class label, with a countdown before
  each recording so the signer can prepare.
* **Step 1.2:** Normalization (Section 3 above) is applied **at collection time**, so saved
  `.npy` files are already normalized sequences, not raw pixels or raw landmarks.
* **Step 1.3:** Sequences are stored as `dataset/<class_name>/seq_###.npy`, one file per sample.

### Phase II: Preprocessing, 1D-CNN Training & TF.js Export (dev only, Python)
* **Step 2.1 (Preprocessing):** `model_training/preprocess.py` loads all `.npy` sequences,
  builds `X` of shape `(N, 24, 128)` and integer-encoded labels `y`, saves a train/val split
  plus a `label_map.json`.
* **Step 2.2 (1D-CNN Architecture):**
  * `Input`: shape `(24, 128)`
  * `Conv1D`: 64 filters, kernel size 3, ReLU, followed by BatchNorm
  * `Conv1D`: 128 filters, kernel size 3, ReLU, followed by BatchNorm, Dropout(0.3)
  * `GlobalAveragePooling1D` (collapses the time axis — keeps the model small and less prone
    to overfitting on a small minor-project dataset)
  * `Dense`: 64 units, ReLU, Dropout(0.3)
  * `Output`: `N` classes, Softmax
* **Step 2.3 (Training):** Adam optimizer, categorical cross-entropy, 80/20 train/val split,
  early stopping on validation loss.
* **Step 2.4 (Export):** `model_training/convert_tfjs.py` runs `tensorflowjs_converter` to
  produce `model.json` + weight shard files, ready to be loaded by TensorFlow.js in the browser.

### Phase III: In-Browser Runtime (this replaces the old "Backend Streaming" phase entirely)
* **Step 3.1:** `web_app/app.js` loads MediaPipe Hands (JS build) with `maxNumHands: 2`.
* **Step 3.2:** On every camera frame, `app.js` builds the 128-value vector (Section 3, Steps
  1–4) and pushes it into a rolling buffer of the last 24 frames.
* **Step 3.3:** Once the buffer is full, the `(24, 128)` tensor is passed to the TensorFlow.js
  model for a forward pass — this is a frozen model, no training happens here (see `docs/GUIDE.md`).
* **Step 3.4 (Debounce):** Predictions are pushed into a short rolling history; a token is only
  emitted when a class dominates that history and average confidence clears a threshold — same
  principle as v1's debounce, just implemented in JavaScript instead of a Python backend.

### Phase IV: Frontend UI & Speech (mostly unchanged from v1, now the *only* runtime layer)
* **Step 4.1:** `<video>` + `<canvas>` via `navigator.mediaDevices.getUserMedia()`.
* **Step 4.2:** Live confidence display + a persistent sentence buffer in the DOM.
* **Step 4.3:** Confirmed tokens are spoken via `window.speechSynthesis`.

### Phase V: Verification, Benchmarking & Testing
* **Benchmark 1:** Frame Rate — target ≥ 20–25 FPS sustained on a mid-range laptop/mobile CPU
  in-browser (lowered from v1's 30 FPS target to reflect the honest cost of running MediaPipe +
  a CNN fully client-side; re-measure and adjust once the real pipeline is running).
* **Benchmark 2:** End-to-End Latency — from gesture completion to spoken/text output, target
  under ~1 second (bounded mostly by the 24-frame/~0.8s buffer window itself, not compute).
* **Benchmark 3:** Validation Accuracy — target ≥ 90% on the held-out validation split for the
  10-class starter list (lowered from v1's 95% target — sequence/motion classification on a
  small self-collected dataset is a harder problem than single-frame static classification, and
  an honest, achievable target is stronger for a report than an inflated one).

---

## 5. Explicit Design Decisions (for your project report)

Document these as deliberate engineering trade-offs, not oversights — this is exactly the kind
of reasoning evaluators want to see:

1. **Why 1D-CNN instead of LSTM/GRU for motion?** A small 1D-CNN with global average pooling is
   lighter-weight and faster to run in TensorFlow.js than a recurrent model, and is sufficient
   for short, fixed-length (~0.8s) gesture windows. LSTMs/GRUs are a reasonable future upgrade
   if longer or more complex motion sequences are added later.
2. **Why zero-fill the missing hand instead of dropping the frame or using a variable-size
   input?** Neural networks need fixed-size input tensors. Zero-filling plus an explicit
   presence flag lets the model learn "this hand is absent" as a distinct, meaningful signal,
   rather than guessing from all-zero coordinates alone (which could be confused with a real
   hand positioned exactly at its own wrist — the flag removes that ambiguity).
3. **Why remove the backend instead of optimizing it?** The backend's core cost was the
   per-frame network round trip, which no amount of backend optimization removes — only
   eliminating the round trip does. Moving inference client-side directly satisfies the
   "no backend, no latency" requirement rather than working around it.
