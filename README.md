# Anvaya
**Zero-GPU, Zero-Backend Real-Time Two-Hand Sign Recognition with Motion Support & Native TTS**

*B.Tech Semester V — Minor Project (Computer Science & Engineering)*

---

## 📌 What changed from the original design

This is v2 of the project. Two earlier design decisions were replaced after evaluating real-world constraints:

| Old Design (v1) | New Design (v2) | Why |
|---|---|---|
| Single-frame MLP classifier | **1D-CNN over a sequence of frames** | A single frame can only capture a static pose. A short sequence (window) lets the model recognize **motion**, not just shape. |
| One hand only (63 values/frame) | **Two hands, with graceful 1-hand fallback** (128 values/frame) | Real sign language regularly uses both hands. A missing hand is treated as a valid, well-defined input state — not an error. |
| Python FastAPI backend + WebSocket streaming | **Fully in-browser (client-side) inference** | Removes network latency and the requirement to run/host a server. The user only needs the internet to *load* the webpage once — after that, camera capture, landmark extraction, inference, and speech all run locally on-device. |

**Explicitly out of scope for this project** (documented here so it's a stated limitation, not a hidden gap):
- Full ASL/ISL grammar, facial-expression grammar, or continuous fluent signing.
- Robustness to heavy occlusion (fully hidden fingers) or very fast motion blur — a known, physical limitation of monocular (single 2D camera) hand tracking, not something normalization or model choice can fully fix.
- Live/continual learning — the model is trained once, offline, and only performs frozen inference at runtime (see `docs/GUIDE.md`, Phase 1 vs Phase 2).

---

## 🛠️ System Architecture (v2)

```
[ Webcam Feed, in-browser via getUserMedia() ]
        │
        ▼
[ MediaPipe Hands (JS/WASM, runs in-browser, up to 2 hands) ]
        │  21 landmarks per hand, per frame
        ▼
[ Per-Hand Normalization (JavaScript, runs in-browser) ]
        │  Wrist-centered translation + Euclidean-max scaling
        │  Missing hand → zero-filled + presence flag = 0
        │  Output: 128-value vector per frame
        ▼
[ Rolling Sequence Buffer (T = 24 frames, ~0.8s) ]
        │  Shape: (24, 128)
        ▼
[ 1D-CNN Inference (TensorFlow.js, runs in-browser) ]
        │  Trained offline, loaded as a frozen model — no learning at runtime
        ▼
[ Debounce / Majority-Vote Stabilizer (JavaScript) ]
        │  Suppresses flicker between predictions
        ▼
┌───────────────┴───────────────┐
▼                                ▼
[ Live Text Buffer / UI ]   [ Native Web Speech API (Client TTS) ]
```

No backend process runs during the presentation. `web_app/` is a static folder — open it (or serve it with any static file server) and everything happens in that browser tab.

---

## 🧰 Tech Stack

| Stage | Technology | Runs where |
|---|---|---|
| Data collection (one-time, dev only) | Python, OpenCV, MediaPipe Hands (Python) | Your laptop, during development |
| Preprocessing & training (one-time, dev only) | NumPy, TensorFlow/Keras | Your laptop, during development |
| Model export | `tensorflowjs_converter` | Your laptop, during development |
| Live inference (presentation / actual use) | MediaPipe Hands (JS), TensorFlow.js, Web Speech API | **Browser, on the presentation device, 100% client-side** |

---

## 📂 Project Structure

```text
anvaya-project/
├── data_collection/
│   └── capture_sequences.py     # Dev-only: records labeled (T,128) sequences per class
├── model_training/
│   ├── preprocess.py            # Builds train/val arrays from recorded sequences
│   ├── train_cnn.py             # Trains the 1D-CNN
│   ├── convert_tfjs.py          # Converts trained Keras model to browser-ready TF.js format
│   └── requirements.txt         # Python deps needed ONLY for data collection / training
├── web_app/                     # <-- THE ACTUAL DEPLOYED PROJECT (no backend)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── model/                   # Trained model files get copied here before presentation
├── docs/
│   ├── ANVAYA_PIPELINE.md       # Full technical / mathematical spec
│   └── GUIDE.md                 # Step-by-step: collect data -> train -> run presentation
└── README.md                    # This file
```

---

## 🚀 Quick Start

Full step-by-step instructions are in **`docs/GUIDE.md`** — read that before running anything. Short version:

1. **Collect data** (dev phase, Python): `python data_collection/capture_sequences.py`
2. **Train the model** (dev phase, Python): `python model_training/preprocess.py` → `python model_training/train_cnn.py` → `python model_training/convert_tfjs.py`
3. **Run the presentation** (no Python needed): open `web_app/index.html` in a browser (or serve the folder statically) and grant camera access.

---

## 📊 Default Configuration (v2)

- Sequence window: **T = 24 frames** (~0.8 seconds at 30 FPS) — long enough to capture a full motion sign, short enough to stay responsive.
- Feature vector per frame: **128 values** = `[hand1: 63 coords + 1 presence flag] + [hand2: 63 coords + 1 presence flag]`.
- Starter class list (10 signs, editable): `Hello, Thank_You, Yes, No, Please, Sorry, Help, Name, A, B` — a deliberate mix of one-hand/two-hand and static/motion signs so the pipeline is properly exercised. Expand this list only after the base pipeline works end-to-end.
- Samples per class: 60–100 sequences recommended for a first working model (see `docs/GUIDE.md` for why this is a floor, not a target).

---

## 👥 Contributors

* **Minor Project Group (B.Tech Semester V)** – Department of Computer Science & Engineering
