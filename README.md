# Project Anvaya
**Real-time two-hand sign recognition with motion support and native text-to-speech, running entirely in the browser.**

*B.Tech Semester V — Minor Project, Computer Science & Engineering*

No backend server. No GPU. The camera, hand tracking, model inference and speech all run inside the browser tab.

---

## What it does

Point a webcam at yourself and sign. Anvaya tracks up to two hands, feeds a short window of hand movement (24 frames, about 0.8 s) into a small 1D-CNN, and shows the recognised sign as text. It can also read the sentence aloud with adjustable speed, pitch and voice.

- **Two hands, with one-hand fallback.** A missing hand is a valid input (zero-filled with a presence flag), not an error.
- **Motion-aware.** The model sees a sequence of frames, not a single pose.
- **Private and offline-capable.** Nothing leaves the device. The internet is only needed to load the page and libraries once.

## Pages

| Page | File | Purpose |
|---|---|---|
| Landing | `web_app/index.html` | Hero, feature strip, "Launch Gesture Studio" button |
| Gesture Studio | `web_app/studio.html` | Live camera, translation box, speech controls, session log |

Learn, Community, About, Sign In, Settings, Help and the footer links are **visual placeholders** for now.

---

## Architecture

```
Webcam (getUserMedia)
  -> MediaPipe Hands (JS/WASM, up to 2 hands, 21 landmarks each)
  -> Per-hand normalization (wrist-centred, max-distance scaled)
  -> 128-value frame vector  [left hand 63 + flag | right hand 63 + flag]
  -> Rolling buffer, 24 frames -> (24, 128)
  -> 1D-CNN (TensorFlow.js, frozen weights, inference only)
  -> Majority-vote debounce + confidence gate
  -> Text box + session log + Web Speech API
```

Training happens once, offline, in Python. During a demo the model never learns. Full maths and design decisions are in `docs/ANVAYA_PIPELINE.md`.

---

## Project structure

```text
anvaya-project/
├── data_collection/                 # DEV ONLY (Python)
│   ├── capture_sequences.py         # records labelled (24,128) sequences from your webcam
│   └── normalizer.py                # normalization + frame-vector logic (mirrored in app.js)
│
├── model_training/                  # DEV ONLY (Python)
│   ├── preprocess.py                # builds train/val arrays + label_map.json
│   ├── train_cnn.py                 # trains the 1D-CNN
│   ├── convert_tfjs.py              # exports the model for the browser
│   └── requirements.txt             # Python dependencies (pinned, tested)
│
├── web_app/                         # THE DEPLOYED PROJECT (static files, no backend)
│   ├── index.html                   # landing page (nav, hero, feature strip, footer)
│   ├── studio.html                  # Gesture Studio page (camera, translation, controls, footer)
│   ├── app.js                       # camera + MediaPipe + inference + debounce
│   ├── controls.js                  # speed / pitch / voice, copy, clear
│   ├── session-log.js               # session history (browser localStorage)
│   ├── assets/                      # (add files here) hero illustrations and other images
│   └── model/                       # trained model files land here (see guides)
│
├── docs/
│   ├── BEGINNER_SETUP_GUIDE.md      # first-time walkthrough: install -> record -> train -> demo
│   ├── GUIDE.md                     # short reference version of the same steps
│   └── ANVAYA_PIPELINE.md           # technical spec and design decisions
└── README.md
```

Script load order in `studio.html` matters: `session-log.js`, then `controls.js`, then `app.js`.

---

## Quick start

Read `docs/BEGINNER_SETUP_GUIDE.md` first if this is your first project. The short version:

```bash
# 1. Setup (Python 3.10, 3.11 or 3.12)
cd model_training
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Record data (60-100 samples per sign), then train and export
cd ../data_collection && python capture_sequences.py
cd ../model_training
python preprocess.py && python train_cnn.py && python convert_tfjs.py
cp -r model_web/* ../web_app/model/ && cp label_map.json ../web_app/model/

# 3. Run the demo (no Python needed for the app itself)
cd ../web_app && python -m http.server 3000
# open http://localhost:3000
```

---

## Tech stack

| Stage | Technology |
|---|---|
| Data collection | Python, OpenCV, MediaPipe Hands 0.10.21 |
| Training | TensorFlow 2.17 (legacy Keras via `tf_keras`), scikit-learn |
| Model export | `convert_tfjs.py` (writes TensorFlow.js format directly) |
| Live app | HTML, Tailwind CSS (CDN, no build step), vanilla JavaScript, MediaPipe Hands (JS), TensorFlow.js 4.20, Web Speech API |

Two version choices are deliberate, so please do not "upgrade" them without testing:
- **MediaPipe 0.10.21**: newer releases removed the `mp.solutions.hands` API the capture script uses.
- **Legacy Keras (`TF_USE_LEGACY_KERAS=1`)**: models saved by Keras 3 cannot be loaded by TensorFlow.js.

---

## What has been verified

Checked with automated tests in a clean environment:
- `pip install -r requirements.txt` resolves with no conflicts; `preprocess.py`, `train_cnn.py` and `convert_tfjs.py` run end to end on synthetic data.
- The exported model loads in TensorFlow.js 4.20 over HTTP (the same call `app.js` makes) and its predictions match Keras to about 1e-8.
- The JavaScript frame-vector code produces the same numbers as the Python training code for two-hand, one-hand and no-hand frames.
- All element IDs used by the JS exist in `studio.html`; all CSS/JS references resolve; all Python and JS files pass syntax checks.

**Not yet verified (needs a real camera and browser):** the visual layout, live MediaPipe detection, real-sign accuracy, and left/right hand mapping on a real webcam. Test these before your presentation.

## Known limitations

- Static and slow, clearly posed signs work best. Very fast motion, motion blur, and fully hidden fingers are physical limits of single-camera hand tracking.
- Accuracy depends mostly on how much varied data you record. 60-100 samples per sign is a floor, not a guarantee.
- Recording runs at your laptop's loop speed while the browser samples at the camera's frame rate. If motion signs underperform, check both run at a similar FPS.
- Sign In, Learn, Community and About are placeholders; there are no user accounts.

---

## Design reference and updating the UI

The UI is built directly from a Google Stitch export (Material-3-style teal palette, glassmorphism cards, Plus Jakarta Sans, Material Symbols icons). There is no separate CSS file — the design system lives in the `tailwind.config` script block at the top of **both** `index.html` and `studio.html` (loaded via the Tailwind CDN script, no build step). Change a colour/spacing/font value there and the whole page follows. Because it's duplicated in two files, **edit the config block in both `index.html` and `studio.html` if you change it** — same rule applies to the shared `<header>`/`<footer>` markup, which is also plain, duplicated HTML (no templating engine).

Both illustrations (the hero doodle and the "Reading your signs..." mascot) are still hotlinked to Stitch's own image host as placeholders. Before final submission, save your own images into `web_app/assets/` and swap the `src="https://lh3.googleusercontent.com/..."` attributes in `index.html`/`studio.html` to point at them instead — depending on a third-party host is fine for now, not for a graded submission.

---

## Contributors

* **Minor Project Group (B.Tech Semester V)** — Department of Computer Science & Engineering