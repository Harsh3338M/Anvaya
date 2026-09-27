# Anvaya — Run Guide

This guide is split into two phases that are **completely separate** and happen at different
times:

- **Phase 1 — Development (you, before the presentation):** collect data, train the model,
  export it for the browser. Needs Python. Happens once (or a few times, as you iterate).
- **Phase 2 — Presentation (the actual demo):** just open a webpage. No Python, no backend, no
  terminal running in the background. The model is already trained and frozen — it only makes
  predictions, it does not learn anything new from the live camera feed.

Do not skip Phase 1 — Phase 2 will not work without a trained model already sitting in
`web_app/model/`.

---

## Phase 1: Development — Collecting Data & Training the Model

### 1.0 One-time setup

```bash
cd anvaya-project/model_training
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1.1 Decide your class list

Default starter list (edit `CLASSES` in `data_collection/capture_sequences.py` to change it):

```
Hello, Thank_You, Yes, No, Please, Sorry, Help, Name, A, B
```

Keep the list small at first (8–12 classes). Add more only after the full pipeline works
end-to-end with a small set — it's much easier to debug 10 classes than 40.

### 1.2 Record sequences

```bash
cd anvaya-project/data_collection
python capture_sequences.py
```

- The script asks you to pick a class label, then shows a webcam preview.
- Press the trigger key to start a countdown, perform the sign, and it auto-records **24
  frames (~0.8 seconds)** as one sample.
- Repeat for **60–100 samples per class**. Vary your hand position, distance from camera, and
  slightly vary the speed/timing of the sign across samples — this is what actually makes the
  model generalize, far more than raw sample count alone.
- For two-hand signs, keep both hands in frame. For one-hand signs, keep the other hand out of
  frame or resting out of camera view — the script will correctly record the missing hand as
  zero-filled with its presence flag off.
- Samples are saved to `data_collection/dataset/<class_name>/seq_###.npy`.

**Reality check:** 60–100 samples per class is a workable floor for a minor project, not a
guarantee of high accuracy. If validation accuracy is poor after training (Step 1.4), the first
fix is almost always "record more, more varied samples" before changing the model architecture.

### 1.3 Preprocess into train/val arrays

```bash
cd ../model_training
python preprocess.py
```

Reads everything under `data_collection/dataset/`, builds `X_train`, `y_train`, `X_val`,
`y_val`, and `label_map.json` (maps class names to integer indices — you'll need this file in
`web_app/` too, so the script copies it there automatically).

### 1.4 Train the 1D-CNN

```bash
python train_cnn.py
```

Trains the model described in `docs/ANVAYA_PIPELINE.md` Section 4, Phase II. Saves the best
checkpoint as `model_training/gesture_cnn.h5` and prints final validation accuracy — check this
number before moving on. If it's low, go back to Step 1.2 and record more/cleaner data before
tweaking hyperparameters; a small, noisy dataset is almost always the actual bottleneck.

### 1.5 Export to browser format

```bash
python convert_tfjs.py
```

Converts `gesture_cnn.h5` into `model_training/model_web/` (a `model.json` plus weight shard
files). Copy that folder's **contents** into `web_app/model/`:

```bash
cp -r model_web/* ../web_app/model/
cp label_map.json ../web_app/model/
```

At this point `web_app/model/` should contain `model.json`, one or more `.bin` weight files,
and `label_map.json`. This is the only hand-off between the Python phase and the browser phase.

---

## Phase 2: Presentation — Running the Live Demo

**No Python. No terminal process running during the demo. No backend.**

### 2.1 Serve the `web_app/` folder

Browsers block camera access on files opened directly as `file://` in some cases, so serve it
with any lightweight static server. One simple option (Python is only being used here as a
static file server, not for any inference logic):

```bash
cd anvaya-project/web_app
python -m http.server 3000
```

Then open `http://localhost:3000` in Chrome, Edge, or Firefox.

### 2.2 Grant camera access

The browser will prompt for camera permission on first load — allow it.

### 2.3 Confirm it's fully offline-capable (optional but worth doing before the actual
presentation)

- MediaPipe Hands and TensorFlow.js are loaded from CDN links in `index.html` by default for
  simplicity. If your presentation venue has unreliable internet, download those library files
  once ahead of time and point the `<script>` tags in `index.html` to the local copies instead
  — everything else (your model, your code) already runs fully offline.
- After the page has loaded once successfully, turn off Wi-Fi and confirm the camera, detection,
  and speech output still work. If you self-hosted the libraries, this will succeed completely.

### 2.4 During the demo

- The live video feed and detected hand landmarks are shown for the audience to see what the
  system is tracking.
- Perform a sign; after roughly 0.8–1 second of holding/performing it, the debounce logic should
  emit a stable text token and speak it aloud.
- If a prediction doesn't come through, that's an expected, documented limitation (fast motion,
  partial occlusion, or a class outside your trained list) — see the "Known Residual
  Limitations" table in `docs/ANVAYA_PIPELINE.md` if asked about it.

---

## Common Questions

**Q: Will the model keep learning from the audience/live demo?**
No. Training happened once, offline, in Phase 1. During Phase 2 the model's weights are frozen
— it only runs inference (a forward pass), never updates itself. This is standard practice for
supervised deep learning systems, not a simplification specific to this project.

**Q: Can I retrain later after collecting more data?**
Yes — repeat Steps 1.2 through 1.5. Each retrain fully replaces the previous model files in
`web_app/model/`; there's no incremental/continual learning happening.

**Q: What if I want to add more classes later?**
Add the new class name to `CLASSES` in `capture_sequences.py`, record samples for it, and rerun
Steps 1.3–1.5. You do need to retrain from scratch on the full class list (old + new) — the
model's output layer size is fixed to the number of classes it was trained on.
