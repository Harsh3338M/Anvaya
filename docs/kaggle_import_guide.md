# Using the Kaggle ISL Dataset as Placeholder Training Data

This guide covers a **separate, optional** data source: a static-image ISL dataset from
Kaggle, used to bootstrap classes you haven't recorded real motion data for yet. If you
haven't read `docs/BEGINNER_SETUP_GUIDE.md` yet, read that first — this guide assumes you
already have Python set up and have recorded at least a little data with
`capture_sequences.py`.

---

## Part 0: Understand what this is (and isn't) for

The Kaggle dataset — [ISL-dataset by ananyaarya22](https://www.kaggle.com/datasets/ananyaarya22/isl-data)
— has **36 classes (digits 0–9, letters A–Z), about 1,000 JPG photos per class**. These are
**static photos**, not videos. A photo has no motion in it.

This project's model is a *motion* model — it looks at a 24-frame window (about 0.8 seconds),
not a single instant. To use static photos with it, each photo is converted into a "held pose"
sequence: the same single frame repeated 24 times, as if you froze in that pose for the whole
window.

**What this is good for:** bulking up classes that genuinely are static poses — most alphabet
letters and digit signs fall into this category. It's a fast way to get a class trained before
you've recorded much of your own data for it.

**What this is NOT good for:** motion-based signs like "Hello" or "Thank You" — a held pose
captures none of the movement that actually defines those signs. Don't expect this dataset to
help with your motion classes at all. Only use it for static ones.

**How it's kept separate, so it can't quietly hurt your real model:** Kaggle data lands in its
own folder (`data_collection/dataset_kaggle_static/`), never mixed into
`data_collection/dataset/` where your real recordings live. `preprocess.py` automatically stops
using Kaggle data for any class once you have 20 or more real samples for it. You'll see exactly
which classes are using which data source every time you run `preprocess.py` — it prints a
report.

---

## Part 1: Download the Dataset

1. Go to [kaggle.com/datasets/ananyaarya22/isl-data](https://www.kaggle.com/datasets/ananyaarya22/isl-data).
2. You'll need a free Kaggle account to download. Sign up/sign in if you haven't.
3. Click **Download** (top right of the page). This downloads a `.zip`, roughly 1GB.
4. Extract the zip somewhere on your computer, e.g. `Downloads/isl-data/`.
5. Check what's inside — it should look like this (folder names may vary slightly in casing):
   ```text
   isl-data/
   ├── 0/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── 1/
   ├── ...
   ├── A/
   ├── B/
   └── ...
   ```
   Each subfolder name is the class (the sign it represents), and it's full of photos of that
   sign. If your extracted folder has an extra wrapping folder (e.g.
   `isl-data/isl-data/0/...`), note the correct inner path — you'll need it in the next step.

---

## Part 2: Run the Importer

Make sure your virtual environment from `docs/BEGINNER_SETUP_GUIDE.md` is active:

```bash
cd model_training
source venv/bin/activate      # Windows: venv\Scripts\activate
cd ../data_collection
```

Run the importer, pointing `--source` at the folder from Part 1 (the one whose direct
subfolders are `0`, `1`, ..., `A`, `B`, ...):

```bash
python import_kaggle_isl.py --source /path/to/Downloads/isl-data
```

This will take a while — it's running hand detection on every image. By default it processes
**up to 200 images per class** (not the full ~1,000), which keeps this to a reasonable one-time
wait while still giving each class plenty of samples. You'll see progress printed per class:

```text
Classes found: 36 | limit per class: 200
  0: 197 written, 3 skipped (no hand detected)
  1: 200 written, 0 skipped
  ...
```

Images get skipped when MediaPipe can't find a hand in them (blurry photos, bad crops, etc.) —
this is expected and normal, not an error.

**Only need a few specific classes?** Use `--classes` to skip the rest and save time:

```bash
python import_kaggle_isl.py --source /path/to/Downloads/isl-data --classes C D E F G
```

**Want more than 200 per class?** Raise the limit (this makes the import take longer):

```bash
python import_kaggle_isl.py --source /path/to/Downloads/isl-data --limit-per-class 500
```

When it finishes, check that `data_collection/dataset_kaggle_static/` now has one folder per
imported class, full of `.npy` files.

---

## Part 3: Preprocess, Train, Convert, Run

This is the **same pipeline you already use** for real recordings — nothing new to learn here.
`preprocess.py` automatically finds and merges both data sources.

```bash
cd ../model_training
python preprocess.py
```

Read its output carefully — this is where you see exactly what's about to be trained on:

```text
Data source per class:
  A: 45 real samples (placeholder ignored)
  B: 3 real + 197 placeholder (below the 20-sample real-data threshold)
  C: 0 real + 200 placeholder -- PLACEHOLDER ONLY, record real motion data for this sign before final submission
  Hello: 68 real samples (placeholder ignored)
```

Read this table each time — it tells you which classes still need real recordings.

Then train and export exactly as before:

```bash
python train_cnn.py
python convert_tfjs.py
cp -r model_web/* ../web_app/model/
cp label_map.json ../web_app/model/
```

Run the web app the same way as always (see `docs/GUIDE.md` or `docs/BEGINNER_SETUP_GUIDE.md`
Part 6):

```bash
cd ../web_app
python -m http.server 3000 or python -m http.server 3002 --bind 127.0.0.1
```

---

## Part 4: Record Real Data Over Time

As you get time, record real motion samples for placeholder-only classes with
`capture_sequences.py`, same as any other class (see `docs/BEGINNER_SETUP_GUIDE.md`, Part 4).
You don't need to do anything to "switch over" — the moment a class has 20+ real samples,
`preprocess.py` automatically stops using its placeholder data on the very next run. Just
re-run Part 3 after recording more data, and check the printed report again.

---

## Part 5: Final Cleanup (do this before your presentation)

Once every class shows `(placeholder ignored)` in the `preprocess.py` report — meaning every
class now has real recorded data — remove the placeholder dataset entirely for a completely
clean final model:

```bash
rm -rf data_collection/dataset_kaggle_static      # Windows: rmdir /s /q data_collection\dataset_kaggle_static
```

Then retrain one final time:

```bash
cd model_training
python preprocess.py && python train_cnn.py && python convert_tfjs.py
cp -r model_web/* ../web_app/model/ && cp label_map.json ../web_app/model/
```

This isn't strictly required — the per-class logic in `preprocess.py` already prevents
placeholder data from affecting a class once it has enough real samples — but deleting the
folder removes any possibility of placeholder data influencing your final model, which is the
safest state to present with.

---

## Troubleshooting

| Problem | Likely Cause |
|---|---|
| `--source not found` | Double-check the path — point it at the folder whose direct subfolders are class names, not a parent folder wrapping it |
| Almost everything gets skipped ("no hand detected") | The dataset's images may be oddly cropped or the class folders may not actually contain hand photos — open a few images manually to check |
| A class trained on placeholder data performs badly live | Expected — a held pose is a weaker signal than real motion, especially for signs that aren't naturally static. Record real data for that class |
| Import takes a very long time | Lower `--limit-per-class`, or use `--classes` to import only what you need right now |