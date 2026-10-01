"""
preprocess.py

DEV-ONLY. Loads recorded sequences and builds X (N, SEQUENCE_LENGTH, 128) and
integer-encoded labels y, splits into train/val, and saves everything the
training script needs.

TWO DATA SOURCES, handled differently on purpose:

  1. data_collection/dataset/                REAL recordings from capture_sequences.py.
  2. data_collection/dataset_kaggle_static/   PLACEHOLDER data from import_kaggle_isl.py
                                              (static images, held-pose sequences --
                                              see that script's docstring for why).

For each class, this script uses ONLY real data if enough of it exists
(REAL_SAMPLES_THRESHOLD or more). Placeholder data for that class is silently
ignored once you have enough real samples -- no manual step needed, per class.
Only classes that are still short on real data fall back to placeholder data
to fill the gap. This is printed out clearly below so it's never ambiguous
what the model was actually trained on.

Once EVERY class has enough real data, delete the whole
data_collection/dataset_kaggle_static/ folder (see docs/KAGGLE_DATASET_GUIDE.md)
for a completely clean final model with zero placeholder-data influence --
this script's per-class logic already avoids real/placeholder conflicts, but
removing the folder entirely is the recommended final step regardless.

Also writes label_map.json, which web_app/ needs too (to turn the model's
output class index back into a human-readable sign name) -- this script
copies it into web_app/model/ automatically.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

SEQUENCE_LENGTH = 24
FRAME_VECTOR_SIZE = 128

# A class stops using placeholder data once it has at least this many real samples.
REAL_SAMPLES_THRESHOLD = 20

THIS_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(THIS_DIR, "..", "data_collection", "dataset")
STATIC_DATASET_DIR = os.path.join(THIS_DIR, "..", "data_collection", "dataset_kaggle_static")
OUTPUT_NPZ = os.path.join(THIS_DIR, "sequences_dataset.npz")
LABEL_MAP_PATH = os.path.join(THIS_DIR, "label_map.json")
WEB_MODEL_DIR = os.path.join(THIS_DIR, "..", "web_app", "model")


def list_npy_files(class_dir):
    if not os.path.isdir(class_dir):
        return []
    return sorted(f for f in os.listdir(class_dir) if f.endswith(".npy"))


def load_sequences(paths, label_map, class_name):
    X, y = [], []
    for p in paths:
        seq = np.load(p)
        if seq.shape != (SEQUENCE_LENGTH, FRAME_VECTOR_SIZE):
            print(f"WARNING: skipping malformed sample {p} with shape {seq.shape}")
            continue
        X.append(seq)
        y.append(label_map[class_name])
    return X, y


def load_dataset():
    real_classes = set(
        d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
    ) if os.path.isdir(DATASET_DIR) else set()

    static_classes = set(
        d for d in os.listdir(STATIC_DATASET_DIR) if os.path.isdir(os.path.join(STATIC_DATASET_DIR, d))
    ) if os.path.isdir(STATIC_DATASET_DIR) else set()

    class_names = sorted(real_classes | static_classes)
    if not class_names:
        raise RuntimeError(
            f"No class folders found under {DATASET_DIR} or {STATIC_DATASET_DIR}. "
            "Run data_collection/capture_sequences.py (and/or import_kaggle_isl.py) first."
        )

    label_map = {name: idx for idx, name in enumerate(class_names)}

    X, y = [], []
    print("Data source per class:")
    for class_name in class_names:
        real_dir = os.path.join(DATASET_DIR, class_name)
        real_files = list_npy_files(real_dir)

        if len(real_files) >= REAL_SAMPLES_THRESHOLD:
            # Enough real data -- placeholder data for this class is ignored entirely.
            paths = [os.path.join(real_dir, f) for f in real_files]
            print(f"  {class_name}: {len(real_files)} real samples (placeholder ignored)")
        else:
            static_dir = os.path.join(STATIC_DATASET_DIR, class_name)
            static_files = list_npy_files(static_dir)
            paths = [os.path.join(real_dir, f) for f in real_files] + \
                     [os.path.join(static_dir, f) for f in static_files]
            if real_files and static_files:
                print(f"  {class_name}: {len(real_files)} real + {len(static_files)} placeholder "
                      f"(below the {REAL_SAMPLES_THRESHOLD}-sample real-data threshold)")
            elif static_files:
                print(f"  {class_name}: 0 real + {len(static_files)} placeholder "
                      f"-- PLACEHOLDER ONLY, record real motion data for this sign before final submission")
            else:
                print(f"  {class_name}: {len(real_files)} real, 0 placeholder "
                      f"-- below threshold and no placeholder data available")

        cx, cy = load_sequences(paths, label_map, class_name)
        X.extend(cx)
        y.extend(cy)

    if not X:
        raise RuntimeError("No valid samples found. Check your dataset folders.")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, label_map


def main():
    X, y, label_map = load_dataset()
    print(f"\nLoaded {X.shape[0]} samples across {len(label_map)} classes.")

    counts_below_20 = []
    for name, idx in label_map.items():
        count = int((y == idx).sum())
        if count < 20:
            counts_below_20.append(name)
    if counts_below_20:
        print(f"WARNING: these classes have fewer than 20 total samples (real+placeholder): "
              f"{', '.join(counts_below_20)}. Consider recording/importing more before training.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.savez(
        OUTPUT_NPZ,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    print(f"\nSaved train/val arrays to {OUTPUT_NPZ}")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map to {LABEL_MAP_PATH}")

    os.makedirs(WEB_MODEL_DIR, exist_ok=True)
    with open(os.path.join(WEB_MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Copied label map to {WEB_MODEL_DIR}/label_map.json")


if __name__ == "__main__":
    main()