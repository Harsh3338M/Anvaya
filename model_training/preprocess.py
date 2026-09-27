"""
preprocess.py

DEV-ONLY. Loads all recorded sequences from data_collection/dataset/,
builds X (N, SEQUENCE_LENGTH, 128) and integer-encoded labels y,
splits into train/val, and saves everything the training script needs.

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

THIS_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(THIS_DIR, "..", "data_collection", "dataset")
OUTPUT_NPZ = os.path.join(THIS_DIR, "sequences_dataset.npz")
LABEL_MAP_PATH = os.path.join(THIS_DIR, "label_map.json")
WEB_MODEL_DIR = os.path.join(THIS_DIR, "..", "web_app", "model")


def load_dataset():
    class_names = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not class_names:
        raise RuntimeError(
            f"No class folders found under {DATASET_DIR}. "
            "Run data_collection/capture_sequences.py first."
        )

    label_map = {name: idx for idx, name in enumerate(class_names)}

    X, y = [], []
    for class_name in class_names:
        class_dir = os.path.join(DATASET_DIR, class_name)
        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".npy"))
        if not files:
            print(f"WARNING: no samples found for class '{class_name}' -- skipping.")
            continue
        for f in files:
            seq = np.load(os.path.join(class_dir, f))
            if seq.shape != (SEQUENCE_LENGTH, FRAME_VECTOR_SIZE):
                print(f"WARNING: skipping malformed sample {f} with shape {seq.shape}")
                continue
            X.append(seq)
            y.append(label_map[class_name])

    if not X:
        raise RuntimeError("No valid samples found. Check your dataset folder.")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, label_map


def main():
    X, y, label_map = load_dataset()
    print(f"Loaded {X.shape[0]} samples across {len(label_map)} classes.")
    print("Samples per class:")
    for name, idx in label_map.items():
        count = int((y == idx).sum())
        print(f"  {name}: {count}")
        if count < 20:
            print(f"    -> WARNING: fewer than 20 samples for '{name}'. "
                  "Consider recording more before training (see docs/GUIDE.md).")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.savez(
        OUTPUT_NPZ,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    print(f"Saved train/val arrays to {OUTPUT_NPZ}")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map to {LABEL_MAP_PATH}")

    # Also drop a copy where the web app can find it later (convert_tfjs.py
    # also copies model files here, so this keeps everything in one place).
    os.makedirs(WEB_MODEL_DIR, exist_ok=True)
    with open(os.path.join(WEB_MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Copied label map to {WEB_MODEL_DIR}/label_map.json")


if __name__ == "__main__":
    main()
