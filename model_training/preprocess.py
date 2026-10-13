"""
preprocess.py

DEV-ONLY. Loads recorded sequences and builds X (N, SEQUENCE_LENGTH, 128) and
integer-encoded labels y, splits into train/val, and saves everything the
training script needs.

DATA SOURCES (handled differently on purpose):

  1. data_collection/dataset/                REAL motion data:
        seq_###.npy       your own recordings (capture_sequences.py)
        inc_<clip>_w##.npy windows cut from INCLUDE videos (import_include_videos.py)
  2. data_collection/dataset_kaggle_static/   PLACEHOLDER data from import_kaggle_isl.py
                                              (static images, held-pose sequences).

For each class, ONLY real data is used once it has REAL_SAMPLES_THRESHOLD or more
samples; placeholder data fills in only for classes still short on real data.
The per-class report printed below always says exactly what the model trains on.
Once every class has real data, delete data_collection/dataset_kaggle_static/
(see docs/KAGGLE_DATASET_GUIDE.md).

TWO THINGS THIS SCRIPT DOES THAT MATTER FOR HONEST ACCURACY:

  * Split by SOURCE CLIP, not by sample. One video yields several overlapping
    windows (inc_<clip>_w00, _w01, ...). A plain random split would put some of a
    clip's windows in training and its near-identical neighbours in validation,
    so the model would be "tested" on material it effectively saw -- inflating
    validation accuracy. Here every window of a clip stays on the same side.
    (Files that aren't windows of a clip -- your own recordings, Kaggle samples --
    each count as their own group.)

  * Classes with NO samples are dropped. capture_sequences.py pre-creates a folder
    for every predefined class (A-Z, 0-9, ...), so most start empty. Giving each a
    model output would waste capacity and add output units that never saw data.

Also writes label_map.json, which web_app/ needs (it turns the model's output
index back into a sign name); it is copied into web_app/model/ automatically.
"""

import os
import re
import json
import numpy as np

SEQUENCE_LENGTH = 24
FRAME_VECTOR_SIZE = 128

# A class stops using placeholder data once it has at least this many real samples.
REAL_SAMPLES_THRESHOLD = 20
VAL_FRACTION = 0.2
SPLIT_SEED = 42

THIS_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(THIS_DIR, "..", "data_collection", "dataset")
STATIC_DATASET_DIR = os.path.join(THIS_DIR, "..", "data_collection", "dataset_kaggle_static")
OUTPUT_NPZ = os.path.join(THIS_DIR, "sequences_dataset.npz")
LABEL_MAP_PATH = os.path.join(THIS_DIR, "label_map.json")
WEB_MODEL_DIR = os.path.join(THIS_DIR, "..", "web_app", "model")

# inc_<clip-id>_w03.npy -> group "<dir>/inc_<clip-id>"
WINDOW_FILE_RE = re.compile(r"^(?P<clip>.+)_w\d+\.npy$")


def list_npy_files(class_dir):
    if not os.path.isdir(class_dir):
        return []
    return sorted(f for f in os.listdir(class_dir) if f.endswith(".npy"))


def group_key(path):
    """Samples sharing a key came from the same source clip and must not be
    split across train/validation. Anything that isn't a clip window is its own group."""
    m = WINDOW_FILE_RE.match(os.path.basename(path))
    if m:
        return os.path.normpath(os.path.join(os.path.dirname(path), m.group("clip")))
    return os.path.normpath(path)


def list_class_dirs(base):
    if not os.path.isdir(base):
        return set()
    return {d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))}


def choose_paths_per_class():
    """Decides, per class, which files to train on (real vs placeholder) and
    prints the report. Returns {class_name: [paths]}."""
    class_names = sorted(list_class_dirs(DATASET_DIR) | list_class_dirs(STATIC_DATASET_DIR))
    if not class_names:
        raise RuntimeError(
            f"No class folders found under {DATASET_DIR} or {STATIC_DATASET_DIR}. "
            "Run capture_sequences.py / import_include_videos.py (and/or import_kaggle_isl.py) first."
        )

    chosen, empty = {}, []
    print("Data source per class:")
    for name in class_names:
        real_dir = os.path.join(DATASET_DIR, name)
        real_files = list_npy_files(real_dir)

        if len(real_files) >= REAL_SAMPLES_THRESHOLD:
            paths = [os.path.join(real_dir, f) for f in real_files]
            print(f"  {name}: {len(real_files)} real samples (placeholder ignored)")
        else:
            static_dir = os.path.join(STATIC_DATASET_DIR, name)
            static_files = list_npy_files(static_dir)
            paths = [os.path.join(real_dir, f) for f in real_files] + \
                    [os.path.join(static_dir, f) for f in static_files]
            if real_files and static_files:
                print(f"  {name}: {len(real_files)} real + {len(static_files)} placeholder "
                      f"(below the {REAL_SAMPLES_THRESHOLD}-sample real-data threshold)")
            elif static_files:
                print(f"  {name}: 0 real + {len(static_files)} placeholder "
                      f"-- PLACEHOLDER ONLY, record real motion data for this sign before final submission")
            elif real_files:
                print(f"  {name}: {len(real_files)} real, 0 placeholder "
                      f"-- below the {REAL_SAMPLES_THRESHOLD}-sample threshold; record more")
            else:
                empty.append(name)
        if paths:
            chosen[name] = paths

    if empty:
        print(f"\nSkipping {len(empty)} classes with no data at all: {', '.join(empty)}")
    return chosen


def load_dataset():
    chosen = choose_paths_per_class()

    per_class = {}  # name -> (list of arrays, list of group keys)
    for name, paths in chosen.items():
        arrays, groups = [], []
        for p in paths:
            seq = np.load(p)
            if seq.shape != (SEQUENCE_LENGTH, FRAME_VECTOR_SIZE):
                print(f"WARNING: skipping malformed sample {p} with shape {seq.shape}")
                continue
            arrays.append(seq)
            groups.append(group_key(p))
        if arrays:
            per_class[name] = (arrays, groups)

    if not per_class:
        raise RuntimeError("No valid samples found. Check your dataset folders.")

    label_map = {name: idx for idx, name in enumerate(sorted(per_class))}
    X, y, groups = [], [], []
    for name, (arrays, grp) in per_class.items():
        X.extend(arrays)
        y.extend([label_map[name]] * len(arrays))
        groups.extend(grp)

    return (np.stack(X).astype(np.float32), np.array(y, dtype=np.int64),
            np.array(groups), label_map)


def split_by_group(X, y, groups):
    """Train/validation split that never separates windows of the same clip.

    Per class, about VAL_FRACTION of its source clips are held out for
    validation -- always at least one clip when the class has two or more, so
    every class that CAN be measured IS measured. (scikit-learn's grouped
    splitters are heuristics with no such per-class guarantee: in testing one
    left a 5-clip class with no validation data at all.) A class with a single
    clip stays entirely in training. Seeded, so reruns split identically.
    Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(SPLIT_SEED)
    val_mask = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        in_class = (y == cls)
        class_groups = np.unique(groups[in_class])      # sorted -> deterministic before shuffling
        if len(class_groups) < 2:
            continue                                     # nothing can be held out
        rng.shuffle(class_groups)
        n_val = int(round(len(class_groups) * VAL_FRACTION))
        n_val = min(max(n_val, 1), len(class_groups) - 1)
        val_groups = class_groups[:n_val]
        val_mask |= in_class & np.isin(groups, val_groups)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def main():
    X, y, groups, label_map = load_dataset()
    print(f"\nLoaded {X.shape[0]} samples from {len(np.unique(groups))} source clips/files "
          f"across {len(label_map)} classes.")

    names = {i: n for n, i in label_map.items()}
    thin = []
    for idx in range(len(label_map)):
        n_groups = len(np.unique(groups[y == idx]))
        if n_groups < 3:
            thin.append(f"{names[idx]} ({n_groups})")
    if thin:
        print(f"WARNING: classes built from fewer than 3 independent clips/recordings "
              f"(validation for these is unreliable): {', '.join(thin)}")

    train_idx, val_idx = split_by_group(X, y, groups)
    print(f"Split by source clip (windows of one clip never straddle the split): "
          f"{len(train_idx)} train / {len(val_idx)} validation samples.")

    missing_val = [names[i] for i in range(len(label_map)) if not (y[val_idx] == i).any()]
    if missing_val:
        print(f"NOTE: no validation samples for: {', '.join(missing_val)} "
              f"(only one clip/recording each) -- their accuracy can't be measured.")

    np.savez(
        OUTPUT_NPZ,
        X_train=X[train_idx], y_train=y[train_idx], groups_train=groups[train_idx],
        X_val=X[val_idx], y_val=y[val_idx], groups_val=groups[val_idx],
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