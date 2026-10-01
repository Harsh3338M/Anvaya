"""
import_kaggle_isl.py

DEV-ONLY, OPTIONAL. Converts a static-image sign-language dataset (built for
image classification) into the sequence format this project's model expects,
so it can be used as PLACEHOLDER training data until real motion samples
exist for the same classes.

Dataset this was written for:
  https://www.kaggle.com/datasets/ananyaarya22/isl-data
  36 classes (digits 0-9, letters A-Z), ~1,000 JPG images per class.

IMPORTANT -- read before running:

  This is a STATIC-IMAGE dataset. A single photo has no motion. Each image is
  converted into a "held pose" sequence by running MediaPipe once on that
  image and repeating the resulting frame SEQUENCE_LENGTH times -- i.e. it
  represents holding that exact pose still for the whole window. This is a
  reasonable stand-in for signs that genuinely ARE static poses (most
  alphabet/digit signs), but it captures zero motion information. It will
  NOT help motion-based signs (e.g. "Hello", "Thank_You") -- those still need
  real recordings from capture_sequences.py.

  Output goes to data_collection/dataset_kaggle_static/, a SEPARATE folder
  from data_collection/dataset/ (your real recordings). They are never
  merged on disk. model_training/preprocess.py decides at training time,
  per class, whether to use this placeholder data -- see its docstring.

Usage:
    python import_kaggle_isl.py --source /path/to/downloaded/isl-data

  --source            Path to the extracted Kaggle dataset (a folder whose
                       subfolders are class names, each full of .jpg images).
  --limit-per-class    Max images to convert per class (default 200). The
                       full dataset is ~1000 images/class; you very likely
                       don't need all of them, and this keeps import time
                       reasonable (36 classes x 1000 images would take much
                       longer for little extra benefit at this project's scale).
  --classes            Optional: only import these specific classes
                       (space-separated), e.g. --classes C D E F
"""

import os
import sys
import argparse
import numpy as np
import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))
from capture_sequences import extract_left_right  # reused as-is, see module docstring above
from normalizer import build_frame_vector, FRAME_VECTOR_SIZE

SEQUENCE_LENGTH = 24  # MUST match capture_sequences.py / train_cnn.py
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset_kaggle_static")

mp_hands = mp.solutions.hands


def normalize_class_name(folder_name):
    """Kaggle folder names are expected to be single characters (e.g. 'a', '7').
    Letters are upper-cased to match this project's convention (capture_sequences.py
    uses 'A', 'B', ...); digits are kept as-is."""
    name = folder_name.strip()
    return name.upper() if name.isalpha() else name


def convert_image(hands, image_path):
    """Returns a (SEQUENCE_LENGTH, FRAME_VECTOR_SIZE) array, or None if no hand
    was detected in the image (these images are skipped -- see Kaggle dataset
    notes: 'some might need pre-processing')."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_xyz, right_xyz = extract_left_right(results)
    if left_xyz is None and right_xyz is None:
        return None  # no hand found -- not usable

    frame_vec = build_frame_vector(left_xyz, right_xyz)  # (FRAME_VECTOR_SIZE,)
    sequence = np.tile(frame_vec, (SEQUENCE_LENGTH, 1))  # "held pose" for the whole window
    return sequence.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Path to the extracted Kaggle dataset folder")
    parser.add_argument("--limit-per-class", type=int, default=200)
    parser.add_argument("--classes", nargs="*", default=None, help="Only import these classes")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        raise RuntimeError(f"--source not found: {args.source}")

    class_folders = sorted(
        d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))
    )
    if args.classes:
        wanted = {c.upper() for c in args.classes}
        class_folders = [c for c in class_folders if normalize_class_name(c) in wanted]

    if not class_folders:
        raise RuntimeError(f"No class folders found under {args.source}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Source: {args.source}")
    print(f"Classes found: {len(class_folders)} | limit per class: {args.limit_per_class}")
    print(f"Output: {OUTPUT_DIR}  (separate from data_collection/dataset/ -- see this script's docstring)\n")

    with mp_hands.Hands(
        static_image_mode=True,  # correct mode for independent, unrelated images (not a video stream)
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:

        total_written, total_skipped = 0, 0
        for folder in class_folders:
            class_name = normalize_class_name(folder)
            src_dir = os.path.join(args.source, folder)
            out_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(out_dir, exist_ok=True)

            images = sorted(
                f for f in os.listdir(src_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )[: args.limit_per_class]

            written, skipped = 0, 0
            for fname in images:
                seq = convert_image(hands, os.path.join(src_dir, fname))
                if seq is None:
                    skipped += 1
                    continue
                out_path = os.path.join(out_dir, f"kaggle_{written:04d}.npy")
                np.save(out_path, seq)
                written += 1

            print(f"  {class_name}: {written} written, {skipped} skipped (no hand detected)")
            total_written += written
            total_skipped += skipped

    print(f"\nDone. {total_written} placeholder sequences written, {total_skipped} images skipped.")
    print("Next: run model_training/preprocess.py -- it will report which classes")
    print("are using this placeholder data vs. your own real recordings.")


if __name__ == "__main__":
    main()