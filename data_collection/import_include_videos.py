"""
import_include_videos.py

DEV-ONLY. Imports sign videos from the INCLUDE dataset (AI4Bharat, IIT Madras;
https://zenodo.org/records/4010759, CC-BY-4.0 -- credit them in your report)
into data_collection/dataset/<label>/ as REAL motion training data.

INCLUDE layout (after unzipping):   <Category>/<NN. word>/MVI_xxxx.MOV
e.g.                                Pronouns/47. they/MVI_5194.MOV
Each video is one signer performing ONE word. This script:

  * unzips the category zips you downloaded (or reads already-extracted folders),
  * turns each folder name into a label ("47. they" -> "They"),
  * REUSES that label's folder if it already exists (case-insensitively --
    "they"/"They"/"THEY" are one class) and creates it only if it doesn't,
  * converts every clip into training windows via video_to_sequences.py,
  * saves them as inc_<clip-id>_w<NN>.npy so they can be told apart from your own
    recordings (seq_*.npy) and so preprocess.py can keep all windows from one
    clip on the same side of the train/validation split.

Re-running is safe: clips that were already imported are skipped.

Usage (run from data_collection/):
    # 1. Preview only -- shows exactly which labels would be created/reused:
    python import_include_videos.py --source ~/Downloads/Pronouns_1of2.zip ~/Downloads/Pronouns_2of2.zip --categories Pronouns --dry-run

    # 2. Do it:
    python import_include_videos.py --source ~/Downloads/Pronouns_1of2.zip ~/Downloads/Pronouns_2of2.zip --categories Pronouns

    # Later, adding more words/categories (same command, different zip/category):
    python import_include_videos.py --source ~/Downloads/Greetings_1of2.zip ~/Downloads/Greetings_2of2.zip --categories Greetings

  --source        One or more .zip files and/or folders (already-extracted INCLUDE).
  --extract-to    Where zips are unpacked (default: data_collection/raw_videos/INCLUDE).
  --categories    Only import these categories (case-insensitive), e.g. Pronouns.
  --words         Only import these words (case-insensitive), e.g. we they.
  --limit-clips   Max clips per word (handy for a quick test run).
  --target-fps    Timeline rate clips are resampled to (default 20). Set this close
                  to the FPS your web app actually shows -- see docs/GUIDE_V2.md.
  --stride        Frames between overlapping windows (default 4).
  --max-windows   Max windows kept per clip (default 8).
  --dry-run       Print the plan and exit without converting anything.
"""

import argparse
import os
import re
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import capture_sequences as cs  # label helpers + DATASET_DIR (read at call time, so it stays the single source of truth)
from video_to_sequences import (
    VIDEO_EXTENSIONS, SEQUENCE_LENGTH, MediaPipeDetector, video_to_windows,
)

DEFAULT_EXTRACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_videos", "INCLUDE")
FILE_PREFIX = "inc_"                 # marks files that came from INCLUDE (easy to find / delete / re-import)
COLLISION_SUFFIX = "_Word"           # single-character words would merge into the letter/digit classes

# Manual label overrides, keyed by the lower-cased word as it appears after the
# "NN. " prefix is stripped. Edit this to rename a label without touching code
# elsewhere, e.g. {"i": "Me", "tv": "Television"}. Overrides skip the
# single-character collision rule below (you're choosing the name yourself).
LABEL_OVERRIDES = {}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def word_from_folder(folder_name):
    """'47. they' -> 'they'  (strips INCLUDE's numeric prefix)."""
    return re.sub(r"^\s*\d+\s*[\.\)\-_:]\s*", "", folder_name).strip()


def label_for_word(word):
    """Turns an INCLUDE word into this project's label, and says why it was
    renamed if it was. Returns (label, note_or_None)."""
    key = word.lower()
    if key in LABEL_OVERRIDES:
        return cs.normalize_custom_label(LABEL_OVERRIDES[key]), "manual override"

    parts = [p for p in re.split(r"[\s_\-]+", word) if p]
    # Capitalise each part, keep the rest as-is ("wet" -> "Wet", "TV" stays "TV")
    candidate = "_".join(p[:1].upper() + p[1:] for p in parts)
    candidate = re.sub(r"[^A-Za-z0-9_\-]", "", candidate)
    if not candidate:
        return None, "empty after cleanup"

    if len(candidate) == 1:
        # Would silently merge into the A-Z / 0-9 fingerspelling class of the
        # same name (e.g. the WORD "I" vs the LETTER "I") -- different signs.
        return cs.normalize_custom_label(candidate + COLLISION_SUFFIX), \
            f"single character, kept apart from the '{candidate}' letter/digit class"

    return cs.normalize_custom_label(candidate), None


# ---------------------------------------------------------------------------
# Finding the data
# ---------------------------------------------------------------------------
def extract_zip(zip_path, dest):
    """Unpacks only the video files from a zip, skipping ones already extracted."""
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and m.filename.lower().endswith(VIDEO_EXTENSIONS)]
        print(f"  {os.path.basename(zip_path)}: {len(members)} video files")
        for i, m in enumerate(members, 1):
            target = os.path.join(dest, m.filename)
            if os.path.exists(target) and os.path.getsize(target) == m.file_size:
                continue
            zf.extract(m, dest)  # ZipFile.extract neutralises '..' and absolute paths
            if i % 25 == 0 or i == len(members):
                print(f"    extracted {i}/{len(members)}")


def resolve_roots(sources, extract_to):
    """Turns --source entries (zips and/or folders) into folders to scan."""
    roots = []
    for src in sources:
        src = os.path.expanduser(src)
        if os.path.isdir(src):
            roots.append(src)
        elif zipfile.is_zipfile(src):
            print(f"Extracting {src} -> {extract_to}")
            extract_zip(src, extract_to)
            if extract_to not in roots:
                roots.append(extract_to)
        else:
            raise RuntimeError(f"--source is neither a folder nor a .zip: {src}")
    return roots


def discover_videos(roots):
    """Yields (category, word_folder, video_path) for every video found.
    Category = grandparent folder, word = parent folder (INCLUDE layout)."""
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in sorted(filenames):
                if not fname.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                word_folder = os.path.basename(dirpath)
                category = os.path.basename(os.path.dirname(dirpath))
                yield category, word_folder, os.path.join(dirpath, fname)


def clip_id(category, word_folder, video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe = lambda s: re.sub(r"[^A-Za-z0-9]", "", s)
    return f"{safe(category)}-{safe(word_folder)}-{re.sub(r'[^A-Za-z0-9_]', '', stem)}"


def already_imported(label_dir, cid):
    prefix = f"{FILE_PREFIX}{cid}_w"
    return os.path.isdir(label_dir) and any(f.startswith(prefix) for f in os.listdir(label_dir))


# ---------------------------------------------------------------------------
# Main routine (detector injectable for tests)
# ---------------------------------------------------------------------------
def run_import(args, detector=None):
    roots = resolve_roots(args.source, os.path.expanduser(args.extract_to))

    wanted_cats = {c.lower() for c in args.categories} if args.categories else None
    wanted_words = {w.lower() for w in args.words} if args.words else None

    # label -> list of (category, word_folder, video_path); also remember notes
    plan = defaultdict(list)
    notes = {}
    label_categories = defaultdict(set)
    for category, word_folder, path in discover_videos(roots):
        if wanted_cats and category.lower() not in wanted_cats:
            continue
        word = word_from_folder(word_folder)
        if wanted_words and word.lower() not in wanted_words:
            continue
        label, note = label_for_word(word)
        if label is None:
            print(f"  skipping folder '{word_folder}': {note}")
            continue
        plan[label].append((category, word_folder, path))
        label_categories[label].add(category)
        if note:
            notes[label] = (word, note)

    if not plan:
        print("No matching videos found. Check --source/--categories, and run with "
              "--dry-run to see what was discovered.")
        return 1

    existing = {l.lower() for l in cs.get_existing_dataset_labels()}
    print(f"\nPlan: {sum(len(v) for v in plan.values())} clips -> {len(plan)} labels")
    print(f"{'label':<18}{'clips':>6}  status")
    for label in sorted(plan):
        clips = plan[label][: args.limit_clips] if args.limit_clips else plan[label]
        status = "reuse existing folder" if label.lower() in existing else "NEW folder"
        extra = f"  [{notes[label][1]}]" if label in notes else ""
        print(f"{label:<18}{len(clips):>6}  {status}{extra}")
    multi = {l: sorted(c) for l, c in label_categories.items() if len(c) > 1}
    for label, cats in multi.items():
        print(f"WARNING: label '{label}' comes from several categories ({', '.join(cats)}) -- "
              f"if those are different signs, rename one via LABEL_OVERRIDES.")

    if args.dry_run:
        print("\nDry run only -- nothing was converted or written.")
        return 0

    own_detector = detector is None
    if own_detector:
        detector = MediaPipeDetector()

    totals = defaultdict(int)
    try:
        for label in sorted(plan):
            clips = plan[label][: args.limit_clips] if args.limit_clips else plan[label]
            label_dir = os.path.join(cs.DATASET_DIR, cs.resolve_existing_casing(label))
            for category, word_folder, path in clips:
                cid = clip_id(category, word_folder, path)
                if already_imported(label_dir, cid):
                    totals["already_imported"] += 1
                    continue
                try:
                    windows, info = video_to_windows(
                        path, detector, target_fps=args.target_fps, stride=args.stride,
                        max_windows=args.max_windows,
                    )
                except IOError as e:
                    print(f"  [{label}] {os.path.basename(path)}: unreadable ({e})")
                    totals["unreadable"] += 1
                    continue

                if info["status"] != "ok":
                    print(f"  [{label}] {os.path.basename(path)}: skipped ({info['status']})")
                    totals[info["status"]] += 1
                    continue

                os.makedirs(label_dir, exist_ok=True)  # created only when there is data to put in it
                for i, w in enumerate(windows):
                    assert w.shape == (SEQUENCE_LENGTH, 128)
                    np.save(os.path.join(label_dir, f"{FILE_PREFIX}{cid}_w{i:02d}.npy"), w)
                totals["clips_imported"] += 1
                totals["windows_written"] += len(windows)
                print(f"  [{label}] {os.path.basename(path)}: {len(windows)} windows "
                      f"({info['active_frames']} active frames)")
    finally:
        if own_detector:
            detector.close()

    print("\nDone.")
    for key in ("clips_imported", "windows_written", "already_imported",
                "no_hands", "too_short", "unreadable"):
        if totals[key]:
            print(f"  {key}: {totals[key]}")
    print("Next: run model_training/preprocess.py (see docs/GUIDE_V2.md).")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", required=True)
    p.add_argument("--extract-to", default=DEFAULT_EXTRACT_DIR)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--words", nargs="*", default=None)
    p.add_argument("--limit-clips", type=int, default=None)
    p.add_argument("--target-fps", type=float, default=20.0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    sys.exit(run_import(build_parser().parse_args()))