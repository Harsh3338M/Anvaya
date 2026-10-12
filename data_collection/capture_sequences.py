"""
capture_sequences.py


Records labeled motion sequences for training the 1D-CNN:
    - Uses MediaPipe Hands (Python) with up to 2 hands.
    - For each frame, builds a normalized 128-value vector (see normalizer.py).
    - Groups frames into fixed-length sequences of SEQUENCE_LENGTH frames.
    - Saves each recorded sequence as dataset/<class_name>/seq_###.npy,
      shape (SEQUENCE_LENGTH, 128).

Handles the "one hand only" case automatically: whichever hand slot
(Left/Right) MediaPipe does not detect in a frame is zero-filled with its
presence flag off. You don't need to do anything special for one-hand signs
other than keeping the other hand out of frame.

BATCH RECORDING: pressing SPACE records a BATCH of BATCH_SIZE samples back to
back (BATCH_GAP_SECONDS apart), instead of just one. Repeat the gesture
naturally each time the on-screen countdown restarts -- small natural
variation between reps is good for training, not a problem. Press ESC at any
point to abort the current batch early (samples already saved are kept).

CUSTOM LABELS: besides the predefined CLASSES list below (including A-Z and
0-9), you can type any custom label at the class-picker menu -- a word, a
number, anything. Typing the same label again later (regardless of case,
e.g. "night" / "Night" / "NIGHT") always reuses the same folder on disk,
rather than creating near-duplicate classes -- see normalize_custom_label().
"""

import os
import re
import time
import cv2
import numpy as np
import mediapipe as mp

from normalizer import build_frame_vector, FRAME_VECTOR_SIZE

# ---------------------------------------------------------------------------
# Config -- edit these for your project
# ---------------------------------------------------------------------------
CLASSES = (
    ["Hello", "Thank_You", "Yes", "No", "Please", "Sorry", "Help", "Name"]
    + [str(d) for d in range(10)]                      # "0".."9"
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]   # "A".."Z"
)
SEQUENCE_LENGTH = 24          # frames per sample (~0.8s at 30fps) -- must match training config
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
COUNTDOWN_SECONDS = 2         # prep time before the FIRST sample in a batch
BATCH_SIZE = 20                # samples recorded per SPACE press
BATCH_GAP_SECONDS = 1.5        # pause before each sample AFTER the first one in a batch
CAM_WIDTH, CAM_HEIGHT = 640, 480

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def landmarks_to_xyz(hand_landmarks):
    """Converts a MediaPipe hand_landmarks object into a (21,3) numpy array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )


def extract_left_right(results):
    """
    Returns (left_xyz, right_xyz), each either a (21,3) array or None,
    using MediaPipe's own handedness classification so hand identity stays
    consistent across frames regardless of detection order.
    """
    left_xyz, right_xyz = None, None
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" or "Right"
            xyz = landmarks_to_xyz(hand_landmarks)
            if label == "Left":
                left_xyz = xyz
            else:
                right_xyz = xyz
    return left_xyz, right_xyz


def next_sample_index(class_dir):
    existing = [f for f in os.listdir(class_dir) if f.startswith("seq_") and f.endswith(".npy")]
    if not existing:
        return 0
    nums = [int(f.replace("seq_", "").replace(".npy", "")) for f in existing]
    return max(nums) + 1


def get_existing_dataset_labels():
    """Folder names already on disk under dataset/ -- includes predefined
    classes already recorded AND any custom labels created in earlier runs."""
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)))


def normalize_custom_label(raw_label):
    """
    Sanitizes a user-typed custom label and matches it CASE-INSENSITIVELY
    against every label already known (the predefined CLASSES list, plus any
    folder already on disk from a previous run) -- so typing "night", "Night"
    or "NIGHT" on different occasions always resolves to the exact same
    folder, never creating near-duplicate classes.

    Returns the label to use, or None if the input was empty/invalid after
    cleanup.
    """
    cleaned = raw_label.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)             # collapse internal whitespace -> underscore
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)  # drop anything unsafe as a folder name
    if not cleaned:
        return None

    known = list(CLASSES) + get_existing_dataset_labels()
    for existing in known:
        if existing.lower() == cleaned.lower():
            return existing  # reuse the exact casing already established

    return cleaned  # first time seeing this label -- use as typed (sanitized)


def choose_class():
    # Predefined classes first, then any custom labels from earlier runs that
    # aren't already in CLASSES, so they're pickable without retyping them.
    all_labels = list(CLASSES) + [c for c in get_existing_dataset_labels() if c not in CLASSES]

    print("\nAvailable classes:")
    for i, c in enumerate(all_labels):
        end = "\n" if (i + 1) % 6 == 0 else "  "
        print(f"[{i:>2}] {c:<10}", end=end)
    if len(all_labels) % 6 != 0:
        print()
    print("  [n] Enter a NEW custom label (any word, letter, or number)")
    print("  [q] quit")

    choice = input("Pick a class index, 'n' for custom, or 'q' to quit: ").strip()

    if choice.lower() == "q":
        return None

    if choice.lower() == "n":
        raw = input("Type the new label (e.g. 'Night', 'Cold', '7'): ")
        label = normalize_custom_label(raw)
        if label is None:
            print("That label was empty after cleanup -- try again.")
            return choose_class()
        if label in all_labels:
            print(f"'{label}' already exists -- reusing that label (new samples add to it).")
        else:
            print(f"New label created: '{label}'")
        return label

    try:
        idx = int(choice)
        return all_labels[idx]
    except (ValueError, IndexError):
        print("Invalid choice, try again.")
        return choose_class()


def wait_with_countdown(cap, seconds, message_prefix):
    """Shows a live camera feed with a countdown overlay for `seconds`.
    Returns False if ESC was pressed during the wait (caller should abort)."""
    steps = max(1, round(seconds / 0.5))
    for step in range(steps):
        ret, frame = cap.read()
        if not ret:
            return True
        frame = cv2.flip(frame, 1)
        remaining = max(0.0, seconds - step * 0.5)
        cv2.putText(frame, f"{message_prefix} {remaining:.1f}s...", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Anvaya - Data Collection", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            return False
        time.sleep(0.5)
    return True


def record_one_sequence(cap, hands, rep, batch_size):
    """Records a single SEQUENCE_LENGTH-frame sample. Returns the sequence
    array, or None if ESC was pressed or the camera failed mid-recording."""
    sequence = []
    while len(sequence) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_xyz, right_xyz = extract_left_right(results)
        frame_vec = build_frame_vector(left_xyz, right_xyz)
        sequence.append(frame_vec)

        disp = frame.copy()
        cv2.putText(disp, f"REC sample {rep}/{batch_size}  frame {len(sequence)}/{SEQUENCE_LENGTH}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow("Anvaya - Data Collection", disp)
        if (cv2.waitKey(1) & 0xFF) == 27:
            return None

    return np.stack(sequence)  # (SEQUENCE_LENGTH, 128)


def resolve_existing_casing(label):
    """Safety net right before writing to disk: if a folder matching this
    label (any case) already exists, use ITS casing instead. Covers the edge
    case where the same custom label was typed with different casing twice
    in one session before anything was recorded under either -- without
    this, both would otherwise create separate, near-duplicate folders the
    first time each gets recorded."""
    for existing in get_existing_dataset_labels():
        if existing.lower() == label.lower():
            return existing
    return label


def record_batch(cap, hands, current_class):
    """Records up to BATCH_SIZE samples for current_class, pausing
    BATCH_GAP_SECONDS between each one after the first. ESC aborts the rest
    of the batch early; samples already saved before that are kept.
    Returns the label actually used (may differ in casing from the input --
    see resolve_existing_casing)."""
    current_class = resolve_existing_casing(current_class)
    class_dir = os.path.join(DATASET_DIR, current_class)
    os.makedirs(class_dir, exist_ok=True)

    print(f"\nStarting batch: up to {BATCH_SIZE} samples of '{current_class}', "
          f"{BATCH_GAP_SECONDS}s between each. Repeat the gesture naturally each "
          f"time -- small variation between reps is good. Press ESC to stop early.")

    saved = 0
    for rep in range(1, BATCH_SIZE + 1):
        wait_seconds = COUNTDOWN_SECONDS if rep == 1 else BATCH_GAP_SECONDS
        if not wait_with_countdown(cap, wait_seconds, f"Sample {rep}/{BATCH_SIZE} in"):
            print("Batch stopped early (ESC).")
            break

        sequence = record_one_sequence(cap, hands, rep, BATCH_SIZE)
        if sequence is None:
            print("Batch stopped early (ESC or camera error).")
            break
        if sequence.shape != (SEQUENCE_LENGTH, FRAME_VECTOR_SIZE):
            print(f"  [{rep}/{BATCH_SIZE}] Recording cut short, discarding sample.")
            continue

        idx = next_sample_index(class_dir)
        out_path = os.path.join(class_dir, f"seq_{idx:03d}.npy")
        np.save(out_path, sequence)
        saved += 1
        print(f"  [{rep}/{BATCH_SIZE}] Saved {out_path}")

    print(f"Batch complete: {saved}/{BATCH_SIZE} samples saved for '{current_class}'.\n")
    return current_class


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        print("Anvaya data collection tool.")
        print(f"Press SPACE to record a batch of {BATCH_SIZE} samples (with the camera window focused).")
        print("Press ESC to quit at any time.\n")

        current_class = choose_class()
        if current_class is None:
            cap.release()
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(display, f"Class: {current_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"SPACE = record batch of {BATCH_SIZE} | c = change class | ESC = quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Anvaya - Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key == ord("c"):
                new_class = choose_class()
                if new_class is not None:
                    current_class = new_class

            elif key == 32:  # SPACE -> record a full batch
                current_class = record_batch(cap, hands, current_class)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()