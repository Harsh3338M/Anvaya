"""
capture_sequences.py

DEV-ONLY tool. You run this on your own laptop during Phase 1 (see docs/GUIDE.md).
It is NOT part of the deployed web app and does not run during the presentation.

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
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp

from normalizer import build_frame_vector, FRAME_VECTOR_SIZE

# ---------------------------------------------------------------------------
# Config -- edit these for your project
# ---------------------------------------------------------------------------
CLASSES = [
    "Hello", "Thank_You", "Yes", "No", "Please",
    "Sorry", "Help", "Name", "A", "B",
]
SEQUENCE_LENGTH = 24          # frames per sample (~0.8s at 30fps) -- must match training config
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
COUNTDOWN_SECONDS = 2         # prep time shown to the signer before recording starts
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


def choose_class():
    print("\nAvailable classes:")
    for i, c in enumerate(CLASSES):
        print(f"  [{i}] {c}")
    print("  [q] quit")
    choice = input("Pick a class index to record: ").strip()
    if choice.lower() == "q":
        return None
    try:
        idx = int(choice)
        return CLASSES[idx]
    except (ValueError, IndexError):
        print("Invalid choice, try again.")
        return choose_class()


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
        print("Press SPACE (with the camera window focused) to start recording a sequence.")
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
            cv2.putText(display, "SPACE = record | c = change class | ESC = quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Anvaya - Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key == ord("c"):
                new_class = choose_class()
                if new_class is not None:
                    current_class = new_class

            elif key == 32:  # SPACE -> record one sequence
                # Countdown so the signer can get ready
                for remaining in range(COUNTDOWN_SECONDS, 0, -1):
                    ret, cframe = cap.read()
                    cframe = cv2.flip(cframe, 1)
                    cv2.putText(cframe, f"Starting in {remaining}...", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("Anvaya - Data Collection", cframe)
                    cv2.waitKey(1)
                    time.sleep(1)

                sequence = []
                print(f"Recording '{current_class}'...")
                while len(sequence) < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    left_xyz, right_xyz = extract_left_right(results)
                    frame_vec = build_frame_vector(left_xyz, right_xyz)
                    sequence.append(frame_vec)

                    disp = frame.copy()
                    cv2.putText(disp, f"REC {len(sequence)}/{SEQUENCE_LENGTH}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Anvaya - Data Collection", disp)
                    cv2.waitKey(1)

                if len(sequence) == SEQUENCE_LENGTH:
                    sequence = np.stack(sequence)  # (SEQUENCE_LENGTH, 128)
                    assert sequence.shape == (SEQUENCE_LENGTH, FRAME_VECTOR_SIZE)

                    class_dir = os.path.join(DATASET_DIR, current_class)
                    idx = next_sample_index(class_dir)
                    out_path = os.path.join(class_dir, f"seq_{idx:03d}.npy")
                    np.save(out_path, sequence)
                    print(f"Saved {out_path}")
                else:
                    print("Recording cut short, discarding sample.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
