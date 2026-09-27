"""
normalizer.py

Per-hand normalization + two-hand frame vector assembly, as defined in
docs/ANVAYA_PIPELINE.md (Section 3).

This exact logic is re-implemented in web_app/app.js for the browser runtime.
If you ever change the math here, mirror the change there too -- the model is
only meaningful if training-time and inference-time normalization match.

Frame vector layout (128 values):
    [ left_hand: 63 coords + 1 presence flag  (64 values) ]
    [ right_hand: 63 coords + 1 presence flag (64 values) ]
"""

import numpy as np

NUM_LANDMARKS = 21
COORDS_PER_HAND = NUM_LANDMARKS * 3  # 63
VALUES_PER_HAND = COORDS_PER_HAND + 1  # 64 (+ presence flag)
FRAME_VECTOR_SIZE = VALUES_PER_HAND * 2  # 128 (two hands)


def normalize_single_hand(landmarks_xyz):
    """
    landmarks_xyz: array-like, shape (21, 3) -- raw (x, y, z) from MediaPipe
    for ONE detected hand.

    Returns a (63,) normalized vector: wrist-centered, then scaled by the
    max Euclidean distance from the wrist.
    """
    pts = np.asarray(landmarks_xyz, dtype=np.float32)
    assert pts.shape == (NUM_LANDMARKS, 3), f"expected (21,3), got {pts.shape}"

    wrist = pts[0].copy()
    shifted = pts - wrist  # Step 1: origin shift

    distances = np.linalg.norm(shifted, axis=1)  # Step 2: euclidean distances
    d_max = distances.max()
    if d_max < 1e-8:
        # Degenerate case (all landmarks collapsed to one point) -- avoid div-by-zero.
        d_max = 1e-8

    scaled = shifted / d_max
    return scaled.flatten()  # (63,)


def build_hand_slot_vector(landmarks_xyz):
    """
    Returns a (64,) vector for one hand: 63 normalized coords + presence flag.
    Pass landmarks_xyz=None if the hand was not detected in this frame.
    """
    if landmarks_xyz is None:
        return np.zeros(VALUES_PER_HAND, dtype=np.float32)  # all zero, flag included

    coords = normalize_single_hand(landmarks_xyz)
    presence_flag = np.array([1.0], dtype=np.float32)
    return np.concatenate([coords, presence_flag])  # (64,)


def build_frame_vector(left_landmarks_xyz, right_landmarks_xyz):
    """
    Assembles the full (128,) per-frame feature vector from up to two hands,
    already assigned to Left/Right slots via MediaPipe's handedness label.

    left_landmarks_xyz / right_landmarks_xyz: (21,3) array or None if that
    hand was not detected in this frame.
    """
    left_vec = build_hand_slot_vector(left_landmarks_xyz)
    right_vec = build_hand_slot_vector(right_landmarks_xyz)
    return np.concatenate([left_vec, right_vec])  # (128,)
