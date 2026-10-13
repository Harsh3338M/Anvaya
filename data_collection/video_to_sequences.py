"""
video_to_sequences.py

DEV-ONLY converter: turns a video clip of ONE sign into training windows of the
exact shape the model expects -- (SEQUENCE_LENGTH, 128) -- using the SAME
normalization as capture_sequences.py (normalizer.py), so video-derived data
and webcam-recorded data are interchangeable.

Used by import_include_videos.py, but written dataset-agnostic so any other
"one clip = one sign" video collection can reuse it later.

Pipeline for one clip:
  1. Decode every frame; downscale wide frames to MAX_FRAME_WIDTH (speed, and
     closer to webcam geometry); MIRROR horizontally, exactly like
     capture_sequences.py does, so left/right hands and x-direction match the
     live webcam path that the web app was built to reproduce.
  2. Run hand detection on every frame in VIDEO/tracking mode (not
     independent-image mode) and build a 128-value frame vector per frame.
  3. Resample the timeline to TARGET_FPS by nearest-frame picking (NOT
     interpolation: interpolating between a "hand present" and a "hand absent"
     frame would invent half-present hands and corrupt the presence flags).
  4. Trim to the ACTIVE segment: first to last frame where any hand is
     visible, plus a small margin. Idle lead-in/lead-out frames are dropped so
     they don't dilute the sign.
  5. Cut that segment into overlapping SEQUENCE_LENGTH windows (the live app
     also classifies a rolling window, so training on windows from different
     moments of the sign matches how it's used). Signs shorter than one window
     are placed at a few different positions inside a zero-padded window
     (zeros = "no hand", the same encoding live capture uses for idle frames).

IMPORTANT -- time scale: the model counts frames, not seconds. TARGET_FPS should
be close to the frame rate your web app actually achieves (see the FPS counter
on studio.html), otherwise dataset signs will look faster or slower than live
signs. See docs/GUIDE_V2.md.
"""

import cv2
import numpy as np

from normalizer import build_frame_vector, FRAME_VECTOR_SIZE, VALUES_PER_HAND

SEQUENCE_LENGTH = 24                 # MUST match capture_sequences.py / train_cnn.py
MAX_FRAME_WIDTH = 640
VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi", ".mkv", ".m4v")

# Presence-flag positions inside the 128-value frame vector (see normalizer.py):
# [left: 63 coords + flag | right: 63 coords + flag]
LEFT_FLAG_INDEX = VALUES_PER_HAND - 1                      # 63
RIGHT_FLAG_INDEX = 2 * VALUES_PER_HAND - 1                 # 127


# ---------------------------------------------------------------------------
# Hand detector (MediaPipe), injectable so the rest can be tested without it
# ---------------------------------------------------------------------------
class MediaPipeDetector:
    """Callable: rgb_frame -> (left_xyz or None, right_xyz or None).

    Matches capture_sequences.py: same handedness handling (extract_left_right)
    and MediaPipe's default model complexity, so detections are comparable."""

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._kwargs = dict(
            static_image_mode=False,      # video mode: tracks hands across frames
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = None
        self.reset()

    def reset(self):
        """Start fresh tracking state -- call before each new video so one
        clip's last hand position can't leak into the next clip."""
        import mediapipe as mp
        if self._hands is not None:
            self._hands.close()
        self._hands = mp.solutions.hands.Hands(**self._kwargs)

    def __call__(self, rgb_frame):
        from capture_sequences import extract_left_right  # lazy: keeps this module light to import
        return extract_left_right(self._hands.process(rgb_frame))

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None


# ---------------------------------------------------------------------------
# Steps 1-2: decode + detect
# ---------------------------------------------------------------------------
def read_video_frame_vectors(video_path, detector, flip=True, max_width=MAX_FRAME_WIDTH):
    """Returns (vectors, src_fps): vectors is (N, 128) float32, one row per
    source frame. Raises IOError if the file can't be opened/decoded."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps < 1 or src_fps > 240:
        src_fps = 30.0  # missing/garbage metadata -> assume a normal frame rate

    if hasattr(detector, "reset"):
        detector.reset()

    vectors = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(round(h * max_width / w))),
                                   interpolation=cv2.INTER_AREA)
            if flip:
                frame = cv2.flip(frame, 1)  # same selfie-mirroring as capture_sequences.py
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            left_xyz, right_xyz = detector(rgb)
            vectors.append(build_frame_vector(left_xyz, right_xyz))
    finally:
        cap.release()

    if not vectors:
        raise IOError(f"video decoded to zero frames: {video_path}")
    return np.stack(vectors).astype(np.float32), float(src_fps)


# ---------------------------------------------------------------------------
# Steps 3-5: pure numpy (unit-testable, no video/MediaPipe needed)
# ---------------------------------------------------------------------------
def resample_to_fps(vectors, src_fps, target_fps):
    """Nearest-frame resampling of the timeline from src_fps to target_fps."""
    n = len(vectors)
    if n == 0:
        return vectors
    duration = n / src_fps
    m = max(1, int(round(duration * target_fps)))
    # floor(x + 0.5) = round-half-UP. (np.round rounds halves to the nearest even
    # number, which makes the frame picks needlessly irregular.)
    idx = np.minimum(np.floor(np.arange(m) / target_fps * src_fps + 0.5).astype(int), n - 1)
    return vectors[idx]


def hand_presence(vectors):
    """Boolean per frame: is at least one hand visible?"""
    return (vectors[:, LEFT_FLAG_INDEX] > 0.5) | (vectors[:, RIGHT_FLAG_INDEX] > 0.5)


def active_segment(vectors, margin):
    """Frames from the first to the last hand-visible frame (+/- margin).
    Returns None if no hand is ever visible."""
    present = np.flatnonzero(hand_presence(vectors))
    if present.size == 0:
        return None
    start = max(0, int(present[0]) - margin)
    end = min(len(vectors) - 1, int(present[-1]) + margin)
    return vectors[start:end + 1]


def segment_to_windows(segment, window=SEQUENCE_LENGTH, stride=4, max_windows=8):
    """Cuts a segment into (window, 128) training windows.

    - Longer than one window: overlapping windows every `stride` frames, always
      including one flush with the end of the segment.
    - Shorter than one window: the segment is placed at the start, middle and
      end of a zero-padded window (idle frames are zeros, like live capture).
    - More than max_windows candidates: evenly sub-sampled."""
    length = len(segment)
    windows = []
    if length < window:
        pad_total = window - length
        for offset in sorted({0, pad_total // 2, pad_total}):
            w = np.zeros((window, FRAME_VECTOR_SIZE), dtype=np.float32)
            w[offset:offset + length] = segment
            windows.append(w)
    else:
        starts = list(range(0, length - window + 1, stride))
        if starts[-1] != length - window:
            starts.append(length - window)
        windows = [segment[s:s + window] for s in starts]

    if len(windows) > max_windows:
        pick = sorted(set(np.linspace(0, len(windows) - 1, max_windows).round().astype(int).tolist()))
        windows = [windows[i] for i in pick]
    return [np.ascontiguousarray(w, dtype=np.float32) for w in windows]


# ---------------------------------------------------------------------------
# Whole-clip convenience wrapper
# ---------------------------------------------------------------------------
def video_to_windows(video_path, detector, target_fps=20.0, stride=4, margin=3,
                     max_windows=8, min_active_frames=6, flip=True):
    """Returns (windows, info). `windows` is a list of (SEQUENCE_LENGTH, 128)
    arrays -- empty if the clip is unusable; info["status"] says why:
      "ok" | "no_hands" | "too_short".
    """
    vectors, src_fps = read_video_frame_vectors(video_path, detector, flip=flip)
    info = {"src_frames": len(vectors), "src_fps": src_fps, "status": "ok", "active_frames": 0}

    resampled = resample_to_fps(vectors, src_fps, target_fps)
    segment = active_segment(resampled, margin)
    if segment is None:
        info["status"] = "no_hands"
        return [], info

    # Count only frames that really have hands (margin frames don't count)
    info["active_frames"] = int(hand_presence(segment).sum())
    if info["active_frames"] < min_active_frames:
        info["status"] = "too_short"
        return [], info

    return segment_to_windows(segment, SEQUENCE_LENGTH, stride, max_windows), info