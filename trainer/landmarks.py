"""Landmark format, README section 6 of the data-collection folder (the contract).

6.2  per-frame 128-value vector: Left 63 + flag, Right 63 + flag, slots by label
6.3  per-hand normalization: aspect correction, wrist origin, / |wrist - MCP9|
6.4  sign window (imported from the recorder's hudyat_common.sign_window)
6.5  resample to 32 frames, presence-aware

The browser app implements exactly these steps; tests/test_landmarks.py pins them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import recorder
from . import settings as cfg

N_FRAMES = 32
FRAME_DIM = 128
HAND_DIM = 63
LEFT_OFFSET, RIGHT_OFFSET = 0, 64
LEFT_FLAG, RIGHT_FLAG = 63, 127
SCALE_EPS = 1e-6


# ----------------------------------------------------------------------------
# 6.3 normalization
# ----------------------------------------------------------------------------
def normalize_hand(lm: np.ndarray, width: int, height: int) -> np.ndarray:
    """(21,3) MediaPipe normalized landmarks -> 63 values (landmark-major x,y,z)."""
    p = np.asarray(lm, dtype=np.float64).reshape(21, 3).copy()
    a = width / height
    p[:, 0] *= a
    p[:, 2] *= a
    p -= p[0]
    s = float(np.sqrt((p[9] ** 2).sum()))
    out = p.reshape(-1)
    if s > SCALE_EPS:
        out = out / s
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# 6.2 per-frame vector
# ----------------------------------------------------------------------------
@dataclass
class Hand:
    label: str              # "Left" / "Right" (already through the recorder's handedness rule)
    score: float
    landmarks: np.ndarray   # (21,3)


def pick_hands(hands: list[Hand]) -> tuple[dict[str, Hand], bool]:
    """{label: hand} by handedness label; a shared label goes to the higher score. (hands, duplicate)."""
    best: dict[str, Hand] = {}
    duplicate = False
    for h in hands:
        if h.label not in ("Left", "Right"):
            continue
        if h.label in best:
            duplicate = True
            if h.score <= best[h.label].score:
                continue
        best[h.label] = h
    return best, duplicate


def frame_vector(hands: list[Hand], width: int, height: int) -> tuple[np.ndarray, bool]:
    """(128-vector, duplicate_label_happened). Slots by label; higher score wins a shared label."""
    v = np.zeros(FRAME_DIM, dtype=np.float32)
    best, duplicate = pick_hands(hands)
    for label, off, flag in (("Left", LEFT_OFFSET, LEFT_FLAG), ("Right", RIGHT_OFFSET, RIGHT_FLAG)):
        if label in best:
            v[off:off + HAND_DIM] = normalize_hand(best[label].landmarks, width, height)
            v[flag] = 1.0
    return v, duplicate


# ----------------------------------------------------------------------------
# 6.5 resampling
# ----------------------------------------------------------------------------
def _slots(width: int) -> list[tuple[np.ndarray, tuple[int, ...], int | None]]:
    """[(value indices, presence flags, flag index to copy)] of a frame layout (docs/FEATURES.md).

    128 = the README 6 frame; 141 = the v3 model input (+ location against the face, as the app resamples it);
    149 = rich (+ raw wrist and size per hand + location against the face). A hand's location values are
    interpolated only when the hand AND the face are in both frames (README 6.5)."""
    left = list(range(LEFT_OFFSET, LEFT_OFFSET + HAND_DIM))
    right = list(range(RIGHT_OFFSET, RIGHT_OFFSET + HAND_DIM))
    if width == FRAME_DIM:
        return [(np.array(left), (LEFT_FLAG,), LEFT_FLAG), (np.array(right), (RIGHT_FLAG,), RIGHT_FLAG)]
    if width == FRAME_DIM + 13:
        face = 140
        return [(np.array(left), (LEFT_FLAG,), LEFT_FLAG), (np.array(right), (RIGHT_FLAG,), RIGHT_FLAG),
                (np.arange(128, 134), (LEFT_FLAG, face), None), (np.arange(134, 140), (RIGHT_FLAG, face), None),
                (np.array([], int), (), face)]
    if width != 149:
        raise ValueError(f"unknown frame layout with {width} values")
    left += [128, 129, 130, 134]
    right += [131, 132, 133, 135]
    face = 148
    return [(np.array(left), (LEFT_FLAG,), LEFT_FLAG), (np.array(right), (RIGHT_FLAG,), RIGHT_FLAG),
            (np.arange(136, 142), (LEFT_FLAG, face), None), (np.arange(142, 148), (RIGHT_FLAG, face), None),
            (np.array([], int), (), face)]


def resample(frames: np.ndarray, n_out: int = N_FRAMES, times: np.ndarray | None = None) -> np.ndarray:
    """(T,D) -> (n_out,D) for D = 128 / 141 / 149. Each hand slot separately (a hand's extra values go with it);
    never blends with an absent hand (or, for the location values, an absent face).

    `times` (optional, length n_out, values in 0..T-1) replaces the uniform positions
    j*(T-1)/(n_out-1); augmentation uses it for time warping.
    """
    frames = np.asarray(frames, dtype=np.float32)
    T, D = frames.shape
    if T == 0:
        raise ValueError("no frames to resample")
    slots = _slots(D)
    out = np.zeros((n_out, D), dtype=np.float32)
    if times is None:
        times = np.arange(n_out) * (T - 1) / (n_out - 1) if n_out > 1 else np.zeros(1)
    for j, t in enumerate(times):
        t = float(min(max(t, 0.0), T - 1))
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, T - 1)
        w = t - i0
        nearest = i0 if w < 0.5 else i1
        for idx, flags, flag in slots:
            if all(frames[i0, f] > 0.5 and frames[i1, f] > 0.5 for f in flags):
                out[j, idx] = (1 - w) * frames[i0, idx] + w * frames[i1, idx]
            else:
                out[j, idx] = frames[nearest, idx]
            if flag is not None:
                out[j, flag] = frames[nearest, flag]
    return out


def present_mask(frames: np.ndarray) -> np.ndarray:
    """True where at least one hand slot is present."""
    return (frames[:, LEFT_FLAG] > 0.5) | (frames[:, RIGHT_FLAG] > 0.5)


@dataclass
class ClipResult:
    sample: np.ndarray          # (32,149) float32, the rich sample (docs/FEATURES.md)
    frames: np.ndarray          # (T,149) all frames before trimming
    hand_counts: np.ndarray     # (T,) number of hands MediaPipe detected per frame
    fps: float
    width: int
    height: int
    trim_start: int
    trim_end: int
    hand_detect_rate: float     # inside the sign window
    both_hands_rate: float
    duplicate_frames: list[int] = field(default_factory=list)
    face_detect_rate: float | None = None   # inside the sign window; None = no face model ran (photos)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def frames_kept(self) -> int:
        return self.trim_end - self.trim_start + 1


def window_and_sample(frames: np.ndarray, hand_counts: np.ndarray, fps: float,
                      data_dir: Path | None = None, params: dict | None = None) -> tuple[np.ndarray, int, int, float, float]:
    """6.4 + 6.5 on per-frame vectors: (sample, start, end, hand_detect_rate, both_hands_rate)."""
    counts = np.asarray(hand_counts)
    start, end, _ = recorder.sign_window(counts > 0, fps, data_dir, params)
    window = counts[start:end + 1]
    hdr = float((window >= 1).mean())
    bhr = float((window >= 2).mean())
    return resample(frames[start:end + 1]), start, end, hdr, bhr


# ----------------------------------------------------------------------------
# MediaPipe
# ----------------------------------------------------------------------------
def create_landmarker(model_path: Path = cfg.MODEL_PATH, data_dir: Path | None = None, params: dict | None = None):
    """HandLandmarker in VIDEO mode, num_hands=2, with the recorder's confidence settings (or adjusted ones)."""
    from mediapipe.tasks.python import BaseOptions, vision
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {model_path}. Run 'SETUP (run once).bat' again.")
    hc = recorder.load_common(data_dir)
    p = params or recorder.default_params(data_dir)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=hc.NUM_HANDS,
        min_hand_detection_confidence=float(p["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(p["min_hand_presence_confidence"]),
        min_tracking_confidence=float(p["min_tracking_confidence"]),
    )
    return vision.HandLandmarker.create_from_options(options)


def face_model_path(data_dir: Path | None = None) -> Path:
    """The face model: this project's models/ folder, else the recorder's (Setup step 5 of its README)."""
    if cfg.FACE_MODEL_PATH.exists() or data_dir is None:
        return cfg.FACE_MODEL_PATH
    other = Path(data_dir) / "models" / "face_landmarker.task"
    return other if other.exists() else cfg.FACE_MODEL_PATH


def create_face_landmarker(data_dir: Path | None = None):
    """FaceLandmarker in VIDEO mode, one face (README 6.7), the same settings as the recorder's take review."""
    from mediapipe.tasks.python import BaseOptions, vision
    path = face_model_path(data_dir if data_dir is not None else cfg.data_dir(cfg.load_settings()))
    if not path.exists():
        raise FileNotFoundError(
            f"MediaPipe face model not found at {path}. Copy face_landmarker.task from the recorder's models "
            f"folder into this project's models folder.")
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)), running_mode=vision.RunningMode.VIDEO, num_faces=1))


def detect_face(landmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
    """The face's landmarks (N,3) on one UNMIRRORED BGR frame, or None if no face was found."""
    import cv2
    import mediapipe as mp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                                         int(timestamp_ms))
    if not result.face_landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)


def detect(landmarker, frame_bgr: np.ndarray, timestamp_ms: int, data_dir: Path | None = None) -> list[Hand]:
    """MediaPipe on one UNMIRRORED BGR frame (README 6.1). Labels used as returned."""
    import cv2
    import mediapipe as mp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(image, int(timestamp_ms))
    hands = []
    for lms, handedness in zip(result.hand_landmarks, result.handedness):
        c = handedness[0]
        hands.append(Hand(label=recorder.actual_hand_label(c.category_name, data_dir),
                          score=float(c.score),
                          landmarks=np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)))
    return hands


def extract_clip(video_path: Path, fps_hint: float | None = None, model_path: Path = cfg.MODEL_PATH,
                 data_dir: Path | None = None, params: dict | None = None) -> ClipResult:
    """Read every frame of a clip, run MediaPipe (hands and face), build the (32,149) rich sample."""
    import cv2
    from .features import FACE_FLAG, frame_vector_rich
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {video_path}")
    fps = float(fps_hint or 0) or float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    vectors, counts, dups = [], [], []
    width = height = 0
    landmarker = create_landmarker(model_path, data_dir, params)
    try:
        face_landmarker = create_face_landmarker(data_dir)
    except Exception:
        landmarker.close()
        cap.release()
        raise
    try:
        i, last_ts = 0, -1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            height, width = frame.shape[:2]
            ts = max(last_ts + 1, int(round(i * 1000.0 / fps)))
            last_ts = ts
            hands = detect(landmarker, frame, ts, data_dir)
            face = detect_face(face_landmarker, frame, ts)
            v, dup = frame_vector_rich(hands, width, height, face)
            if dup:
                dups.append(i)
            vectors.append(v)
            counts.append(len(hands))
            i += 1
    finally:
        landmarker.close()
        face_landmarker.close()
        cap.release()
    if not vectors:
        raise IOError(f"video has no readable frames: {video_path}")
    frames = np.stack(vectors)
    counts = np.asarray(counts)
    sample, start, end, hdr, bhr = window_and_sample(frames, counts, fps, data_dir, params)
    fdr = float((frames[start:end + 1, FACE_FLAG] > 0.5).mean())
    return ClipResult(sample, frames, counts, fps, width, height, start, end, hdr, bhr, dups, fdr)


# ----------------------------------------------------------------------------
# Photos (uploaded class-folder datasets)
# ----------------------------------------------------------------------------
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGE_PAD = 1 / 3            # border added around tight hand crops so MediaPipe finds the hand


def mirror_hand_vector(v63: np.ndarray) -> np.ndarray:
    """x -> -x of a normalized hand (a left hand becomes the matching right hand)."""
    h = np.asarray(v63, np.float32).reshape(21, 3).copy()
    h[:, 0] *= -1
    return h.reshape(-1)


def image_frame(hands: list[Hand], width: int, height: int) -> np.ndarray:
    """128-vector for a photo: the most confident hand, stored as a RIGHT hand (mirrored if labelled Left).

    Photos have no body or camera context, so their handedness label is unreliable (and dataset tools
    often add mirrored copies). Our signers are right-handed and sign the right hand as "Right" (6.1),
    so every photo hand goes into the Right slot.
    """
    v = np.zeros(FRAME_DIM, dtype=np.float32)
    if not hands:
        return v
    best = max(hands, key=lambda h: h.score)
    vec = normalize_hand(best.landmarks, width, height)
    if best.label == "Left":
        vec = mirror_hand_vector(vec)
    v[RIGHT_OFFSET:RIGHT_OFFSET + HAND_DIM] = vec
    v[RIGHT_FLAG] = 1.0
    return v


def create_image_landmarker(model_path: Path = cfg.MODEL_PATH, params: dict | None = None):
    """HandLandmarker in IMAGE mode (photos), with the same confidence values as for videos."""
    from mediapipe.tasks.python import BaseOptions, vision
    p = params or recorder.default_params()
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=float(p["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(p["min_hand_presence_confidence"])))


def extract_image(image_path: Path, model_path: Path = cfg.MODEL_PATH, data_dir: Path | None = None,
                  params: dict | None = None, landmarker=None) -> ClipResult:
    """A photo becomes a held-still sign: the same frame repeated 32 times. `landmarker` (IMAGE mode) can be
    passed in to reuse one for many photos."""
    import cv2
    import mediapipe as mp
    img = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {image_path}")
    h0, w0 = img.shape[:2]
    py, px = int(h0 * IMAGE_PAD), int(w0 * IMAGE_PAD)
    img = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_REPLICATE)
    height, width = img.shape[:2]
    own = landmarker is None
    lm = landmarker or create_image_landmarker(model_path, params or recorder.default_params(data_dir))
    try:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    finally:
        if own:
            lm.close()
    hands = [Hand(h[0].category_name, float(h[0].score), np.array([[q.x, q.y, q.z] for q in l], np.float32))
             for l, h in zip(res.hand_landmarks, res.handedness)]
    from .features import photo_frame_rich
    frame = photo_frame_rich(max(hands, key=lambda h: h.score) if hands else None, width, height)
    sample = np.repeat(frame[None], N_FRAMES, axis=0)
    found = 1.0 if hands else 0.0
    return ClipResult(sample, frame[None], np.array([len(hands)]), 30.0, w0, h0, 0, 0, found,
                      1.0 if len(hands) >= 2 else 0.0)
