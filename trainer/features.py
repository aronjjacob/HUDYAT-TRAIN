"""The model's features, mirroring and the dominant hand (docs/FEATURES.md is the contract; tests pin it).

Extraction stores one RICH sample per clip, (32, 149) float32:

    0-127    the v1 frame of README section 6 (unchanged)
    128-130  Left wrist, raw:  (x * W/H, y, z * W/H) of landmark 0, before the wrist is subtracted
    131-133  Right wrist, raw
    134      Left hand size:  |landmark 9 - landmark 0| in the same coordinates (the 6.3 scale)
    135      Right hand size
    136-141  Left hand location against the face (README 6.7): wrist, thumb tip, index tip (x, y)
    142-147  Right hand location
    148      Face present flag
    An absent hand's values are all zero; without a face all 13 location values are zero.

From it, the model input is derived (`derive`), 128 values per frame, in one of two feature versions the HUDYAT
app knows, so a model trained here gets in the app the numbers it was trained on:

    v1  "hand-relative-v1": every landmark relative to its wrist (the wrist itself is 0).
    v2  "hand-relative-sequence-v1" (the format of Russel's trainer): landmarks 1-20 exactly as v1; the wrist is its
        movement since the hand's first valid frame of the clip, divided by the hand's median 2D palm length over
        the clip. A still hand (a photo) gives exactly v1.
    v3  "hand-face-location-v1": v1 + the 13 location values of README 6.7 (141 values). The only version that
        tells apart signs made with the same hand at different places (Father at the forehead, Mother at the
        chin). Used for Family; the app runs its face model only for v3 models.
 Values 0-127 are divided 3D (README 6.3); the app divides by the 2D wrist -> landmark 9 length, so each
present hand is divided once more by the 2D length of its own (already 3D-normalized) landmark 9. A hand whose 2D
length is ~0 counts as absent, as in the app. (The app also stores the hands at other indices; the export reorders
the model's first layer for that, see trainer/app_export.py.)

Samples are stored AS RECORDED (README 6.1: left-handed signers are never flipped). The canonical
dominant hand is applied when a sample is used: `canonical(x, "left")` mirrors it into the right-handed form.
"""

from __future__ import annotations

import re

import numpy as np

from .landmarks import (FRAME_DIM, HAND_DIM, LEFT_FLAG, LEFT_OFFSET, RIGHT_FLAG, RIGHT_OFFSET, SCALE_EPS, Hand,
                        frame_vector, pick_hands)

RICH_DIM = 149
L_WRIST, R_WRIST = slice(128, 131), slice(131, 134)
L_SCALE, R_SCALE = 134, 135
L_LOC, R_LOC = slice(136, 142), slice(142, 148)
FACE_FLAG = 148
LOC_DIM = 13                          # README 6.7: 6 per hand + the face flag
FACE_NOSE, FACE_FOREHEAD, FACE_CHIN = 1, 10, 152
LOC_POINTS = [0, 4, 8]                # wrist, thumb tip, index tip

VERSIONS = (1, 2, 3)
NORMALIZATIONS = {1: "hand-relative-v1", 2: "hand-relative-sequence-v1",     # the names the HUDYAT app gives them
                  3: "hand-face-location-v1"}
APP_VERSIONS = (1, 2, 3)              # the ones the HUDYAT app computes (v3: with its face model)
NORMALIZATION = NORMALIZATIONS[1]
HANDS = ("right", "left")


# Runs and models record the normalization they were made with ("normalization"). Ones without it are from before
# that was recorded: those made after the switch to hand-relative-v1 (2026-10-21 03:13) use it; older ones (such as
# the backup of 2026-10-21 02:48) used the 3D one and batch normalization epsilon 1e-3, so they must not be used.
SWITCHED_TO_APP_FEATURES = "2026-10-21T03:13"


def made_at(conf: dict) -> str:
    """When a run (created), a final model (date) or a preprocess.json (its model_run_id) was made, as ISO text."""
    if conf.get("created") or conf.get("date"):
        return str(conf.get("created") or conf.get("date"))
    m = re.search(r"(\d{8})-(\d{6})$", str(conf.get("model_run_id", "")))
    return f"{m[1][:4]}-{m[1][4:6]}-{m[1][6:]}T{m[2][:2]}:{m[2][2:4]}:{m[2][4:]}" if m else ""


def normalization_of(conf: dict | None) -> str | None:
    """The normalization a run or model was made with, or None when it predates hand-relative-v1."""
    if not conf:
        return None
    if conf.get("normalization"):
        return str(conf["normalization"])
    when = made_at(conf)
    return NORMALIZATION if when and when >= SWITCHED_TO_APP_FEATURES else None


def require_app_features(conf: dict | None, what: str, hint: str) -> None:
    """Refuse a run or model made with other features than the app's (it would get the wrong numbers)."""
    from .dataset import PlainError
    if normalization_of(conf) not in NORMALIZATIONS.values():
        raise PlainError(f"{what} was made with older features than the HUDYAT app uses.", hint)


def feature_dim(version: int = 1) -> int:
    if int(version) not in VERSIONS:
        raise ValueError(f"unknown feature version {version}")
    return FRAME_DIM + LOC_DIM if int(version) == 3 else FRAME_DIM


def normalization(version: int = 1) -> str:
    """The app's name for a feature version's normalization."""
    feature_dim(version)
    return NORMALIZATIONS[int(version)]


def version_of(name: str) -> int:
    """The feature version of an app normalization name (an export's preprocessing.normalization)."""
    for v, n in NORMALIZATIONS.items():
        if n == name:
            return v
    raise ValueError(f"not a feature version of this trainer: {name!r}")


# ----------------------------------------------------------------------------
# Rich frame
# ----------------------------------------------------------------------------
def wrist_and_size(lm: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    """Raw wrist (aspect-corrected) and the wrist-MCP9 distance of one hand (21,3)."""
    p = np.asarray(lm, dtype=np.float64).reshape(21, 3).copy()
    a = width / height
    p[:, 0] *= a
    p[:, 2] *= a
    return p[0], float(np.sqrt(((p[9] - p[0]) ** 2).sum()))


def location_features(face: np.ndarray | None, best: dict[str, Hand], width: int, height: int) -> np.ndarray:
    """README 6.7: the 13 values (Left 6, Right 6, face flag). face: (N,3) MediaPipe face landmarks or None.
    Each point is ((x - nose.x) / s, (y - nose.y) / s), s = the 2D forehead -> chin length, x aspect-corrected."""
    out = np.zeros(LOC_DIM, dtype=np.float32)
    if face is None or len(face) <= FACE_CHIN:
        return out
    a = width / height
    f = np.asarray(face, dtype=np.float64)[:, :2] * [a, 1.0]
    nose = f[FACE_NOSE]
    s = float(np.hypot(*(f[FACE_FOREHEAD] - f[FACE_CHIN])))
    if s < SCALE_EPS:
        return out
    for h, label in enumerate(("Left", "Right")):
        if label in best:
            p = np.asarray(best[label].landmarks, dtype=np.float64)[LOC_POINTS, :2] * [a, 1.0]
            out[h * 6:h * 6 + 6] = ((p - nose) / s).reshape(-1)
    out[12] = 1.0
    return out


def frame_vector_rich(hands: list[Hand], width: int, height: int,
                      face: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """(149-vector, duplicate_label_happened): the v1 frame + raw wrist and size per hand + the location against
    the face (zeros when no face is given or found)."""
    v1, duplicate = frame_vector(hands, width, height)
    out = np.zeros(RICH_DIM, dtype=np.float32)
    out[:FRAME_DIM] = v1
    best, _ = pick_hands(hands)
    for label, wr, sc in (("Left", L_WRIST, L_SCALE), ("Right", R_WRIST, R_SCALE)):
        if label in best:
            w, s = wrist_and_size(best[label].landmarks, width, height)
            out[wr] = w
            out[sc] = s
    out[L_LOC.start:] = location_features(face, best, width, height)
    return out, duplicate


def photo_frame_rich(hand: Hand | None, width: int, height: int) -> np.ndarray:
    """A photo's frame: its hand always in the Right slot (mirrored if labelled Left), see landmarks.image_frame.
    Photos are hand crops: no face, so the location values stay zero."""
    from .landmarks import image_frame
    out = np.zeros(RICH_DIM, dtype=np.float32)
    if hand is None:
        return out
    out[:FRAME_DIM] = image_frame([hand], width, height)
    w, s = wrist_and_size(hand.landmarks, width, height)
    if hand.label == "Left":
        w = w * [-1, 1, 1]
    out[R_WRIST] = w
    out[R_SCALE] = s
    return out


# ----------------------------------------------------------------------------
# Model input
# ----------------------------------------------------------------------------
def derive(x: np.ndarray, version: int = 1) -> np.ndarray:
    """Model input from rich samples: (T,149) -> (T,128), or (N,T,149) -> (N,T,128). The app's hand-relative-v1:
    every present hand's 63 values divided by the 2D length of its landmark 9 (wrist -> middle-finger MCP).
    v2 (hand-relative-sequence-v1) then replaces each wrist by its movement over the clip (`_wrist_path`).
    v3 (hand-face-location-v1) is v1 followed by the 13 location values, (T,141); a hand that v1 counts as
    absent gets location zeros too."""
    feature_dim(version)
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != RICH_DIM:
        raise ValueError(f"expected rich samples with {RICH_DIM} values per frame, got {x.shape[-1]}")
    if int(version) == 2 and x.ndim < 2:
        raise ValueError("feature v2 needs the frames of a clip, (T,149) or (N,T,149)")
    out = x[..., :FRAME_DIM].astype(np.float64)
    for off, flag, wrist, size in ((LEFT_OFFSET, LEFT_FLAG, L_WRIST, L_SCALE), (RIGHT_OFFSET, RIGHT_FLAG, R_WRIST, R_SCALE)):
        hand = out[..., off:off + HAND_DIM]
        k = np.hypot(hand[..., 27], hand[..., 28])                 # landmark 9's x and y
        present = (out[..., flag] > 0.5) & (k > SCALE_EPS)
        hand[...] = np.where(present[..., None], hand / np.where(present, k, 1.0)[..., None], 0.0)
        out[..., flag] = present
        if int(version) == 2:
            # The 2D palm length in the raw wrist's units: values 0-127 are divided by the 3D size, so landmark 9's
            # 2D length k is the 2D palm length / the 3D size.
            hand[..., 0:3] = _wrist_path(x[..., wrist].astype(np.float64), k * x[..., size], present)
    if int(version) == 3:
        loc = x[..., L_LOC.start:].astype(np.float64)
        loc[..., 0:6] *= out[..., LEFT_FLAG:LEFT_FLAG + 1]
        loc[..., 6:12] *= out[..., RIGHT_FLAG:RIGHT_FLAG + 1]
        out = np.concatenate([out, loc], axis=-1)
    return out.astype(np.float32)


def _wrist_path(wrist: np.ndarray, palm: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Per clip (the T axis): each present frame's raw wrist minus the wrist of the first present frame, divided by
    the median palm length of the present frames; absent frames 0."""
    out = np.zeros_like(wrist)
    w, p, v, o = (a.reshape(-1, *a.shape[-2:]) if a.ndim == wrist.ndim else a.reshape(-1, a.shape[-1])
                  for a in (wrist, palm, present, out))
    for i in range(len(w)):
        if v[i].any():
            o[i, v[i]] = (w[i, v[i]] - w[i, np.argmax(v[i])]) / np.median(p[i, v[i]])
    return out


# ----------------------------------------------------------------------------
# Mirror (works on both layouts: 128 or 149 values per frame; any leading shape)
# ----------------------------------------------------------------------------
_X_HAND = np.concatenate([np.arange(LEFT_OFFSET, LEFT_OFFSET + HAND_DIM, 3),
                          np.arange(RIGHT_OFFSET, RIGHT_OFFSET + HAND_DIM, 3)])
_SWAP_HANDS = np.r_[np.arange(64, 128), np.arange(0, 64)]


def mirror(x: np.ndarray) -> np.ndarray:
    """Left-right mirror image: x -> -x everywhere, then the Left and Right parts swap (flags too).

    The raw wrist positions (rich layout) are only ever used as differences, so negating x is the mirror image.
    The location values are measured from the nose, so negating their x mirrors them too (the face flag stays)."""
    x = np.array(x, dtype=np.float32, copy=True)
    d = x.shape[-1]
    if d not in (FRAME_DIM, FRAME_DIM + LOC_DIM, RICH_DIM):
        raise ValueError(f"unknown frame layout with {d} values")
    x[..., _X_HAND] *= -1
    order = np.arange(d)
    order[:FRAME_DIM] = _SWAP_HANDS
    if d == FRAME_DIM + LOC_DIM:                    # the v3 model input: 128-133 Left location, 134-139 Right
        x[..., 128:140:2] *= -1
        order[128:134], order[134:140] = np.arange(134, 140), np.arange(128, 134)
    if d == RICH_DIM:
        x[..., [128, 131]] *= -1
        order[128:131], order[131:134] = np.arange(131, 134), np.arange(128, 131)
        order[134], order[135] = 135, 134
        x[..., [136, 138, 140, 142, 144, 146]] *= -1
        order[L_LOC], order[R_LOC] = np.arange(R_LOC.start, R_LOC.stop), np.arange(L_LOC.start, L_LOC.stop)
    return x[..., order]


# ----------------------------------------------------------------------------
# Dominant hand
# ----------------------------------------------------------------------------
def canonical(x: np.ndarray, hand: str) -> np.ndarray:
    """The right-dominant form of a sample signed with `hand` ("right" / "left")."""
    if hand == "right":
        return x
    if hand == "left":
        return mirror(x)
    raise ValueError(f"hand must be 'right' or 'left', not {hand!r} (resolve 'auto' with auto_hand first)")


def _path(x: np.ndarray, present: np.ndarray, wrist: slice, size: int) -> float:
    pts = x[present, wrist].astype(np.float64)
    if len(pts) < 2:
        return 0.0
    s = float(np.median(x[present, size]))
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) / (s if s > SCALE_EPS else 1.0)


def auto_hand(x: np.ndarray) -> str:
    """Which hand signs, from one rich window (T,149): "right" or "left".

    Mostly one hand visible -> that hand (labelled Left -> "left"). Mostly two hands -> the hand whose wrist
    travels further (path length / hand size). No hands or a tie -> "right"."""
    x = np.asarray(x)
    lp, rp = x[:, LEFT_FLAG] > 0.5, x[:, RIGHT_FLAG] > 0.5
    anyh = lp | rp
    if not anyh.any():
        return "right"
    one = lp ^ rp
    if one.sum() * 2 > anyh.sum():
        return "left" if (one & lp).sum() > (one & rp).sum() else "right"
    return "left" if _path(x, lp, L_WRIST, L_SCALE) > _path(x, rp, R_WRIST, R_SCALE) else "right"


def resolve_hand(x: np.ndarray, hand: str) -> str:
    """"auto" -> auto_hand(x); "right" / "left" as given."""
    return auto_hand(x) if hand == "auto" else hand
