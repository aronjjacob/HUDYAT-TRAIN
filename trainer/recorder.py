"""Access to the recorder's hudyat_common.py (imported from the data-collection folder).

The trim rule (sign_window), the handedness rule and the MediaPipe settings must be
identical to the recorder's, so they are imported from its own file. If that file
cannot be imported (folder missing, import error) the COPIES below are used;
tests/test_recorder_parity.py checks that the copies give identical results.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import settings as cfg

# ----------------------------------------------------------------------------
# Copies (used only when hudyat_common.py cannot be imported)
# ----------------------------------------------------------------------------
TRIM_MARGIN_S = 0.1
TRIM_MIN_FRAMES = 8
NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
SWAP_HANDEDNESS_LABELS = False
HAND_DETECT_WARN_RATE = 0.80
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def sign_window_copy(hand_present, fps: float, margin_s: float = TRIM_MARGIN_S,
                     min_frames: int = TRIM_MIN_FRAMES) -> tuple[int, int, bool]:
    """Copy of hudyat_common.sign_window (README 6.4); margin / minimum length can be adjusted."""
    present = np.asarray(hand_present, dtype=bool)
    total = len(present)
    if total == 0:
        raise ValueError("empty clip")
    if not present.any():
        return 0, total - 1, False
    margin = _round_half_up(margin_s * fps)
    first = int(np.argmax(present))
    last = int(total - 1 - np.argmax(present[::-1]))
    start, end = max(0, first - margin), min(total - 1, last + margin)
    need = min(int(min_frames), total)
    if end - start + 1 < need:
        centre = (start + end) / 2
        start = _round_half_up(centre - (need - 1) / 2)
        start = max(0, min(start, total - need))
        end = start + need - 1
    return start, end, (start > 0 or end < total - 1)


def actual_hand_label_copy(label: str) -> str:
    if not SWAP_HANDEDNESS_LABELS:
        return label
    return {"Left": "Right", "Right": "Left"}.get(label, label)


FALLBACK = SimpleNamespace(
    sign_window=sign_window_copy, actual_hand_label=actual_hand_label_copy,
    TRIM_MARGIN_S=TRIM_MARGIN_S, TRIM_MIN_FRAMES=TRIM_MIN_FRAMES, NUM_HANDS=NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE=MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE=MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE=MIN_TRACKING_CONFIDENCE,
    SWAP_HANDEDNESS_LABELS=SWAP_HANDEDNESS_LABELS, HAND_DETECT_WARN_RATE=HAND_DETECT_WARN_RATE,
    HAND_CONNECTIONS=HAND_CONNECTIONS, imported_from=None,
)

_cache: dict[str, object] = {}


def load_common(data_dir: Path | None = None):
    """The recorder's hudyat_common module, or FALLBACK if it cannot be imported."""
    if data_dir is None:
        data_dir = cfg.data_dir()
    if data_dir is None:
        return FALLBACK
    path = (Path(data_dir) / "hudyat_common.py").resolve()
    key = str(path)
    if key in _cache:
        return _cache[key]
    if not path.exists():
        return FALLBACK
    # Never write __pycache__ into the data-collection folder (it is read-only for us).
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("hudyat_common", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("hudyat_common", module)
        spec.loader.exec_module(module)
        module.imported_from = str(path)
    except Exception:  # noqa: BLE001 - any import problem -> use the copies
        return FALLBACK
    finally:
        sys.dont_write_bytecode = old_flag
    _cache[key] = module
    return module


def sign_window(hand_present, fps: float, data_dir: Path | None = None,
                params: dict | None = None) -> tuple[int, int, bool]:
    """The recorder's own sign_window; the trainer's copy only when the trim values were adjusted."""
    hc = load_common(data_dir)
    if params:
        m, n = float(params["trim_margin_s"]), int(params["trim_min_frames"])
        if (m, n) != (float(hc.TRIM_MARGIN_S), int(hc.TRIM_MIN_FRAMES)):
            return sign_window_copy(hand_present, fps, m, n)
    return hc.sign_window(hand_present, fps)


def default_params(data_dir: Path | None = None) -> dict:
    """The recorder's preprocessing values (README 6.4 + MediaPipe settings)."""
    hc = load_common(data_dir)
    return {"trim_margin_s": float(hc.TRIM_MARGIN_S), "trim_min_frames": int(hc.TRIM_MIN_FRAMES),
            "min_hand_detection_confidence": float(hc.MIN_HAND_DETECTION_CONFIDENCE),
            "min_hand_presence_confidence": float(hc.MIN_HAND_PRESENCE_CONFIDENCE),
            "min_tracking_confidence": float(hc.MIN_TRACKING_CONFIDENCE)}


def preprocess_params(settings: dict | None = None, data_dir: Path | None = None) -> dict:
    """Values used for extraction: the recorder's, with any adjustments saved in trainer_settings.json."""
    s = settings or cfg.load_settings()
    p = default_params(data_dir if data_dir is not None else cfg.data_dir(s))
    for k, v in (s.get("preprocess") or {}).items():
        if k in p and v is not None:
            p[k] = type(p[k])(v)
    return p


def params_id(p: dict) -> str:
    """Short id stored with every sample, e.g. m0.10-f8-d0.50-p0.50-t0.50."""
    return (f"m{p['trim_margin_s']:.2f}-f{int(p['trim_min_frames'])}-d{p['min_hand_detection_confidence']:.2f}"
            f"-p{p['min_hand_presence_confidence']:.2f}-t{p['min_tracking_confidence']:.2f}")


def parse_params_id(pid: str, data_dir: Path | None = None) -> dict:
    """Inverse of params_id ('' = the recorder's values, for samples made before the id was stored)."""
    import re
    m = re.fullmatch(r"m([\d.]+)-f(\d+)-d([\d.]+)-p([\d.]+)-t([\d.]+)", (pid or "").strip())
    if not m:
        return default_params(data_dir)
    return {"trim_margin_s": float(m[1]), "trim_min_frames": int(m[2]), "min_hand_detection_confidence": float(m[3]),
            "min_hand_presence_confidence": float(m[4]), "min_tracking_confidence": float(m[5])}


def is_default(p: dict, data_dir: Path | None = None) -> bool:
    return params_id(p) == params_id(default_params(data_dir))


def actual_hand_label(label: str, data_dir: Path | None = None) -> str:
    return load_common(data_dir).actual_hand_label(label)


def recorder_camera_settings(data_dir: Path | None = None) -> dict:
    """camera / backend from the recorder's settings.json (defaults 0 / msmf)."""
    import json
    out = {"camera": 0, "backend": "msmf", "camera_name": ""}
    d = data_dir or cfg.data_dir()
    if d and (Path(d) / "settings.json").exists():
        try:
            out.update(json.loads((Path(d) / "settings.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return out
