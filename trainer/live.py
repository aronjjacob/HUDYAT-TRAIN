"""Live sign window (README 6.6) and classification. Shared by the webcam and the --video path.

Two recognizers: the trainer's own model (Classifier) and the HUDYAT app's exported model files (AppClassifier,
the network and input layout the app runs). The app recognizer has two captures, so the same tester can compare:

  fixed  the recommended capture: the sign window ends when the hands go down (LiveSegmenter), is trimmed to the
         frames with a hand (6.4) and resampled without blending with an absent hand (6.5), and a left-handed
         signer is mirrored to the right-handed form (Hand: Right / Left / Auto)
  app    what the app does today (client/src/features/fsl-recognition): a fixed 2-second clip from the first
         frame with a hand (AppClipSegmenter), resampled by frame index with absent hands blended in, no mirror

Both accept a guess with the app's rule: at least 80 % sure and at least 15 points ahead of the second guess.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import settings as cfg
from .landmarks import window_and_sample

PRE_BUFFER_S = 0.1     # ring buffer before the first hand frame (matches the 0.1 s trim margin)
END_ABSENT_S = 0.2     # "hands disappeared" = no hand for this long (recommended in 6.6)
MAX_WINDOW_S = 3.0     # a window never lasts longer than this
MIN_HANDS_S = 0.2      # a window with less hand time than this is a detection blip, not a sign: no prediction
APP_CLIP_S = 2.0      # the app's fixed clip for moving signs (category-recognition.tsx)
APP_MIN_SCORE, APP_MIN_MARGIN = 0.80, 0.15      # the app's acceptance gates (gesture-lock.ts DEFAULT_GATES)
LIVE_FIELDS = ["time", "tester", "tester_hand", "target", "prediction", "confidence", "correct", "response_time_ms",
               "model_run_id", "feature_version", "recognizer", "capture", "margin", "time_from_hand_ms"]
CAPTURES = {"fixed": "Fixed: sign window, mirror", "app": "As the app today: 2-second clip"}
# A live-test setup = (recognizer, capture). The paper reports the first one that has attempts as its live result.
SETUPS = {("app", "fixed"): "App model · fixed capture", ("trainer", "fixed"): "Trainer model",
          ("app", "app"): "App model · app capture today"}


def setup_label(recognizer: str, capture: str) -> str:
    return SETUPS.get((recognizer, capture), f"{recognizer} · {capture}")


def setup_rule(recognizer: str, threshold: float) -> str:
    """When an attempt counts as correct in this setup."""
    if recognizer == "app":
        return (f"the top guess is the target, ≥ {APP_MIN_SCORE:.0%} sure and ≥ {APP_MIN_MARGIN * 100:.0f} points "
                "ahead of the second guess (the app's rule)")
    return f"the top guess is the target with ≥ {threshold:.0%} confidence"


def score_attempts(df, threshold: float):
    """`correct` again from each attempt's prediction, by its setup's rule: the trainer model at `threshold` (the
    settings' confidence threshold), the app model at the app's gates. Attempts saved under another threshold are
    counted by the current one."""
    df = df.copy()
    hit = df["prediction"] == df["target"]
    margin = pd_numeric(df["margin"])
    app = df["recognizer"] == "app"
    df["correct"] = (hit & ((~app & (df["confidence"] >= threshold))
                            | (app & (df["confidence"] >= APP_MIN_SCORE) & (margin >= APP_MIN_MARGIN)))).astype(int)
    return df


def pd_numeric(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


@dataclass
class Window:
    frames: np.ndarray        # (T,128)
    counts: np.ndarray        # (T,) hands per frame
    fps: float
    reason: str               # "hands_gone" | "max_length" | "flush"


class LiveSegmenter:
    """Feed one frame at a time; returns a Window when a sign has been completed, else None.

    A sign is completed when the hands appear and then disappear, or after 3 s. There is never a window while the
    hands are idle: after a 3-second window the hands must go down before the next one can start (hands held up
    in view are not a new sign), and a window with less than 0.2 s of hands is dropped."""

    def __init__(self, fps: float) -> None:
        self.set_fps(fps)
        self.ring: deque = deque(maxlen=self.n_pre)
        self.active: list | None = None
        self.absent_frames = 0
        self.waiting_for_rest = False       # True after a 3-second window until the hands have been down a moment

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.n_pre = max(1, _round_half_up(PRE_BUFFER_S * self.fps))
        self.n_end = max(1, _round_half_up(END_ABSENT_S * self.fps))
        self.n_max = max(1, _round_half_up(MAX_WINDOW_S * self.fps))
        self.n_min_hands = max(1, _round_half_up(MIN_HANDS_S * self.fps))

    @property
    def signing(self) -> bool:
        return self.active is not None

    def push(self, vector: np.ndarray, n_hands: int) -> Window | None:
        item = (np.asarray(vector, np.float32), int(n_hands))
        if self.active is None and self.waiting_for_rest:
            self.absent_frames = 0 if n_hands > 0 else self.absent_frames + 1
            if self.absent_frames >= self.n_end:
                self.waiting_for_rest = False
                self.absent_frames = 0
            self.ring.append(item)
            return None
        if self.active is None:
            if n_hands > 0:
                self.active = list(self.ring) + [item]
                self.ring.clear()
                self.absent_frames = 0
                return self._maybe_end_on_length()
            self.ring.append(item)
            return None
        self.active.append(item)
        self.absent_frames = 0 if n_hands > 0 else self.absent_frames + 1
        if self.absent_frames >= self.n_end:
            return self._finish("hands_gone")
        return self._maybe_end_on_length()

    def _maybe_end_on_length(self) -> Window | None:
        if self.active is not None and len(self.active) >= self.n_max:
            return self._finish("max_length")
        return None

    def flush(self) -> Window | None:
        """End of a video file: the open window counts as completed (the hands are gone with the video)."""
        return self._finish("flush") if self.active else None

    def _finish(self, reason: str) -> Window | None:
        frames = np.stack([f for f, _ in self.active])
        counts = np.array([c for _, c in self.active])
        self.active = None
        self.absent_frames = 0
        self.waiting_for_rest = reason == "max_length"
        if int((counts > 0).sum()) < self.n_min_hands:
            return None
        return Window(frames, counts, self.fps, reason)


class AppClipSegmenter:
    """The app's capture today: a clip starts at the first frame with a hand and is complete after 2 seconds,
    whatever the hands do; the next clip starts at the next frame with a hand. Same interface as LiveSegmenter."""

    waiting_for_rest = False

    def __init__(self, fps: float) -> None:
        self.set_fps(fps)
        self.active: list | None = None

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.n_max = max(1, _round_half_up(APP_CLIP_S * self.fps))

    @property
    def signing(self) -> bool:
        return self.active is not None

    def push(self, vector: np.ndarray, n_hands: int) -> Window | None:
        if self.active is None:
            if n_hands == 0:
                return None
            self.active = []
        self.active.append((np.asarray(vector, np.float32), int(n_hands)))
        return self._finish("clip_2s") if len(self.active) >= self.n_max else None

    def flush(self) -> Window | None:
        return self._finish("flush") if self.active else None

    def _finish(self, reason: str) -> Window | None:
        frames = np.stack([f for f, _ in self.active])
        counts = np.array([c for _, c in self.active])
        self.active = None
        return Window(frames, counts, self.fps, reason) if len(frames) >= 8 else None   # the app needs 8 frames


def app_resample(frames: np.ndarray, n_out: int = 32) -> np.ndarray:
    """The app's resample (category-core.js) on 128-value frames in the trainer's slots: every coordinate linear
    between frames floor(t) and floor(t)+1 (an absent hand's zeros included), each flag from the nearest frame
    (floor(t + 0.5)), and a hand whose flag is 0 zeroed."""
    from .landmarks import HAND_DIM, LEFT_FLAG, LEFT_OFFSET, RIGHT_FLAG, RIGHT_OFFSET
    f = np.asarray(frames, dtype=np.float64)
    T = len(f)
    out = np.zeros((n_out, f.shape[1]), dtype=np.float64)
    for t in range(n_out):
        pos = t * (T - 1) / (n_out - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, T - 1)
        r = pos - lo
        out[t] = f[lo] * (1 - r) + f[hi] * r
        near = int(np.floor(pos + 0.5))
        for off, flag in ((LEFT_OFFSET, LEFT_FLAG), (RIGHT_OFFSET, RIGHT_FLAG)):
            out[t, flag] = f[near, flag]
            if not out[t, flag]:
                out[t, off:off + HAND_DIM] = 0.0
    return out.astype(np.float32)


def app_today_input(window: Window, version: int = 1) -> np.ndarray:
    """What the app feeds its model today: every frame normalized (v2: over the whole clip), the whole 2-second clip
    resampled its way."""
    from .app_export import to_app_layout
    from .features import derive
    return to_app_layout(app_resample(derive(window.frames, version)))


def recommended_input(window: Window, hand: str = "auto", params: dict | None = None,
                      canonical: bool = True, version: int = 1) -> tuple[np.ndarray, dict]:
    """The recommended input (docs/APP_RECOGNITION.md has it in JavaScript for the app): every frame normalized like
    the app, the window cut to the frames with a hand (6.4), resampled without blending with an absent hand (6.5),
    and a left-handed signer mirrored (hand: right / left / auto). (32, 128) in the app's layout.

    Feature v2 is converted after resampling, like its training samples (app_export.WRIST_PATH). Feature v3 is
    converted per frame like v1, with the location values against the face (32, 141)."""
    from .app_export import to_app_layout
    from .features import derive, mirror, resolve_hand
    from .landmarks import resample
    sample, start, end, _, _ = window_and_sample(window.frames, window.counts, window.fps, params=params)
    used = resolve_hand(sample, hand) if canonical else "right"
    x = derive(sample, 2) if version == 2 else resample(derive(window.frames, 3 if version == 3 else 1)[start:end + 1])
    if used == "left":
        x = mirror(x)
    return to_app_layout(x), {"hand_used": used, "trim": (start, end)}


class AppClassifier:
    """The HUDYAT app's model files (an export of trainer/app_export.py): the same network, weights, input layout
    and acceptance gates as the app. capture: "fixed" (recommended) or "app" (as the app today)."""

    def __init__(self, export_dir: Path, capture: str = "fixed") -> None:
        import keras

        from .app_export import read_export
        from .model import build_model, fast_predict
        if capture not in CAPTURES:
            raise ValueError(f"capture must be one of {list(CAPTURES)}")
        self.capture = capture
        meta, ws = read_export(Path(export_dir))
        self.meta = meta
        self.labels = list(meta["classes"])
        self.model = build_model(len(self.labels))
        i = 0
        for layer in self.model.layers:                     # layer by layer, like the app's loadWeights
            n = len(layer.get_weights())
            layer.set_weights(ws[i:i + n])
            i += n
        if i != len(ws):
            raise ValueError("Unexpected model weight count.")
        self.predict = fast_predict(self.model)
        pre = meta["preprocessing"]
        from .features import version_of
        self.feature_version = version_of(pre["normalization"])
        self.version = meta["version"]
        self.model_run_id = (pre.get("trainer") or {}).get("model_run_id") or ""
        if self.feature_version == 3:
            self.capture = capture = "fixed"            # the app's 2-second clip predates the face location
        self.canonical = bool((pre.get("handedness") or {}).get("canonical", True)) and capture == "fixed"
        self.threshold = APP_MIN_SCORE
        ex, sw = pre.get("extraction", {}), pre.get("sign_window", {})
        self.params = {"trim_margin_s": sw.get("trim_margin_s", 0.1), "trim_min_frames": sw.get("trim_min_frames", 8),
                       "min_hand_detection_confidence": ex.get("detection_confidence", 0.5),
                       "min_hand_presence_confidence": ex.get("detection_confidence", 0.5),
                       "min_tracking_confidence": 0.5}

    def model_input(self, window: Window, hand: str = "auto") -> tuple[np.ndarray, dict]:
        """(32, 128) in the app's layout, and what was done to get it."""
        if self.capture == "app":
            return app_today_input(window, self.feature_version), {"hand_used": "none",
                                                                   "trim": (0, len(window.frames) - 1)}
        return recommended_input(window, hand, self.params, self.canonical, self.feature_version)

    def classify(self, window: Window, hand: str = "auto") -> dict:
        x, info = self.model_input(window, hand)
        prob = self.predict(x[None])[0]
        order = np.argsort(prob)[::-1]
        top = [(self.labels[i], float(prob[i])) for i in order[:3]]
        margin = float(prob[order[0]] - (prob[order[1]] if len(order) > 1 else 0.0))
        return {"top3": top, "prediction": top[0][0], "confidence": top[0][1], "margin": margin,
                "sure": top[0][1] >= APP_MIN_SCORE and margin >= APP_MIN_MARGIN, "gates": "app",
                "n_frames": len(window.frames),
                **info}


class Classifier:
    """The final model of a category, used exactly as its preprocess.json says (the browser app does the same):
    rich frames -> sign window (6.4) -> 32 frames -> canonical hand (Right / Left / Auto) -> feature version."""

    def __init__(self, model_dir: Path, threshold: float = 0.85) -> None:
        import json

        from .dataset import PlainError
        model_dir = Path(model_dir)
        pre = model_dir / "preprocess.json"
        if not pre.exists():
            raise PlainError("This model was trained before v2 and has no preprocess.json.",
                             "Train the model again on the Train model page, then start the live test.")
        import keras

        from .model import fast_predict
        from .train import load_labels
        self.pre = json.loads(pre.read_text(encoding="utf-8"))
        from .features import require_app_features
        require_app_features(self.pre, "This model", "Train the model again on the Train model page, then start the live test.")
        self.model = keras.models.load_model(model_dir / "model.keras")
        self.predict = fast_predict(self.model)                          # compiled + warmed up
        self.labels = load_labels(model_dir)
        self.threshold = threshold
        self.version = self.feature_version = int(self.pre.get("feature_version", 1))
        self.canonical = bool(self.pre.get("handedness", {}).get("canonical", False))
        self.model_run_id = self.pre.get("model_run_id", "")
        keys = ("trim_margin_s", "trim_min_frames", "min_hand_detection_confidence", "min_hand_presence_confidence",
                "min_tracking_confidence")
        self.params = {k: self.pre[k] for k in keys} if all(k in self.pre for k in keys) else None

    def classify(self, window: Window, hand: str = "auto") -> dict:
        """hand: right / left / auto (the tester's hand setting). Returns the top 3 and the hand that was used."""
        from .features import canonical, derive, resolve_hand
        sample, start, end, hdr, _ = window_and_sample(window.frames, window.counts, window.fps, params=self.params)
        used = resolve_hand(sample, hand) if self.canonical else "right"
        x = canonical(sample, used) if self.canonical else sample
        prob = self.predict(derive(x, self.version)[None])[0]
        order = np.argsort(prob)[::-1][:3]
        top = [(self.labels[i], float(prob[i])) for i in order]
        margin = float(prob[order[0]] - prob[order[1]]) if len(order) > 1 else top[0][1]
        return {"top3": top, "prediction": top[0][0], "confidence": top[0][1], "margin": margin,
                "sure": top[0][1] >= self.threshold, "trim": (start, end), "n_frames": len(window.frames),
                "hand_detect_rate": hdr, "hand_used": used, "sample": sample}


def save_attempt(paths: cfg.Paths, category: str, tester: str, tester_hand: str, target: str, prediction: str,
                 confidence: float, correct: bool, response_ms: float, model_run_id: str = "",
                 feature_version: int | str = "", recognizer: str = "", capture: str = "", margin: float | None = None,
                 time_from_hand_ms: float | None = None) -> None:
    """Append one Test-mode attempt to work/live_tests/<category>.csv. An older file without the newer columns gets
    them added (old rows keep them blank)."""
    paths.live_tests.mkdir(parents=True, exist_ok=True)
    p = paths.live_tests / f"{category}.csv"
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != LIVE_FIELDS:
            old = read_attempts(p)
            old[LIVE_FIELDS].to_csv(p, index=False)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LIVE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "tester": tester, "tester_hand": tester_hand,
                    "target": target, "prediction": prediction, "confidence": f"{confidence:.4f}",
                    "correct": int(bool(correct)), "response_time_ms": f"{response_ms:.1f}",
                    "model_run_id": model_run_id, "feature_version": feature_version, "recognizer": recognizer,
                    "capture": capture, "margin": "" if margin is None else f"{margin:.4f}",
                    "time_from_hand_ms": "" if time_from_hand_ms is None else f"{time_from_hand_ms:.1f}"})


def read_attempts(path: Path):
    """The attempts of a live-test file (any version) with every LIVE_FIELDS column; empty if there is none."""
    import pandas as pd
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LIVE_FIELDS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in LIVE_FIELDS:
        if c not in df.columns:
            df[c] = ""
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")
    df["time_from_hand_ms"] = pd.to_numeric(df["time_from_hand_ms"], errors="coerce")
    df["recognizer"] = df["recognizer"].replace("", "trainer")
    df["capture"] = df["capture"].replace("", "fixed")
    df["tester_hand"] = df["tester_hand"].str.strip().str.lower()
    return df[LIVE_FIELDS + [c for c in df.columns if c not in LIVE_FIELDS]]


def by_model(df) -> list[dict]:
    """Attempts per model (model run + feature version), in run order: model_run_id, feature_version (None when an
    old file did not record it), attempts, accuracy."""
    rows = []
    for (run, fv), g in df.groupby([df["model_run_id"].replace("", "not recorded"), df["feature_version"]], sort=True):
        rows.append({"model_run_id": run, "feature_version": int(fv) if str(fv).isdigit() else None,
                     "attempts": len(g), "accuracy": float(g["correct"].mean())})
    return rows


def is_correct(prediction: str, confidence: float, target: str, threshold: float = 0.85) -> bool:
    """Test mode: correct only when the top guess is the target AND it is at least `threshold` sure."""
    return prediction == target and confidence >= threshold


def now() -> float:
    return time.perf_counter()
