"""Live sign window (README 6.6) and classification. Shared by the webcam and the --video path."""

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
LIVE_FIELDS = ["time", "tester", "target", "prediction", "confidence", "correct", "response_time_ms"]


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


@dataclass
class Window:
    frames: np.ndarray        # (T,128)
    counts: np.ndarray        # (T,) hands per frame
    fps: float
    reason: str               # "hands_gone" | "max_length" | "flush"


class LiveSegmenter:
    """Feed one frame at a time; returns a Window when one has ended, else None."""

    def __init__(self, fps: float) -> None:
        self.set_fps(fps)
        self.ring: deque = deque(maxlen=self.n_pre)
        self.active: list | None = None
        self.absent_frames = 0

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.n_pre = max(1, _round_half_up(PRE_BUFFER_S * self.fps))
        self.n_end = max(1, _round_half_up(END_ABSENT_S * self.fps))
        self.n_max = max(1, _round_half_up(MAX_WINDOW_S * self.fps))

    @property
    def signing(self) -> bool:
        return self.active is not None

    def push(self, vector: np.ndarray, n_hands: int) -> Window | None:
        item = (np.asarray(vector, np.float32), int(n_hands))
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
        return self._finish("flush") if self.active else None

    def _finish(self, reason: str) -> Window:
        frames = np.stack([f for f, _ in self.active])
        counts = np.array([c for _, c in self.active])
        self.active = None
        self.absent_frames = 0
        return Window(frames, counts, self.fps, reason)


class Classifier:
    def __init__(self, model_dir: Path, threshold: float = 0.70) -> None:
        import keras
        from .model import fast_predict
        from .train import load_labels
        self.model = keras.models.load_model(Path(model_dir) / "model.keras")
        self.predict = fast_predict(self.model)                          # compiled + warmed up
        self.labels = load_labels(model_dir)
        self.threshold = threshold
        import json
        conf = Path(model_dir) / "config.json"
        self.params = json.loads(conf.read_text(encoding="utf-8")).get("preprocess") if conf.exists() else None

    def classify(self, window: Window) -> dict:
        sample, start, end, hdr, _ = window_and_sample(window.frames, window.counts, window.fps, params=self.params)
        prob = self.predict(sample[None])[0]
        order = np.argsort(prob)[::-1][:3]
        top = [(self.labels[i], float(prob[i])) for i in order]
        return {"top3": top, "prediction": top[0][0], "confidence": top[0][1],
                "sure": top[0][1] >= self.threshold, "trim": (start, end), "n_frames": len(window.frames),
                "hand_detect_rate": hdr, "sample": sample}


def save_attempt(paths: cfg.Paths, category: str, tester: str, target: str, prediction: str,
                 confidence: float, correct: bool, response_ms: float) -> None:
    paths.live_tests.mkdir(parents=True, exist_ok=True)
    p = paths.live_tests / f"{category}.csv"
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LIVE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "tester": tester, "target": target,
                    "prediction": prediction, "confidence": f"{confidence:.4f}", "correct": int(bool(correct)),
                    "response_time_ms": f"{response_ms:.1f}"})


def is_correct(prediction: str, confidence: float, target: str, threshold: float = 0.70) -> bool:
    return prediction == target and confidence >= threshold


def now() -> float:
    return time.perf_counter()
