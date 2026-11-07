"""The live test's state, without a camera: Free / Test mode, the hand setting, the target sign, the trials, the last
10 attempts and the end-of-run summary. The window (live_window.py) draws this and sends it keys and clicks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

HAND_CYCLE = ("right", "left", "auto")


@dataclass
class Attempt:
    target: str | None
    prediction: str
    confidence: float
    top3: list
    sure: bool
    correct: bool | None           # None in Free mode
    response_ms: float
    hand_used: str = "right"

    @property
    def tone(self) -> str:
        """ok (green) / unsure (amber "Not sure, try again") / wrong (red, Test mode only)."""
        if not self.sure:
            return "unsure"
        return "wrong" if self.correct is False else "ok"


@dataclass
class LiveSession:
    labels: list[str]
    mode: str = "free"                     # free / test
    target: str | None = None
    trials: int = 25
    hand: str = "auto"                     # right / left / auto
    threshold: float = 0.85
    paused: bool = False
    finished: bool = False
    attempts: list[Attempt] = field(default_factory=list)          # Test mode attempts of this run
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    last: Attempt | None = None
    signing: bool = False
    waiting_for_rest: bool = False         # after a 3-second window: lower the hands before the next sign
    window_fraction: float = 0.0           # share of the 3-second limit used by the current window
    attempted: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.target not in self.labels:
            self.target = self.labels[0] if self.labels else None

    # --- counts -----------------------------------------------------------------------------------
    @property
    def trial(self) -> int:
        return len(self.attempts)

    @property
    def correct(self) -> int:
        return sum(bool(a.correct) for a in self.attempts)

    @property
    def running_accuracy(self) -> float | None:
        return self.correct / self.trial if self.trial else None

    # --- keys / clicks ------------------------------------------------------------------------------
    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def next_sign(self) -> str | None:
        if self.labels:
            i = self.labels.index(self.target) if self.target in self.labels else -1
            self.target = self.labels[(i + 1) % len(self.labels)]
        return self.target

    def pick(self, sign: str) -> None:
        if sign in self.labels:
            self.target = sign

    def cycle_hand(self) -> str:
        self.hand = HAND_CYCLE[(HAND_CYCLE.index(self.hand) + 1) % len(HAND_CYCLE)] if self.hand in HAND_CYCLE \
            else "auto"
        return self.hand

    def toggle_mode(self) -> str:
        self.mode = "free" if self.mode == "test" else "test"
        self.finished = False
        return self.mode

    def restart(self) -> None:
        """A new Test run (after the summary)."""
        self.attempts.clear()
        self.attempted.clear()
        self.history.clear()
        self.last = None
        self.finished = False

    # --- results ----------------------------------------------------------------------------------
    def record(self, r: dict) -> Attempt | None:
        """One classified sign window. None when paused or the Test run is over."""
        if self.paused or self.finished:
            return None
        test = self.mode == "test"
        # the app model brings the app's own rule (score and margin); the trainer model uses the threshold
        accepted = bool(r["sure"]) if r.get("gates") == "app" else r["confidence"] >= self.threshold
        correct = (r["prediction"] == self.target and accepted) if test else None
        a = Attempt(self.target if test else None, r["prediction"], float(r["confidence"]), list(r["top3"]),
                    bool(r["sure"]), correct, float(r.get("response_ms") or 0.0), r.get("hand_used", "right"))
        self.last = a
        self.history.append(a)
        if test:
            self.attempts.append(a)
            self.attempted.add(self.target)
            if self.trial >= self.trials:
                self.finished = True
        return a

    def sign_states(self) -> dict[str, str]:
        """For the sign list: current / done (tried in this run) / next."""
        return {s: "current" if s == self.target else "done" if s in self.attempted else "next" for s in self.labels}

    def status_caption(self) -> str:
        if self.finished:
            return "Test finished: press Space for a new run, Q to close"
        if self.paused:
            return "Paused: press Space to go on"
        if self.signing:
            return "Signing…"
        if self.waiting_for_rest:
            return "Lower your hands, then sign again"
        if self.last is not None:
            a = self.last
            if a.tone == "unsure":
                return "Not sure, try again"
            name = a.prediction.replace("_", " ").title()
            if a.correct is None:
                return f"{name}  ·  raise your hands for the next sign"
            return ("Correct" if a.correct else f"Wrong: that looked like {name}") + \
                "  ·  raise your hands for the next sign"
        return "Raise your hands to sign"

    def summary(self) -> dict:
        per: dict[str, dict] = {}
        for a in self.attempts:
            p = per.setdefault(a.target, {"attempts": 0, "correct": 0, "ms": []})
            p["attempts"] += 1
            p["correct"] += int(bool(a.correct))
            p["ms"].append(a.response_ms)
        for p in per.values():
            p["accuracy"] = p["correct"] / p["attempts"]
            p["mean_response_ms"] = sum(p["ms"]) / len(p["ms"])
            del p["ms"]
        ms = [a.response_ms for a in self.attempts]
        return {"attempts": self.trial, "correct": self.correct, "accuracy": self.running_accuracy,
                "mean_response_ms": sum(ms) / len(ms) if ms else None, "per_sign": per}
