"""Live sign window (README 6.6): start with a 0.1 s pre-buffer, end after 0.2 s without a hand or 3 s."""

import csv
import json

import numpy as np

from trainer import settings as cfg
from trainer.live import LiveSegmenter, is_correct, save_attempt


def vec(i, present):
    v = np.zeros(128, np.float32)
    if present:
        v[0] = i
        v[63] = 1
    return v


def feed(seg, pattern):
    out = []
    for i, p in enumerate(pattern):
        w = seg.push(vec(i, p), int(p))
        if w is not None:
            out.append((i, w))
    return out


def test_window_with_prebuffer_and_end_after_absence():
    seg = LiveSegmenter(30)                 # pre 3 frames, end after 6 absent, max 90
    pattern = [False] * 10 + [True] * 20 + [False] * 10
    wins = feed(seg, pattern)
    assert len(wins) == 1
    i, w = wins[0]
    assert i == 10 + 20 + 6 - 1             # ends on the 6th absent frame
    assert len(w.frames) == 3 + 20 + 6
    assert w.reason == "hands_gone"
    assert list(w.counts[:3]) == [0, 0, 0] and w.counts[3] == 1


def test_short_dropout_does_not_split_a_sign():
    seg = LiveSegmenter(30)
    pattern = [True] * 10 + [False] * 4 + [True] * 10 + [False] * 8
    wins = feed(seg, pattern)
    assert len(wins) == 1 and len(wins[0][1].frames) == 10 + 4 + 10 + 6


def test_max_three_seconds_then_new_window():
    seg = LiveSegmenter(30)
    wins = feed(seg, [True] * 200)
    assert [len(w.frames) for _, w in wins] == [90, 90]
    assert all(w.reason == "max_length" for _, w in wins)
    assert seg.signing                      # third window still open
    assert len(seg.flush().frames) == 20


def test_fps_dependent_lengths():
    seg = LiveSegmenter(15)
    assert (seg.n_pre, seg.n_end, seg.n_max) == (2, 3, 45)


def test_correct_rule_and_csv(tmp_path, monkeypatch):
    assert is_correct("red", 0.71, "red")
    assert not is_correct("red", 0.69, "red")
    assert not is_correct("blue", 0.99, "red")
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"work_dir": str(tmp_path / "work")}))
    monkeypatch.setenv("HUDYAT_TRAINER_SETTINGS", str(s))
    paths = cfg.Paths()
    save_attempt(paths, "colors", "Ana", "red", "red", 0.9, True, 123.4)
    save_attempt(paths, "colors", "Ana", "red", "blue", 0.8, False, 99.0)
    rows = list(csv.DictReader(open(paths.live_tests / "colors.csv", encoding="utf-8")))
    assert [r["correct"] for r in rows] == ["1", "0"]
    assert set(rows[0]) == {"time", "tester", "target", "prediction", "confidence", "correct", "response_time_ms"}
