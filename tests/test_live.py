"""Live sign window (README 6.6): start with a 0.1 s pre-buffer, end after 0.2 s without a hand or 3 s."""

import csv
import json

import numpy as np

from trainer import settings as cfg
from trainer.live import LIVE_FIELDS, LiveSegmenter, is_correct, read_attempts, save_attempt


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


def test_max_three_seconds_then_no_window_while_hands_stay_up():
    seg = LiveSegmenter(30)
    wins = feed(seg, [True] * 300)          # hands held up for 10 s: one sign, not one every 3 s
    assert [len(w.frames) for _, w in wins] == [90]
    assert wins[0][1].reason == "max_length"
    assert not seg.signing and seg.waiting_for_rest and seg.flush() is None


def test_after_a_3_second_window_the_hands_must_go_down_first():
    seg = LiveSegmenter(30)
    pattern = [True] * 100 + [False] * 3 + [True] * 20 + [False] * 10 + [True] * 20 + [False] * 10
    wins = feed(seg, pattern)
    # 3 frames down is shorter than the 0.2 s rest: still the same idle hands; 10 frames down re-arms
    assert [w.reason for _, w in wins] == ["max_length", "hands_gone"]
    assert wins[1][0] == 100 + 3 + 20 + 10 + 20 + 6 - 1


def test_no_window_while_hands_are_away_or_for_a_detection_blip():
    seg = LiveSegmenter(30)                 # a sign needs at least 0.2 s (6 frames) of hands
    assert feed(seg, [False] * 200) == []
    assert feed(seg, [False] * 10 + [True] * 2 + [False] * 20) == []
    assert not seg.signing
    wins = feed(seg, [True] * 6 + [False] * 6)
    assert len(wins) == 1


def test_fps_dependent_lengths():
    seg = LiveSegmenter(15)
    assert (seg.n_pre, seg.n_end, seg.n_max, seg.n_min_hands) == (2, 3, 45, 3)


def test_correct_rule_and_csv(tmp_path, monkeypatch):
    assert is_correct("red", 0.85, "red")                     # default threshold 85 %
    assert not is_correct("red", 0.84, "red")
    assert not is_correct("blue", 0.99, "red")
    assert is_correct("red", 0.71, "red", threshold=0.70)     # the threshold from Settings
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"work_dir": str(tmp_path / "work")}))
    monkeypatch.setenv("HUDYAT_TRAINER_SETTINGS", str(s))
    paths = cfg.Paths()
    save_attempt(paths, "colors", "Ana", "right", "red", "red", 0.9, True, 123.4, "final-colors-1", 2)
    save_attempt(paths, "colors", "Ben", "left", "red", "blue", 0.8, False, 99.0, "final-colors-1", 2)
    rows = list(csv.DictReader(open(paths.live_tests / "colors.csv", encoding="utf-8")))
    assert [r["correct"] for r in rows] == ["1", "0"]
    assert list(rows[0]) == LIVE_FIELDS == ["time", "tester", "tester_hand", "target", "prediction", "confidence",
                                            "correct", "response_time_ms", "model_run_id", "feature_version",
                                            "recognizer", "capture", "margin", "time_from_hand_ms"]
    assert rows[1]["tester_hand"] == "left" and rows[1]["model_run_id"] == "final-colors-1"
    df = read_attempts(paths.live_tests / "colors.csv")
    assert list(df["correct"]) == [1, 0] and df["response_time_ms"].iloc[0] == 123.4


def test_old_live_files_are_still_read(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text("time,tester,target,prediction,confidence,correct,response_time_ms\n"
                 "2026-01-01T10:00:00,Ana,red,red,0.9,1,500\n")
    df = read_attempts(p)
    assert list(df.columns)[:len(LIVE_FIELDS)] == LIVE_FIELDS
    assert df["tester_hand"].iloc[0] == "" and df["correct"].iloc[0] == 1
    assert read_attempts(tmp_path / "missing.csv").empty
