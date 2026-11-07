"""The live window's logic (no camera): modes, hand setting, targets, trials, history, summary; and the classifier
following preprocess.json (feature version, canonical hand with Auto)."""

import json

import numpy as np
import pytest

from trainer import features as F
from trainer.live_session import LiveSession

LABELS = ["red", "blue", "green"]


def result(pred, conf=0.9, ms=120.0):
    top = [(pred, conf)] + [(x, (1 - conf) / 2) for x in LABELS if x != pred][:2]
    return {"prediction": pred, "confidence": conf, "top3": top, "sure": conf >= 0.85, "hand_used": "right",
            "response_ms": ms}


def test_free_mode_shows_guesses_without_scoring():
    s = LiveSession(LABELS, mode="free")
    a = s.record(result("blue"))
    assert a.correct is None and s.trial == 0 and not s.finished
    assert s.last is a and [h.correct for h in s.history] == [None]


def test_test_mode_scores_trials_and_finishes():
    s = LiveSession(LABELS, mode="test", target="red", trials=3)
    assert s.status_caption() == "Raise your hands to sign"
    s.record(result("red"))
    s.record(result("blue"))
    assert s.trial == 2 and s.correct == 1 and s.running_accuracy == pytest.approx(0.5)
    s.record(result("red", conf=0.5))                   # right sign but not sure -> not correct
    assert s.finished and s.correct == 1
    summ = s.summary()
    assert summ["attempts"] == 3 and summ["accuracy"] == pytest.approx(1 / 3)
    assert summ["per_sign"]["red"]["attempts"] == 3 and summ["mean_response_ms"] == pytest.approx(120.0)
    assert s.record(result("red")) is None              # finished: no more attempts


def test_keys_next_sign_hand_mode_pause():
    s = LiveSession(LABELS, mode="test", target="red", trials=5, hand="auto")
    s.next_sign()
    assert s.target == "blue"
    s.pick("green")
    assert s.target == "green"
    s.next_sign()
    assert s.target == "red"                            # wraps around
    assert [s.cycle_hand() for _ in range(3)] == ["right", "left", "auto"]
    s.toggle_mode()
    assert s.mode == "free"
    s.toggle_pause()
    assert s.paused and s.status_caption() == "Paused: press Space to go on"
    assert s.record(result("red")) is None              # paused: ignored
    s.toggle_pause()
    assert not s.paused


def test_attempted_signs_are_marked_done():
    s = LiveSession(LABELS, mode="test", target="red", trials=10)
    s.record(result("blue"))
    s.next_sign()
    s.record(result("blue"))
    assert s.sign_states() == {"red": "done", "blue": "current", "green": "next"}


def test_history_keeps_the_last_ten():
    s = LiveSession(LABELS, mode="test", target="red", trials=50)
    for i in range(14):
        s.record(result("red" if i % 2 else "blue"))
    assert len(s.history) == 10


def test_result_colour_rule():
    s = LiveSession(LABELS, mode="test", target="red", trials=5)
    assert s.record(result("red")).tone == "ok"
    assert s.record(result("red", conf=0.5)).tone == "unsure"
    assert s.record(result("blue")).tone == "wrong"
    f = LiveSession(LABELS, mode="free")
    assert f.record(result("blue")).tone == "ok" and f.record(result("blue", conf=0.4)).tone == "unsure"


def test_threshold_is_85_percent_and_gates_correct():
    from trainer import settings as cfg
    assert cfg.DEFAULTS["confidence_threshold"] == 0.85
    s = LiveSession(LABELS, mode="test", target="red", trials=5)
    assert s.threshold == 0.85
    a = s.record(result("red", conf=0.84))              # right sign, 84 %: not sure -> not correct
    assert a.correct is False and a.tone == "unsure" and s.status_caption() == "Not sure, try again"
    a = s.record(result("red", conf=0.85))              # right sign, at the threshold -> correct
    assert a.correct is True and a.tone == "ok"
    a = s.record(result("blue", conf=0.99))             # sure but the wrong sign -> not correct
    assert a.correct is False and a.tone == "wrong"
    assert s.correct == 1 and s.trial == 3


def test_threshold_follows_settings():
    s = LiveSession(LABELS, mode="test", target="red", threshold=0.6)
    assert s.record(result("red", conf=0.65)).correct is True


def test_caption_asks_to_lower_the_hands_after_a_3_second_window():
    s = LiveSession(LABELS, mode="free")
    s.waiting_for_rest = True
    assert s.status_caption() == "Lower your hands, then sign again"


# --- the classifier follows preprocess.json ------------------------------------------------------
@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from trainer.model import build_model
    d = tmp_path_factory.mktemp("model")
    m = build_model(3)
    m.save(d / "model.keras")
    (d / "labels.json").write_text(json.dumps(LABELS))
    from trainer import recorder
    pre = {"feature_version": 1, "features_per_frame": 128, **recorder.default_params(None),
           "handedness": {"canonical": True, "default_hand": "auto"}, "labels": LABELS, "model_run_id": "final-x",
           "normalization": "hand-relative-v1"}
    (d / "preprocess.json").write_text(json.dumps(pre))
    return d


def window(mirrored=False):
    from test_features import rich_window
    from trainer.live import Window
    r = rich_window(T=40, left=False, right=True, move_right=0.01)
    if mirrored:
        r = F.mirror(r)
    return Window(r, (r[:, 63] + r[:, 127]).astype(int), 30.0, "hands_gone")


def test_classifier_mirrors_a_left_handed_window_back(model_dir):
    from trainer.live import Classifier
    c = Classifier(model_dir)
    assert c.version == 1 and c.canonical and c.model_run_id == "final-x"
    right = c.classify(window(), "right")
    left_auto = c.classify(window(mirrored=True), "auto")
    left_set = c.classify(window(mirrored=True), "left")
    assert left_auto["hand_used"] == "left" and right["hand_used"] == "right"
    for r in (left_auto, left_set):
        np.testing.assert_allclose([p for _, p in r["top3"]], [p for _, p in right["top3"]], atol=1e-5)


def test_classifier_refuses_a_model_without_preprocess_json(tmp_path):
    from trainer.dataset import PlainError
    from trainer.live import Classifier
    (tmp_path / "labels.json").write_text("[]")
    with pytest.raises(PlainError):
        Classifier(tmp_path)


# --- the window's drawing (no camera) ------------------------------------------------------------
def test_screens_draw_and_register_buttons(tmp_path):
    import cv2
    from trainer.live_window import draw_screen
    frame = np.full((720, 1280, 3), 90, np.uint8)
    s = LiveSession(LABELS, mode="test", target="red", trials=2)
    for name, prep in (("idle", lambda: None), ("result", lambda: s.record(result("blue"))),
                       ("summary", lambda: s.record(result("red")))):
        prep()
        canvas, hit = draw_screen(frame, [], s, {"category": "colors", "hands_seen": True})
        assert canvas.shape == (720, 1280, 3) and canvas.dtype == np.uint8
        cv2.imwrite(str(tmp_path / f"{name}.png"), canvas)
    assert s.finished
    keys = {hit.key_at(x, y) for x in range(0, 1280, 8) for y in range(0, 720, 8)} - {None}
    assert {"quit", "pause"} <= keys                                   # summary: New run + Close
    s2 = LiveSession(LABELS, mode="free")
    _, hit2 = draw_screen(None, [], s2, {"category": "colors"})
    keys2 = {hit2.key_at(x, y) for x in range(0, 1280, 6) for y in range(0, 720, 6)} - {None}
    assert {"quit", "mode", "hand", "next", "pause", "sign:red", "sign:blue", "sign:green"} <= keys2
