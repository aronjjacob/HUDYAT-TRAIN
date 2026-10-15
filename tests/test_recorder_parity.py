"""The trim rule and constants must be identical to the recorder's hudyat_common.py.

The trainer imports the recorder's own file; the copies in trainer/recorder.py are
only a fallback and must give identical results.
"""

import numpy as np
import pytest

from trainer import recorder
from trainer.settings import DEFAULT_DATA_DIR

REAL = DEFAULT_DATA_DIR
needs_recorder = pytest.mark.skipif(not (REAL / "hudyat_common.py").exists(),
                                    reason="recorder folder not on this computer")


@needs_recorder
def test_imports_the_recorders_own_file():
    hc = recorder.load_common(REAL)
    assert hc is not recorder.FALLBACK
    assert hc.imported_from.lower().endswith("hudyat_common.py")


@needs_recorder
def test_import_writes_nothing_into_the_recorder_folder(tmp_path):
    import shutil
    import sys
    fake = tmp_path / "rec"
    fake.mkdir()
    shutil.copy2(REAL / "hudyat_common.py", fake / "hudyat_common.py")
    sys.modules.pop("hudyat_common", None)
    recorder._cache.clear()
    hc = recorder.load_common(fake)
    assert hc is not recorder.FALLBACK
    assert sorted(p.name for p in fake.iterdir()) == ["hudyat_common.py"]     # no __pycache__
    recorder._cache.clear()


@needs_recorder
def test_constants_match():
    hc = recorder.load_common(REAL)
    fb = recorder.FALLBACK
    for name in ("TRIM_MARGIN_S", "TRIM_MIN_FRAMES", "NUM_HANDS", "MIN_HAND_DETECTION_CONFIDENCE",
                 "MIN_HAND_PRESENCE_CONFIDENCE", "MIN_TRACKING_CONFIDENCE", "SWAP_HANDEDNESS_LABELS",
                 "HAND_DETECT_WARN_RATE"):
        assert getattr(hc, name) == getattr(fb, name), name
    assert [tuple(c) for c in hc.HAND_CONNECTIONS] == [tuple(c) for c in fb.HAND_CONNECTIONS]


@needs_recorder
def test_sign_window_copy_identical_to_recorder():
    hc = recorder.load_common(REAL)
    rng = np.random.default_rng(1)
    cases = [[], [True], [False] * 5, [True] * 3, [False] * 89 + [True]]
    for _ in range(2000):
        T = int(rng.integers(1, 260))
        p = rng.random(T) < rng.random()
        if rng.random() < 0.5:                       # contiguous block like a real sign
            p[:] = False
            a = int(rng.integers(0, T))
            p[a:a + int(rng.integers(1, 40))] = True
        cases.append(p.tolist())
    for present in cases:
        for fps in (15.0, 19.7, 25.0, 29.97, 30.0, 59.94, 60.0):
            if not present:
                with pytest.raises(Exception):
                    hc.sign_window(present, fps)
                with pytest.raises(Exception):
                    recorder.FALLBACK.sign_window(present, fps)
                continue
            assert tuple(hc.sign_window(present, fps)) == tuple(recorder.FALLBACK.sign_window(present, fps))


@needs_recorder
def test_handedness_rule_identical():
    hc = recorder.load_common(REAL)
    for lab in ("Left", "Right", "Other"):
        assert hc.actual_hand_label(lab) == recorder.FALLBACK.actual_hand_label(lab)


def test_fallback_used_when_folder_missing(tmp_path):
    assert recorder.load_common(tmp_path / "missing") is recorder.FALLBACK


def test_adjusted_trim_values_use_the_copy_with_new_values():
    present = [False] * 20 + [True] * 10 + [False] * 20
    p = dict(recorder.default_params(None), trim_margin_s=0.2, trim_min_frames=16)
    assert recorder.sign_window(present, 30, None, p)[:2] == (14, 35)          # margin 6 frames at 30 fps
    assert recorder.sign_window(present, 30, None, recorder.default_params(None))[:2] == (17, 32)
    short = [False] * 20 + [True] + [False] * 20
    s, e, _ = recorder.sign_window(short, 4, None, p)
    assert e - s + 1 == 16


def test_params_id_round_trip_and_default():
    d = recorder.default_params(None)
    assert recorder.params_id(d) == "m0.10-f8-d0.50-p0.50-t0.50"
    assert recorder.parse_params_id(recorder.params_id(d)) == d
    assert recorder.parse_params_id("") == d                                 # samples made before the id existed
    q = dict(d, min_hand_detection_confidence=0.65, trim_min_frames=12)
    assert recorder.parse_params_id(recorder.params_id(q)) == q
    assert recorder.is_default(d) and not recorder.is_default(q)


def test_preprocess_params_from_settings():
    s = {"preprocess": {"trim_margin_s": 0.25, "unknown": 1}}
    p = recorder.preprocess_params(s, None)
    assert p["trim_margin_s"] == 0.25 and p["trim_min_frames"] == 8 and "unknown" not in p
