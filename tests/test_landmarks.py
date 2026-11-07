"""Landmark format, pinned to README section 6 of the data-collection folder."""

import math

import numpy as np
import pytest

from trainer import landmarks as L
from trainer.recorder import FALLBACK


# --- ports of the README's browser reference code (6.3, 6.4), used as the oracle ---------------
def normalize_hand_js(lm, W, H):
    a = W / H
    p = [[q[0] * a, q[1], q[2] * a] for q in lm]
    wx, wy, wz = p[0]
    for q in p:
        q[0] -= wx
        q[1] -= wy
        q[2] -= wz
    s = math.hypot(*p[9])
    out = [v for q in p for v in q]
    return [v / s for v in out] if s > 1e-6 else out


def js_round(x):            # Math.round = half up
    return math.floor(x + 0.5)


def sign_window_js(present, fps):
    T = len(present)
    first = present.index(True) if True in present else -1
    if first < 0:
        return 0, T - 1
    last = T - 1 - present[::-1].index(True)
    m = js_round(0.1 * fps)
    start, end = max(0, first - m), min(T - 1, last + m)
    need = min(8, T)
    if end - start + 1 < need:
        centre = (start + end) / 2
        start = max(0, min(js_round(centre - (need - 1) / 2), T - need))
        end = start + need - 1
    return start, end


def hand(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (21, 3)).astype(np.float32)


# --- 6.3 normalization ----------------------------------------------------------------------
@pytest.mark.parametrize("W,H", [(1280, 720), (640, 480), (1920, 1080)])
def test_normalize_matches_readme_reference(W, H):
    lm = hand(1)
    ours = L.normalize_hand(lm, W, H)
    ref = np.array(normalize_hand_js(lm.astype(np.float64).tolist(), W, H), dtype=np.float32)
    assert ours.shape == (63,) and ours.dtype == np.float32
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)


def test_normalize_worked_example():
    # wrist at (0.5, 0.5, 0); MCP9 0.1 to the right (x is aspect-corrected by 1280/720)
    lm = np.zeros((21, 3), np.float32)
    lm[:] = [0.5, 0.5, 0.0]
    lm[9] = [0.6, 0.5, 0.0]
    lm[4] = [0.5, 0.4, 0.01]
    out = L.normalize_hand(lm, 1280, 720).reshape(21, 3)
    a = 1280 / 720
    s = 0.1 * a
    np.testing.assert_allclose(out[0], [0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(out[9], [1, 0, 0], atol=1e-6)               # wrist-MCP9 distance = 1
    np.testing.assert_allclose(out[4], [0, -0.1 / s, 0.01 * a / s], rtol=1e-5)


def test_normalize_zero_scale_guard():
    lm = np.full((21, 3), 0.3, np.float32)       # all points on the wrist: distance 0 -> no division
    out = L.normalize_hand(lm, 1280, 720)
    assert np.all(np.isfinite(out)) and np.allclose(out, 0)


# --- 6.2 per-frame vector -------------------------------------------------------------------
def test_frame_vector_slots_by_label_not_order():
    left, right = hand(1), hand(2)
    v1, _ = L.frame_vector([L.Hand("Right", 0.9, right), L.Hand("Left", 0.9, left)], 1280, 720)
    v2, _ = L.frame_vector([L.Hand("Left", 0.9, left), L.Hand("Right", 0.9, right)], 1280, 720)
    assert v1.shape == (128,) and v1.dtype == np.float32
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_allclose(v1[0:63], L.normalize_hand(left, 1280, 720))
    np.testing.assert_allclose(v1[64:127], L.normalize_hand(right, 1280, 720))
    assert v1[63] == 1.0 and v1[127] == 1.0


def test_frame_vector_missing_hand_is_zeros_with_flag_0():
    v, dup = L.frame_vector([L.Hand("Right", 0.9, hand(2))], 1280, 720)
    assert not dup
    assert np.all(v[0:64] == 0)                  # Left: 63 zeros + flag 0
    assert v[127] == 1.0
    v0, _ = L.frame_vector([], 1280, 720)
    assert np.all(v0 == 0)


def test_frame_vector_duplicate_label_higher_score_wins():
    a, b = hand(3), hand(4)
    v, dup = L.frame_vector([L.Hand("Left", 0.6, a), L.Hand("Left", 0.8, b)], 1280, 720)
    assert dup
    np.testing.assert_allclose(v[0:63], L.normalize_hand(b, 1280, 720))
    assert v[127] == 0 and np.all(v[64:127] == 0)


def test_index_formula():
    lm = hand(5)
    v, _ = L.frame_vector([L.Hand("Right", 0.9, lm)], 1280, 720)
    n = L.normalize_hand(lm, 1280, 720)
    for k in (0, 9, 20):
        for a in range(3):
            assert v[64 + 3 * k + a] == n[3 * k + a]


# --- 6.4 sign window ------------------------------------------------------------------------
@pytest.mark.parametrize("fps,margin", [(30, 3), (60, 6), (25, 3), (15, 2)])
def test_sign_window_margin(fps, margin):
    present = [False] * 20 + [True] * 30 + [False] * 40
    start, end, trimmed = FALLBACK.sign_window(present, fps)
    assert (start, end) == (20 - margin, 49 + margin) and trimmed


def test_sign_window_gaps_kept_and_whole_clip_when_no_hand():
    present = [False] * 10 + [True, False, False, True] + [False] * 10
    s, e, _ = FALLBACK.sign_window(present, 30)
    assert s == 7 and e == 16
    assert FALLBACK.sign_window([False] * 50, 30)[:2] == (0, 49)


def test_sign_window_min_8_frames_around_centre():
    present = [False] * 40
    present[20] = True                               # m=0 at fps 4 -> 1 frame -> widen to 8
    s, e, _ = FALLBACK.sign_window(present, 4)
    assert e - s + 1 == 8 and s <= 20 <= e
    present = [True] + [False] * 30                  # at the edge: shifted to stay inside
    assert FALLBACK.sign_window(present, 4)[:2] == (0, 7)
    assert FALLBACK.sign_window([True] * 5, 30)[:2] == (0, 4)   # shorter than 8: kept whole


def test_sign_window_random_vs_readme_js():
    rng = np.random.default_rng(0)
    for _ in range(500):
        T = int(rng.integers(1, 130))
        fps = float(rng.choice([15, 19.7, 25, 29.97, 30, 60]))
        present = (rng.random(T) < rng.random() * 0.3).tolist()
        assert FALLBACK.sign_window(present, fps)[:2] == sign_window_js(present, fps)


# --- 6.5 resampling -------------------------------------------------------------------------
def frames_with(values_left, flags_left, values_right=None, flags_right=None):
    T = len(values_left)
    f = np.zeros((T, 128), np.float32)
    for i in range(T):
        f[i, 0:63] = values_left[i]
        f[i, 63] = flags_left[i]
        if values_right is not None:
            f[i, 64:127] = values_right[i]
            f[i, 127] = flags_right[i]
    return f


def test_resample_shape_and_identity_for_32_frames():
    f = frames_with(np.arange(32, dtype=np.float32), [1] * 32)
    out = L.resample(f)
    assert out.shape == (32, 128) and out.dtype == np.float32
    np.testing.assert_allclose(out, f)


def test_resample_linear_interpolation_when_both_present():
    f = frames_with([0.0, 31.0], [1, 1])            # T=2: t = j/31 -> value j
    out = L.resample(f)
    np.testing.assert_allclose(out[:, 0], np.arange(32), rtol=1e-6)
    assert np.all(out[:, 63] == 1)


def test_resample_never_blends_with_absent_hand():
    # frame 0 present (value 5), frame 1 absent (zeros): nearest frame copied, no blending
    f = frames_with([5.0, 0.0], [1, 0])
    out = L.resample(f)
    j = np.arange(32)
    w = j / 31
    nearest_is_0 = w < 0.5
    np.testing.assert_array_equal(out[nearest_is_0, 0], 5.0)
    np.testing.assert_array_equal(out[~nearest_is_0, 0], 0.0)
    np.testing.assert_array_equal(out[:, 63], nearest_is_0.astype(np.float32))
    assert set(np.unique(out[:, 0])) <= {0.0, 5.0}


def test_resample_slots_independent():
    f = frames_with([1.0, 3.0], [1, 1], [7.0, 0.0], [1, 0])
    out = L.resample(f)
    np.testing.assert_allclose(out[:, 0], 1 + 2 * np.arange(32) / 31, rtol=1e-6)   # Left blended
    assert set(np.unique(out[:, 64])) <= {0.0, 7.0}                                  # Right copied


def test_window_and_sample_uses_sign_window():
    T = 90
    counts = np.zeros(T, int)
    counts[30:60] = 1
    frames = np.zeros((T, 128), np.float32)
    frames[30:60, 63] = 1
    frames[30:60, 0] = np.arange(30)
    sample, s, e, hdr, bhr = L.window_and_sample(frames, counts, 30.0, data_dir=None)
    assert (s, e) == (27, 62)
    assert sample.shape == (32, 128)
    assert hdr == pytest.approx(30 / 36) and bhr == 0.0


def test_extract_clip_synthetic_video_without_hands(tmp_path):
    """A generated video with no hands: whole clip kept, all zeros, detection rate 0."""
    import cv2
    p = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(20):
        frame = np.full((240, 320, 3), 40 + i * 5, np.uint8)
        w.write(frame)
    w.release()
    r = L.extract_clip(p, 30.0, data_dir=None)
    assert r.n_frames == 20 and (r.trim_start, r.trim_end) == (0, 19)
    assert r.sample.shape == (32, 149) and r.sample.dtype == np.float32      # rich sample (docs/FEATURES.md)
    assert r.frames.shape == (20, 149)
    assert np.all(r.sample == 0) and r.hand_detect_rate == 0.0
    assert (r.width, r.height) == (320, 240)
