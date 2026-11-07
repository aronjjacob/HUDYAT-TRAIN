"""Features (docs/FEATURES.md): the rich 149-value frame, the app's model input, mirror, canonical hand, Auto."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from trainer import features as F
from trainer import landmarks as L

DOC = Path(__file__).resolve().parent.parent / "docs" / "FEATURES.md"


def hand(label, seed, score=0.9, shift=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    lm = rng.uniform(0.3, 0.6, (21, 3)).astype(np.float32)
    lm[:, 0] += shift[0]
    lm[:, 1] += shift[1]
    lm[:, 2] = rng.uniform(-0.05, 0.05, 21)
    lm[0, 2] = 0.0
    return L.Hand(label, score, lm)


def face_points(nose=(0.5, 0.4), forehead_y=0.25, chin_y=0.55, seed=0):
    """A synthetic 478-point face: only the nose (1), forehead (10) and chin (152) matter for README 6.7."""
    f = np.random.default_rng(seed).uniform(0.4, 0.6, (478, 3)).astype(np.float32)
    f[F.FACE_NOSE] = [nose[0], nose[1], 0.0]
    f[F.FACE_FOREHEAD] = [nose[0], forehead_y, 0.0]
    f[F.FACE_CHIN] = [nose[0], chin_y, 0.0]
    return f


def rich_window(T=32, left=True, right=True, seed=0, move_left=0.0, move_right=0.0, face=False):
    """T rich frames; each present hand drifts along x by move_* per frame (in normalized image units).
    face=True adds a still face, so the location values are filled in."""
    frames = []
    for t in range(T):
        hs = []
        if left:
            hs.append(hand("Left", seed, shift=(move_left * t, 0.0)))
        if right:
            hs.append(hand("Right", seed + 1, shift=(move_right * t, 0.01 * t)))
        frames.append(F.frame_vector_rich(hs, 1280, 720, face_points(seed=seed) if face else None)[0])
    return np.stack(frames)


# --- rich frame -----------------------------------------------------------------------------
def test_rich_frame_extends_v1_exactly():
    hs = [hand("Left", 1), hand("Right", 2), hand("Right", 3, score=0.5)]      # duplicate Right: lower score dropped
    v1, dup1 = L.frame_vector(hs, 1280, 720)
    rich, dup2 = F.frame_vector_rich(hs, 1280, 720)
    assert rich.shape == (F.RICH_DIM,) == (149,) and rich.dtype == np.float32
    np.testing.assert_array_equal(rich[:128], v1)
    assert dup1 and dup2


def test_rich_frame_wrist_and_scale():
    h = hand("Right", 4)
    rich, _ = F.frame_vector_rich([h], 1280, 720)
    a = 1280 / 720
    p = h.landmarks.astype(np.float64) * [a, 1, a]
    np.testing.assert_allclose(rich[F.R_WRIST], p[0], rtol=1e-6)
    np.testing.assert_allclose(rich[F.R_SCALE], np.linalg.norm(p[9] - p[0]), rtol=1e-6)
    assert not rich[F.L_WRIST].any() and rich[F.L_SCALE] == 0          # absent Left hand stays zero


# --- model input: the app's hand-relative-v1 ------------------------------------------------
def app_normalize_frame(hands, W, H):
    """Port of the HUDYAT app's normalizeFrame (client/src/features/fsl-recognition/normalize.ts), in the
    trainer's slots: per hand, pixels minus the wrist divided by the 2D wrist -> landmark 9 length."""
    out = np.zeros(128)
    for h in hands:
        off, flag = (0, 63) if h.label == "Left" else (64, 127)
        p = h.landmarks.astype(np.float64) * [W, H, W]
        palm = np.hypot(*(p[9, :2] - p[0, :2]))
        if palm < 1e-6:
            continue
        out[off:off + 63] = ((p - p[0]) / palm).ravel()
        out[flag] = 1
    return out


@pytest.mark.parametrize("W,H", [(1280, 720), (640, 480)])
def test_model_input_is_the_apps_hand_relative_v1(W, H):
    hs = [hand("Left", 1), hand("Right", 2)]
    rich, _ = F.frame_vector_rich(hs, W, H)
    np.testing.assert_allclose(F.derive(rich[None], 1)[0], app_normalize_frame(hs, W, H), atol=1e-5)
    assert F.feature_dim(1) == F.feature_dim(2) == 128 and F.feature_dim(3) == 141 and F.VERSIONS == (1, 2, 3)
    with pytest.raises(ValueError):
        F.derive(rich[None], 4)


def test_model_input_shapes_and_absent_hands():
    r = rich_window(right=False)
    x = F.derive(r, 1)
    assert x.shape == (32, 128) and x.dtype == np.float32
    assert not x[:, 64:128].any()                                  # absent Right stays zero, flag 0
    assert (x[:, 63] == 1).all()
    assert F.derive(np.stack([r, r])).shape == (2, 32, 128)
    flat = r.copy()
    flat[:, 27:29] = 0.0                                            # landmark 9 on top of the wrist (in 2D)
    assert not F.derive(flat, 1)[:, :64].any()                      # the app drops such a hand, so do we


# --- model input: v2 = hand-relative-sequence-v1 (Russel's trainer; the app's normalizeSequence) ---------------
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hand_relative_sequence_parity.json"


def moving_hands(T, W, H, seed=0):
    """T frames of [Left, Right] hands; both wrists travel (Right also shrinks: it moves away from the camera)."""
    out = []
    for t in range(T):
        left = hand("Left", seed, shift=(0.004 * t, -0.002 * t))
        right = hand("Right", seed + 1, shift=(-0.006 * t, 0.003 * t))
        right.landmarks = (right.landmarks - right.landmarks[0]) * (1 - 0.01 * t) + right.landmarks[0]
        out.append([left, right])
    return out


def app_normalize_sequence(frames, W, H):
    """Port of the app's normalizeSequence (normalize.ts) = Russel's normalizeFrames, in the trainer's slots:
    landmarks 1-20 as normalizeFrame; the wrist = its pixel movement since the hand's first valid frame divided by
    the median 2D palm length of the hand's valid frames."""
    out = np.stack([app_normalize_frame(hs, W, H) for hs in frames])
    for label, off, flag in (("Left", 0, 63), ("Right", 64, 127)):
        valid = [t for t in range(len(frames)) if out[t, flag]]
        if not valid:
            continue
        px = {t: next(h for h in frames[t] if h.label == label).landmarks.astype(np.float64) * [W, H, W] for t in valid}
        scale = np.median([np.hypot(*(px[t][9, :2] - px[t][0, :2])) for t in valid])
        for t in valid:
            out[t, off:off + 3] = (px[t][0] - px[valid[0]][0]) / scale
    return out


@pytest.mark.parametrize("W,H", [(1280, 720), (640, 480)])
def test_v2_is_the_apps_hand_relative_sequence(W, H):
    frames = moving_hands(20, W, H)
    rich = np.stack([F.frame_vector_rich(hs, W, H)[0] for hs in frames])
    np.testing.assert_allclose(F.derive(rich, 2), app_normalize_sequence(frames, W, H), atol=1e-5)
    np.testing.assert_allclose(F.derive(rich, 2)[:, 3:63], F.derive(rich, 1)[:, 3:63])    # landmarks 1-20 as v1
    assert F.normalization(1) == "hand-relative-v1" and F.normalization(2) == "hand-relative-sequence-v1"
    assert F.version_of("hand-relative-sequence-v1") == 2 and F.version_of("hand-relative-v1") == 1
    with pytest.raises(ValueError):
        F.version_of("image")


def test_v2_matches_the_parity_cases_exported_with_russels_alphabet_model():
    """The camera parity cases of alphabet 46e7e387a78ee614 (frames and expected input in the app's layout)."""
    from trainer.app_export import to_app_layout
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["preprocessing"]["normalization"] == F.normalization(2)
    for c in data["cases"]:
        W, H = c["width"], c["height"]
        frames = np.array(c["frames"], dtype=np.float32)
        rich = np.stack([F.frame_vector_rich([L.Hand(label, 1.0, f[o:o + 63].reshape(21, 3))
                                              for label, o, flag in (("Left", 0, 126), ("Right", 63, 127)) if f[flag]],
                                             W, H)[0] for f in frames])
        np.testing.assert_allclose(to_app_layout(F.derive(rich, 2)), np.array(c["expected"]), atol=1e-4)


def test_v2_of_a_still_hand_is_v1():
    """A photo (one frame, or the same frame repeated) has no wrist movement: v2 gives exactly v1's numbers."""
    rich = rich_window(T=1)
    np.testing.assert_allclose(F.derive(rich, 2), F.derive(rich, 1), atol=1e-6)
    still = np.repeat(rich_window(T=1), 32, axis=0)
    np.testing.assert_allclose(F.derive(still, 2), F.derive(still, 1), atol=1e-6)
    np.testing.assert_allclose(F.derive(np.stack([still, still]), 2), F.derive(np.stack([still, still]), 1), atol=1e-6)


def test_v2_wrist_starts_at_zero_and_absent_hands_stay_zero():
    r = rich_window(move_left=0.01, move_right=0.02)
    r[:5, :64] = 0.0                                    # the Left hand appears at frame 5
    r[:5, F.L_WRIST] = 0.0
    r[:5, F.L_SCALE] = 0.0
    x = F.derive(r, 2)
    assert not x[:5, :64].any()
    np.testing.assert_allclose(x[5, 0:3], 0.0, atol=1e-6)
    np.testing.assert_allclose(x[0, 64:67], 0.0, atol=1e-6)
    assert x[31, 0] > 0 and x[31, 64] > 0               # both wrists moved right
    assert x.shape == (32, 128) and x.dtype == np.float32
    with pytest.raises(ValueError):
        F.derive(r[0], 2)                               # v2 needs the frames of a clip


def test_v2_mirror_commutes_with_the_model_input():
    r = rich_window(move_left=0.01, move_right=-0.015)
    np.testing.assert_allclose(F.derive(F.mirror(r), 2), F.mirror(F.derive(r, 2)), atol=1e-5)


def test_v2_converting_before_or_after_resampling():
    """The trainer converts the stored 32 frames (after resampling); Russel's trainer converts every camera frame and
    resamples after. Landmarks 1-20 agree; the wrist path can start later on the 32 frames (they may skip the frame
    where the hand appeared): by up to one 32-frame step of wrist movement. So a model must get its own order
    (exports say which, see app_export.WRIST_PATH)."""
    for T, gap, seed in ((45, 3, 1), (75, 3, 2), (90, 4, 3), (40, 0, 4)):
        rows = []
        for t in range(T):
            hs = []
            if gap <= t < T - gap:                        # no hand in the 0.1 s margins of the cut clip
                h = hand("Right", seed, shift=(0.08 * np.sin(t / 6), 0.05 * np.cos(t / 4)))
                h.landmarks = (h.landmarks - h.landmarks[0]) * (1 + 0.15 * np.sin(t / 9)) + h.landmarks[0]
                hs.append(h)
            rows.append(F.frame_vector_rich(hs, 1280, 720)[0])
        rich = np.stack(rows)
        per_frame = F.derive(rich, 2)
        one_frame = np.abs(np.diff(per_frame[gap:T - gap, 64:67], axis=0)).max()
        diff = np.abs(F.derive(L.resample(rich), 2) - L.resample(per_frame))
        assert diff[:, 67:].max() < 1e-5
        assert diff[:, 64:67].max() <= one_frame * ((T - 1) / 31 + 1) + 1e-4


# --- mirror ---------------------------------------------------------------------------------
@pytest.mark.parametrize("version", [None, 1])
def test_mirror_twice_is_identity(version):
    r = rich_window(move_left=0.01, move_right=0.02)
    x = r if version is None else F.derive(r, version)
    np.testing.assert_allclose(F.mirror(F.mirror(x)), x, atol=1e-6)


def test_mirror_flips_x_and_swaps_hands():
    r = rich_window(right=False)
    m = F.mirror(r)
    assert not m[:, :64].any()                     # the Left hand moved to the Right slot
    assert (m[:, L.RIGHT_FLAG] == 1).all()
    lh = r[:, :63].reshape(32, 21, 3)
    rh = m[:, 64:127].reshape(32, 21, 3)
    np.testing.assert_allclose(rh[..., 0], -lh[..., 0])
    np.testing.assert_allclose(rh[..., 1:], lh[..., 1:])
    np.testing.assert_allclose(m[:, F.R_WRIST][:, 0], -r[:, F.L_WRIST][:, 0])
    np.testing.assert_allclose(m[:, F.R_SCALE], r[:, F.L_SCALE])
    assert not m[:, F.L_WRIST].any() and not m[:, F.L_SCALE].any()


def test_mirror_commutes_with_the_model_input():
    r = rich_window(move_left=0.01, move_right=-0.015)
    np.testing.assert_allclose(F.derive(F.mirror(r), 1), F.mirror(F.derive(r, 1)), atol=1e-5)


def test_mirror_keeps_absent_hand_zero():
    r = rich_window(left=False)
    for x in (r, F.derive(r, 1)):
        m = F.mirror(x)
        assert not m[:, 64:128].any()


# --- canonical hand + Auto -------------------------------------------------------------------
def test_canonical_mirrors_only_left():
    r = rich_window(move_left=0.01)
    np.testing.assert_array_equal(F.canonical(r, "right"), r)
    np.testing.assert_allclose(F.canonical(r, "left"), F.mirror(r))
    with pytest.raises(ValueError):
        F.canonical(r, "auto")


@pytest.mark.parametrize("left,right,ml,mr,expected", [
    (True, False, 0.0, 0.0, "left"),          # one hand labelled Left -> left-handed
    (False, True, 0.0, 0.0, "right"),         # one hand labelled Right -> right-handed
    (True, True, 0.02, 0.0, "left"),          # two hands, the left one moves -> left-handed
    (True, True, 0.0, 0.02, "right"),         # two hands, the right one moves -> right-handed
    (False, False, 0.0, 0.0, "right"),        # no hands -> right (the default)
])
def test_auto_hand(left, right, ml, mr, expected):
    assert F.auto_hand(rich_window(left=left, right=right, move_left=ml, move_right=mr)) == expected


def test_auto_hand_mostly_one_hand_with_short_two_hand_overlap():
    r = rich_window(move_left=0.0, move_right=0.0)
    r[:24, 64:128] = 0.0                            # the Right hand only shows up in the last 8 frames
    r[:24, F.R_WRIST] = 0.0
    r[:24, F.R_SCALE] = 0.0
    assert F.auto_hand(r) == "left"


def test_auto_hand_of_a_mirrored_right_handed_window_is_left():
    r = rich_window(move_left=0.0, move_right=0.03)
    assert F.auto_hand(r) == "right"
    assert F.auto_hand(F.mirror(r)) == "left"


def test_resample_rich_keeps_extras_with_their_hand():
    r = rich_window(T=50, move_left=0.01)
    r[10:20, :64] = 0.0
    r[10:20, F.L_WRIST] = 0.0
    r[10:20, F.L_SCALE] = 0.0
    out = L.resample(r)
    assert out.shape == (32, F.RICH_DIM)
    absent = out[:, L.LEFT_FLAG] < 0.5
    assert absent.any()
    assert not out[absent][:, F.L_WRIST].any() and not out[absent][:, F.L_SCALE].any()
    np.testing.assert_allclose(out[:, :128], L.resample(r[:, :128]))


# --- location against the face (README 6.7, feature v3) --------------------------------------
def at(label, x, y):
    """A hand whose wrist, thumb tip and index tip are all at (x, y)."""
    lm = np.full((21, 3), 0.0, np.float32)
    lm[:, 0], lm[:, 1] = x, y
    lm[9] += [0.02, -0.05, 0.0]                                   # a real palm length (v1 needs it)
    return L.Hand(label, 0.9, lm)


def test_location_is_from_the_nose_in_face_sizes():
    W, H = 1280, 720
    face = face_points(nose=(0.5, 0.4), forehead_y=0.25, chin_y=0.55)           # face size 0.3
    rich, _ = F.frame_vector_rich([at("Right", 0.6, 0.25)], W, H, face)
    a = W / H
    np.testing.assert_allclose(rich[F.R_LOC], np.tile([(0.6 - 0.5) * a / 0.3, (0.25 - 0.4) / 0.3], 3), rtol=1e-5)
    assert not rich[F.L_LOC].any() and rich[F.FACE_FLAG] == 1.0


def test_father_at_the_forehead_and_mother_at_the_chin_differ():
    face = face_points()
    father, _ = F.frame_vector_rich([at("Right", 0.5, 0.25)], 1280, 720, face)    # at the forehead
    mother, _ = F.frame_vector_rich([at("Right", 0.5, 0.55)], 1280, 720, face)    # at the chin
    still_f, still_m = np.repeat(father[None], 32, 0), np.repeat(mother[None], 32, 0)
    for v in (1, 2):                                                 # v1 / v2 cannot tell them apart ...
        np.testing.assert_allclose(F.derive(still_f, v), F.derive(still_m, v), atol=1e-6)
    assert father[F.R_LOC][1] == pytest.approx(-0.5) and mother[F.R_LOC][1] == pytest.approx(0.5)  # ... v3 can
    x_f, x_m = F.derive(still_f, 3), F.derive(still_m, 3)
    np.testing.assert_allclose(x_f[:, :128], x_m[:, :128], atol=1e-6)
    assert not np.allclose(x_f[:, 128:], x_m[:, 128:])


def test_location_scale_free():
    """The same pose filmed closer (everything 1.5x larger around the nose) gives the same values."""
    near = face_points(nose=(0.5, 0.4), forehead_y=0.4 - 0.225, chin_y=0.4 + 0.225)
    far, _ = F.frame_vector_rich([at("Right", 0.56, 0.3)], 1280, 720, face_points())
    close, _ = F.frame_vector_rich([at("Right", 0.59, 0.25)], 1280, 720, near)
    np.testing.assert_allclose(far[136:], close[136:], atol=1e-5)


def test_no_face_gives_zero_location():
    for face in (None, np.zeros((0, 3), np.float32), face_points(forehead_y=0.4, chin_y=0.4)):
        rich, _ = F.frame_vector_rich([at("Right", 0.5, 0.3)], 1280, 720, face)
        assert not rich[136:].any()
        assert rich[L.RIGHT_FLAG] == 1.0                              # the hand itself is still there


def test_v3_is_v1_plus_the_location():
    r = rich_window(move_left=0.01, move_right=0.02, face=True)
    x = F.derive(r, 3)
    assert x.shape == (32, 141) and F.feature_dim(3) == 141
    np.testing.assert_array_equal(x[:, :128], F.derive(r, 1))
    np.testing.assert_allclose(x[:, 128:], r[:, 136:], atol=1e-6)
    assert F.normalization(3) == "hand-face-location-v1" and F.version_of("hand-face-location-v1") == 3
    assert 3 in F.APP_VERSIONS                                     # the app runs its face model for v3 (Family)


def test_v3_location_of_an_absent_hand_is_zero():
    r = rich_window(face=True)
    r[:, L.LEFT_OFFSET + 27:L.LEFT_OFFSET + 29] = 0.0                # Left landmark 9 at the wrist: v1 drops it
    x = F.derive(r, 3)
    assert not x[:, 128:134].any() and x[:, 134:140].any()


def test_mirror_swaps_and_flips_the_location():
    r = rich_window(right=False, face=True)
    m = F.mirror(r)
    np.testing.assert_allclose(m[:, F.R_LOC][:, 0::2], -r[:, F.L_LOC][:, 0::2])
    np.testing.assert_allclose(m[:, F.R_LOC][:, 1::2], r[:, F.L_LOC][:, 1::2])
    assert not m[:, F.L_LOC].any() and (m[:, F.FACE_FLAG] == 1).all()
    np.testing.assert_allclose(F.mirror(m), r, atol=1e-6)


def test_v3_mirror_commutes_with_the_model_input():
    r = rich_window(move_left=0.01, move_right=-0.015, face=True)
    x = F.derive(r, 3)
    np.testing.assert_allclose(F.derive(F.mirror(r), 3), F.mirror(x), atol=1e-5)
    np.testing.assert_allclose(F.mirror(F.mirror(x)), x, atol=1e-6)


def test_resample_never_blends_location_with_an_absent_face():
    r = rich_window(T=50, move_right=0.01, face=True)
    r[20:30, 136:] = 0.0                                              # face lost for 10 frames
    out = L.resample(r)
    no_face = out[:, F.FACE_FLAG] < 0.5
    assert no_face.any() and (~no_face).any()
    assert not out[no_face][:, 136:148].any()                         # copied zeros, never half-blended
    present = out[~no_face][:, F.R_LOC]
    assert (np.abs(present) > 1e-3).all()


# --- the JavaScript reference in docs/FEATURES.md -------------------------------------------
def js_reference() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("<!-- js-reference -->")
    block = text[start:].split("```js", 1)[1].split("```", 1)[0]
    return block


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_python_matches_the_js_reference(tmp_path):
    rng = np.random.default_rng(7)
    cases = []
    for k in range(6):
        r = rich_window(T=40, seed=k, left=k != 3, right=k != 4, move_left=0.01 * (k % 3), move_right=0.015 * (k % 2),
                        face=k != 5)
        r[rng.random(40) < 0.15, :64] = 0.0                    # some Left dropouts, with their extras zeroed too
        bad = r[:, L.LEFT_FLAG] < 0.5
        r[bad, 128:131] = 0.0
        r[bad, 134] = 0.0
        r[bad, 136:142] = 0.0
        r[rng.random(40) < 0.1, 136:] = 0.0                    # some frames without a face
        cases.append(r)
    hs3 = (hand("Left", 1), hand("Right", 2), hand("Right", 3, score=0.4))
    hands = [[{"label": h.label, "score": h.score, "landmarks": h.landmarks.tolist()} for h in hs3]] * 2
    faces = [None, face_points(seed=4).tolist()]
    (tmp_path / "in.json").write_text(json.dumps({"cases": [c.tolist() for c in cases], "hands": hands,
                                                  "faces": faces}))
    script = js_reference() + """
const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = {cases: inp.cases.map(c => {
  const s = resampleRich(c, 32);
  return {resampled: s, input: modelInput(s), mirrored: mirrorFeatures(modelInput(s)), mirroredRich: mirrorFeatures(s),
          hand: autoHand(s), inputV2: modelInputSequence(s), inputV3: modelInputLocation(s),
          mirroredV3: mirrorFeatures(modelInputLocation(s))};
}), frames: inp.hands.map((h, i) => frameVectorRich(h, 1280, 720, inp.faces[i]))};
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""
    (tmp_path / "ref.js").write_text(script, encoding="utf-8")
    subprocess.run(["node", str(tmp_path / "ref.js"), str(tmp_path / "in.json"), str(tmp_path / "out.json")],
                   check=True, capture_output=True, timeout=60)
    out = json.loads((tmp_path / "out.json").read_text())
    for c, o in zip(cases, out["cases"]):
        s = L.resample(c)
        np.testing.assert_allclose(np.array(o["resampled"]), s, atol=1e-5)
        np.testing.assert_allclose(np.array(o["input"]), F.derive(s, 1), atol=1e-5)
        np.testing.assert_allclose(np.array(o["mirrored"]), F.mirror(F.derive(s, 1)), atol=1e-5)
        np.testing.assert_allclose(np.array(o["mirroredRich"]), F.mirror(s), atol=1e-5)
        assert o["hand"] == F.auto_hand(s)
        np.testing.assert_allclose(np.array(o["inputV2"]), F.derive(s, 2), atol=1e-5)
        np.testing.assert_allclose(np.array(o["inputV3"]), F.derive(s, 3), atol=1e-5)
        np.testing.assert_allclose(np.array(o["mirroredV3"]), F.mirror(F.derive(s, 3)), atol=1e-5)
    for h, face, js in zip(hands, faces, out["frames"]):
        hs = [L.Hand(x["label"], x["score"], np.array(x["landmarks"], np.float32)) for x in h]
        f = None if face is None else np.array(face, np.float32)
        np.testing.assert_allclose(np.array(js), F.frame_vector_rich(hs, 1280, 720, f)[0], atol=1e-5)
    assert out["frames"][1][F.FACE_FLAG] == 1 and any(out["frames"][1][136:148])
