"""Live test with the HUDYAT app's model: the app's capture today (2-second clip, its resample) and the recommended
one (sign window, no blending, mirror), the app's acceptance rule, and the copies of the app's code."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from trainer import app_export as X
from trainer import features as F
from trainer import live
from trainer import settings as cfg

from test_app_export import TFJS_TOOL
from test_experiments import Ctx, env  # noqa: F401  (fixture)
from test_features import rich_window

APP_CORE = Path.home() / "Documents" / "HUDYAT" / "client" / "src" / "features" / "fsl-recognition" / "category-core.js"


def js_function(src: str, name: str) -> str:
    """The text of `function name(...) {...}` (with or without `export`), by brace matching."""
    i = src.index(f"function {name}(")
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        depth += {"{": 1, "}": -1}.get(src[k], 0)
        if depth == 0:
            return re.sub(r"\s+", " ", src[i:k + 1])
    raise ValueError(name)


@pytest.mark.skipif(not APP_CORE.exists(), reason="the HUDYAT app folder is not on this computer")
@pytest.mark.parametrize("tool,names", [("app_predict.js", ["buildModel", "loadWeights"]),
                                        ("app_resample.js", ["resample"])])
def test_copies_of_the_apps_code_are_still_identical(tool, names):
    app = APP_CORE.read_text(encoding="utf-8")
    app = re.sub(r"\n\s*// Keras exports weights per layer;[^\n]*", "", app)       # a comment our copy leaves out
    ours = (TFJS_TOOL / tool).read_text(encoding="utf-8")
    for n in names:
        assert js_function(ours, n) == js_function(app, n), f"{n} in {tool} differs from the app's"


def test_app_resample_is_the_apps(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed")
    rng = np.random.default_rng(1)
    clips = []
    for T in (8, 45, 61):
        x = F.derive(rich_window(T=T, seed=T, move_left=0.01, move_right=0.02))
        x[rng.random(T) < 0.25, :64] = 0.0                     # the Left hand drops out now and then
        clips.append(x)
    (tmp_path / "in.json").write_text(json.dumps({"clips": [X.to_app_layout(c).tolist() for c in clips]}))
    subprocess.run([node, str(TFJS_TOOL / "app_resample.js"), str(tmp_path / "in.json"), str(tmp_path / "out.json")],
                   check=True, capture_output=True, timeout=60)
    js = json.loads((tmp_path / "out.json").read_text())["out"]
    for c, o in zip(clips, js):
        np.testing.assert_allclose(np.array(o), X.to_app_layout(live.app_resample(c)), atol=1e-5)


def test_app_resample_blends_with_an_absent_hand_and_the_fix_does_not():
    from trainer.landmarks import resample
    x = F.derive(rich_window(T=33, right=False))
    x[16:, :64] = 0.0                                          # the hand leaves half-way
    app, fixed = live.app_resample(x), resample(x)
    t = 15                                                     # between frame 15 (hand) and 16 (none)
    assert app[t, 63] == fixed[t, 63] == 1
    assert np.abs(app[t, :63]).sum() < np.abs(fixed[t, :63]).sum()      # the app's values shrink towards zero


def test_app_clip_is_two_seconds_from_the_first_hand():
    seg = live.AppClipSegmenter(30.0)
    assert seg.n_max == 60
    for _ in range(10):
        assert seg.push(np.zeros(149), 0) is None and not seg.signing
    out = [seg.push(np.ones(149), 1 if i < 20 else 0) for i in range(60)]   # hands down after 0.7 s: still 2 s
    assert all(o is None for o in out[:-1]) and out[-1] is not None
    assert len(out[-1].frames) == 60 and out[-1].reason == "clip_2s"


@pytest.fixture
def export(env):
    from trainer import final
    final.run(Ctx(), "cat", opts={"epochs": 2, "aug_factor": 1, "batch_size": 8})
    chk = X.export_category(Ctx(), "cat")
    return cfg.Paths().exports / chk["export_name"]


def window_of(r, fps=30.0):
    return live.Window(r, (r[:, 63] + r[:, 127]).astype(int), fps, "hands_gone")


def test_app_classifier_runs_the_exported_weights(export):
    import keras
    clf = live.AppClassifier(export, "fixed")
    w = window_of(rich_window(T=40, left=False, move_right=0.01))
    x, info = clf.model_input(w, "right")
    assert x.shape == (32, 128) and info["hand_used"] == "right"
    trainer_model = keras.models.load_model(cfg.Paths().model_dir("cat") / "model.keras")
    back = np.empty_like(x)
    back[..., X.APP_ORDER] = x                                 # app layout -> trainer layout
    np.testing.assert_allclose(clf.predict(x[None])[0], trainer_model.predict(back[None], verbose=0)[0], atol=1e-5)
    r = clf.classify(w, "right")
    assert r["gates"] == "app" and r["sure"] == (r["confidence"] >= 0.8 and r["margin"] >= 0.15)
    assert r["margin"] == pytest.approx(r["top3"][0][1] - r["top3"][1][1], abs=1e-6)


def test_fixed_capture_mirrors_a_left_hand_and_the_apps_does_not(export):
    r = rich_window(T=40, left=False, move_right=0.01)
    right, left = window_of(r), window_of(F.mirror(r))
    fixed = live.AppClassifier(export, "fixed")
    a, ia = fixed.model_input(right, "auto")
    b, ib = fixed.model_input(left, "auto")
    assert ia["hand_used"] == "right" and ib["hand_used"] == "left"
    np.testing.assert_allclose(a, b, atol=1e-5)               # the left-handed sign reaches the model as right-handed
    today = live.AppClassifier(export, "app")
    c, ic = today.model_input(left, "auto")
    assert ic["hand_used"] == "none" and c[:, 126].all() and not c[:, 127].any()   # still a Left hand in the app


def test_fixed_capture_trims_to_the_sign_and_the_apps_keeps_the_idle_frames(export):
    r = rich_window(T=60, left=False, move_right=0.01)
    r[40:] = 0.0                                               # the hands went down after 40 frames
    w = window_of(r)
    _, fixed = live.AppClassifier(export, "fixed").model_input(w)
    x, _ = live.AppClassifier(export, "app").model_input(w)
    assert fixed["trim"][1] < 45                               # cut 0.1 s after the last hand
    assert not x[-8:, 127].any()                               # the app's clip ends with empty frames



# --- docs/APP_RECOGNITION.md: the JavaScript for the app, checked against the trainer ------------------------------
DOC = Path(__file__).resolve().parent.parent / "docs" / "APP_RECOGNITION.md"

# the app's normalizeFrame (client/src/features/fsl-recognition/normalize.ts) without its TypeScript types
NORMALIZE_FRAME_JS = """
function normalizeFrame(frame,width,height){
 const out=new Float32Array(128);
 for(let slot=0;slot<2;slot++){
  if(!frame[126+slot])continue;
  const base=slot*63,px=i=>frame[base+i*3]*width,py=i=>frame[base+i*3+1]*height,pz=i=>frame[base+i*3+2]*width;
  const ox=px(0),oy=py(0),oz=pz(0);
  const palm=Math.hypot(px(9)-ox,py(9)-oy);
  if(!Number.isFinite(palm)||palm<1e-6)continue;
  for(let i=0;i<21;i++)out.set([(px(i)-ox)/palm,(py(i)-oy)/palm,(pz(i)-oz)/palm],base+i*3);
  out[126+slot]=1;
 }
 return Array.from(out);
}
"""


def doc_js() -> str:
    text = DOC.read_text(encoding="utf-8")
    return text[text.index("<!-- app-js -->"):].split("```js", 1)[1].split("```", 1)[0]


def raw_frames(T, seed, left, right, gap=(), move=0.004):
    """T frames of MediaPipe-like hands: (trainer rich frames, hand counts, the app's packResult frames)."""
    from trainer import landmarks as L
    rng = np.random.default_rng(seed)
    base = {lab: rng.uniform(0.35, 0.6, (21, 3)).astype(np.float32) for lab in ("Left", "Right")}
    for b in base.values():
        b[:, 2] = rng.uniform(-0.05, 0.05, 21)
        b[0, 2] = 0.0
    rich, counts, raws = [], [], []
    for t in range(T):
        hs = []
        for lab, on, sign in (("Left", left, 1), ("Right", right, -1)):
            if on and t not in gap:
                lm = base[lab].copy()
                lm[:, 0] += sign * move * t
                lm[:, 1] += 0.002 * t
                hs.append(L.Hand(lab, 0.9, lm))
        rich.append(F.frame_vector_rich(hs, 1280, 720)[0])
        counts.append(len(hs))
        raw = np.zeros(128)
        for h in hs:
            slot = 0 if h.label == "Left" else 1
            raw[slot * 63:slot * 63 + 63] = h.landmarks.astype(np.float64).ravel()
            raw[126 + slot] = 1
        raws.append(raw)
    return np.stack(rich), np.array(counts), np.stack(raws)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_the_documented_javascript_matches_the_trainer(tmp_path):
    from trainer.landmarks import resample
    fps = 30.0
    # 1. the segmenter: idle, a sign, a blip, a 3-second hold, rest, another sign
    pres = [0] * 10 + [1] * 30 + [0] * 12 + [1] * 3 + [0] * 10 + [1] * 100 + [0] * 10 + [1] * 25 + [0] * 8
    seg, py_windows = live.LiveSegmenter(fps), []
    for i, p in enumerate(pres):
        w = seg.push(np.full(149, i, np.float32), p)
        if w is not None:
            py_windows.append([int(w.frames[0, 0]), len(w.frames), w.reason])
    # 2. clips: right-handed with dropouts, left-handed (Left label only), both hands held still
    cases = [raw_frames(40, 1, False, True, gap={12, 13}), raw_frames(50, 2, True, False, move=0.006),
             raw_frames(45, 3, True, True, gap={20}, move=0.0)]
    clips = []
    for rich, counts, raws in cases:
        clips.append({"rich": rich, "counts": counts, "raws": raws})
    inp = {"presence": pres, "fps": fps,
           "clips": [{"raws": c["raws"].tolist()} for c in clips]}
    (tmp_path / "in.json").write_text(json.dumps(inp))
    script = NORMALIZE_FRAME_JS + doc_js() + """
const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const seg = createSignSegmenter(inp.fps), windows = [];
inp.presence.forEach((p, i) => { const w = seg.push(i, p === 1); if (w) windows.push([w.items[0], w.items.length, w.reason]); });
const clips = inp.clips.map(c => {
  const items = c.raws.map(raw => ({raw, norm: normalizeFrame(raw, 1280, 720)}));
  const [s, e] = signWindow(items.map(it => hasHand(it.raw)), inp.fps);
  const cut = items.slice(s, e + 1).map(it => it.norm);
  return {window: [s, e], resampled: resampleSign(cut), mirrored: mirrorFrames(resampleSign(cut)),
          auto: autoHand(c.raws.slice(s, e + 1), 1280, 720),
          right: recommendedInput(items, inp.fps, 1280, 720, 'right'),
          auto_input: recommendedInput(items, inp.fps, 1280, 720, 'auto')};
});
fs.writeFileSync(process.argv[3], JSON.stringify({windows, clips}));
"""
    (tmp_path / "doc.js").write_text(script, encoding="utf-8")
    subprocess.run(["node", str(tmp_path / "doc.js"), str(tmp_path / "in.json"), str(tmp_path / "out.json")],
                   check=True, capture_output=True, timeout=60)
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["windows"] == py_windows and len(py_windows) == 3          # the blip gives no window
    for c, o in zip(clips, out["clips"]):
        w = live.Window(c["rich"], c["counts"], fps, "hands_gone")
        sample, start, end, _, _ = live.window_and_sample(c["rich"], c["counts"], fps)
        assert o["window"] == [start, end]
        x = resample(F.derive(c["rich"], 1)[start:end + 1])
        np.testing.assert_allclose(np.array(o["resampled"]), X.to_app_layout(x), atol=1e-4)
        np.testing.assert_allclose(np.array(o["mirrored"]), X.to_app_layout(F.mirror(x)), atol=1e-4)
        assert o["auto"] == F.auto_hand(sample)
        for key, hand in (("right", "right"), ("auto_input", "auto")):
            py, info = live.recommended_input(w, hand)
            assert o[key]["hand"] == info["hand_used"]
            np.testing.assert_allclose(np.array(o[key]["input"]), py, atol=1e-4)
    assert [o["auto"] for o in out["clips"]] == ["right", "left", "right"]      # both hands still: a tie -> right


def test_a_feature_v2_model_gets_the_numbers_it_was_trained_on(env):
    """Live test of a v2 export: the cut window resampled to 32 frames, mirrored if left-handed, then converted,
    like the stored training samples (not converted per camera frame, see test_features)."""
    from trainer import final
    from trainer.landmarks import window_and_sample
    final.run(Ctx(), "cat", opts={"epochs": 1, "aug_factor": 1, "batch_size": 8, "feature_version": 2})
    d = cfg.Paths().exports / X.export_category(Ctx(), "cat")["export_name"]
    r = rich_window(T=50, left=False, move_right=0.01)
    r[:4] = 0.0                                                # the hand appears after 4 frames
    for rich, hand in ((r, "right"), (F.mirror(r), "left")):
        w = window_of(rich)
        clf = live.AppClassifier(d, "fixed")
        assert clf.feature_version == 2
        x, info = clf.model_input(w, "auto")
        sample = window_and_sample(w.frames, w.counts, w.fps, params=clf.params)[0]
        assert info["hand_used"] == hand
        np.testing.assert_allclose(x, X.to_app_layout(F.derive(F.canonical(sample, hand), 2)), atol=1e-5)
    trainer_clf = live.Classifier(cfg.Paths().model_dir("cat"))
    assert trainer_clf.feature_version == 2
    np.testing.assert_allclose(trainer_clf.classify(window_of(r), "right")["top3"][0][1],
                               live.AppClassifier(d, "fixed").classify(window_of(r), "right")["top3"][0][1], atol=1e-4)


def test_the_live_test_can_pick_any_export_of_a_category(env, monkeypatch):
    """Training v2 after v1 replaces the trainer model, but both exports stay: the Live test lists both, with their
    feature version and model run, and runs the one picked."""
    from trainer import dataset as ds
    from trainer import final, live_window
    final.run(Ctx(), "cat", opts={"epochs": 1, "aug_factor": 1, "batch_size": 8})
    v1 = X.export_category(Ctx(), "cat")
    final.run(Ctx(), "cat", opts={"epochs": 1, "aug_factor": 1, "batch_size": 8, "feature_version": 2})
    v2 = X.export_category(Ctx(), "cat")
    paths = cfg.Paths()
    found = X.list_exports(paths, "cat")
    assert [e["name"] for e in found] == [v2["export_name"], v1["export_name"]]          # newest first
    assert [e["feature_version"] for e in found] == [2, 1]
    assert [e["model_run_id"] for e in found] == [v2["model_run_id"], v1["model_run_id"]]
    assert found[0]["current"] and not found[1]["current"]          # v2 is the trainer's model now
    assert X.list_exports(paths, "other") == []
    assert X.export_dir(paths, "cat", v1["export_name"]) == paths.exports / v1["export_name"]
    assert X.export_dir(paths, "cat", None) == paths.exports / v2["export_name"]
    for bad in ("../cat-dabest-recorder-v1", "nope", v1["export_name"].replace("cat", "other")):
        with pytest.raises(ds.PlainError):
            X.export_dir(paths, "cat", bad)
    assert live.AppClassifier(X.export_dir(paths, "cat", v1["export_name"])).feature_version == 1
    seen = {}
    monkeypatch.setattr(live_window, "run", lambda ctx, category, **kw: seen.update(kw) or {"message": "ok"})
    from trainer import cli
    assert cli.main("live", ["--category", "cat", "--recognizer", "app", "--export", v1["export_name"]]) == 0
    assert seen["export"] == v1["export_name"]


def test_live_attempts_are_reported_per_model(env):
    paths = cfg.Paths()
    for run, fv, ok in (("final-cat-1", 1, 1), ("final-cat-1", 1, 0), ("final-cat-2", 2, 1)):
        live.save_attempt(paths, "cat", "T1", "right", "A", "A", 0.9, ok, 300.0, run, fv, "app", "fixed", 0.5, 800.0)
    df = live.read_attempts(paths.live_tests / "cat.csv")
    rows = live.by_model(df)
    assert [(r["model_run_id"], r["feature_version"], r["attempts"], r["accuracy"]) for r in rows] == [
        ("final-cat-1", 1, 2, 0.5), ("final-cat-2", 2, 1, 1.0)]


# the app's locationFeatures + normalizeFrameLocation (client/src/features/fsl-recognition/normalize.ts) without types
NORMALIZE_LOCATION_JS = """
function locationFeatures(frame,face,width,height){
 const out=new Float32Array(13);
 if(!face||face.length<=152||!(height>0))return out;
 const a=width/height,nose=face[1];
 const size=Math.hypot((face[10].x-face[152].x)*a,face[10].y-face[152].y);
 if(!Number.isFinite(size)||size<1e-6)return out;
 for(let slot=0;slot<2;slot++){
  if(!frame[126+slot])continue;
  [0,4,8].forEach((point,i)=>{
   const base=slot*63+point*3;
   out[slot*6+i*2]=(frame[base]-nose.x)*a/size;
   out[slot*6+i*2+1]=(frame[base+1]-nose.y)/size;
  });
 }
 out[12]=1;
 return out;
}
function normalizeFrameLocation(frame,face,width,height){
 const out=new Float32Array(141),hands=normalizeFrame(frame,width,height),location=locationFeatures(frame,face,width,height);
 out.set(hands);
 for(let slot=0;slot<2;slot++)if(!hands[126+slot])location.fill(0,slot*6,slot*6+6);
 out.set(location,128);
 return Array.from(out);
}
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_the_documented_javascript_gives_feature_v3_like_the_trainer(tmp_path):
    """Family (hand-face-location-v1): the app's per-frame location values, resampled and mirrored by the documented
    sign capture, equal the trainer's recommended input for an exported v3 model."""
    from test_features import face_points
    from trainer import landmarks as L
    fps, rng = 30.0, np.random.default_rng(7)
    cases = []
    for T, left, right, gap, no_face in ((40, False, True, {12, 13}, {5, 20, 21}), (36, True, False, set(), {30}),
                                         (45, True, True, {20}, set(range(0, 45, 7)))):
        base = {lab: rng.uniform(0.35, 0.6, (21, 3)).astype(np.float32) for lab in ("Left", "Right")}
        rich, counts, raws, faces = [], [], [], []
        for t in range(T):
            hs = []
            for lab, on, sign in (("Left", left, 1), ("Right", right, -1)):
                if on and t not in gap:
                    lm = base[lab].copy()
                    lm[:, 0] += sign * 0.004 * t
                    lm[:, 1] -= 0.006 * t                       # the hand rises towards the forehead
                    hs.append(L.Hand(lab, 0.9, lm))
            face = None if t in no_face else face_points(nose=(0.5 + 0.001 * t, 0.4), seed=t)
            rich.append(F.frame_vector_rich(hs, 1280, 720, face)[0])
            counts.append(len(hs))
            raw = np.zeros(128)
            for h in hs:
                slot = 0 if h.label == "Left" else 1
                raw[slot * 63:slot * 63 + 63] = h.landmarks.astype(np.float64).ravel()
                raw[126 + slot] = 1
            raws.append(raw.tolist())
            faces.append(None if face is None else [{"x": float(p[0]), "y": float(p[1])} for p in face[:153]])
        cases.append((live.Window(np.stack(rich), np.array(counts), fps, "hands_gone"), raws, faces))
    (tmp_path / "in.json").write_text(json.dumps({"fps": fps, "clips": [{"raws": r, "faces": f} for _, r, f in cases]}))
    script = NORMALIZE_FRAME_JS + NORMALIZE_LOCATION_JS + doc_js() + """
const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const clips = inp.clips.map(c => {
  const items = c.raws.map((raw, i) => ({raw, norm: normalizeFrameLocation(raw, c.faces[i], 1280, 720)}));
  return Object.fromEntries(['right', 'left', 'auto'].map(h => [h, recommendedInput(items, inp.fps, 1280, 720, h)]));
});
fs.writeFileSync(process.argv[3], JSON.stringify({clips}));
"""
    (tmp_path / "doc.js").write_text(script, encoding="utf-8")
    subprocess.run(["node", str(tmp_path / "doc.js"), str(tmp_path / "in.json"), str(tmp_path / "out.json")],
                   check=True, capture_output=True, timeout=60)
    out = json.loads((tmp_path / "out.json").read_text())["clips"]
    for (w, _, _), o in zip(cases, out):
        for hand in ("right", "left", "auto"):
            py, info = live.recommended_input(w, hand, version=3)
            assert py.shape == (32, 141) and o[hand]["hand"] == info["hand_used"]
            np.testing.assert_allclose(np.array(o[hand]["input"]), py, atol=1e-4)
    first = np.array(out[0]["right"]["input"])
    assert first[:, 140].any() and (first[:, 135] < 0).any()       # a face was found; the Right wrist rose above the nose
