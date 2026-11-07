"""Export for the HUDYAT app: the files load and predict like the trained model (numpy copy of the app's network;
the app's own buildModel + loadWeights in real TF.js when Node is present)."""

import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from trainer import app_export as X
from trainer import experiments as E
from trainer import settings as cfg
from trainer.model import build_model
from trainer.settings import PROJECT_DIR

from test_experiments import SIGNS, Ctx, env  # noqa: F401  (fixture)

TFJS_TOOL = PROJECT_DIR / "tools" / "tfjs_check"


def trained(n=5, seed=0):
    model = build_model(n)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(64, 32, 128)).astype(np.float32)
    model.fit(x, np.eye(n)[rng.integers(0, n, 64)], epochs=1, verbose=0)          # non-trivial BN statistics
    return model, x[:6]


def test_app_layout_moves_the_right_hand_and_the_flags():
    x = np.arange(128, dtype=np.float32)
    a = X.to_app_layout(x)
    np.testing.assert_array_equal(a[:63], x[:63])                  # Left hand
    np.testing.assert_array_equal(a[63:126], x[64:127])            # Right hand right after it
    assert a[126] == x[63] and a[127] == x[127]                    # then the two flags
    assert sorted(X.APP_ORDER) == list(range(128))


def test_app_network_on_app_frames_equals_the_model_on_trainer_frames():
    model, x = trained()
    ws = X.app_weights(model)
    assert [list(w.shape) for w in ws][:2] == [[3, 128, 64], [64]] and len(ws) == 28
    np.testing.assert_allclose(X.forward_app(ws, X.to_app_layout(x)), model.predict(x, verbose=0), atol=1e-5)


def test_batch_norm_epsilon_is_the_apps():
    model = build_model(3)
    assert [l.epsilon for l in model.layers if "batch_normalization" in l.name] == [1e-4, 1e-4]


@pytest.fixture
def exported(env):
    """A real (tiny) final model of the test category, exported."""
    from trainer import final
    final.run(Ctx(), "cat", opts={"epochs": 2, "aug_factor": 1, "batch_size": 8})
    check = X.export_category(Ctx(), "cat")
    return check, cfg.Paths().exports / check["export_name"]


def test_export_files_and_catalog_entry(exported):
    check, d = exported
    assert check["ok"] and check["real_clips"] and d.name == "cat-dabest-recorder-v1"
    assert {p.name for p in d.iterdir()} == {"model.json", "weights.bin", "parity.json", "preprocessing.json",
                                             "catalog_entry.json", "verify.json", "README.txt"}
    meta = json.loads((d / "model.json").read_text())
    blob = (d / "weights.bin").read_bytes()
    assert meta["format"] == "fsl-conv1d-bilstm-v1" and meta["inputShape"] == [32, 128] and meta["category"] == "cat"
    assert meta["classes"] == list(SIGNS)
    assert meta["weightBytes"] == len(blob) == sum(s["bytes"] for s in meta["weights"])
    assert meta["weightSha256"] == hashlib.sha256(blob).hexdigest() and meta["version"] == meta["weightSha256"][:16]
    pre = meta["preprocessing"]
    assert pre["normalization"] == "hand-relative-v1" and pre["extraction"]["swap_hands"] is False
    assert 0 < pre["extraction"]["detection_confidence"] <= 1
    assert pre["extraction"]["model_sha256"] == hashlib.sha256(cfg.MODEL_PATH.read_bytes()).hexdigest()
    assert json.loads((d / "preprocessing.json").read_text()) == pre
    entry = json.loads((d / "catalog_entry.json").read_text())
    assert entry["id"] == "cat" and entry["version"] == meta["version"] and entry["classes"] == len(SIGNS)
    assert entry["modelSha256"] == hashlib.sha256((d / "model.json").read_bytes()).hexdigest()
    assert entry["weightsSha256"] == meta["weightSha256"]
    assert entry["bytes"] == (d / "model.json").stat().st_size + len(blob)
    assert entry["modelUrl"] == f"./models/cat/{meta['version']}/model.json"
    assert entry["handSha256"] == pre["extraction"]["model_sha256"]
    parity = json.loads((d / "parity.json").read_text())
    assert np.array(parity["inputs"]).shape[1:] == (32, 128)
    assert np.array(parity["expected"]).shape == (len(parity["inputs"]), len(SIGNS))
    assert X.last_export(cfg.Paths(), "cat")["version"] == meta["version"]


def test_parity_inputs_are_in_the_apps_layout(exported):
    _, d = exported
    p = np.array(json.loads((d / "parity.json").read_text())["inputs"])
    # the test clips are right-handed: the Right flag is the app's index 127, the Left flag (126) stays 0
    assert (p[:, :, 127] == 1).any() and not p[:, :, 126].any()


def test_export_refuses_a_model_without_the_apps_features(env):
    from trainer import dataset as ds
    from trainer import final
    final.run(Ctx(), "cat", opts={"epochs": 1, "aug_factor": 1, "batch_size": 8})
    p = cfg.Paths().model_dir("cat") / "config.json"
    conf = json.loads(p.read_text())
    conf["feature_version"], conf["input_shape"] = 4, [32, 137]
    p.write_text(json.dumps(conf))
    with pytest.raises(ds.PlainError, match="app's features"):
        X.export_category(Ctx(), "cat")


def test_export_of_a_feature_v3_model(env):
    """v3 (hand + face location, Family): 141 values per frame, the face model named in the export and the catalog."""
    import hashlib
    from trainer import final
    from trainer.landmarks import face_model_path
    final.run(Ctx(), "cat", opts={"epochs": 1, "aug_factor": 1, "batch_size": 8, "feature_version": 3})
    check = X.export_category(Ctx(), "cat")
    d = cfg.Paths().exports / check["export_name"]
    assert check["ok"] and check["real_clips"] and d.name == "cat-dabest-recorder-v3"
    meta = json.loads((d / "model.json").read_text())
    pre = meta["preprocessing"]
    assert meta["inputShape"] == [32, 141] and pre["normalization"] == "hand-face-location-v1" and pre["features"] == 141
    face_sha = hashlib.sha256(face_model_path(cfg.data_dir()).read_bytes()).hexdigest()
    assert pre["face"]["model_sha256"] == face_sha and (pre["face"]["nose"], pre["face"]["forehead"], pre["face"]["chin"]) == (1, 10, 152)
    assert len(pre["layout"]) == 7 and pre["trainer"]["feature_version"] == 3
    entry = json.loads((d / "catalog_entry.json").read_text())
    assert entry["faceUrl"] == f"./shared/{face_sha}.task" and entry["faceSha256"] == face_sha and entry["faceBytes"] > 0
    assert np.array(json.loads((d / "parity.json").read_text())["inputs"]).shape[1:] == (32, 141)
    assert "face detector" in (d / "README.txt").read_text()
    assert list(X.to_app_layout(np.arange(141))[128:]) == list(range(128, 141))    # the location values stay in place


def test_models_record_their_normalization_and_older_ones_are_refused(env):
    """A model from before hand-relative-v1 (like the 2026-10-21 backup) is also v1 and 32 x 128: only its
    normalization tells it apart. Export and the live test refuse it."""
    from trainer import dataset as ds
    from trainer import final, live
    final.run(Ctx(), "cat", opts={"epochs": 1, "aug_factor": 1, "batch_size": 8})
    mdir = cfg.Paths().model_dir("cat")
    conf = json.loads((mdir / "config.json").read_text())
    pre = json.loads((mdir / "preprocess.json").read_text())
    assert conf["normalization"] == pre["normalization"] == "hand-relative-v1"
    del conf["normalization"], pre["normalization"]
    conf["date"], pre["model_run_id"] = "2026-10-20T15:00:00", "final-cat-20261020-150000"
    (mdir / "config.json").write_text(json.dumps(conf))
    (mdir / "preprocess.json").write_text(json.dumps(pre))
    with pytest.raises(ds.PlainError, match="older features"):
        X.export_category(Ctx(), "cat")
    with pytest.raises(ds.PlainError, match="older features"):
        live.Classifier(mdir)
    conf["date"], pre["model_run_id"] = "2026-10-21T06:38:03", "final-cat-20261021-063008"   # after the switch
    (mdir / "config.json").write_text(json.dumps(conf))
    (mdir / "preprocess.json").write_text(json.dumps(pre))
    assert X.export_category(Ctx(), "cat")["ok"]
    live.Classifier(mdir)


def test_the_apps_own_loader_gives_the_same_answers(exported, tmp_path):
    """tools/tfjs_check/app_predict.js = the app's buildModel + loadWeights, in real TF.js."""
    node = shutil.which("node")
    if not node or not (TFJS_TOOL / "node_modules" / "@tensorflow" / "tfjs").exists():
        pytest.skip("Node.js or tools/tfjs_check/node_modules not installed")
    _, d = exported
    out = tmp_path / "out.json"
    subprocess.run([node, str(TFJS_TOOL / "app_predict.js"), str(d), str(out)], check=True, cwd=TFJS_TOOL,
                   capture_output=True, timeout=300)
    expected = np.array(json.loads((d / "parity.json").read_text())["expected"])
    np.testing.assert_allclose(np.array(json.loads(out.read_text())["pred"]), expected, atol=X.PARITY_TOLERANCE)


def test_export_runs_from_the_cli_for_several_categories(exported):
    out = X.run(Ctx(), ["cat"])
    assert out["exports"] == ["cat-dabest-recorder-v1"] and "HUDYAT app" in out["message"]
    assert E.DEFAULT_OPTS["feature_version"] == 1


def test_export_of_a_feature_v2_model(env):
    """A v2 model (hand-relative-sequence-v1) exports with that normalization and the order of its wrist path."""
    from trainer import final
    final.run(Ctx(), "cat", opts={"epochs": 1, "aug_factor": 1, "batch_size": 8, "feature_version": 2})
    check = X.export_category(Ctx(), "cat")
    d = cfg.Paths().exports / check["export_name"]
    assert check["ok"] and check["real_clips"] and d.name == "cat-dabest-recorder-v2"
    pre = json.loads((d / "model.json").read_text())["preprocessing"]
    assert pre["normalization"] == "hand-relative-sequence-v1" and pre["wrist_path"] == X.WRIST_PATH
    assert pre["trainer"]["feature_version"] == 2
    assert "first valid frame" in pre["coordinates"]
