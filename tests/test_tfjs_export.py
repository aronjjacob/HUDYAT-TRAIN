"""TF.js export: the files load and predict like Keras (numpy reader; real TF.js when Node is present)."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from trainer.model import build_model
from trainer.settings import PROJECT_DIR
from trainer.tfjs_export import forward_numpy, read_tfjs, write_tfjs

TFJS_TOOL = PROJECT_DIR / "tools" / "tfjs_check"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("tfjs")
    model = build_model(5)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 32, 128)).astype(np.float32)
    model.fit(X, np.eye(5)[rng.integers(0, 5, 64)], epochs=1, verbose=0)   # non-trivial BN statistics
    mj = write_tfjs(model, ["a", "b", "c", "d", "none"], out / "tfjs")
    return model, X[:6], mj


def test_files_and_manifest(exported):
    model, _, mj = exported
    d = mj.parent
    assert sorted(p.name for p in d.iterdir()) == ["group1-shard1of1.bin", "labels.json", "model.json"]
    j = json.loads(mj.read_text())
    assert j["format"] == "layers-model"
    n = sum(int(np.prod(w["shape"])) for w in j["weightsManifest"][0]["weights"])
    assert (d / "group1-shard1of1.bin").stat().st_size == 4 * n == 4 * model.count_params()
    assert json.loads((d / "labels.json").read_text())[-1] == "none"


def test_numpy_forward_matches_keras(exported):
    model, X, mj = exported
    topo, w = read_tfjs(mj)
    np.testing.assert_allclose(forward_numpy(topo, w, X), model.predict(X, verbose=0), atol=1e-5)


def tfjs_predict(mj, X, tmp_path):
    node = shutil.which("node")
    if not node or not (TFJS_TOOL / "node_modules" / "@tensorflow" / "tfjs").exists():
        pytest.skip("Node.js or tools/tfjs_check/node_modules not installed")
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"x": X.tolist()}))
    subprocess.run([node, str(TFJS_TOOL / "tfjs_predict.js"), str(mj.parent), str(inp), str(out)],
                   check=True, cwd=TFJS_TOOL, capture_output=True, timeout=300)
    return np.array(json.loads(out.read_text())["pred"], np.float32)


def test_real_tfjs_matches_keras(exported, tmp_path):
    model, X, mj = exported
    np.testing.assert_allclose(tfjs_predict(mj, X, tmp_path), model.predict(X, verbose=0), atol=1e-4)
