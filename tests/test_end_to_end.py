"""End-to-end with REAL hand videos in a TEMPORARY fake data-collection folder.

FSL-105 clips are copied (read-only) from the recorder folder and relabelled S1-S7
(tests/fake_dataset.py). The whole pipeline runs through the same functions the app's
buttons start: extract -> split -> train (few epochs) -> evaluate -> export, then the
live test is fed a recorded video through the hidden --video path.
These numbers are a pipeline check only, never real results.

Slow (a few minutes). Skip with:  python -m pytest -m "not slow"
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import fake_dataset as fd
from trainer.settings import PROJECT_DIR

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not fd.real_available(), reason="recorder folder with FSL-105 clips not found")]

CAT = fd.CATEGORY


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data = fd.build(root)
    settings = fd.make_settings(root, data)
    old = os.environ.get("HUDYAT_TRAINER_SETTINGS")
    os.environ["HUDYAT_TRAINER_SETTINGS"] = str(settings)
    from trainer import settings as cfg
    yield {"root": root, "data": data, "settings": settings, "paths": cfg.Paths()}
    if old is None:
        os.environ.pop("HUDYAT_TRAINER_SETTINGS", None)
    else:
        os.environ["HUDYAT_TRAINER_SETTINGS"] = old


def run_cli(project, *args, timeout=900):
    env = dict(os.environ, HUDYAT_TRAINER_SETTINGS=str(project["settings"]), TF_CPP_MIN_LOG_LEVEL="2")
    return subprocess.run([sys.executable, *args], cwd=PROJECT_DIR, env=env, capture_output=True, text=True,
                          timeout=timeout)


def test_1_extract(project):
    from trainer import extract, jobs
    from trainer import dataset as ds
    r = jobs.run_job("extract", extract.run, workers=4)
    paths = project["paths"]
    idx = ds.read_index(paths)
    n_kept = len(fd.SIGNS) * fd.PER_SIGNER * 7 + fd.PER_SIGNER * 7      # colors + none
    assert len(idx) == n_kept == r["extracted"]
    assert list(idx.columns[:13]) == ["file", "category", "sign_id", "signer", "take", "fps", "n_frames",
                                      "trim_start", "trim_end", "frames_kept", "hand_detect_rate",
                                      "both_hands_rate", "status"]
    assert set(idx["signer"]) == set("S1 S2 S3 S4 S5 S6 S7".split())           # no PUB, no needs_rerecord
    assert "alphabet_static_test" not in set(idx["category"])
    for f in idx["file"]:
        a = np.load(paths.work / f)
        assert a.shape == (32, 128) and a.dtype == np.float32
        assert set(np.unique(a[:, [63, 127]])) <= {0.0, 1.0}
    assert (idx["trim_end"].astype(int) >= idx["trim_start"].astype(int)).all()
    # incremental: a second run extracts nothing
    assert jobs.run_job("extract", extract.run, workers=2)["extracted"] == 0
    # nothing was written into the (fake) data-collection folder
    assert not any(p.suffix == ".npy" for p in project["data"].rglob("*"))


def test_2_split(project):
    from trainer import jobs, split
    jobs.run_job("split", split.run)
    sp = json.loads(project["paths"].split_json(CAT).read_text())
    assert sp["labels"] == fd.SIGNS + ["none"]
    for name, signers in (("train", {"S1", "S2", "S3", "S4"}), ("val", {"S5"}), ("test", {"S6", "S7"})):
        assert sp[name] and {it["signer"] for it in sp[name]} <= signers


def test_3_train(project):
    from trainer import jobs, train
    jobs.run_job("train", train.run, [CAT], epochs=6, batch_size=16, aug_factor=2, seed=42)
    d = project["paths"].model_dir(CAT)
    for f in ("model.keras", "labels.json", "config.json", "history.csv", "training_plot.png"):
        assert (d / f).exists(), f
    cfgj = json.loads((d / "config.json").read_text())
    assert cfgj["options"]["seed"] == 42 and cfgj["input_shape"] == [32, 128]
    assert "tensorflow" in cfgj["versions"] and cfgj["counts"]
    with open(d / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert 1 <= len(rows) <= 6 and {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(rows[0])


def test_4_evaluate(project):
    from trainer import evaluate, jobs
    jobs.run_job("evaluate", evaluate.run, [CAT])
    r = project["paths"].report_dir(CAT)
    for f in ("report.md", "metrics.json", "confusion_matrix.png", "confusion_matrix.csv"):
        assert (r / f).exists(), f
    m = json.loads((r / "metrics.json").read_text())
    assert set(m["per_signer"]) <= {"S6", "S7"} and m["per_signer"]
    n = len(m["labels"])
    assert np.array(m["pooled"]["confusion"]).shape == (n, n)
    assert m["ms_per_sample"] > 0 and isinstance(m["pass"], bool)
    with open(project["paths"].summary_csv) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["category"] == CAT and rows[0]["result"] in ("PASS", "FAIL")


def test_5_export_matches_keras(project, tmp_path):
    import keras
    from trainer import jobs, tfjs_export
    from test_tfjs_export import tfjs_predict
    jobs.run_job("export", tfjs_export.run, [CAT])
    d = project["paths"].model_dir(CAT)
    t = d / "tfjs"
    assert {"model.json", "group1-shard1of1.bin", "labels.json", "verify.json"} <= {p.name for p in t.iterdir()}
    assert json.loads((t / "verify.json").read_text())["ok"]
    assert json.loads((t / "labels.json").read_text()) == json.loads((d / "labels.json").read_text())
    sp = json.loads(project["paths"].split_json(CAT).read_text())
    X = np.stack([np.load(project["paths"].work / it["file"]) for it in sp["test"][:5]])
    k = keras.models.load_model(d / "model.keras").predict(X, verbose=0)
    topo, w = tfjs_export.read_tfjs(t / "model.json")
    np.testing.assert_allclose(tfjs_export.forward_numpy(topo, w, X), k, atol=1e-5)
    np.testing.assert_allclose(tfjs_predict(t / "model.json", X, tmp_path), k, atol=1e-4)


def _concat_clips(files, out):
    import cv2
    w = None
    for f in files:
        cap = cv2.VideoCapture(str(f))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if w is None:
                w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fr.shape[1], fr.shape[0]))
            w.write(fr)
        cap.release()
    w.release()


def test_6_live_test_from_video(project):
    """Hidden --video option: same code path as the webcam; one CSV row per attempt."""
    data = project["data"]
    clips = [data / f"dataset/raw/{CAT}/{s}/S6/{s}_S6_01.mp4" for s in fd.SIGNS]
    video = project["root"] / "live_input.mp4"
    _concat_clips(clips, video)
    r = run_cli(project, "live_test.py", "--category", CAT, "--mode", "test", "--tester", "pytest",
                "--target", "red", "--trials", "25", "--video", str(video), "--headless")
    assert r.returncode == 0, r.stdout + r.stderr
    attempts = [ln for ln in r.stdout.splitlines() if ln.startswith("attempt ")]
    assert len(attempts) >= 1
    csv_path = project["paths"].live_tests / f"{CAT}.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert len(rows) == len(attempts)
    for row in rows:
        assert row["target"] == "red" and row["tester"] == "pytest"
        assert row["prediction"] in fd.SIGNS + ["none"]
        assert 0 <= float(row["confidence"]) <= 1 and float(row["response_time_ms"]) >= 0
        assert row["correct"] == str(int(row["prediction"] == "red" and float(row["confidence"]) >= 0.70))


def test_7_cli_scripts_work_without_ui(project):
    r = run_cli(project, "split.py", "--category", CAT)
    assert r.returncode == 0, r.stderr
    r = run_cli(project, "evaluate.py", "--category", CAT)             # refused without --confirm
    assert r.returncode == 2 and "confirm" in r.stderr.lower()
