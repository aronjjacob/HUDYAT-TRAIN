"""End-to-end with REAL hand videos in a TEMPORARY data-collection folder (tests/fake_dataset.py).

Our S1-S3 recordings are copied (read-only), S3 marked left-handed with mirrored videos. The whole pipeline runs
through the same functions the app's buttons start, with few epochs:
  Preprocess -> Test 1 -> Test 2 -> Final model -> Export for the app (its own loader vs Keras) -> Live (--video) -> Paper.
These numbers are a pipeline check only, never real results.

Slow (several minutes). Skip with:  python -m pytest -m "not slow"
"""

import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import fake_dataset as fd
from trainer.settings import PROJECT_DIR

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not fd.real_available(), reason="recorder folder with S1-S3 clips not found")]

CAT = fd.CATEGORY
FAST = {"epochs": 12, "batch_size": 16, "aug_factor": 2, "patience": 6, "seed": 42}
SHOTS = PROJECT_DIR / "docs" / "screenshots" / "v2"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data = fd.build(root)
    settings = fd.make_settings(root, data)
    old = os.environ.get("HUDYAT_TRAINER_SETTINGS")
    os.environ["HUDYAT_TRAINER_SETTINGS"] = str(settings)
    from trainer import settings as cfg
    yield {"root": root, "data": data, "settings": settings, "paths": cfg.Paths(), "runs": {}}
    if old is None:
        os.environ.pop("HUDYAT_TRAINER_SETTINGS", None)
    else:
        os.environ["HUDYAT_TRAINER_SETTINGS"] = old


def run_cli(project, *args, timeout=900):
    env = dict(os.environ, HUDYAT_TRAINER_SETTINGS=str(project["settings"]), TF_CPP_MIN_LOG_LEVEL="2")
    return subprocess.run([sys.executable, *args], cwd=PROJECT_DIR, env=env, capture_output=True, text=True,
                          timeout=timeout)


def test_1_preprocess(project):
    from trainer import dataset as ds
    from trainer import extract, jobs
    r = jobs.run_job("extract", extract.run, workers=4)
    idx = ds.read_index(project["paths"])
    n = len(fd.SIGNS) * len(fd.SIGNERS) * fd.PER_SIGNER                          # signs only: none clips ignored
    assert len(idx) == n == r["extracted"]
    assert set(idx["signer"]) == set(fd.SIGNERS)                                  # no PUB, no needs_rerecord
    assert (idx["category"] != "none").all()
    assert (idx["format"] == ds.SAMPLE_FORMAT).all()
    for f in idx["file"][:5]:
        a = np.load(project["paths"].work / f)
        assert a.shape == (32, 149) and a.dtype == np.float32
    assert jobs.run_job("extract", extract.run)["extracted"] == 0                  # incremental


def test_2_hands_per_signer(project):
    """The mirrored S3 videos really are left-handed: Auto finds the left hand for most of S3's sign clips."""
    import pandas as pd
    from trainer import dataset as ds
    from trainer.handstats import hand_stats
    idx = ds.read_index(project["paths"])
    idx = idx[idx["category"] == CAT]
    hands = ds.clip_hands(project["data"])
    clips = pd.DataFrame({"file": idx["file"], "signer": idx["signer"],
                          "hand": [hands[f] for f in idx["source_file"]], "source_file": idx["source_file"]})
    per, bad = hand_stats(project["paths"].work, clips)
    s3 = per.set_index("signer").loc[fd.LEFT]
    assert s3["hand"] == "left" and s3["left_pct"] > s3["right_pct"]
    assert (bad["signer"] == fd.LEFT).sum() <= fd.PER_SIGNER                      # few mismatches, if any


def test_3_test1_random(project):
    from trainer import experiments as E
    from trainer import jobs
    r = jobs.run_job("experiment", E.run, "random", CAT, dict(FAST))
    d = E.run_dir(project["paths"], "random", CAT, r["run_id"])
    conf, res = E.load_config(d), E.load_results(d)
    assert res["complete"] and len(res["rounds"]) == 1
    assert conf["opts"]["seed"] == 42 and "tensorflow" in conf["versions"]
    files = [it["file"] for p in ("train", "val", "test") for it in conf["rounds"][0][p]]
    assert len(files) == len(set(files))
    assert (d / "figures" / "confusion_summed.png").exists() and (d / "figures" / "curves.png").exists()
    project["runs"]["random"] = r["run_id"]


def test_4_test2_loso(project):
    from trainer import experiments as E
    from trainer import jobs
    r = jobs.run_job("experiment", E.run, "loso", CAT, dict(FAST))
    d = E.run_dir(project["paths"], "loso", CAT, r["run_id"])
    conf, res = E.load_config(d), E.load_results(d)
    assert [x["held_out"] for x in res["rounds"]] == fd.SIGNERS
    for rd in conf["rounds"]:
        assert rd["held_out"] not in {it["signer"] for it in rd["train"] + rd["val"]}
        assert {it["signer"] for it in rd["test"]} == {rd["held_out"]}
    assert res["summary"]["accuracy"]["sd"] is not None
    h = res["handedness"]
    assert h["auto_correct"]["recorded"] >= 0.8                       # Auto finds the signing hand on real clips
    s3 = next(x for x in res["rounds"] if x["held_out"] == fd.LEFT)
    print(f"\nLOSO mean {res['summary']['accuracy']['mean']:.1%}; S3 (left-handed) as recorded: "
          f"without canonical {s3['handedness']['recorded']['none']['accuracy']:.1%}, with Auto "
          f"{s3['handedness']['recorded']['auto']['accuracy']:.1%}")
    project["runs"]["loso"] = r["run_id"]


def test_5_final_model_and_export(project, tmp_path):
    import keras
    from test_app_export import TFJS_TOOL
    from trainer import app_export, final, jobs
    run_id = project["runs"]["loso"]
    jobs.run_job("train", final.run, CAT, from_run=run_id)
    d = project["paths"].model_dir(CAT)
    conf = json.loads((d / "config.json").read_text())
    assert conf["from_run"] == run_id and conf["signers"] == fd.SIGNERS and conf["input_shape"] == [32, 128]
    assert conf["labels"] == fd.SIGNS                          # the dataset has none clips: ignored, N signs -> N outputs
    assert keras.models.load_model(d / "model.keras").output_shape == (None, len(fd.SIGNS))
    pre = json.loads((d / "preprocess.json").read_text())
    assert pre["feature_version"] == 1 and pre["handedness"]["canonical"] and pre["labels"] == conf["labels"]
    jobs.run_job("export", app_export.run, [CAT])
    t = project["paths"].exports / app_export.export_name(project["paths"], CAT, 1)
    assert t.name == f"{CAT}-dabest-recorder-v1"
    assert {"model.json", "weights.bin", "parity.json", "catalog_entry.json", "preprocessing.json"} <= \
        {p.name for p in t.iterdir()}
    chk = json.loads((t / "verify.json").read_text())
    assert chk["ok"] and chk["real_clips"]
    node = shutil.which("node")
    if node and (TFJS_TOOL / "node_modules" / "@tensorflow" / "tfjs").exists():
        out = tmp_path / "out.json"
        subprocess.run([node, str(TFJS_TOOL / "app_predict.js"), str(t), str(out)], check=True, cwd=TFJS_TOOL,
                       capture_output=True, timeout=300)
        np.testing.assert_allclose(np.array(json.loads(out.read_text())["pred"]),
                                   np.array(json.loads((t / "parity.json").read_text())["expected"]), atol=2e-4)


def test_6_live_window_from_video(project):
    """Hidden --video: the same code path as the webcam. A right-handed clip, then its mirrored copy with Hand = Auto
    (the window must detect and mirror the left hand). Screens are saved for the docs."""
    data = project["data"]
    right = data / f"dataset/raw/{CAT}/hello/S1/hello_S1_01.mp4"
    mirrored = project["root"] / "hello_S1_01_mirrored.mp4"
    fd.mirror_video(right, mirrored)
    shots = project["root"] / "shots"
    r = run_cli(project, "live_test.py", "--category", CAT, "--video", str(right), "--hand", "auto", "--headless",
                "--screenshot", str(shots / "free"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "hand right" in r.stdout
    r = run_cli(project, "live_test.py", "--category", CAT, "--mode", "test", "--tester", "pytest", "--tester-hand",
                "left", "--target", "hello", "--trials", "1", "--video", str(mirrored), "--hand", "auto",
                "--headless", "--screenshot", str(shots / "test"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "hand left" in r.stdout                                    # Auto found the left hand and mirrored it
    rows = list(csv.DictReader(open(project["paths"].live_tests / f"{CAT}.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["tester_hand"] == "left" and rows[0]["feature_version"] == "1"
    assert rows[0]["model_run_id"].startswith(f"final-{CAT}")
    # the HUDYAT app's exported model on the same left-handed video: the fixed capture mirrors it, the app's does not
    for capture, hand in (("fixed", "hand left"), ("app", "hand none")):
        r = run_cli(project, "live_test.py", "--category", CAT, "--video", str(mirrored), "--hand", "auto",
                    "--headless", "--recognizer", "app", "--capture", capture)
        assert r.returncode == 0, r.stdout + r.stderr
        assert hand in r.stdout and "App model" in r.stdout
    SHOTS.mkdir(parents=True, exist_ok=True)
    for src, name in ((shots / "free" / "free.png", "live_free.png"), (shots / "test" / "summary.png",
                                                                       "live_summary.png")):
        assert src.exists(), src
        shutil.copy2(src, SHOTS / name)


def test_7_test_mode_screen(project):
    """A two-trial run, so the Test screen (with a result) is saved before the summary."""
    data = project["data"]
    shots = project["root"] / "shots" / "test2"
    video = project["root"] / "two_signs.mp4"
    import cv2
    w = None
    for f in (data / f"dataset/raw/{CAT}/hello/S2/hello_S2_01.mp4", data / f"dataset/raw/{CAT}/hello/S2/hello_S2_02.mp4"):
        cap = cv2.VideoCapture(str(f))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if w is None:
                w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (fr.shape[1], fr.shape[0]))
            w.write(fr)
        cap.release()
    w.release()
    r = run_cli(project, "live_test.py", "--category", CAT, "--mode", "test", "--tester", "pytest", "--target",
                "hello", "--trials", "3", "--video", str(video), "--headless", "--screenshot", str(shots))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (shots / "test.png").exists()
    shutil.copy2(shots / "test.png", SHOTS / "live_test.png")


def test_8_paper(project):
    from trainer import jobs, paper
    out = jobs.run_job("paper", paper.run, CAT)
    d = paper.out_dir(project["paths"], CAT)
    for name in ("dataset_clips", "test1_recall", "test2_round_accuracy", "handedness",
                 "live_accuracy", "benchmark"):
        assert (d / "figures" / f"{name}.png").exists() and (d / "figures" / f"{name}.pdf").exists(), name
    s = (d / "results_summary.md").read_text(encoding="utf-8")
    assert project["runs"]["loso"] in s
    for t in ("table6_random", "table7_loso", "table7_loso_per_fold", "errors_loso"):
        assert (d / "tables" / f"{t}.csv").exists(), t
    assert out["zip"].endswith("paper.zip")


def test_9_cli_without_ui(project):
    r = run_cli(project, "experiment.py", "--test", "loso")                        # no category: plain error
    assert r.returncode == 2 and "category" in r.stderr.lower()
    r = run_cli(project, "experiment.py", "--help")
    assert r.returncode == 0 and "--no-canonical" in r.stdout
