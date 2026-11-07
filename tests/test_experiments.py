"""Tests 1 and 2: which clips each round trains / validates / tests on, resume, the handedness check, results.

A stand-in for Keras ("fit") keeps these fast: its "model" reads the sign from the Right-hand slot, so it is right
only for right-dominant input. That makes the handedness check predictable."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from trainer import dataset as ds
from trainer import experiments as E
from trainer import features as F
from trainer import settings as cfg
from trainer.experiments import Fitted

SIGNS = ("a", "b", "c")


class Ctx:
    def __init__(self):
        self.lines, self.fractions = [], []

    def progress(self, fraction, text="", **_):
        self.fractions.append(fraction)

    def log(self, msg):
        self.lines.append(msg)


def right_handed_sample(label_id: int) -> np.ndarray:
    """One right hand, landmark 1's y = the sign number (mirroring never changes y), moving a little."""
    x = np.zeros((32, F.RICH_DIM), np.float32)
    x[:, 64 + 3 * 1 + 1] = label_id
    x[:, 64 + 3 * 1] = 0.2
    x[:, 64 + 3 * 9] = 1.0                     # landmark 9 at (1, 0): its 2D length is 1, so the app's scaling keeps y
    x[:, 127] = 1
    x[:, F.R_WRIST] = np.stack([np.linspace(0.5, 0.7, 32), np.full(32, 0.5), np.zeros(32)], 1)
    x[:, F.R_SCALE] = 0.1
    return x


@pytest.fixture
def env(tmp_path, monkeypatch):
    """3 signers; S3 is left-handed (its clips are stored as recorded: the mirror image)."""
    data = tmp_path / "data"
    (data / "dataset").mkdir(parents=True)
    (data / "signs.csv").write_text("category,sign_id,sign_label,reps,hands,mirror_ok\n"
                                    + "".join(f"cat,{s},{s.upper()},4,1,0\n" for s in SIGNS)
                                    + "none,none,None,4,1,0\nother,z,Z,4,1,0\n")
    (data / "signers.csv").write_text("code,hand\nS1,right\nS2,right\nS3,left\n")
    work = tmp_path / "work"
    meta, index = [], []
    for g in ("S1", "S2", "S3", "PUB"):
        hand = "left" if g == "S3" else "right"
        for i, s in enumerate(SIGNS):
            for t in range(4):
                src = f"dataset/raw/cat/{s}/{g}/{s}_{g}_{t:02d}.mp4"
                meta.append({"signer": g, "signing_hand": hand, "category": "cat", "sign_id": s, "take": t,
                             "file_path": src, "status": "kept"})
                if g == "PUB":
                    continue
                rel = f"landmarks/cat/{s}/{g}/{s}_{g}_{t:02d}.npy"
                x = right_handed_sample(i)
                (work / rel).parent.mkdir(parents=True, exist_ok=True)
                np.save(work / rel, F.mirror(x) if hand == "left" else x)
                index.append({"file": rel, "category": "cat", "sign_id": s, "signer": g, "take": t, "status": "ok",
                              "hand_detect_rate": 1.0, "source_file": src, "format": ds.SAMPLE_FORMAT})
    with open(data / "dataset" / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    sf = tmp_path / "trainer_settings.json"
    sf.write_text(json.dumps({"data_dir": str(data), "work_dir": str(work)}))
    monkeypatch.setenv("HUDYAT_TRAINER_SETTINGS", str(sf))
    ds.write_index(cfg.Paths(), index)
    return data


class FakeFit:
    """Records what each round trained on; 'predicts' from the Right-hand slot."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls, self.fail_on_call = [], fail_on_call

    def __call__(self, ctx, train_items, val_items, Xtr, ytr, Xva, yva, labels, opts, mirror_ok, out_dir, tag):
        self.calls.append({"tag": tag, "train": [it["file"] for it in train_items],
                           "val": [it["file"] for it in val_items], "Xtr": Xtr.copy(), "mirror_ok": dict(mirror_ok)})
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("stopped")
        n = len(labels)

        def predict(X):
            X = np.asarray(X)
            right = X[:, :, 127].mean(1) > 0.5
            y_of = np.where(right[:, None], X[:, :, 64 + 4], X[:, :, 4])       # landmark 1's y of the hand shown
            lab = np.clip(np.round(y_of.mean(1)), 0, n - 1).astype(int)
            pred = np.where(right, lab, (lab + 1) % n)
            return np.eye(n, dtype=np.float32)[pred]
        hist = {"loss": [1.0, 0.5], "accuracy": [0.5, 0.9], "val_loss": [1.1, 0.6], "val_accuracy": [0.4, 0.8]}
        return Fitted(predict, hist, None)


def add_old_none_samples(data):
    """A work folder from before: none clips in metadata and extracted none samples in the index."""
    meta = pd.read_csv(data / "dataset" / "metadata.csv", dtype=str)
    paths = cfg.Paths()
    index = ds.read_index(paths).to_dict("records")
    rows = []
    for g in ("S1", "S2", "S3"):
        for t in range(4):
            src = f"dataset/raw/none/none/{g}/none_{g}_{t:02d}.mp4"
            rel = f"landmarks/none/none/{g}/none_{g}_{t:02d}.npy"
            rows.append({"signer": g, "signing_hand": "right", "category": "none", "sign_id": "none", "take": t,
                         "file_path": src, "status": "kept"})
            (paths.work / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(paths.work / rel, right_handed_sample(0))
            index.append({"file": rel, "category": "none", "sign_id": "none", "signer": g, "take": t, "status": "ok",
                          "hand_detect_rate": 1.0, "source_file": src, "format": ds.SAMPLE_FORMAT})
    pd.concat([meta, pd.DataFrame(rows)]).to_csv(data / "dataset" / "metadata.csv", index=False)
    ds.write_index(paths, index)


# --- clips --------------------------------------------------------------------------------------
def test_clip_table_ignores_none_clips(env):
    add_old_none_samples(env)
    clips, labels = E.clip_table("cat")
    assert labels == list(SIGNS) and "none" not in set(clips["label"])
    assert not clips["file"].str.contains("/none/").any()
    with pytest.raises(ds.PlainError):
        E.clip_table("none")                                          # none is not a category with a model


def test_clip_table_hands_and_labels(env):
    clips, labels = E.clip_table("cat")
    assert labels == list(SIGNS)                                      # only the signs: no none class
    assert set(clips["signer"]) == {"S1", "S2", "S3"}                 # PUB never
    assert set(clips[clips["signer"] == "S3"]["hand"]) == {"left"}
    assert set(clips[clips["signer"] != "S3"]["hand"]) == {"right"}


def test_clip_table_refuses_old_128_samples(env):
    paths = cfg.Paths()
    idx = ds.read_index(paths)
    idx.loc[0, "format"] = ""
    ds.write_index(paths, idx.to_dict("records"))
    with pytest.raises(ds.PlainError, match="before the new sample format") as e:
        E.clip_table("cat")
    assert "Extract landmarks" in e.value.hint


# --- Test 2 (LOSO) ------------------------------------------------------------------------------
def test_loso_rounds_never_train_on_the_held_out_signer(env):
    fit = FakeFit()
    res = E.run(Ctx(), "loso", "cat", {"seed": 3}, fit=fit)
    assert [c["tag"] for c in fit.calls] == ["S1", "S2", "S3"]
    for c in fit.calls:
        assert not any(f"/{c['tag']}/" in f for f in c["train"] + c["val"])
        assert not set(c["train"]) & set(c["val"])
    out = json.loads((E.run_dir(cfg.Paths(), "loso", "cat", res["run_id"]) / "results.json").read_text())
    assert [r["held_out"] for r in out["rounds"]] == ["S1", "S2", "S3"]
    d = E.run_dir(cfg.Paths(), "loso", "cat", res["run_id"])
    for name in ("round_accuracy", "confusion_S1", "confusion_S3_normalized"):
        assert (d / "figures" / f"{name}.png").exists()


def test_canonical_mirrors_left_handed_training_clips(env):
    fit = FakeFit()
    E.run(Ctx(), "loso", "cat", {"seed": 3, "canonical": True}, fit=fit)
    for c in fit.calls:                                    # every training sample is right-dominant
        assert (c["Xtr"][:, :, 127] == 1).all() and not c["Xtr"][:, :, 63].any()
    fit2 = FakeFit()
    E.run(Ctx(), "loso", "cat", {"seed": 3, "canonical": False}, fit=fit2)
    assert any(c["Xtr"][:, :, 63].any() for c in fit2.calls)           # S3 stays as recorded


def test_mirror_augmentation_is_off_by_default_and_global_when_on(env):
    fit = FakeFit()
    E.run(Ctx(), "random", "cat", {"seed": 1}, fit=fit)
    assert not any(fit.calls[0]["mirror_ok"].values())
    fit = FakeFit()
    E.run(Ctx(), "random", "cat", {"seed": 1, "mirror_aug": True}, fit=fit)
    assert all(fit.calls[0]["mirror_ok"].values())                     # no sign has mirror_ok yet: all signs


def test_results_mean_sd_and_handedness_check(env):
    res = E.run(Ctx(), "loso", "cat", {"seed": 3, "canonical": True}, fit=FakeFit())
    out = E.load_results(E.run_dir(cfg.Paths(), "loso", "cat", res["run_id"]))
    assert out["summary"]["accuracy"]["mean"] == pytest.approx(1.0)
    assert out["summary"]["accuracy"]["sd"] == pytest.approx(0.0)
    assert out["pass"] is True
    h = out["handedness"]
    assert h["recorded"]["auto"]["accuracy"]["mean"] == pytest.approx(1.0)
    assert h["mirrored"]["auto"]["accuracy"]["mean"] == pytest.approx(1.0)    # Auto finds the hand, mirror fixes it
    s1 = next(r for r in out["rounds"] if r["held_out"] == "S1")
    assert s1["handedness"]["mirrored"]["none"]["accuracy"] == pytest.approx(0.0)   # right-handed, mirrored, no fix
    s3 = next(r for r in out["rounds"] if r["held_out"] == "S3")              # the left-handed signer as recorded
    assert s3["handedness"]["recorded"]["none"]["accuracy"] == pytest.approx(0.0)
    assert s3["handedness"]["recorded"]["auto"]["accuracy"] == pytest.approx(1.0)
    assert h["auto_correct"]["recorded"] == pytest.approx(1.0) and h["auto_correct"]["mirrored"] == pytest.approx(1.0)
    assert len(out["pooled"]["confusion"]) == len(SIGNS)


def test_loso_resume_skips_finished_rounds(env):
    with pytest.raises(RuntimeError):
        E.run(Ctx(), "loso", "cat", {"seed": 3}, fit=FakeFit(fail_on_call=2))
    run_id = E.list_runs(cfg.Paths())[0]["run_id"]
    assert E.list_runs(cfg.Paths())[0]["state"] == "stopped"
    fit = FakeFit()
    E.run(Ctx(), "loso", "cat", resume=run_id, fit=fit)
    assert [c["tag"] for c in fit.calls] == ["S2", "S3"]
    assert E.list_runs(cfg.Paths())[0]["state"] == "done"


def test_a_run_from_before_the_apps_features_cannot_be_resumed(env):
    from trainer import dataset as ds
    with pytest.raises(RuntimeError):
        E.run(Ctx(), "loso", "cat", {"seed": 3}, fit=FakeFit(fail_on_call=2))
    run = E.list_runs(cfg.Paths())[0]
    p = E.find_run(cfg.Paths(), run["run_id"]) / "config.json"
    conf = json.loads(p.read_text())
    assert conf["normalization"] == "hand-relative-v1"
    del conf["normalization"]
    conf["created"] = "2026-10-20T15:00:00"
    p.write_text(json.dumps(conf))
    with pytest.raises(ds.PlainError, match="older features"):
        E.run(Ctx(), "loso", "cat", resume=run["run_id"], fit=FakeFit())


# --- Test 1 -------------------------------------------------------------------------------------
def test_random_split_run_saves_config_and_results(env):
    res = E.run(Ctx(), "random", "cat", {"seed": 5}, fit=FakeFit())
    d = E.run_dir(cfg.Paths(), "random", "cat", res["run_id"])
    conf = json.loads((d / "config.json").read_text())
    assert conf["opts"]["seed"] == 5 and conf["opts"]["feature_version"] == 1
    assert conf["labels"] == list(SIGNS) and conf["signers"] == ["S1", "S2", "S3"]
    r = conf["rounds"][0]
    files = [it["file"] for p in ("train", "val", "test") for it in r[p]]
    assert len(files) == len(set(files)) == 36
    assert set(r["counts"]) == {"train", "val", "test"}
    assert "versions" in conf
    out = E.load_results(d)
    assert len(out["rounds"]) == 1 and out["summary"]["accuracy"]["sd"] is None
    assert (d / "round_all" / "predictions.csv").exists()
    assert (d / "round_all" / "history.csv").exists()
    for name in ("confusion_summed", "confusion_summed_normalized", "per_sign_recall", "curves", "handedness"):
        assert (d / "figures" / f"{name}.png").stat().st_size > 1000


def test_a_feature_v2_run_records_its_normalization_and_trains_on_v2(env):
    fit = FakeFit()
    res = E.run(Ctx(), "random", "cat", {"seed": 5, "feature_version": 2}, fit=fit)
    conf = json.loads((E.run_dir(cfg.Paths(), "random", "cat", res["run_id"]) / "config.json").read_text())
    assert conf["opts"]["feature_version"] == 2 and conf["normalization"] == "hand-relative-sequence-v1"
    assert res["run_id"].endswith("-v2") and E.list_runs(cfg.Paths())[0]["feature_version"] == 2


def test_list_runs_for_compare(env):
    E.run(Ctx(), "loso", "cat", {"seed": 3}, fit=FakeFit())
    E.run(Ctx(), "loso", "cat", {"seed": 4}, fit=FakeFit())
    runs = E.list_runs(cfg.Paths(), category="cat")
    assert sorted(r["seed"] for r in runs) == [3, 4] and {r["feature_version"] for r in runs} == {1}
    assert all(r["state"] == "done" and r["accuracy_mean"] == pytest.approx(1.0) for r in runs)
    assert {r["test"] for r in runs} == {"loso"}


def test_photo_dataset_offers_only_test_1(env):
    paths = cfg.Paths()
    idx = ds.read_index(paths)
    with pytest.raises(E.SplitError):
        E.check_test_possible(pd.DataFrame({"file": ["x"], "label": ["a"], "signer": ["IMP"]}), "loso")
    E.check_test_possible(E.clip_table("cat")[0], "loso")
    assert not idx.empty


def test_real_keras_tiny_run_and_final_model(env):
    """The real training function end to end on the tiny dataset (2 epochs): files, shapes, a loadable model."""
    import keras
    from trainer import final
    res = E.run(Ctx(), "random", "cat", {"seed": 2, "epochs": 2, "aug_factor": 1, "batch_size": 8})
    out = E.load_results(E.run_dir(cfg.Paths(), "random", "cat", res["run_id"]))
    assert out["complete"] and len(out["histories"]["all"]["loss"]) == 2
    final.run(Ctx(), "cat", from_run=res["run_id"])
    m = keras.models.load_model(cfg.Paths().model_dir("cat") / "model.keras")
    assert m.input_shape == (None, 32, 128) and m.output_shape == (None, len(SIGNS))
