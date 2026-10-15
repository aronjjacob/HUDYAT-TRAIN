"""Which clips are used: S1-S7 only, status kept, not rejected, detection >= 80% except none,
no alphabet_static_test. The data folder is only read."""

import csv
import json

import pytest

from trainer import dataset as ds
from trainer import settings as cfg

META = ["signer", "category", "sign_id", "take", "file_path", "status", "expert_status"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "dataset").mkdir(parents=True)
    (data / "signs.csv").write_text("category,sign_id,sign_label,reps,hands,mirror_ok\n"
                                    "colors,red,Red,15,1,1\ncolors,blue,Blue,15,1,0\nnone,none,None,10,1,0\n"
                                    "alphabet_static_test,static_a,A,5,1,0\n")
    rows = [("S1", "colors", "red", 1, "kept", ""), ("S1", "colors", "red", 2, "needs_rerecord", ""),
            ("PUB", "colors", "red", 1, "kept", ""), ("S2", "colors", "blue", 1, "kept", "rejected"),
            ("S2", "colors", "blue", 2, "kept", "approved"), ("S3", "none", "none", 1, "kept", ""),
            ("S3", "alphabet_static_test", "static_a", 1, "kept", ""), ("S4", "colors", "blue", 1, "kept", ""),
            ("S4", "colors", "red", 1, "kept", "")]
    with open(data / "dataset" / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(META)
        for g, c, s, t, st, ex in rows:
            w.writerow([g, c, s, t, f"dataset/raw/{c}/{s}/{g}/{s}_{g}_{t:02d}.mp4", st, ex])
    sf = tmp_path / "trainer_settings.json"
    sf.write_text(json.dumps({"data_dir": str(data), "work_dir": str(tmp_path / "work")}))
    monkeypatch.setenv("HUDYAT_TRAINER_SETTINGS", str(sf))
    return data


def test_categories(env):
    signs = ds.load_signs(env)
    assert ds.model_categories(signs) == ["colors"]
    assert ds.extract_categories(signs) == ["colors", "none"]


def test_kept_rows_filters(env):
    k = ds.kept_rows(env)
    got = set(zip(k["signer"], k["sign_id"], k["take"]))
    assert got == {("S1", "red", "1"), ("S2", "blue", "2"), ("S3", "none", "1"), ("S4", "blue", "1"),
                   ("S4", "red", "1")}


def _index(env, rates):
    paths = cfg.Paths()
    rows = []
    for _, m in ds.kept_rows(env).iterrows():
        rows.append({"file": ds.npy_rel_path(m), "category": m["category"], "sign_id": m["sign_id"],
                     "signer": m["signer"], "take": m["take"], "fps": 30, "status": "ok",
                     "hand_detect_rate": rates.get((m["signer"], m["sign_id"]), 0.95), "both_hands_rate": 0,
                     "source_file": m["file_path"]})
    ds.write_index(paths, rows)
    return paths


def test_low_detection_excluded_except_none(env):
    paths = _index(env, {("S4", "blue"): 0.79, ("S3", "none"): 0.0})
    u = ds.usable_samples(paths, env)
    reason = dict(zip(zip(u["signer"], u["sign_id"]), u["excluded_reason"]))
    assert "below 80%" in reason[("S4", "blue")]
    assert reason[("S3", "none")] == ""
    assert reason[("S1", "red")] == ""


def test_validation_rejection_excludes_and_last_decision_wins(env):
    paths = _index(env, {})
    f = "dataset/raw/colors/red/S1/red_S1_01.mp4"
    ds.add_validation(paths, f, "rejected", "wrong handshape", "Ana")
    u = ds.usable_samples(paths, env)
    assert u.set_index("source_file").loc[f, "excluded_reason"] == "rejected in Validation"
    ds.add_validation(paths, f, "approved", "", "Ana")
    u = ds.usable_samples(paths, env)
    assert u.set_index("source_file").loc[f, "excluded_reason"] == ""
    assert paths.validation_csv.parent == paths.work                 # stored in this project
    assert not (env / "work").exists()


def test_npy_path_layout(env):
    k = ds.kept_rows(env)
    r = k[(k["signer"] == "S1")].iloc[0]
    assert ds.npy_rel_path(r) == "landmarks/colors/red/S1/red_S1_01.npy"


def test_static_test_set_only_when_asked(env):
    signs = ds.load_signs(env)
    assert "alphabet_static_test" not in ds.extract_categories(signs)
    assert ds.extract_categories(signs, include_static=True)[-1] == "alphabet_static_test"
    assert "static_a" not in set(ds.kept_rows(env)["sign_id"])
    assert "static_a" in set(ds.kept_rows(env, include_static=True)["sign_id"])


def test_extract_plan_per_category(env):
    from trainer import extract
    paths = cfg.Paths()
    todo, _ = extract.plan(paths, env)
    assert {t["category"] for t in todo} == {"colors", "none"}
    todo, _ = extract.plan(paths, env, categories=["none"])
    assert {t["category"] for t in todo} == {"none"}
    todo, _ = extract.plan(paths, env, categories=["alphabet_static_test"])
    assert [t["sign_id"] for t in todo] == ["static_a"]


def test_extraction_survives_a_crashed_helper(env, monkeypatch):
    """A helper killed (e.g. out of memory) must not stop the job: the rest continues with fewer helpers."""
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    from trainer import extract, jobs

    def fake_worker(job):
        return dict(job["row"], status=ds.STATUS_OK, hand_detect_rate="1.000", _dups=[])

    class CrashingPool:
        calls = 0

        def __init__(self, max_workers):
            CrashingPool.calls += 1
            self.first = CrashingPool.calls == 1

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, job):
            f = Future()
            if self.first and getattr(self, "n", 0) >= 2:
                f.set_exception(BrokenProcessPool("helper killed"))
            else:
                f.set_result(fn(job))
            self.n = getattr(self, "n", 0) + 1
            return f

    monkeypatch.setattr(extract, "_worker", fake_worker)
    monkeypatch.setattr(extract, "ProcessPoolExecutor", CrashingPool)
    monkeypatch.setattr(cfg, "MODEL_PATH", env / "signs.csv")          # any existing file
    r = jobs.run_job("extract", extract.run, workers=4)
    assert r["extracted"] == len(ds.kept_rows(env)) and CrashingPool.calls == 2
    idx = ds.read_index(cfg.Paths())
    assert len(idx) == len(ds.kept_rows(env))
