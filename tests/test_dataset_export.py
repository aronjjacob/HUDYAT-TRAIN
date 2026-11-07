"""Export a category's landmarks (.npy) as a dataset, then open that export again and train on it without videos.
The recordings-style dataset it came from keeps working, untouched."""

import csv
import json
import zipfile

import numpy as np
import pytest

from test_experiments import Ctx, FakeFit, env  # noqa: F401  (env is a fixture)
from trainer import dataset as ds
from trainer import dataset_export
from trainer import experiments as E
from trainer import extract
from trainer import settings as cfg
from trainer.upload import import_zip, scan_zip

REJECTED = "dataset/raw/cat/a/S1/a_S1_00.mp4"
OLD = "dataset/raw/cat/b/S2/b_S2_01.mp4"


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


@pytest.fixture
def exported(env):  # noqa: F811
    paths = cfg.Paths()
    ds.add_validation(paths, REJECTED, "rejected", "wrong handshape", "Expert A")
    idx = ds.read_index(paths).to_dict("records")
    for r in idx:
        if r["source_file"] == OLD:
            r["format"] = "rich136"
    ds.write_index(paths, idx)
    res = dataset_export.run(Ctx(), ["cat"])
    out = paths.datasets / res["datasets"][0]
    return env, paths, out


def test_export_has_only_usable_current_samples_and_says_why_others_are_left_out(exported):
    data, paths, out = exported
    meta = read(out / "dataset" / "metadata.csv")
    files = {r["file_path"] for r in meta}
    assert len(meta) == 3 * 3 * 4 - 2 and REJECTED not in files and OLD not in files
    assert all(r["signer"] in ("S1", "S2", "S3") for r in meta)              # never PUB
    assert "review" not in meta[0] and not (out / "validation.csv").exists()     # training data only
    why = {r["source_file"]: r for r in read(out / "excluded.csv")}
    assert why[REJECTED]["reason"] == "rejected in Validation" and "reviewer" not in why[REJECTED]
    assert why[OLD]["reason"].startswith("old format")
    idx = read(out / "landmarks_index.csv")
    assert {r["source_file"] for r in idx} == files and len(list(out.rglob("*.npy"))) == len(meta)
    for r in idx[:3]:                                                          # copied unchanged
        assert np.array_equal(np.load(out / r["file"]), np.load(paths.work / r["file"]))
    info = json.loads((out / "dataset.json").read_text())
    assert info["landmarks_only"] and [c["sign_id"] for c in info["classes"]] == ["a", "b", "c"]
    assert info["sample"]["shape"] == [32, 149] and info["n_samples"] == len(meta) and info["n_excluded"] == 2
    assert (out / "README.txt").exists() and not list(paths.datasets.glob("*.tmp"))
    with zipfile.ZipFile(paths.datasets / f"{out.name}.zip") as z:
        assert "dataset.json" in z.namelist()                                 # no extra folder: short paths
    assert dataset_export.list_exports(paths, "cat")[0]["name"] == out.name


def test_nothing_to_export_is_a_plain_error(env):  # noqa: F811
    paths = cfg.Paths()
    ds.write_index(paths, [])
    with pytest.raises(ds.PlainError):
        dataset_export.run(Ctx(), ["cat"])


def test_the_export_trains_without_videos_and_the_recordings_keep_working(exported):
    data, paths, out = exported
    zp = paths.datasets / f"{out.name}.zip"
    with open(zp, "rb") as f:
        assert scan_zip(f).layout == "landmarks only"
    with open(zp, "rb") as f:
        res = import_zip(f, zp.name, cfg.load_settings())
    assert res.layout == "landmarks only" and cfg.landmarks_only(res.root)
    n = len(read(out / "dataset" / "metadata.csv"))

    cfg.save_settings({"data_dir": str(res.root)})                            # use the export
    p2 = cfg.Paths()
    assert p2.work != paths.work and cfg.check_data_dir(res.root) == []
    todo, _ = extract.plan(p2, res.root)
    assert todo == []                                                          # nothing to extract
    u = ds.usable_samples(p2, res.root)
    assert len(u) == n and (u["excluded_reason"] == "").all()
    with pytest.raises(ds.PlainError, match="landmarks only"):
        extract.run(Ctx())                                                     # no videos: never re-extract
    clips, labels = E.clip_table("cat")
    assert len(clips) == n and labels == ["a", "b", "c"]
    r = E.run(Ctx(), "loso", "cat", {"seed": 3}, fit=FakeFit())
    assert r["run_id"]

    cfg.save_settings({"data_dir": str(data)})                                # back to the recordings
    assert cfg.Paths().work == paths.work and not cfg.landmarks_only(data)
    assert len(ds.read_index(paths)) == 3 * 3 * 4                             # untouched by the import
    assert not E.list_runs(paths)                                              # the export's run stays with it
    with pytest.raises(ds.PlainError, match="new sample format"):
        E.clip_table("cat")                                                    # the recordings still ask to re-extract OLD
