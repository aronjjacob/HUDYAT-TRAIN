"""Combining datasets: photos A-C (upload) + J/Z videos (recordings) into one model."""

import csv

import numpy as np
import pytest

from trainer import dataset as ds
from trainer import recorder
from trainer import settings as cfg
from trainer.combine import Part, combine, default_leave_out, items_of
from trainer.split import build_split
from trainer.upload import import_folder


@pytest.fixture
def two(tmp_path):
    s = {"work_dir": str(tmp_path / "work")}
    # upload: photo folders A, B, J (J as a still photo)
    src = tmp_path / "photos"
    for letter in "ABJ":
        (src / letter).mkdir(parents=True)
        for i in range(10):
            (src / letter / f"{i}.jpg").write_bytes(b"x")
    up = import_folder(src, "photos", s).root
    # pretend the photos were extracted: index + samples in the upload's own work folder
    su = dict(s, data_dir=str(up))
    pu = cfg.Paths(su)
    rows = []
    for _, m in ds.kept_rows(up).iterrows():
        rel = ds.npy_rel_path(m)
        (pu.work / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(pu.work / rel, np.ones((32, 128), np.float32))
        rows.append({"file": rel, "category": m["category"], "sign_id": m["sign_id"], "signer": m["signer"],
                     "status": "ok", "hand_detect_rate": 1.0, "source_file": m["file_path"], "split": m["split"],
                     "preprocess": recorder.params_id(recorder.default_params())})
    ds.write_index(pu, rows)
    # recordings: alphabet_dynamic J, Z + none, signers S1 and S5
    rec = tmp_path / "rec"
    (rec / "dataset").mkdir(parents=True)
    (rec / "signs.csv").write_text("category,sign_id,sign_label,reps,hands,mirror_ok\n"
                                   "alphabet_dynamic,letter_j,J,15,1,0\nalphabet_dynamic,letter_z,Z,15,1,0\n"
                                   "none,none,None,10,1,0\n")
    with open(rec / "dataset" / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signer", "category", "sign_id", "take", "file_path", "status"])
        for g in ("S1", "S5"):
            for c, sid in (("alphabet_dynamic", "letter_j"), ("alphabet_dynamic", "letter_z"), ("none", "none")):
                w.writerow([g, c, sid, 1, f"dataset/raw/{c}/{sid}/{g}/{sid}_{g}_01.mp4", "kept"])
    return s, up, rec


def test_items_and_default_leave_out(two):
    s, up, rec = two
    items = items_of(Part(up, "photos")) + items_of(Part(rec, "alphabet_dynamic"))
    assert [(i.label, i.kind, i.n_files) for i in items] == [("A", "photos", 10), ("B", "photos", 10),
                                                              ("J", "photos", 10), ("J", "videos", 2),
                                                              ("Z", "videos", 2)]
    leave = default_leave_out(items)
    assert {i.label for i in items if i.id in leave} == {"J"} and all(
        i.kind == "photos" for i in items if i.id in leave)


def test_combine_merges_reuses_and_keeps_splits(two):
    s, up, rec = two
    items = items_of(Part(up, "photos")) + items_of(Part(rec, "alphabet_dynamic"))
    leave = default_leave_out(items)
    r = combine("alphabet", [i for i in items if i.id not in leave], none_from=rec, settings=s)
    assert r.name == "alphabet" and cfg.check_data_dir(r.root) == []
    signs = ds.load_signs(r.root)
    assert [(x.category, x.sign_id, x.sign_label) for x in signs] == [
        ("alphabet", "a", "A"), ("alphabet", "b", "B"), ("alphabet", "j", "J"), ("alphabet", "z", "Z"),
        ("none", "none", "None")]
    kept = ds.kept_rows(r.root)
    assert len(kept) == 20 + 4 + 2                                            # A, B photos + J, Z videos + none
    assert r.n_reused == 20                                                   # the photo samples, not re-extracted
    j = kept[kept["sign_id"] == "j"]
    assert set(j["signer"]) == {"S1", "S5"} and set(j["split"]) == {""}        # recordings: split by signer
    sc = dict(s, data_dir=str(r.root))
    u = ds.usable_samples(cfg.Paths(sc), r.root, sc)
    assert len(u) == 20 and set(u["sign_id"]) == {"a", "b"}
    sp = build_split("alphabet", u, signs)
    assert sp["labels"] == ["a", "b", "j", "z", "none"]
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == 20
    # the source datasets are untouched
    assert len(ds.kept_rows(up)) == 30 and not (rec / "work").exists()
