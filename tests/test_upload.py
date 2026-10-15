"""Uploading a dataset (.zip or folder): copied into <work>/imports/<name>/data and made usable."""

import csv
import io
import zipfile
from collections import Counter

import pytest

from trainer import dataset as ds
from trainer import settings as cfg
from trainer.dataset import PlainError
from trainer.split import build_split
from trainer.upload import assign_splits, group_key, import_folder, import_zip, list_imports


def make_zip(files: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf


@pytest.fixture
def s(tmp_path):
    return {"work_dir": str(tmp_path / "work")}


def rows(root):
    return list(csv.DictReader(open(root / "dataset" / "metadata.csv", encoding="utf-8")))


def test_copy_of_recorder_folder_used_as_is(s, tmp_path):
    meta = "signer,category,sign_id,take,file_path,status\nS1,colors,red,1,dataset/raw/colors/red/S1/red_S1_01.mp4,kept\n"
    z = make_zip({"HUDYAT - HAND TRACKER/signs.csv": "category,sign_id,sign_label,reps,hands,mirror_ok\ncolors,red,Red,15,1,1\n",
                  "HUDYAT - HAND TRACKER/dataset/metadata.csv": meta,
                  "HUDYAT - HAND TRACKER/dataset/raw/colors/red/S1/red_S1_01.mp4": b"x"})
    r = import_zip(z, "HAND TRACKER.zip", s)
    assert r.name == "hand_tracker" and r.layout == "recorder copy"
    assert r.root == tmp_path / "work" / "imports" / "hand_tracker" / "data" / "HUDYAT - HAND TRACKER"
    assert (r.root / "dataset" / "metadata.csv").read_text() == meta          # untouched
    assert cfg.check_data_dir(r.root) == [] and r.n_used == 1
    assert ds.load_signs(r.root)[0].mirror_ok == 1


def test_videos_per_signer_builds_metadata_and_signs(s):
    files = {f"raw/{c}/{g_s}/{g}/{g_s}_{g}_{t:02d}.mp4": b"x"
             for c, g_s in (("colors", "red"), ("colors", "blue"), ("none", "none"))
             for g in ("S1", "S5", "PUB") for t in (1, 2)}
    r = import_zip(make_zip(files), "raw.zip", s)
    assert r.layout.startswith("videos per signer") and cfg.check_data_dir(r.root) == []
    rr = rows(r.root)
    assert len(rr) == 12 and {x["signer"] for x in rr} == {"S1", "S5"}      # PUB folders ignored
    assert {x["take"] for x in rr} == {"1", "2"} and all(x["split"] == "" for x in rr)
    assert [(x.category, x.sign_id) for x in ds.load_signs(r.root)] == [("colors", "blue"), ("colors", "red"),
                                                                          ("none", "none")]
    assert any("PUB" in n for n in r.notes)
    assert len(ds.kept_rows(r.root)) == 12


def test_class_folders_of_photos(s, tmp_path):
    src = tmp_path / "alphabet-static" / "Collated"
    for letter in "ABC":
        (src / letter).mkdir(parents=True)
        for i in range(20):
            (src / letter / f"{i}_jpg.rf.aa{i}.jpg").write_bytes(b"x")
            (src / letter / f"{i}_jpg.rf.bb{i}.jpg").write_bytes(b"x")        # a copy of the same original
    (src / "notes.txt").write_text("ignored")
    r = import_folder(tmp_path / "alphabet-static", None, s)
    assert r.name == "alphabet_static" and r.layout.startswith("one folder per sign")
    assert (tmp_path / "alphabet-static" / "Collated" / "A").exists()             # source untouched
    assert cfg.check_data_dir(r.root) == []
    signs = ds.load_signs(r.root)
    assert [(x.category, x.sign_id, x.sign_label) for x in signs] == [("alphabet_static", "a", "A"),
                                                                       ("alphabet_static", "b", "B"),
                                                                       ("alphabet_static", "c", "C")]
    rr = rows(r.root)
    assert len(rr) == 120 and {x["signer"] for x in rr} == {"IMP"} and {x["kind"] for x in rr} == {"image"}
    per = Counter(x["split"] for x in rr)
    assert per["train"] == 84 and per["val"] == 18 and per["test"] == 18
    # copies of one original are always in the same split
    by_group = {}
    for x in rr:
        by_group.setdefault((x["sign_id"], group_key(x["file_path"].rsplit("/", 1)[1].rsplit(".", 1)[0])), set()).add(x["split"])
    assert all(len(v) == 1 for v in by_group.values())
    assert len(ds.kept_rows(r.root)) == 120                                       # usable without signers S1-S7
    # the split step uses the fixed split; no none class in this dataset
    kept = ds.kept_rows(r.root)
    usable = kept.assign(file=[ds.npy_rel_path(x) for _, x in kept.iterrows()], excluded_reason="")
    sp = build_split("alphabet_static", usable, signs)
    assert sp["labels"] == ["a", "b", "c"] and len(sp["train"]) == 84 and len(sp["test"]) == 18
    assert sp["warnings"] == []
    assert [n for n, _ in list_imports(s)] == ["alphabet_static"]
    assert cfg.import_name(r.root, s) == "alphabet_static"
    assert cfg.work_dir(dict(s, data_dir=str(r.root))) == tmp_path / "work" / "imports" / "alphabet_static" / "work"


def test_group_key_and_small_signs():
    assert group_key("100_jpg.rf.4a2c7f39") == "100"
    assert group_key("J_99 (2)") == "j_99"
    assert group_key("Image_1667018687.444107") == "image_1667018687.444107"
    from pathlib import Path
    sp = assign_splits({"a": [Path(f"{i}.jpg") for i in range(3)]})
    assert sorted(sp.values()) == ["test", "train", "val"]


def test_unsafe_paths_are_skipped(s, tmp_path):
    files = {"../evil.txt": b"x", "raw/colors/red/S1/red_S1_01.mp4": b"x"}
    r = import_zip(make_zip(files), "x.zip", s)
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "work" / "imports" / "evil.txt").exists()
    assert r.n_used == 1


def test_unusable_upload_gives_plain_error_and_leaves_nothing(s, tmp_path):
    with pytest.raises(PlainError, match="No photos or videos"):
        import_zip(make_zip({"notes.txt": b"hello"}), "notes.zip", s)
    with pytest.raises(PlainError, match="not a readable .zip"):
        import_zip(io.BytesIO(b"not a zip"), "bad.zip", s)
    with pytest.raises(PlainError, match="at least two sign folders"):
        import_zip(make_zip({"only/a.jpg": b"x"}), "one.zip", s)
    assert list((tmp_path / "work" / "imports").iterdir()) == []
