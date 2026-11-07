"""Uploading a dataset (.zip or folder): copied into <work>/imports/<name>/data and made usable."""

import csv
import io
import zipfile
from collections import Counter

import pytest

from trainer import dataset as ds
from trainer import settings as cfg
from trainer.dataset import PlainError
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
    assert len(ds.kept_rows(r.root)) == 8                                     # the none clips are ignored
    assert ds.model_categories(ds.load_signs(r.root)) == ["colors"]


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
    # Test 1 on photos: copies of one photo always land in the same set; no none class in this dataset
    import pandas as pd
    from pathlib import Path
    from trainer import splits
    kept = ds.kept_rows(r.root)
    clips = pd.DataFrame({"file": [ds.npy_rel_path(x) for _, x in kept.iterrows()], "label": kept["sign_id"].values,
                          "signer": kept["signer"].values,
                          "group": [f"{lab}|{group_key(Path(f).stem)}" for lab, f in zip(kept["sign_id"], kept["file_path"])]})
    sp = splits.random_split(clips, seed=42)
    assert sum(len(sp[p]) for p in splits.PARTS) == len(clips) == 120
    where = {it["group"]: p for p in splits.PARTS for it in sp[p]}
    assert all(where[it["group"]] == p for p in splits.PARTS for it in sp[p])
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


# --- scan before import (Data page: show what was found, import only on "Import") -------------------
from trainer.upload import scan_folder, scan_zip  # noqa: E402


def test_scan_recorder_copy_folder(tmp_path, s):
    root = tmp_path / "copy" / "HUDYAT - HAND TRACKER"
    (root / "dataset").mkdir(parents=True)
    (root / "signs.csv").write_text("category,sign_id,sign_label,reps,hands,mirror_ok\ncolors,red,Red,15,1,0\n"
                                    "colors,blue,Blue,15,1,0\n")
    (root / "signers.csv").write_text("code,hand\nS1,right\nS2,left\n")
    (root / "dataset" / "metadata.csv").write_text(
        "signer,signing_hand,category,sign_id,take,file_path,status\n"
        "S1,right,colors,red,1,a.mp4,kept\nS1,right,colors,blue,1,b.mp4,kept\nS2,left,colors,red,1,c.mp4,kept\n"
        "S2,left,colors,red,2,d.mp4,needs_rerecord\nPUB,,colors,red,1,e.mp4,kept\n")
    r = scan_folder(tmp_path / "copy")
    assert r.ok and r.layout == "recorder copy"
    assert r.signers == {"S1": "right", "S2": "left"}
    assert r.n_clips == 3 and r.signs == {"red": 2, "blue": 1}
    assert any("PUB" in n for n in r.notes)
    assert not cfg.imports_dir(s).exists()                       # nothing copied


def test_scan_signer_folders_zip(s):
    z = make_zip({f"raw/colors/{sg}/{sg2}/{sg}_{sg2}_{t:02d}.mp4": b"x"
                  for sg in ("red", "blue") for sg2 in ("S1", "S2", "PUB") for t in range(2)})
    r = scan_zip(z)
    assert r.ok and r.layout == "videos per signer"
    assert set(r.signers) == {"S1", "S2"} and r.n_clips == 8
    assert r.signs == {"red": 4, "blue": 4}
    assert any("PUB" in n for n in r.notes)


def test_scan_photo_class_folders(tmp_path):
    for lab in "ABC":
        (tmp_path / "fsl" / lab).mkdir(parents=True)
        for i in range(3):
            (tmp_path / "fsl" / lab / f"{lab}{i}.jpg").write_bytes(b"x")
    (tmp_path / "fsl" / "readme.jpg").write_bytes(b"x")
    r = scan_folder(tmp_path / "fsl")
    assert r.ok and r.layout == "one folder per sign"
    assert r.signers == {} and r.n_clips == 9 and r.n_photos == 9
    assert r.signs == {"A": 3, "B": 3, "C": 3}
    assert any("not inside a sign folder" in n for n in r.notes)
    assert any("Test 1" in n for n in r.notes)                     # photos: no signer IDs, only Test 1


def test_scan_reports_problems_in_plain_language(tmp_path):
    (tmp_path / "empty").mkdir()
    r = scan_folder(tmp_path / "empty")
    assert not r.ok and "No photos or videos" in r.problems[0]
    (tmp_path / "one" / "A").mkdir(parents=True)
    (tmp_path / "one" / "A" / "x.jpg").write_bytes(b"x")
    r = scan_folder(tmp_path / "one")
    assert not r.ok and "two sign folders" in r.problems[0]
    assert not scan_zip(io.BytesIO(b"not a zip")).ok
