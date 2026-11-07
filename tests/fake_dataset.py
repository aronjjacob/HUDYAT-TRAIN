"""Builds a TEMPORARY data-collection folder from REAL hand videos for the end-to-end and UI tests.

Our own recordings (signers S1-S3, greetings) are COPIED from the recorder folder (which is only read) and relabelled:
S1 and S2 stay right-handed; S3 is marked LEFT-handed and its videos are MIRRORED (cv2.flip on every frame), so the
dominant hand really is the left one. FSL-105 public clips stand in for `none` clips, which the trainer ignores. A
needs_rerecord take and a PUB row are added that must be ignored too. Results from this data are only a pipeline check, never real results.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from trainer.settings import DEFAULT_DATA_DIR

REAL = DEFAULT_DATA_DIR
CATEGORY = "greetings"
SIGNS = ["good_morning", "hello", "thank_you"]
SIGNERS = ["S1", "S2", "S3"]
LEFT = "S3"                                                          # marked left-handed: videos mirrored
NONE_LIKE = [("word_concepts", "yes"), ("word_concepts", "no")]      # FSL-105 stand-ins for none clips (ignored)
PER_SIGNER = 4                                                       # takes per sign and signer

META_FIELDS = ["timestamp", "signer", "signing_hand", "category", "sign_id", "sign_label", "hands_required",
               "take", "file_path", "camera_index", "camera_name", "width", "height", "fps_actual", "n_frames",
               "duration_s", "codec", "orientation", "hand_detect_rate", "both_hands_rate", "mean_brightness",
               "status"]


def _rows(**match) -> list[dict]:
    with open(REAL / "dataset" / "metadata.csv", newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f)
                if all(r.get(k) == v for k, v in match.items()) and (REAL / r["file_path"]).exists()]


def real_available() -> bool:
    if not ((REAL / "hudyat_common.py").exists() and (REAL / "dataset" / "metadata.csv").exists()):
        return False
    return all(len(_rows(signer=g, category=CATEGORY, sign_id=s, status="kept")) >= PER_SIGNER
               for g in SIGNERS for s in SIGNS)


def mirror_video(src: Path, dst: Path) -> None:
    """A left-right mirrored copy (what a left-handed signer doing the same sign looks like)."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        out.write(cv2.flip(frame, 1))
    cap.release()
    if out is not None:
        out.release()


def build(root: Path, per_signer: int = PER_SIGNER) -> Path:
    """Create <root>/data (a temporary recorder folder) and return it."""
    data = Path(root) / "data"
    if data.exists():
        shutil.rmtree(data)
    (data / "dataset" / "raw").mkdir(parents=True)
    (data / "models").mkdir()
    for f in ("hudyat_common.py", "README.md"):
        shutil.copy2(REAL / f, data / f)
    (data / "settings.json").write_text(json.dumps({"camera": 0, "camera_name": "test", "backend": "msmf"}))
    with open(data / "signers.csv", "w", newline="", encoding="utf-8") as f:
        f.write("code,hand,display_name_optional,notes\n"
                + "".join(f"{g},{'left' if g == LEFT else 'right'},,\n" for g in SIGNERS))
    with open(data / "signs.csv", "w", newline="", encoding="utf-8") as f:
        f.write("category,sign_id,sign_label,reps,hands,mirror_ok\n")
        for s in SIGNS:
            f.write(f"{CATEGORY},{s},{s.replace('_', ' ').title()},15,1,0\n")
        f.write("none,none,None,10,1,0\n")
        f.write("alphabet_static_test,static_a,A (static),5,1,0\n")
    rows = []
    for s in SIGNS:
        for g in SIGNERS:
            for take, r in enumerate(_rows(signer=g, category=CATEGORY, sign_id=s, status="kept")[:per_signer], 1):
                rel = f"dataset/raw/{CATEGORY}/{s}/{g}/{s}_{g}_{take:02d}.mp4"
                (data / rel).parent.mkdir(parents=True, exist_ok=True)
                if g == LEFT:
                    mirror_video(REAL / r["file_path"], data / rel)
                else:
                    shutil.copy2(REAL / r["file_path"], data / rel)
                rows.append({**{k: r.get(k, "") for k in META_FIELDS}, "signer": g,
                             "signing_hand": "left" if g == LEFT else "right", "take": take, "file_path": rel,
                             "status": "kept"})
    none_i = 0
    need = -(-per_signer * len(SIGNERS) // len(NONE_LIKE))
    for src_cat, src_sign in NONE_LIKE:
        src = _rows(signer="PUB", category=src_cat, sign_id=src_sign)
        if len(src) < need:
            raise RuntimeError(f"not enough PUB clips for {src_cat}/{src_sign}")
        for r in src[:need]:
            g = SIGNERS[none_i % len(SIGNERS)]
            take = none_i // len(SIGNERS) + 1
            none_i += 1
            rel = f"dataset/raw/none/none/{g}/none_{g}_{take:02d}.mp4"
            (data / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REAL / r["file_path"], data / rel)
            rows.append({**{k: r.get(k, "") for k in META_FIELDS}, "signer": g,
                         "signing_hand": "left" if g == LEFT else "right", "category": "none", "sign_id": "none",
                         "sign_label": "None", "take": take, "file_path": rel, "hands_required": 1, "status": "kept"})
    extra = []                    # rows that must be ignored: a needs_rerecord take and a PUB row
    for base, changes, tag in ((rows[0], {"status": "needs_rerecord"}, "S1_99"), (rows[1], {"signer": "PUB"}, "PUB_01")):
        rel = f"dataset/raw/{base['category']}/{base['sign_id']}/{changes.get('signer', 'S1')}/{base['sign_id']}_{tag}.mp4"
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / base["file_path"], data / rel)
        extra.append(dict(base, file_path=rel, take=int(tag[-2:]), **changes))
    rows += extra
    with open(data / "dataset" / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return data


def make_settings(root: Path, data: Path) -> Path:
    """trainer_settings.json for the temp project (work folder inside root)."""
    p = Path(root) / "trainer_settings.json"
    p.write_text(json.dumps({"data_dir": str(data), "work_dir": str(Path(root) / "work"),
                             "camera": 0, "backend": "msmf"}, indent=2))
    return p
