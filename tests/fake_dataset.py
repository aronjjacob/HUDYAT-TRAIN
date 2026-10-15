"""Builds a TEMPORARY fake data-collection folder for the end-to-end and UI tests.

Real FSL-105 hand videos (the recorder's imported PUB clips) are COPIED (read-only)
into a temp folder and relabelled as signers S1-S7 with a matching metadata.csv,
signs.csv and the recorder's own hudyat_common.py. The real data folder is never
written. Results from this data are only a pipeline check, never real results.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from trainer.settings import DEFAULT_DATA_DIR, SIGNERS

REAL = DEFAULT_DATA_DIR
CATEGORY = "colors"
SIGNS = ["red", "blue", "green"]
NONE_LIKE = [("word_concepts", "yes"), ("word_concepts", "no")]    # stand-ins for the none class
PER_SIGNER = 2                                                      # 2 clips x 7 signers = 14 per sign

META_FIELDS = ["timestamp", "signer", "signing_hand", "category", "sign_id", "sign_label", "hands_required",
               "take", "file_path", "camera_index", "camera_name", "width", "height", "fps_actual", "n_frames",
               "duration_s", "codec", "orientation", "hand_detect_rate", "both_hands_rate", "mean_brightness",
               "status"]


def real_available() -> bool:
    return (REAL / "hudyat_common.py").exists() and (REAL / "dataset" / "metadata.csv").exists()


def _pub_rows(category: str, sign_id: str) -> list[dict]:
    with open(REAL / "dataset" / "metadata.csv", newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f)
                if r["signer"] == "PUB" and r["category"] == category and r["sign_id"] == sign_id
                and (REAL / r["file_path"]).exists()]


def build(root: Path, per_signer: int = PER_SIGNER) -> Path:
    """Create <root>/data (fake recorder folder) and return it."""
    data = Path(root) / "data"
    if data.exists():
        shutil.rmtree(data)
    (data / "dataset" / "raw").mkdir(parents=True)
    (data / "models").mkdir()
    for f in ("hudyat_common.py", "README.md"):
        shutil.copy2(REAL / f, data / f)
    (data / "settings.json").write_text(json.dumps({"camera": 0, "camera_name": "test", "backend": "msmf"}))
    with open(data / "signers.csv", "w", newline="", encoding="utf-8") as f:
        f.write("code,hand,display_name_optional,notes\n" + "".join(f"{s},right,,\n" for s in SIGNERS))
    with open(data / "signs.csv", "w", newline="", encoding="utf-8") as f:
        f.write("category,sign_id,sign_label,reps,hands,mirror_ok\n")
        for s in SIGNS:
            f.write(f"{CATEGORY},{s},{s.title()},15,1,{1 if s == 'red' else 0}\n")
        f.write("none,none,None,10,1,0\n")
        f.write("alphabet_static_test,static_a,A (static),5,1,0\n")
    rows = []
    plan = [(CATEGORY, s, CATEGORY, s) for s in SIGNS]
    plan += [("none", "none", c, s) for c, s in NONE_LIKE]
    none_take: dict[str, int] = {}
    need = per_signer * len(SIGNERS)
    none_i = 0                                          # none clips continue across sources: every signer gets some
    for cat, sign, src_cat, src_sign in plan:
        src = _pub_rows(src_cat, src_sign)
        n = need if cat != "none" else -(-need // len(NONE_LIKE))
        if len(src) < n:
            raise RuntimeError(f"not enough PUB clips for {src_cat}/{src_sign}")
        for i, r in enumerate(src[:n]):
            if cat == "none":
                signer = SIGNERS[none_i % len(SIGNERS)]
                none_i += 1
            else:
                signer = SIGNERS[i % len(SIGNERS)]
            key = f"{sign}/{signer}"
            take = none_take[key] = none_take.get(key, 0) + 1
            rel = f"dataset/raw/{cat}/{sign}/{signer}/{sign}_{signer}_{take:02d}.mp4"
            (data / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REAL / r["file_path"], data / rel)
            rows.append({**{k: r.get(k, "") for k in META_FIELDS}, "signer": signer, "signing_hand": "right",
                         "category": cat, "sign_id": sign, "sign_label": sign.title(), "take": take,
                         "file_path": rel, "hands_required": 1, "status": "kept"})
    # rows that must be ignored: a needs_rerecord take and a PUB row
    extra = []
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
