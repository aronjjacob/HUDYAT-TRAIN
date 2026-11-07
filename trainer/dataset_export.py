"""Export the extracted landmarks of one category as a dataset: the .npy samples plus what is needed to use them.

Writes work/datasets/<category>-<dataset>-landmarks-<YYYYMMDD-HHMM>/ and a .zip of it next to the folder:

  signs.csv                 the category's signs (as in the recorder)
  signers.csv               code, hand
  dataset/metadata.csv      one row per exported sample (the recorder's columns)
  landmarks/<category>/<sign>/<signer>/<clip>.npy   the usable samples, 32 frames x 149 values, float32
  landmarks_index.csv       their index rows (detection rates, trim, preprocessing id, format)
  excluded.csv              clips left out and why
  dataset.json              landmarks_only, classes, shape, preprocessing, counts, source
  FEATURES.md, README.txt

The folder is laid out like a recorder folder plus the trainer's own files, so the Data page can open it again
(upload.py, "landmarks only"): it then trains without the videos.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import dataset as ds
from . import recorder
from . import settings as cfg
from .landmarks import N_FRAMES

DIM = 149                                   # values per frame of a rich149 sample (docs/FEATURES.md section 1)
EXCLUDED_FIELDS = ["source_file", "sign_id", "signer", "take", "reason"]

README = """HUDYAT landmark dataset: {category}
{rule}
{n} samples of {n_signs} signs from {n_signers} signers ({signers}). Made {created} from the dataset "{source}".

What is in it
  landmarks/<category>/<sign>/<signer>/<clip>.npy
      one sample per clip: a float32 array of shape (32, 149), the hands (and face location) in 32 frames
      evenly spread over the sign. FEATURES.md section 1 says what each of the 149 values is.
  dataset/metadata.csv   one row per sample: signer, signing hand, sign, take, the original video's path
                         (file_path; the video itself is not included)
  landmarks_index.csv    per sample: the .npy path (file), hand detection rate, trim window, preprocessing settings
  signs.csv              the signs: sign_id, the label shown, takes per signer
  signers.csv            each signer's signing hand
  excluded.csv           clips left out of this dataset and why
  dataset.json           the same in one file: classes, sample shape, preprocessing settings, counts

Only clips usable for training are included. Clips rejected in the trainer, with low hand detection or not yet
extracted in the current format are left out and listed in excluded.csv.

Load it in Python
  import numpy as np, pandas as pd
  idx = pd.read_csv("landmarks_index.csv")
  X = np.stack([np.load(f) for f in idx["file"]])        # (n, 32, 149)
  y = idx["sign_id"]

Train on it in the HUDYAT Trainer
  Data page > Add or combine datasets > upload this .zip (or add the folder). It opens as "landmarks only": nothing to
  extract, go straight to Test the model or Train model.
"""


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cfg.PROJECT_DIR, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plan(paths: cfg.Paths, data_dir: Path, category: str, settings: dict | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """(index rows to export, excluded clips with their reason) for one category."""
    usable = ds.usable_samples(paths, data_dir, settings)
    usable = usable[usable["category"] == category] if not usable.empty else usable
    excluded = []

    def skip(src, sign, signer, take, reason):
        excluded.append({"source_file": src, "sign_id": sign, "signer": signer, "take": take, "reason": reason})

    ok = []
    for r in usable.to_dict("records"):
        if r["excluded_reason"]:
            skip(r["source_file"], r["sign_id"], r["signer"], r["take"], r["excluded_reason"])
        elif r["format"] != ds.SAMPLE_FORMAT:
            skip(r["source_file"], r["sign_id"], r["signer"], r["take"], "old format, re-extract")
        elif not (paths.work / r["file"]).exists():
            skip(r["source_file"], r["sign_id"], r["signer"], r["take"], "sample file missing, re-extract")
        else:
            ok.append(r)
    done = set(usable["source_file"]) if not usable.empty else set()
    kept = ds.kept_rows(data_dir, include_static=True)
    for _, m in kept[kept["category"] == category].iterrows():
        if m["file_path"] not in done:
            skip(m["file_path"], m["sign_id"], m["signer"], m.get("take", ""), "not extracted yet")
    return pd.DataFrame(ok, columns=list(usable.columns)), excluded


def export_category(ctx, category: str) -> dict:
    s = cfg.load_settings()
    data_dir = cfg.data_dir(s)
    paths = cfg.Paths(s)
    signs = [x for x in ds.load_signs(data_dir) if x.category == category]
    if not signs:
        raise ds.PlainError(f"Unknown category: {category}.", "Pick a category from the list.")
    rows, excluded = plan(paths, data_dir, category, s)
    if rows.empty:
        raise ds.PlainError(f"No extracted samples to export in {category}.",
                            "Extract them on the Preprocess page first (Re-extract all for samples in the old format).")
    created = datetime.now()
    paths.datasets.mkdir(parents=True, exist_ok=True)
    base = f"{category}-{paths.dataset}-landmarks-{created:%Y%m%d-%H%M}"
    name, n = base, 2
    while (paths.datasets / name).exists() or (paths.datasets / f"{name}.zip").exists():
        name, n = f"{base}_{n}", n + 1
    out, tmp = paths.datasets / name, paths.datasets / f"{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        # the samples
        for i, r in enumerate(rows.to_dict("records")):
            (tmp / r["file"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths.work / r["file"], tmp / r["file"])
            if i % 100 == 0:
                ctx.progress(0.8 * i / len(rows), f"{category}: copying {i + 1} / {len(rows)} samples")
        _write_csv(tmp / "landmarks_index.csv", ds.INDEX_FIELDS, rows.to_dict("records"))

        # metadata: the recorder's rows of these clips
        files = set(rows["source_file"])
        meta = ds.read_metadata(data_dir)
        meta = meta[meta["file_path"].isin(files)].copy()
        (tmp / "dataset").mkdir(parents=True, exist_ok=True)
        meta.to_csv(tmp / "dataset" / "metadata.csv", index=False)

        _write_csv(tmp / "signs.csv", ["category", "sign_id", "sign_label", "reps", "hands", "mirror_ok"],
                   [{"category": x.category, "sign_id": x.sign_id, "sign_label": x.sign_label, "reps": x.reps,
                     "hands": x.hands, "mirror_ok": x.mirror_ok} for x in signs])
        hands = ds.clip_hands(data_dir)
        signers = sorted(set(rows["signer"]))
        by_signer = {g: sorted({hands.get(f, "right") for f in meta[meta["signer"] == g]["file_path"]}) for g in signers}
        _write_csv(tmp / "signers.csv", ["code", "hand"],
                   [{"code": g, "hand": h[0] if len(h) == 1 else ""} for g, h in by_signer.items()])
        _write_csv(tmp / "excluded.csv", EXCLUDED_FIELDS, excluded)

        labels = {x.sign_id: x.sign_label for x in signs}
        counts = rows.groupby(["sign_id", "signer"]).size()
        params = recorder.preprocess_params(s, data_dir)
        info = {
            "landmarks_only": True,
            "name": name,
            "category": category,
            "classes": [{"sign_id": x.sign_id, "label": x.sign_label} for x in signs],
            "n_samples": len(rows),
            "sample": {"shape": [N_FRAMES, DIM], "dtype": "float32", "format": ds.SAMPLE_FORMAT,
                       "description": "docs/FEATURES.md section 1 (copied as FEATURES.md)"},
            "preprocessing": {"settings": params, "ids_used": sorted(set(rows["preprocess"].astype(str)) - {""}),
                              "min_hand_detect_rate": float(s.get("min_hand_detect_rate", 0.8))},
            "counts": {sid: {g: int(counts.get((sid, g), 0)) for g in signers} for sid in labels},
            "n_excluded": len(excluded),
            "source": {"dataset": paths.dataset, "data_dir": str(data_dir)},
            "created": created.isoformat(timespec="seconds"),
            "trainer_commit": _git_commit(),
        }
        (tmp / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        features = cfg.PROJECT_DIR / "docs" / "FEATURES.md"
        if features.exists():
            shutil.copy2(features, tmp / "FEATURES.md")
        (tmp / "README.txt").write_text(README.format(
            category=category, rule="=" * (len(category) + 25), n=len(rows), n_signs=rows["sign_id"].nunique(),
            n_signers=len(signers), signers=", ".join(signers), created=info["created"].replace("T", " "),
            source=paths.dataset), encoding="utf-8")

        ctx.progress(0.85, f"{category}: zipping")
        zp = paths.datasets / f"{name}.zip.tmp"
        # files at the top of the .zip, no extra folder: unpacked paths stay short (Windows 260-character limit)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tmp.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmp).as_posix())
        tmp.rename(out)
        zp.rename(paths.datasets / f"{name}.zip")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        (paths.datasets / f"{name}.zip.tmp").unlink(missing_ok=True)
        raise
    ctx.log(f"[{category}] {len(rows)} samples exported to {out} ({len(excluded)} clips left out)")
    return {"name": name, "n_samples": len(rows), "n_excluded": len(excluded)}


def run(ctx, categories: list[str]) -> dict:
    done = []
    for i, c in enumerate(categories):
        ctx.progress(i / len(categories), f"Exporting {c}")
        done.append(export_category(ctx, c))
    left = sum(d["n_excluded"] for d in done)
    return {"message": f"Exported {ds.plural(sum(d['n_samples'] for d in done), 'sample')} as "
                       f"{ds.plural(len(done), 'dataset')}: {', '.join(d['name'] for d in done)}."
                       + (f" {ds.plural(left, 'clip')} left out, listed in excluded.csv." if left else ""),
            "datasets": [d["name"] for d in done]}


def list_exports(paths: cfg.Paths, category: str) -> list[dict]:
    """The landmark datasets exported from `category`, newest first: name, folder, zip, n_samples, created."""
    out = []
    if paths.datasets.exists():
        for d in paths.datasets.iterdir():
            p = d / "dataset.json"
            if not (d.is_dir() and p.exists()):
                continue
            try:
                info = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if info.get("category") == category:
                out.append({"name": d.name, "folder": d, "zip": paths.datasets / f"{d.name}.zip",
                            "n_samples": info.get("n_samples", 0), "n_excluded": info.get("n_excluded", 0),
                            "created": info.get("created", "")})
    return sorted(out, key=lambda x: x["created"], reverse=True)
