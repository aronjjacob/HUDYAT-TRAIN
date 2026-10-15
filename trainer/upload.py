"""Uploaded datasets (Dataset page): a .zip or a folder, copied into <work>/imports/<name>/data.

Each uploaded dataset gets its own outputs in <work>/imports/<name>/work (see settings.work_dir), so
datasets never mix. Accepted layouts (found automatically, at any depth):

  1. a copy of a HUDYAT - HAND TRACKER folder: signs.csv + dataset/metadata.csv (+ dataset/raw/...)
     -> used as is (signer split S1-S4 / S5 / S6-S7).
  2. videos in <category>/<sign>/<S1..S7>/<file>  (e.g. the recorder's dataset/raw)
     -> metadata.csv and signs.csv are made from the folder names (signer split).
  3. class folders: one folder per sign with photos and/or videos in it, e.g. Collated/A/*.jpg
     -> one category named after the dataset; no signers, so every file gets a FIXED split:
        70 % train / 15 % validation / 15 % test, chosen per sign with a fixed seed, and all copies
        of one original (Roboflow "x_jpg.rf.<hash>", "x (2).jpg") always in the same split.

The source (zip or folder) is only read.
"""

from __future__ import annotations

import csv
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import settings as cfg
from .dataset import PlainError

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MEDIA_EXT = VIDEO_EXT + IMAGE_EXT
IMPORT_SIGNER = "IMP"                      # signer code of class-folder data (no real signers)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42
META_FIELDS = ["timestamp", "signer", "signing_hand", "category", "sign_id", "sign_label", "hands_required", "take",
               "file_path", "fps_actual", "n_frames", "status", "split", "kind"]


@dataclass
class UploadResult:
    root: Path
    name: str = ""
    layout: str = ""
    n_files: int = 0
    n_used: int = 0
    labels: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", Path(str(name)).stem.lower()).strip("_")
    return s[:40] or "dataset"


def _new_dataset_dir(name: str, settings: dict | None = None) -> tuple[str, Path]:
    base = cfg.imports_dir(settings)
    base.mkdir(parents=True, exist_ok=True)
    s, n = slug(name), 2
    final = s
    while (base / final).exists():
        final = f"{s}_{n}"
        n += 1
    (base / final / "data").mkdir(parents=True)
    return final, base / final / "data"


def _is_media(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in MEDIA_EXT and ".tmp." not in p.name


def unpack_zip(zip_file, dest: Path) -> None:
    """Safe unpack (entries pointing outside `dest` are skipped)."""
    try:
        z = zipfile.ZipFile(zip_file)
    except zipfile.BadZipFile as e:
        raise PlainError("This file is not a readable .zip file.", "Zip the folder again and upload it.") from e
    root = dest.resolve()
    with z:
        for m in z.infolist():
            if m.is_dir() or m.filename.startswith("__MACOSX/"):
                continue
            target = (dest / m.filename).resolve()
            if root not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)


def copy_folder(src: Path, dest: Path, progress=None) -> None:
    """Copy the media files (and csv files) of `src` into dest/<src name>/, keeping the folder layout."""
    src = Path(src)
    if not src.is_dir():
        raise PlainError(f"The folder {src} does not exist.", "Choose the folder again.")
    files = [p for p in src.rglob("*") if p.is_file() and (p.suffix.lower() in MEDIA_EXT or p.suffix.lower() == ".csv"
                                                            or p.name == "hudyat_common.py")]
    if not files:
        raise PlainError(f"No photos or videos found in {src}.", "Choose the folder that holds the sign folders.")
    for i, p in enumerate(files):
        t = dest / src.name / p.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)
        if progress and (i % 200 == 0 or i == len(files) - 1):
            progress((i + 1) / len(files), f"Copying {i + 1} / {len(files)} files")


# ----------------------------------------------------------------------------
# Layout detection
# ----------------------------------------------------------------------------
def find_recorder_root(d: Path) -> Path | None:
    for cand in [d] + sorted(p for p in d.rglob("*") if p.is_dir() and len(p.relative_to(d).parts) <= 3):
        if (cand / "signs.csv").exists() and (cand / "dataset" / "metadata.csv").exists():
            return cand
    return None


def group_key(stem: str) -> str:
    """Copies of one original share a key: 'x_jpg.rf.<hash>' -> 'x', 'x (2)' -> 'x'."""
    s = re.split(r"[._](?:jpe?g|png)\.rf\.|\.rf\.", stem, maxsplit=1)[0]
    s = re.sub(r"\s*\(\d+\)$", "", s)
    return s.lower()


def assign_splits(files_by_label: dict[str, list[Path]], seed: int = SPLIT_SEED) -> dict[Path, str]:
    """Per sign: groups of copies shuffled with a fixed seed, then 70 / 15 / 15 % by file count."""
    out: dict[Path, str] = {}
    rng = random.Random(seed)
    for label in sorted(files_by_label):
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in sorted(files_by_label[label]):
            groups[group_key(f.stem)].append(f)
        keys = sorted(groups)
        rng.shuffle(keys)
        total = sum(len(g) for g in groups.values())
        b0 = SPLIT_FRACTIONS[0] * total
        b1 = (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]) * total
        names, done = [], 0
        for k in keys:
            names.append("train" if done < b0 else "val" if done < b1 else "test")
            done += len(groups[k])
        if len(keys) >= 3:                       # every sign gets validation and test examples
            if "test" not in names:
                names[-1] = "test"
            if "val" not in names:
                names[-2] = "val"
        for k, sp in zip(keys, names):
            for f in groups[k]:
                out[f] = sp
    return out


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _row(cat, sign, label, signer, take, rel, split="", kind="video") -> dict:
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "signer": signer, "signing_hand": "",
            "category": cat, "sign_id": sign, "sign_label": label, "hands_required": 1, "take": take,
            "file_path": rel, "fps_actual": "", "n_frames": "", "status": "kept", "split": split, "kind": kind}


def prepare(data: Path, name: str) -> UploadResult:
    """Detect the layout of the files in `data` and write what the trainer needs. Returns the dataset root."""
    data = Path(data)
    rec = find_recorder_root(data)
    if rec is not None:
        res = UploadResult(rec, layout="recorder copy")
        with open(rec / "dataset" / "metadata.csv", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        res.n_files = len(rows)
        res.n_used = sum(1 for r in rows if r.get("signer", "").strip() in cfg.SIGNERS and r.get("status") == "kept")
        res.labels = dict(Counter(r.get("sign_id", "") for r in rows))
        if not (rec / "hudyat_common.py").exists():
            res.notes.append("No hudyat_common.py: the trainer's tested copies of the trim and handedness rules are used.")
        return res

    labels_text: dict[str, str] = {}
    media = sorted(p for p in data.rglob("*") if _is_media(p))
    if not media:
        raise PlainError("No photos or videos found in the upload.",
                         "Put each sign in its own folder (e.g. A\\, B\\, ...) with its photos or videos inside.")
    signer_tree = [p for p in media if len(p.relative_to(data).parts) >= 4 and re.fullmatch(r"S\d+|PUB", p.parent.name)]
    if len(signer_tree) >= 0.8 * len(media):
        # layout 2: <category>/<sign>/<signer>/<file>
        rows, unknown = [], set()
        for p in signer_tree:
            cat, sign, signer = p.parts[-4], p.parts[-3], p.parts[-2]
            if signer not in cfg.SIGNERS:
                unknown.add(signer)
                continue
            m = re.search(r"_(\d+)$", p.stem)
            rows.append(_row(cat, sign, sign.replace("_", " ").title(), signer, int(m.group(1)) if m else len(rows) + 1,
                             p.relative_to(data).as_posix(), kind="image" if p.suffix.lower() in IMAGE_EXT else "video"))
        if not rows:
            raise PlainError("No videos of signers S1–S7 found.", "Signer folders must be named S1 ... S7.")
        res = UploadResult(data, layout="videos per signer (S1–S7)", n_files=len(media), n_used=len(rows))
        if unknown:
            res.notes.append(f"Ignored folder(s) {', '.join(sorted(unknown))}: only signers S1–S7 are used.")
        pairs = list(dict.fromkeys((r["category"], r["sign_id"]) for r in rows))
    else:
        # layout 3: class folders
        cat = slug(name)
        by_label: dict[str, list[Path]] = defaultdict(list)
        for p in media:
            if p.parent == data:
                continue
            by_label[p.parent.name].append(p)
        if len(by_label) < 2:
            raise PlainError("The upload needs at least two sign folders.",
                             "Put each sign in its own folder (e.g. A\\, B\\, ...) with its photos or videos inside.")
        splits = assign_splits(by_label)
        rows = []
        for label in sorted(by_label):
            sid = slug(label)
            for i, p in enumerate(sorted(by_label[label]), 1):
                rows.append(_row(cat, sid, label, IMPORT_SIGNER, i, p.relative_to(data).as_posix(), splits[p],
                                 "image" if p.suffix.lower() in IMAGE_EXT else "video"))
        res = UploadResult(data, layout="one folder per sign (fixed 70/15/15 split)", n_files=len(media),
                           n_used=len(rows))
        n_img = sum(r["kind"] == "image" for r in rows)
        if n_img:
            res.notes.append(f"{n_img} photo(s): each becomes a held-still 32-frame sample; the hand is stored as a "
                             "right hand (photos of a left hand are mirrored).")
        loose = sum(1 for p in media if p.parent == data)
        if loose:
            res.notes.append(f"Ignored {loose} file(s) that are not inside a sign folder.")
        pairs = list(dict.fromkeys((cat, slug(lab)) for lab in sorted(by_label)))
        labels_text.update({slug(lab): lab for lab in by_label})
    _write_csv(data / "dataset" / "metadata.csv", META_FIELDS, rows)
    if not (data / "signs.csv").exists():
        lines = ["category,sign_id,sign_label,reps,hands,mirror_ok"]
        for c, s in pairs:
            lab = labels_text.get(s, s) if res.layout.startswith("one folder") else s.replace("_", " ").title()
            reps = 10 if c == cfg.NONE_CATEGORY else 15
            lines.append(f"{c},{s},{lab},{reps},1,0")
        (data / "signs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    res.labels = dict(Counter(r["sign_id"] for r in rows))
    return res


def import_zip(zip_file, name: str, settings: dict | None = None) -> UploadResult:
    final, data = _new_dataset_dir(name, settings)
    try:
        unpack_zip(zip_file, data)
        res = prepare(data, final)
    except BaseException:
        shutil.rmtree(data.parent, ignore_errors=True)
        raise
    res.name = final
    return res


def import_folder(src: Path, name: str | None = None, settings: dict | None = None, progress=None) -> UploadResult:
    src = Path(src)
    final, data = _new_dataset_dir(name or src.name, settings)
    try:
        copy_folder(src, data, progress)
        res = prepare(data, final)
    except BaseException:
        shutil.rmtree(data.parent, ignore_errors=True)
        raise
    res.name = final
    return res


def list_imports(settings: dict | None = None) -> list[tuple[str, Path]]:
    """[(name, dataset root)] of the uploaded datasets."""
    base = cfg.imports_dir(settings)
    out = []
    if base.exists():
        for d in sorted(p for p in base.iterdir() if p.is_dir() and (p / "data").is_dir()):
            root = find_recorder_root(d / "data") or d / "data"
            if (root / "signs.csv").exists():
                out.append((d.name, root))
    return out
