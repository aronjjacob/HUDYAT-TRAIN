"""Combine datasets into one new dataset (one model), e.g. the static A-Y photos + the J and Z videos.

Signs are merged by their label (sign_label, case-insensitive): "J" from the photos and "J" from the
recordings (sign_id letter_j) are the same sign. Leave out the copies you do not want (e.g. the J / Z
photos, since J and Z are movements). A combined model has only signs, never a none class.

Nothing is copied except already-extracted samples: the new metadata.csv points to the original files
(absolute paths), and samples made with the current preprocessing settings are reused, so only the
missing ones need extracting. Each part keeps its split: recordings by signer, uploads their fixed split.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import dataset as ds
from . import recorder
from . import settings as cfg
from .upload import META_FIELDS, _new_dataset_dir, slug


@dataclass(frozen=True)
class Part:
    """All signs of one category of one dataset."""
    data_dir: Path
    category: str
    dataset_label: str = ""


@dataclass
class Item:
    """One sign of one part (what the user can include / leave out)."""
    key: str              # merge key: the sign label, upper case
    label: str
    part: Part
    sign_id: str
    n_files: int
    kind: str             # "photos" / "videos" / "photos + videos"

    @property
    def id(self) -> str:
        return f"{self.part.data_dir}|{self.part.category}|{self.sign_id}"


@dataclass
class CombineResult:
    root: Path
    name: str
    category: str
    n_files: int = 0
    n_reused: int = 0
    signs: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def items_of(part: Part) -> list[Item]:
    signs = [s for s in ds.load_signs(part.data_dir) if s.category == part.category]
    kept = ds.kept_rows(part.data_dir, include_static=True)
    kept = kept[kept["category"] == part.category] if not kept.empty else kept
    out = []
    for s in signs:
        rows = kept[kept["sign_id"] == s.sign_id] if not kept.empty else kept
        img = int(rows["file_path"].str.lower().str.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")).sum()) \
            if not rows.empty else 0
        kind = "photos" if img and img == len(rows) else ("videos" if not img else "photos + videos")
        label = s.sign_label.strip() or s.sign_id
        out.append(Item(label.upper(), label, part, s.sign_id, len(rows), kind))
    return out


def default_leave_out(items: list[Item]) -> set[str]:
    """Where a sign has both photos and videos, leave the photos out (videos carry the movement)."""
    by_key: dict[str, list[Item]] = {}
    for it in items:
        by_key.setdefault(it.key, []).append(it)
    out = set()
    for group in by_key.values():
        if any(i.kind != "photos" for i in group) and any(i.kind == "photos" for i in group):
            out |= {i.id for i in group if i.kind == "photos"}
    return out


def _source_samples(data_dir: Path, settings: dict) -> dict[str, dict]:
    """Index rows (ok, file exists) of a source dataset, by source file."""
    s = dict(settings, data_dir=str(data_dir))
    paths = cfg.Paths(s)
    idx = ds.read_index(paths)
    out = {}
    for r in idx.to_dict("records"):
        if r.get("status") != ds.STATUS_FAILED and (paths.work / r["file"]).exists():
            out[str(r["source_file"])] = dict(r, _work=str(paths.work))
    return out


def combine(name: str, items: list[Item], settings: dict | None = None) -> CombineResult:
    if not items:
        raise ds.PlainError("Nothing chosen to combine.", "Tick at least two signs.")
    s = settings or cfg.load_settings()
    final, data = _new_dataset_dir(name, s)
    cat = slug(final)
    try:
        params = recorder.preprocess_params(s, None)
        pid = recorder.params_id(params)
        rows, signs, reused, index_rows = [], {}, 0, []
        new_work = cfg.Paths(dict(s, data_dir=str(data))).work
        sources: dict[str, dict] = {}
        parts = list(dict.fromkeys(it.part for it in items))
        for part in parts:
            if str(part.data_dir) not in sources:
                sources[str(part.data_dir)] = _source_samples(part.data_dir, s)
            kept = ds.kept_rows(part.data_dir, include_static=True)
            kept = kept[kept["category"] == part.category]
            wanted = {it.sign_id: (slug(it.label), it.label) for it in items if it.part == part}
            mirror = {x.sign_id: x.mirror_ok for x in ds.load_signs(part.data_dir) if x.category == part.category}
            for _, m in kept[kept["sign_id"].isin(list(wanted))].iterrows():
                new_sid, label = wanted[m["sign_id"]]
                src_file = Path(part.data_dir) / m["file_path"]
                row = {k: m.get(k, "") for k in META_FIELDS}
                row.update(category=cat, sign_id=new_sid, sign_label=label, file_path=str(src_file),
                           status="kept", split=m.get("split", "") or "",
                           kind="image" if src_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                           else "video")
                rows.append(row)
                signs.setdefault((cat, new_sid), (label, mirror.get(m["sign_id"], 0)))
                # reuse the sample when it was made with the current preprocessing settings
                src = sources[str(part.data_dir)].get(str(m["file_path"]))
                if src is not None and (src.get("preprocess") or recorder.params_id(recorder.default_params())) == pid:
                    rel = ds.npy_rel_path(row)
                    (new_work / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(src["_work"]) / src["file"], new_work / rel)
                    index_rows.append(dict(src, file=rel, category=cat, sign_id=new_sid, source_file=str(src_file),
                                           split=row["split"], preprocess=pid))
                    reused += 1
        (data / "dataset").mkdir(parents=True, exist_ok=True)
        with open(data / "dataset" / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        lines = ["category,sign_id,sign_label,reps,hands,mirror_ok"]
        for (c, sid), (label, mir) in sorted(signs.items(), key=lambda kv: kv[0][1]):
            lines.append(f"{c},{sid},{label},15,1,{int(mir)}")
        (data / "signs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if index_rows:
            ds.write_index(cfg.Paths(dict(s, data_dir=str(data))),
                           [{k: v for k, v in r.items() if not k.startswith("_")} for r in index_rows])
        (data / "combined_from.json").write_text(json.dumps({
            "created": datetime.now().isoformat(timespec="seconds"),
            "parts": [{"dataset": str(it.part.data_dir), "category": it.part.category, "sign": it.sign_id,
                       "as": slug(it.label)} for it in items]}, indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(data.parent, ignore_errors=True)
        raise
    res = CombineResult(data, final, cat, len(rows), reused,
                        {lab: int(sum(r["sign_label"] == lab for r in rows)) for (_, _), (lab, _) in signs.items()})
    if len(rows) > reused:
        res.notes.append(f"{len(rows) - reused} file(s) still need preprocessing (Preprocess page).")
    return res

