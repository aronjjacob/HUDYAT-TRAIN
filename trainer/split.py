"""Step 4: split by signer (fixed): train S1-S4, validation S5, test S6-S7. Every model gets 'none'."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import dataset as ds
from . import settings as cfg

SPLITS = ("train", "val", "test")


def split_of(signer: str) -> str | None:
    for name, signers in cfg.SPLIT_SIGNERS.items():
        if signer in signers:
            return name
    return None


def labels_for(category: str, signs: list[ds.Sign]) -> list[str]:
    """The category's signs, + none when the dataset has a none class (the recordings always do)."""
    has_none = any(s.category == cfg.NONE_CATEGORY for s in signs)
    return [s.sign_id for s in signs if s.category == category] + ([cfg.NONE_SIGN] if has_none else [])


def build_split(category: str, usable: pd.DataFrame, signs: list[ds.Sign]) -> dict:
    """Pure function (tested): the split dictionary for one category."""
    labels = labels_for(category, signs)
    rows = usable[(usable["excluded_reason"] == "")
                  & (((usable["category"] == category) & usable["sign_id"].isin(labels))
                     | (usable["category"] == cfg.NONE_CATEGORY))]
    fixed = rows["split"].isin(SPLITS) if "split" in rows.columns else pd.Series(False, index=rows.index)
    rows = rows[rows["signer"].isin(cfg.SIGNERS) | fixed]       # never PUB or anything else
    out = {s: [] for s in SPLITS}
    counts = {lab: {s: 0 for s in SPLITS} for lab in labels}
    for _, r in rows.sort_values("file").iterrows():
        given = str(r.get("split", "") or "")
        sp = given if given in SPLITS else split_of(r["signer"])       # uploaded data: its fixed split
        if sp is None:
            continue
        label = cfg.NONE_SIGN if r["category"] == cfg.NONE_CATEGORY else r["sign_id"]
        out[sp].append({"file": r["file"], "label": label, "signer": r["signer"]})
        counts[label][sp] += 1
    warnings = []
    by_signer = not fixed.any()
    names = ({"train": "train (S1-S4)", "val": "validation (S5)", "test": "test (S6-S7)"} if by_signer
             else {"train": "train", "val": "validation", "test": "test"})
    for lab in labels:
        missing = [s for s in SPLITS if counts[lab][s] == 0]
        if missing:
            warnings.append(f"{lab}: no clips in {', '.join(names[m] for m in missing)}")
    mirror = {s.sign_id: s.mirror_ok for s in signs if s.category == category}
    mirror[cfg.NONE_SIGN] = next((s.mirror_ok for s in signs if s.sign_id == cfg.NONE_SIGN), 0)
    return {"category": category, "labels": labels, "signers": cfg.SPLIT_SIGNERS if by_signer else None,
            "split_by": "signer" if by_signer else "fixed 70/15/15 split of the upload", "mirror_ok": mirror,
            "counts": counts, "warnings": warnings, **out,
            "created": datetime.now().isoformat(timespec="seconds")}


def run(ctx, categories: list[str] | None = None) -> dict:
    s = cfg.load_settings()
    data_dir = cfg.data_dir(s)
    problems = cfg.check_data_dir(data_dir)
    if problems:
        raise ds.PlainError(problems[0], "Choose the dataset on the Dataset page (a folder or a .zip upload).")
    paths = cfg.Paths(s)
    usable = ds.usable_samples(paths, data_dir, s)
    if usable.empty:
        raise ds.PlainError("No extracted landmarks yet.", "Run step 2 (Preprocess) first.")
    signs = ds.load_signs(data_dir)
    all_cats = ds.model_categories(signs)
    cats = categories or all_cats
    unknown = [c for c in cats if c not in all_cats]
    if unknown:
        raise ds.PlainError(f"Unknown category: {', '.join(unknown)}.", "Pick a category from the list.")
    paths.splits.mkdir(parents=True, exist_ok=True)
    summary = {}
    for i, c in enumerate(cats):
        sp = build_split(c, usable, signs)
        Path(paths.split_json(c)).write_text(json.dumps(sp, indent=2), encoding="utf-8")
        summary[c] = {k: len(sp[k]) for k in SPLITS}
        ctx.log(f"{c}: train {summary[c]['train']}, val {summary[c]['val']}, test {summary[c]['test']}"
                + (f"  ({len(sp['warnings'])} warning(s))" if sp["warnings"] else ""))
        for w in sp["warnings"]:
            ctx.log(f"   warning: {w}")
        ctx.progress((i + 1) / len(cats), c)
    return {"message": f"Split saved for {len(cats)} categor{'y' if len(cats) == 1 else 'ies'}.", "summary": summary}


def load_split(paths: cfg.Paths, category: str) -> dict | None:
    p = paths.split_json(category)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
