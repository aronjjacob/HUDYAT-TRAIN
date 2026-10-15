"""Reading the data-collection folder (read-only) and this project's index / validation files."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import settings as cfg

INDEX_FIELDS = ["file", "category", "sign_id", "signer", "take", "fps", "n_frames", "trim_start",
                "trim_end", "frames_kept", "hand_detect_rate", "both_hands_rate", "status",
                "source_file", "width", "height", "extracted_at", "preprocess", "split"]
SPLIT_NAMES = ("train", "val", "test")
VALIDATION_FIELDS = ["time", "source_file", "decision", "reason", "reviewer"]

STATUS_OK = "ok"
STATUS_LOW = "low_detection"
STATUS_FAILED = "failed"


class PlainError(Exception):
    """An error shown to the team in plain language, with what to do next."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class Sign:
    category: str
    sign_id: str
    sign_label: str
    reps: int
    hands: int
    mirror_ok: int


def load_signs(data_dir: Path) -> list[Sign]:
    path = Path(data_dir) / "signs.csv"
    if not path.exists():
        raise PlainError(f"signs.csv not found in {data_dir}.",
                         "Choose the HUDYAT - HAND TRACKER folder on the Dataset page.")
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out.append(Sign(row["category"].strip(), row["sign_id"].strip(), row["sign_label"].strip(),
                            int(row.get("reps") or 15), int(row.get("hands") or 1),
                            int((row.get("mirror_ok") or "0").strip() or 0)))
    return out


def categories(signs: list[Sign]) -> list[str]:
    return list(dict.fromkeys(s.category for s in signs))


def model_categories(signs: list[Sign]) -> list[str]:
    """Categories that get a model: all except none (added to each) and the static test."""
    return [c for c in categories(signs) if c != cfg.NONE_CATEGORY and c not in cfg.EXCLUDED_CATEGORIES]


def extract_categories(signs: list[Sign], include_static: bool = False) -> list[str]:
    """Categories whose clips are extracted: every model category + none (+ the static test set if asked)."""
    out = model_categories(signs) + ([cfg.NONE_CATEGORY] if cfg.NONE_CATEGORY in categories(signs) else [])
    if include_static:
        out += [c for c in categories(signs) if c in cfg.EXCLUDED_CATEGORIES]
    return out


def read_metadata(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "dataset" / "metadata.csv"
    if not path.exists():
        return pd.DataFrame(columns=["signer", "category", "sign_id", "take", "file_path", "status"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    for c in ("signer", "category", "sign_id", "status", "file_path", "split"):
        if c in df.columns:
            df[c] = df[c].str.strip()
    if "expert_status" in df.columns:
        df["expert_status"] = df["expert_status"].str.strip().str.lower()
    return df


def recorded_rows(data_dir: Path) -> pd.DataFrame:
    """Rows of our own signers S1-S7 (never PUB or anything else), plus uploaded class-folder data, which has
    no signers but a fixed split column (train / val / test)."""
    df = read_metadata(data_dir)
    if df.empty:
        return df
    ok = df["signer"].isin(cfg.SIGNERS)
    if "split" in df.columns:
        ok |= df["split"].isin(SPLIT_NAMES)
    return df[ok].copy()


def kept_rows(data_dir: Path, signs: list[Sign] | None = None, include_static: bool = False) -> pd.DataFrame:
    """status == kept, S1-S7, a category this app extracts, expert_status != rejected."""
    df = recorded_rows(data_dir)
    df = df[df["status"] == "kept"]
    if "expert_status" in df.columns:
        df = df[df["expert_status"] != "rejected"]
    if signs is None:
        signs = load_signs(data_dir)
    df = df[df["category"].isin(extract_categories(signs, include_static))]
    return df.copy()


def npy_rel_path(row) -> str:
    """landmarks/<category>/<sign_id>/<signer>/<file>.npy (relative to the work folder)."""
    stem = Path(str(row["file_path"]).replace("\\", "/")).stem
    return f"landmarks/{row['category']}/{row['sign_id']}/{row['signer']}/{stem}.npy"


# ----------------------------------------------------------------------------
# landmarks_index.csv
# ----------------------------------------------------------------------------
def read_index(paths: cfg.Paths) -> pd.DataFrame:
    if not paths.index_csv.exists():
        return pd.DataFrame(columns=INDEX_FIELDS)
    df = pd.read_csv(paths.index_csv, dtype={"signer": str, "take": str, "split": str, "preprocess": str},
                     keep_default_na=False)
    for c in ("hand_detect_rate", "both_hands_rate", "fps"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def write_index(paths: cfg.Paths, rows: list[dict]) -> None:
    paths.index_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.index_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, paths.index_csv)


# ----------------------------------------------------------------------------
# validation.csv (this project's own review log; append-only, last decision wins)
# ----------------------------------------------------------------------------
def read_validation(paths: cfg.Paths) -> pd.DataFrame:
    if not paths.validation_csv.exists():
        return pd.DataFrame(columns=VALIDATION_FIELDS)
    return pd.read_csv(paths.validation_csv, dtype=str, keep_default_na=False)


def latest_decisions(paths: cfg.Paths) -> dict[str, dict]:
    df = read_validation(paths)
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        out[r["source_file"]] = dict(r)
    return out


def add_validation(paths: cfg.Paths, source_file: str, decision: str, reason: str, reviewer: str) -> None:
    if decision not in ("approved", "rejected"):
        raise ValueError(decision)
    paths.validation_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not paths.validation_csv.exists()
    with open(paths.validation_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VALIDATION_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "source_file": source_file,
                    "decision": decision, "reason": reason, "reviewer": reviewer})


def rejected_files(paths: cfg.Paths) -> set[str]:
    return {f for f, d in latest_decisions(paths).items() if d.get("decision") == "rejected"}


# ----------------------------------------------------------------------------
# Usable clips for training
# ----------------------------------------------------------------------------
def usable_samples(paths: cfg.Paths, data_dir: Path, settings: dict | None = None) -> pd.DataFrame:
    """Index rows that may be used: extracted ok, still kept in metadata, not rejected,
    detection >= threshold (except none). Adds a column 'excluded_reason' on a copy of all rows."""
    s = settings or cfg.load_settings()
    thr = float(s.get("min_hand_detect_rate", 0.8))
    idx = read_index(paths)
    if idx.empty:
        return idx.assign(excluded_reason=[])
    kept = kept_rows(data_dir, include_static=True)
    kept_files = set(kept["file_path"])
    rejected = rejected_files(paths)
    reasons = []
    for _, r in idx.iterrows():
        if r["status"] == STATUS_FAILED:
            reasons.append("extraction failed")
        elif r["source_file"] not in kept_files:
            reasons.append("no longer kept / expert rejected")
        elif r["source_file"] in rejected:
            reasons.append("rejected in Validation")
        elif r["category"] != cfg.NONE_CATEGORY and not (r["hand_detect_rate"] >= thr):
            reasons.append(f"hand detection below {thr:.0%}")
        else:
            reasons.append("")
    idx = idx.copy()
    idx["excluded_reason"] = reasons
    return idx
