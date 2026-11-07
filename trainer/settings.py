"""Paths and trainer_settings.json.

Everything the trainer writes goes under WORK_DIR (default: <project>/work).
The data-collection folder is only ever READ.

Tests (and the UI test) point the app at a temporary project by setting the
environment variable HUDYAT_TRAINER_SETTINGS to another settings file; its
"work_dir" then decides where outputs go.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path.home() / "Documents" / "HUDYAT - HAND TRACKER"
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
FACE_MODEL_PATH = PROJECT_DIR / "models" / "face_landmarker.task"   # face location (README 6.7); else the recorder's

SIGNERS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
NONE_CATEGORY = "none"                            # not a sign: ignored if a dataset has it (models know only signs)
EXCLUDED_CATEGORIES = ("alphabet_static_test",)   # test set of the separate static A-Y photo model

DEFAULTS = {
    "data_dir": "",                 # chosen on first run
    "work_dir": "work",             # relative to the project folder, or absolute
    "min_hand_detect_rate": 0.80,   # clips below this (inside the sign window) are left out
    "camera": None,                 # None = use the recorder's settings.json
    "backend": None,
    "confidence_threshold": 0.85,   # live test: below this shows "Not sure, try again"
    "default_hand": "auto",         # live test: the tester's hand (right / left / auto)
}


def settings_path() -> Path:
    return Path(os.environ.get("HUDYAT_TRAINER_SETTINGS", PROJECT_DIR / "trainer_settings.json"))


DATA_DIR_ENV = "HUDYAT_DATA_DIR"      # set for a background job: it keeps working on the dataset it started with
IMPORTS = "imports"                  # uploaded datasets: <work root>/imports/<name>/data (+ its own .../work)


def load_settings() -> dict:
    s = dict(DEFAULTS)
    p = settings_path()
    if p.exists():
        try:
            s.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    if os.environ.get(DATA_DIR_ENV):
        s["data_dir"] = os.environ[DATA_DIR_ENV]
    return s


def save_settings(updates: dict) -> dict:
    s = load_settings()
    s.update(updates)
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return s


def data_dir(settings: dict | None = None) -> Path | None:
    s = settings or load_settings()
    d = (s.get("data_dir") or "").strip()
    return Path(d) if d else None


def work_root(settings: dict | None = None) -> Path:
    """The top work folder (outputs of the recorder dataset; uploaded datasets live in imports/ under it)."""
    s = settings or load_settings()
    w = Path(s.get("work_dir") or "work")
    if not w.is_absolute():
        w = settings_path().parent / w
    return w


def imports_dir(settings: dict | None = None) -> Path:
    return work_root(settings) / IMPORTS


def import_name(data: Path | None, settings: dict | None = None) -> str | None:
    """Name of an uploaded dataset if `data` is one (<root>/imports/<name>/data), else None."""
    if data is None:
        return None
    try:
        rel = Path(data).resolve().relative_to(imports_dir(settings).resolve())
    except (ValueError, OSError):
        return None
    return rel.parts[0] if rel.parts else None


def work_dir(settings: dict | None = None) -> Path:
    """Where this dataset's outputs go: the work root for the recorder data, imports/<name>/work for uploads."""
    s = settings or load_settings()
    name = import_name(data_dir(s), s)
    return imports_dir(s) / name / "work" if name else work_root(s)


class Paths:
    """All output locations under the work folder."""

    def __init__(self, settings: dict | None = None) -> None:
        self.work = work_dir(settings)
        self.dataset = import_name(data_dir(settings), settings) or "recorder"     # name used in export folders
        self.landmarks = self.work / "landmarks"
        self.index_csv = self.work / "landmarks_index.csv"
        self.duplicate_log = self.work / "landmarks_duplicate_handedness.log"
        self.splits = self.work / "splits"
        self.models = self.work / "models"
        self.reports = self.work / "reports"
        self.summary_csv = self.reports / "summary.csv"
        self.validation_csv = self.work / "validation.csv"
        self.live_tests = self.work / "live_tests"
        self.experiments = self.work / "experiments"          # Test 1 / Test 2 runs: <test>/<category>/<run_id>/
        self.paper = self.work / "paper"                      # figures and tables for the paper
        self.exports = self.work / "exports"                  # named folders for the HUDYAT app (app_export.py)
        self.datasets = self.work / "datasets"                # landmark datasets to share (dataset_export.py)
        self.jobs = self.work / "jobs"
        self.lock = work_root(settings) / "job.lock"        # one job at a time, whatever the dataset
        self.preview_cache = self.work / "preview_cache"

    def model_dir(self, category: str) -> Path:
        return self.models / category

    def report_dir(self, category: str) -> Path:
        return self.reports / category

    def split_json(self, category: str) -> Path:
        return self.splits / f"{category}.json"


def check_data_dir(path: Path | None) -> list[str]:
    """Plain-language problems with a data-collection folder ([] = fine)."""
    if path is None:
        return ["No dataset chosen yet."]
    problems = []
    if not path.is_dir():
        return [f"The folder {path} does not exist."]
    if not (path / "signs.csv").exists():
        problems.append(f"signs.csv is missing from {path}. Is this the HUDYAT - HAND TRACKER folder?")
    if not (path / "dataset" / "metadata.csv").exists():
        problems.append("dataset/metadata.csv not found: nothing has been recorded in this folder yet.")
    return problems


def landmarks_only(data_dir: Path | None) -> bool:
    """A dataset exported by dataset_export.py: the samples come with it, there are no videos to extract."""
    p = Path(data_dir) / "dataset.json" if data_dir else None
    if not p or not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("landmarks_only"))
    except (OSError, ValueError):
        return False


def detect_data_dir() -> Path | None:
    """The recorder folder at its usual place, if it is a valid data-collection folder."""
    return DEFAULT_DATA_DIR if not check_data_dir(DEFAULT_DATA_DIR) else None
