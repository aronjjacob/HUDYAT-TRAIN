"""What exists on disk, the pages of the menu, and the category picker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

from trainer import dataset as ds
from trainer import settings as cfg

# The menu: every page can be opened at any time (a page that needs something says what, in plain words).
PAGES = [("data", "Data", ":material/folder_open:"), ("preprocess", "Preprocess", ":material/back_hand:"),
         ("test", "Test the model", ":material/fact_check:"), ("final", "Train model", ":material/model_training:"),
         ("live", "Live test", ":material/videocam:"), ("paper", "Results for the paper", ":material/article:")]
PAGE_OBJS: dict = {}                   # page key -> st.Page, filled by trainer_app.py (for the "Next" links)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=8)
def _metadata(data_dir: str, _mt: float) -> pd.DataFrame:
    return ds.read_metadata(Path(data_dir))


@st.cache_data(show_spinner=False, max_entries=8)
def _signs(data_dir: str, _mt: float) -> list[ds.Sign]:
    return ds.load_signs(Path(data_dir))


@st.cache_data(show_spinner=False, max_entries=8)
def _hands(data_dir: str, _mt: float, _mt2: float) -> dict[str, str]:
    return ds.clip_hands(Path(data_dir))


def metadata(data_dir: Path) -> pd.DataFrame:
    return _metadata(str(data_dir), _mtime(data_dir / "dataset" / "metadata.csv")).copy()


def signs(data_dir: Path) -> list[ds.Sign]:
    return _signs(str(data_dir), _mtime(data_dir / "signs.csv"))


def clip_hands(data_dir: Path) -> dict[str, str]:
    return _hands(str(data_dir), _mtime(data_dir / "dataset" / "metadata.csv"), _mtime(data_dir / "signers.csv"))


def kept_rows(data_dir: Path, include_static: bool = False) -> pd.DataFrame:
    """Same rule as trainer.dataset.kept_rows, from the cached metadata."""
    df = metadata(data_dir)
    if df.empty or "signer" not in df.columns:
        return df
    ours = df["signer"].isin(cfg.SIGNERS)
    if "split" in df.columns:                           # uploaded photo data: no signers
        ours |= df["split"].isin(ds.SPLIT_NAMES)
    df = df[ours & (df["status"] == "kept")]
    if "expert_status" in df.columns:
        df = df[df["expert_status"] != "rejected"]
    return df[df["category"].isin(ds.extract_categories(signs(data_dir), include_static))].copy()


@dataclass
class Snapshot:
    settings: dict
    paths: cfg.Paths
    data_dir: Path | None
    problems: list[str]
    signs: list = field(default_factory=list)
    kept: pd.DataFrame = field(default_factory=pd.DataFrame)
    index: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_todo: int = 0                    # clips to extract (new, failed, or in the old 128 format)
    n_old: int = 0                     # of those: extracted before v2
    n_rich: int = 0                    # samples ready for the tests
    by_cat: dict = field(default_factory=dict)            # {category: (kept clips, to extract)}
    model_cats: list[str] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)       # categories with a final model (v2)
    exported: list[str] = field(default_factory=list)
    live_cats: list[str] = field(default_factory=list)
    photos: bool = False               # uploaded photo dataset: no signers, Test 1 only
    name: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def signers(self) -> list[str]:
        if self.kept.empty:
            return []
        from trainer.splits import signer_order
        return signer_order(g for g in self.kept["signer"] if g in cfg.SIGNERS)


def snapshot() -> Snapshot:
    s = cfg.load_settings()
    paths = cfg.Paths(s)
    d = cfg.data_dir(s)
    snap = Snapshot(s, paths, d, cfg.check_data_dir(d))
    if snap.problems:
        return snap
    try:
        snap.signs = signs(d)
        snap.kept = kept_rows(d)
    except Exception as e:  # noqa: BLE001 - unreadable csv -> plain message
        snap.problems = [f"Could not read the data folder: {e}"]
        return snap
    snap.model_cats = ds.model_categories(snap.signs)
    snap.photos = not snap.kept.empty and not snap.kept["signer"].isin(cfg.SIGNERS).any()
    snap.name = cfg.import_name(d) or "HAND TRACKER recordings"
    snap.index = ds.read_index(paths)
    rich = set()
    if not snap.index.empty:
        ok = snap.index[(snap.index["status"] != ds.STATUS_FAILED) & (snap.index["format"] == ds.SAMPLE_FORMAT)]
        rich = {f for f in ok["file"] if (paths.work / f).exists()}
        old = snap.index[(snap.index["status"] != ds.STATUS_FAILED) & (snap.index["format"] != ds.SAMPLE_FORMAT)]
        old_files = set(old["file"])
    else:
        old_files = set()
    if not snap.kept.empty:
        files = [ds.npy_rel_path(r) for _, r in snap.kept.iterrows()]
        snap.n_todo = sum(f not in rich for f in files)
        snap.n_old = sum(f in old_files for f in files)
        snap.n_rich = len(files) - snap.n_todo
        for c, f in zip(snap.kept["category"], files):
            n, t = snap.by_cat.get(c, (0, 0))
            snap.by_cat[c] = (n + 1, t + (f not in rich))
    from trainer.experiments import list_runs
    from trainer.train import model_exists
    snap.runs = list_runs(paths)
    snap.finals = [c for c in snap.model_cats
                   if model_exists(paths, c) and (paths.model_dir(c) / "preprocess.json").exists()]
    from trainer.app_export import last_export
    snap.exported = [c for c in snap.finals if last_export(paths, c)]
    snap.live_cats = sorted(p.stem for p in paths.live_tests.glob("*.csv")) if paths.live_tests.exists() else []
    return snap


def steps_done(snap: Snapshot) -> dict[str, bool]:
    """Which steps of the usual order have been done at least once (pages stay open either way)."""
    paper = snap.paths.paper
    return {"data": snap.ok and not snap.kept.empty,
            "preprocess": bool(snap.n_rich) and not snap.n_todo,
            "test": any(r["complete"] for r in snap.runs),
            "final": bool(snap.finals),
            "live": bool(snap.live_cats),
            "paper": paper.exists() and any(paper.glob("*/paper.json"))}


def next_step(snap: Snapshot) -> str | None:
    """The first step that hasn't been done yet."""
    done = steps_done(snap)
    return next((k for k, _, _ in PAGES if not done[k]), None)


def step_label(key: str) -> str:
    i = [k for k, _, _ in PAGES].index(key)
    return f"Step {i + 1} of {len(PAGES)}"


# ----------------------------------------------------------------------------
# One category picker per page, remembered across pages
# ----------------------------------------------------------------------------
SHARED = "shared_cat"


def pick_category(options: list[str], page: str, label: str = "Category", fmt=None, box=None,
                  help: str | None = None) -> str:
    box = box or st
    key = f"cat_{page}"
    shared = st.session_state.get(SHARED)
    if st.session_state.get(key) not in options:
        st.session_state[key] = shared if shared in options else options[0]
    kw = {"format_func": fmt} if fmt else {}
    val = box.selectbox(label, options, key=key, help=help, **kw)
    st.session_state[SHARED] = val
    return val
