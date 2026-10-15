"""What exists on disk, and the state of each of the 8 steps (done / ready / locked) for the sidebar."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

from trainer import dataset as ds
from trainer import settings as cfg
from trainer.train import trained_categories

STEPS = [("dataset", "Dataset"), ("preprocess", "Preprocess"), ("validation", "Validation"), ("split", "Split"),
         ("train", "Train"), ("evaluate", "Evaluate"), ("export", "Export"), ("live", "Live Test")]
DONE, READY, LOCKED = "done", "ready", "locked"


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


def metadata(data_dir: Path) -> pd.DataFrame:
    return _metadata(str(data_dir), _mtime(data_dir / "dataset" / "metadata.csv")).copy()


def signs(data_dir: Path) -> list[ds.Sign]:
    return _signs(str(data_dir), _mtime(data_dir / "signs.csv"))


def kept_rows(data_dir: Path, include_static: bool = False) -> pd.DataFrame:
    """Same rule as trainer.dataset.kept_rows, from the cached metadata."""
    df = metadata(data_dir)
    if df.empty or "signer" not in df.columns:
        return df
    ours = df["signer"].isin(cfg.SIGNERS)
    if "split" in df.columns:                           # uploaded class-folder data: fixed split, no signers
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
    n_todo: int = 0
    model_cats: list[str] = field(default_factory=list)
    split_cats: list[str] = field(default_factory=list)
    trained: list[str] = field(default_factory=list)
    evaluated: list[str] = field(default_factory=list)
    exported: list[str] = field(default_factory=list)
    live_cats: list[str] = field(default_factory=list)
    n_reviewed: int = 0
    done_files: set = field(default_factory=set)
    fixed_split: bool = False          # uploaded class-folder data: train / val / test fixed at upload (no signers)
    name: str = ""                     # label of the dataset in use

    @property
    def ok(self) -> bool:
        return not self.problems


def snapshot() -> Snapshot:
    s = cfg.load_settings()
    paths = cfg.Paths(s)
    d = cfg.data_dir(s)
    problems = cfg.check_data_dir(d)
    snap = Snapshot(s, paths, d, problems)
    if problems:
        return snap
    try:
        snap.signs = signs(d)
        snap.kept = kept_rows(d)
    except Exception as e:  # noqa: BLE001 - unreadable csv -> plain message
        snap.problems = [f"Could not read the data-collection folder: {e}"]
        return snap
    snap.model_cats = ds.model_categories(snap.signs)
    snap.fixed_split = (not snap.kept.empty and "split" in snap.kept.columns
                        and bool(snap.kept["split"].isin(ds.SPLIT_NAMES).any()))
    snap.name = cfg.import_name(d) or "HAND TRACKER recordings"
    snap.index = ds.read_index(paths)
    done = set()
    if not snap.index.empty:
        ok = snap.index[snap.index["status"] != ds.STATUS_FAILED]
        done = {f for f in ok["file"] if (paths.work / f).exists()}
    snap.done_files = done
    if not snap.kept.empty:
        snap.n_todo = int(sum(ds.npy_rel_path(r) not in done for _, r in snap.kept.iterrows()))
    snap.split_cats = [c for c in snap.model_cats if paths.split_json(c).exists()]
    snap.trained = trained_categories(paths)
    snap.evaluated = [c for c in snap.trained if (paths.report_dir(c) / "metrics.json").exists()]
    snap.exported = [c for c in snap.trained if (paths.model_dir(c) / "tfjs" / "model.json").exists()]
    snap.live_cats = sorted(p.stem for p in paths.live_tests.glob("*.csv")) if paths.live_tests.exists() else []
    if not snap.kept.empty:
        dec = ds.latest_decisions(paths)
        snap.n_reviewed = int(sum(f in dec for f in snap.kept["file_path"]))
    return snap


def step_states(snap: Snapshot, focus_cat: str = "__all__") -> dict[str, tuple[str, str]]:
    """{step: (state, short reason)}, for all categories or for the category chosen in the sidebar."""
    st_: dict[str, tuple[str, str]] = {}
    if not snap.ok:
        reason = "Choose the dataset first."
        return {k: (LOCKED, reason) for k, _ in STEPS}
    if focus_cat != "__all__":
        return _focus_states(snap, focus_cat)
    has_clips = not snap.kept.empty
    st_["dataset"] = (DONE, "") if has_clips else (READY, "Nothing recorded yet.")
    if not has_clips:
        st_["preprocess"] = (LOCKED, "No kept S1-S7 clips yet.")
    elif snap.n_todo == 0:
        st_["preprocess"] = (DONE, "")
    else:
        st_["preprocess"] = (READY, f"{snap.n_todo} clip(s) to extract.")
    if not has_clips:
        st_["validation"] = (LOCKED, "No kept S1-S7 clips yet.")
    elif snap.n_reviewed >= len(snap.kept):
        st_["validation"] = (DONE, "")
    else:
        st_["validation"] = (READY, f"{len(snap.kept) - snap.n_reviewed} clip(s) not reviewed (optional).")
    if snap.index.empty:
        st_["split"] = (LOCKED, "Extract landmarks first (step 2).")
    elif snap.split_cats and len(snap.split_cats) == len(snap.model_cats):
        st_["split"] = (DONE, "")
    else:
        st_["split"] = (READY, "")
    if not snap.split_cats:
        st_["train"] = (LOCKED, "Split first (step 4).")
    elif all(c in snap.trained for c in snap.split_cats):
        st_["train"] = (DONE, "")
    else:
        st_["train"] = (READY, "")
    for key, have in (("evaluate", snap.evaluated), ("export", snap.exported)):
        if not snap.trained:
            st_[key] = (LOCKED, "Train a model first (step 5).")
        elif all(c in have for c in snap.trained):
            st_[key] = (DONE, "")
        else:
            st_[key] = (READY, "")
    if not snap.trained:
        st_["live"] = (LOCKED, "Train a model first (step 5).")
    else:
        st_["live"] = (DONE, "") if snap.live_cats else (READY, "")
    return st_


def next_step(states: dict[str, tuple[str, str]]) -> str | None:
    for k, title in STEPS:
        if states[k][0] == READY and k != "validation":
            return k
    return None


# ----------------------------------------------------------------------------
# The category the team is working on (sidebar). "All categories" = every page shows its own picker.
# ----------------------------------------------------------------------------
ALL_CATS = "__all__"


def focus(snap: Snapshot) -> str:
    """The category chosen at the top of the sidebar (ALL_CATS if none / no longer valid)."""
    if len(snap.model_cats) == 1:                       # e.g. an uploaded dataset: one model, nothing to choose
        return snap.model_cats[0]
    f = st.session_state.get("focus_sel")
    if f is None:
        f = snap.settings.get("focus_category") or ALL_CATS
    return f if f in snap.model_cats else ALL_CATS


def page_category(snap: Snapshot, label: str, options: list[str], key: str, fmt=None, container=None) -> str:
    """The page's category: the sidebar's choice if one is set, else this page's own picker."""
    f = focus(snap)
    box = container or st
    if f != ALL_CATS:
        where = ("the only model of this dataset" if len(snap.model_cats) == 1
                 else "chosen at the top of the sidebar")
        box.html(f'<div class="note" style="margin:4px 0 8px 0">Category: <b style="color:#1A1A1A">{f}</b> '
                 f'· {where}</div>')
        return f
    return box.selectbox(label, options, key=key, format_func=fmt) if fmt else box.selectbox(label, options, key=key)


def _focus_states(snap: Snapshot, f: str) -> dict[str, tuple[str, str]]:
    st_: dict[str, tuple[str, str]] = {}
    kept = snap.kept
    mine = kept[kept["category"] == f] if not kept.empty else kept
    st_["dataset"] = (DONE, "") if not mine.empty else (READY, f"Nothing recorded for {f} yet.")
    if mine.empty:
        st_["preprocess"] = (LOCKED, f"No kept S1-S7 clips of {f} yet.")
        st_["validation"] = (LOCKED, f"No kept S1-S7 clips of {f} yet.")
    else:
        need = kept[kept["category"].isin([f, cfg.NONE_CATEGORY])]           # none only if the dataset has it
        todo = int(sum(ds.npy_rel_path(r) not in snap.done_files for _, r in need.iterrows()))
        st_["preprocess"] = (DONE, "") if todo == 0 else (READY, f"{todo} clip(s) of {f} + none to extract.")
        dec = ds.latest_decisions(snap.paths)
        left = int(sum(x not in dec for x in mine["file_path"]))
        st_["validation"] = (DONE, "") if left == 0 else (READY, f"{left} clip(s) of {f} not reviewed (optional).")
    has_samples = any(x.startswith(f"landmarks/{f}/") for x in snap.done_files)
    if not has_samples:
        st_["split"] = (LOCKED, f"Extract {f} first (step 2).")
    else:
        st_["split"] = (DONE, "") if f in snap.split_cats else (READY, "")
    if f not in snap.split_cats:
        st_["train"] = (LOCKED, f"Split {f} first (step 4).")
    else:
        st_["train"] = (DONE, "") if f in snap.trained else (READY, "")
    for key, have in (("evaluate", snap.evaluated), ("export", snap.exported), ("live", snap.live_cats)):
        if f not in snap.trained:
            st_[key] = (LOCKED, f"Train {f} first (step 5).")
        else:
            st_[key] = (DONE, "") if f in have else (READY, "")
    return st_
