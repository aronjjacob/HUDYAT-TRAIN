"""Datasets: the HAND TRACKER recordings (used in place, read only) and uploaded datasets (copied into
<work>/imports/<name>). Switch between them, or add one from a .zip or a folder."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from trainer import recorder
from trainer import settings as cfg
from trainer.dataset import PlainError
from trainer.upload import find_recorder_root, import_folder, import_zip, list_imports

from .common import banner, note, section

PICK_FOLDER = (
    "import tkinter as tk, sys\n"
    "from tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "p = filedialog.askdirectory(title='Choose a dataset folder', initialdir=sys.argv[1])\n"
    "print(p or '')\n"
)
LAYOUTS_HTML = (
    '<div class="note" style="font-size:13.5px;line-height:1.7">'
    "<b>What can be added</b> (as a .zip or a folder):<br>"
    "• <b>One folder per sign</b> with photos and/or videos inside, e.g. <code>Collated\\A\\*.jpg</code>, "
    "<code>Collated\\B\\*.jpg</code> ... It becomes one model with a fixed split: 70 % train, 15 % validation, "
    "15 % test (copies of the same photo stay together).<br>"
    "• <b>Videos per signer</b>: <code>category\\sign\\S1…S7\\file.mp4</code> (like the recorder's "
    "<code>dataset\\raw</code>). Split by signer: S1–S4 train, S5 validation, S6–S7 test.<br>"
    "• <b>A copy of a HUDYAT - HAND TRACKER folder</b> (signs.csv + dataset).<br>"
    "Uploads are copied into this project (<code>work\\imports</code>); the original is never changed.</div>")


def pick_folder(initial: str) -> str:
    """Native Windows folder dialog (the app runs on this computer)."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([sys.executable, "-c", PICK_FOLDER, initial or str(Path.home())], capture_output=True,
                       text=True, creationflags=flags, timeout=600)
    return r.stdout.strip()


def use(path: Path, message: str = "") -> None:
    cfg.save_settings({"data_dir": str(path), "focus_category": None})
    recorder._cache.clear()
    st.cache_data.clear()
    for k in list(st.session_state):
        if k in ("focus_sel", "ds_pick") or k.startswith(("pp_", "va_", "sp_", "tr_", "ev_", "ex_", "lv_", "ds_cat")):
            st.session_state.pop(k, None)
    if message:
        st.session_state.ds_flash = message
    st.rerun()


def flash() -> None:
    """A one-time message. Always takes the same slot, so elements below it keep their state (open boxes)."""
    msg = st.session_state.pop("ds_flash", None)
    with st.container():
        if msg:
            banner("success", msg)


def auto_detect() -> None:
    """First start: use the recorder folder at its usual place without asking."""
    s = cfg.load_settings()
    if not (s.get("data_dir") or "").strip():
        found = cfg.detect_data_dir()
        if found:
            cfg.save_settings({"data_dir": str(found)})
            st.session_state.ds_flash = f"Using the recordings in {found}."


def available() -> list[tuple[str, Path]]:
    """[(label, folder)]: the recordings (if found or in use) + every uploaded dataset."""
    out: list[tuple[str, Path]] = []
    current = cfg.data_dir()
    found = cfg.detect_data_dir()
    if found:
        out.append(("HAND TRACKER recordings", found))
    if current and not cfg.import_name(current) and (not found or Path(current) != found) and current.exists():
        out.append((f"Recordings in {current.name}", current))
    out += [(f"{name} (uploaded)", root) for name, root in list_imports()]
    return out


def label_of(path: Path | None) -> str:
    for lab, p in available():
        if path is not None and Path(p) == Path(path):
            return lab
    return str(path) if path else "none chosen"


def add_panel(key: str = "add") -> None:
    """Upload a .zip or add a folder."""
    st.html(LAYOUTS_HTML)
    tab_zip, tab_folder = st.tabs(["Upload a .zip", "Add a folder"])
    with tab_zip:
        up = st.file_uploader("Dataset (.zip)", type=["zip"], key=f"{key}_zip")
        name = st.text_input("Name", value=Path(up.name).stem if up else "", key=f"{key}_zip_name",
                             placeholder="e.g. alphabet_static")
        if st.button("Upload and use", key=f"{key}_do_zip", type="primary", disabled=up is None):
            with st.spinner(f"Unpacking {up.name}... (large files take a while)"):
                try:
                    res = import_zip(up, name or up.name)
                except PlainError as e:
                    banner("error", e.message, e.hint)
                    return
            _done(res)
    with tab_folder:
        pending = st.session_state.pop(f"{key}_pending", None)
        if pending is not None:
            st.session_state[f"{key}_folder"] = pending
        c1, c2 = st.columns([5, 1.3], vertical_alignment="bottom")
        c1.text_input("Folder", key=f"{key}_folder", placeholder=r"C:\Users\...\Downloads\alphabet-static")
        if c2.button("Browse...", key=f"{key}_browse", use_container_width=True):
            picked = pick_folder(st.session_state.get(f"{key}_folder") or str(Path.home() / "Downloads"))
            if picked:
                st.session_state[f"{key}_pending"] = str(Path(picked))
                st.rerun()
        typed = (st.session_state.get(f"{key}_folder") or "").strip()
        src = Path(typed) if typed else None
        if src is not None and not src.is_dir():
            banner("warning", f"The folder {src} does not exist.")
            return
        rec = find_recorder_root(src) if src is not None and (src / "signs.csv").exists() else None
        if rec is not None:
            note("This is a HUDYAT - HAND TRACKER folder: it is used where it is (read only, nothing is copied).")
            if st.button("Use this folder", key=f"{key}_use_rec", type="primary"):
                use(rec, f"Using the recordings in {rec}.")
        else:
            name = st.text_input("Name", value=src.name if src else "", key=f"{key}_folder_name",
                                 placeholder="e.g. alphabet_static")
            if st.button("Copy and use this folder", key=f"{key}_do_folder", type="primary", disabled=src is None):
                bar = st.progress(0.0, text="Copying...")
                try:
                    res = import_folder(src, name or src.name, progress=lambda f, t: bar.progress(f, text=t))
                except PlainError as e:
                    banner("error", e.message, e.hint)
                    return
                _done(res)


def _done(res) -> None:
    msg = f"Added '{res.name}': {res.n_used} file(s) in {len(res.labels)} sign(s) ({res.layout})."
    if res.notes:
        msg += " " + " ".join(res.notes)
    use(res.root, msg)


def switcher(key: str = "ds_pick", container=None) -> None:
    """Choose which dataset the app works on."""
    box = container or st
    opts = available()
    if not opts:
        return
    current = cfg.data_dir()
    paths = [str(p) for _, p in opts]
    labels = {str(p): lab for lab, p in opts}
    if current is not None and str(current) not in paths:
        paths.insert(0, str(current))
        labels[str(current)] = str(current)
    if st.session_state.get(key) not in paths:
        st.session_state[key] = str(current) if current is not None and str(current) in paths else paths[0]
    picked = box.selectbox("Dataset", paths, key=key, format_func=lambda p: labels.get(p, p),
                           help="Switch between your recordings and uploaded datasets. Each one keeps its own "
                                "samples, models and reports.")
    if current is None or picked != str(current):
        use(Path(picked), f"Switched to {labels.get(picked, picked)}.")


def panel(key: str = "ds") -> None:
    """Used where no dataset is chosen yet, and in Settings."""
    found = cfg.detect_data_dir()
    current = cfg.data_dir()
    if found and found != current:
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.html(f'<div class="note"><b>Recordings found:</b> {found}</div>')
        if c2.button("Use the HAND TRACKER recordings", key=f"{key}_use_found", type="primary",
                     use_container_width=True):
            use(found, f"Using the recordings in {found}.")
    if len(available()) > 1:
        switcher(f"{key}_pick")
    add_panel(f"{key}_add")


def missing(snap) -> None:
    """Shown on a page when no usable dataset is chosen: say why, and offer the choices right here."""
    banner("warning", snap.problems[0] if snap.problems else "No dataset chosen yet.",
           "Choose or add a dataset. This page fills in as soon as one is chosen.")
    section("Choose a dataset")
    panel("ds_page")
