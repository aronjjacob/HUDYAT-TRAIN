"""Datasets: the HAND TRACKER recordings (used in place, read only) and uploaded datasets (copied into
<work>/imports/<name>). Switch between them, or add one from a .zip or a folder (scan first, then Import)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from trainer import recorder
from trainer import settings as cfg
from trainer.dataset import PlainError
from trainer.upload import find_recorder_root, import_folder, import_zip, list_imports, scan_folder, scan_zip

from .common import banner, note, plural, section, stats

PICK_FOLDER = (
    "import tkinter as tk, sys\n"
    "from tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "p = filedialog.askdirectory(title='Choose a dataset folder', initialdir=sys.argv[1])\n"
    "print(p or '')\n"
)
LAYOUTS_HTML = (
    '<div class="note" style="font-size:13.5px;line-height:1.7">'
    "<b>What can be added</b>, as a .zip or a folder:<br>"
    "• <b>A copy of a HUDYAT - HAND TRACKER folder</b> with signs.csv and dataset.<br>"
    r"• <b>Videos per signer</b>: <code>category\sign\S1…S7\file.mp4</code>, like the recorder's "
    r"<code>dataset\raw</code>.<br>"
    "• <b>A photo dataset</b> with one folder per letter, e.g. the Kaggle FSL alphabet: "
    r"<code>A\*.jpg</code>, <code>B\*.jpg</code> … Photos have no signer IDs, so only Test 1 is possible.<br>"
    "You see what was found first; nothing is copied until you press <b>Import</b>. The original is never "
    "changed.</div>")
SESSION_PREFIXES = ("pp_", "dt_", "ts_", "fm_", "lv_", "pa_", "cat_")


def pick_folder(initial: str) -> str:
    """Native Windows folder dialog (the app runs on this computer)."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([sys.executable, "-c", PICK_FOLDER, initial or str(Path.home())], capture_output=True,
                       text=True, creationflags=flags, timeout=600)
    return r.stdout.strip()


def use(path: Path, message: str = "") -> None:
    cfg.save_settings({"data_dir": str(path)})
    recorder._cache.clear()
    st.cache_data.clear()
    for k in list(st.session_state):
        if k in ("ds_pick", "shared_cat") or k.startswith(SESSION_PREFIXES):
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
    out += [(f"{name} · uploaded", root) for name, root in list_imports()]
    return out


def label_of(path: Path | None) -> str:
    for lab, p in available():
        if path is not None and Path(p) == Path(path):
            return lab
    return str(path) if path else "none chosen"


def show_scan(res) -> None:
    """What a scan found, in plain language."""
    who = (f"{len(res.signers)}" if res.signers else "none, photos" if res.n_photos else "none")
    stats([("Layout", res.layout or "—", ""), ("Signers", who, ""), ("Signs", len(res.signs), ""),
           ("Photos" if res.n_photos and res.n_photos == res.n_clips else "Clips", res.n_clips,
            "" if res.n_clips else "err")])
    if res.signers:
        note("Signers: " + ", ".join(f"{g} {h + '-handed' if h else 'right-handed, hand not given'}"
                                     for g, h in res.signers.items()) + ".")
    for p in res.problems:
        banner("error", p, "Fix the folder or the .zip and try again.")
    for n in res.notes:
        note(n)


def add_panel(key: str = "add") -> None:
    """Upload a .zip or add a folder: what is in it is shown first, Import copies it."""
    st.html(LAYOUTS_HTML)
    tab_zip, tab_folder = st.tabs(["Upload a .zip", "Add a folder"])
    with tab_zip:
        up = st.file_uploader("Dataset .zip", type=["zip"], key=f"{key}_zip")
        if up is not None:
            sig = (up.name, up.size)
            if st.session_state.get(f"{key}_zip_sig") != sig:
                with st.spinner(f"Looking inside {up.name}..."):
                    st.session_state[f"{key}_zip_scan"] = scan_zip(up)
                st.session_state[f"{key}_zip_sig"] = sig
            res = st.session_state[f"{key}_zip_scan"]
            show_scan(res)
            name = st.text_input("Name", value=Path(up.name).stem, key=f"{key}_zip_name",
                                 placeholder="e.g. fsl_alphabet")
            if st.button("Import", key=f"{key}_do_zip", type="primary", disabled=not res.ok):
                with st.spinner(f"Unpacking {up.name}... Large files take a while."):
                    try:
                        done = import_zip(up, name or up.name)
                    except PlainError as e:
                        banner("error", e.message, e.hint)
                        return
                _done(done)
    with tab_folder:
        pending = st.session_state.pop(f"{key}_pending", None)
        if pending is not None:
            st.session_state[f"{key}_folder"] = pending
        c1, c2 = st.columns([5, 1.3], vertical_alignment="bottom")
        c1.text_input("Folder", key=f"{key}_folder", placeholder=r"C:\Users\...\Downloads\fsl-alphabet")
        if c2.button("Browse...", key=f"{key}_browse", use_container_width=True):
            picked = pick_folder(st.session_state.get(f"{key}_folder") or str(Path.home() / "Downloads"))
            if picked:
                st.session_state[f"{key}_pending"] = str(Path(picked))
                st.rerun()
        typed = (st.session_state.get(f"{key}_folder") or "").strip()
        src = Path(typed) if typed else None
        if src is None:
            return
        if not src.is_dir():
            banner("warning", f"The folder {src} does not exist.")
            return
        rec = find_recorder_root(src) if (src / "signs.csv").exists() else None
        if rec is not None:
            note("This is a HUDYAT - HAND TRACKER folder: it is used where it is, read only; nothing is copied.")
            if st.button("Use this folder", key=f"{key}_use_rec", type="primary"):
                use(rec, f"Using the recordings in {rec}.")
            return
        with st.spinner("Looking inside the folder..."):
            res = scan_folder(src)
        show_scan(res)
        name = st.text_input("Name", value=src.name, key=f"{key}_folder_name", placeholder="e.g. fsl_alphabet")
        if st.button("Import", key=f"{key}_do_folder", type="primary", disabled=not res.ok):
            bar = st.progress(0.0, text="Copying...")
            try:
                done = import_folder(src, name or src.name, progress=lambda f, t: bar.progress(f, text=t))
            except PlainError as e:
                banner("error", e.message, e.hint)
                return
            _done(done)


def _done(res) -> None:
    msg = f"Imported '{res.name}': {plural(res.n_used, 'file')} in {plural(len(res.labels), 'sign')}."
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
                                "samples, tests and models.")
    if current is None or picked != str(current):
        use(Path(picked), f"Switched to {labels.get(picked, picked)}.")


def panel(key: str = "ds") -> None:
    """Choose the recordings, switch, or add a dataset (used on Data when nothing is chosen, and in Settings)."""
    found = cfg.detect_data_dir()
    current = cfg.data_dir()
    if found and found != current:
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.html('<div class="note"><b>Recordings found</b> in Documents\\HUDYAT - HAND TRACKER.</div>')
        if c2.button("Use the HAND TRACKER recordings", key=f"{key}_use_found", type="primary",
                     use_container_width=True):
            use(found, "Using the HAND TRACKER recordings.")
    if len(available()) > 1:
        switcher(f"{key}_pick")
    add_panel(f"{key}_add")


def missing(snap) -> None:
    """Shown on a page when no usable dataset is chosen: say why, and offer the choices right here."""
    banner("warning", snap.problems[0] if snap.problems else "No dataset chosen yet.",
           "Choose or add a dataset. This page fills in as soon as one is chosen.")
    section("Choose a dataset")
    panel("ds_page")
