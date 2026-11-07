"""Settings: dataset (folder / upload), camera, thresholds."""

from __future__ import annotations

import hashlib

import streamlit as st

from trainer import recorder
from trainer import settings as cfg

from . import data_source
from .common import banner, note, page_header, section


@st.cache_data(show_spinner=False)
def _sha256(path: str, _mt: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render() -> None:
    s = cfg.load_settings()
    page_header(None, "Settings", "Dataset, camera and thresholds.")

    section("Dataset")
    data_source.flash()
    current = cfg.data_dir(s)
    if current and not cfg.check_data_dir(current):
        banner("success", f"In use: {current}")
    data_source.panel("set")
    problems = cfg.check_data_dir(current)
    chosen = current

    section("Camera for the live test")
    rec = recorder.recorder_camera_settings(chosen if not problems else None)
    cams = list(range(6))
    cur_cam = s.get("camera") if s.get("camera") is not None else rec.get("camera", 0)
    cur_be = s.get("backend") or rec.get("backend", "msmf")
    if int(cur_cam) not in cams:
        cams.append(int(cur_cam))
    c1, c2 = st.columns(2)
    cam = c1.selectbox("Camera", cams, index=cams.index(int(cur_cam)) if int(cur_cam) in cams else 0,
                       format_func=lambda i: f"Camera {i}" + (f" · {rec['camera_name']}" if i == rec.get("camera")
                                                             and rec.get("camera_name") else ""), key="cam")
    be = c2.selectbox("Backend", ["msmf", "dshow"], index=0 if cur_be == "msmf" else 1, key="backend",
                      format_func=lambda b: {"msmf": "MSMF · Media Foundation, default", "dshow": "DirectShow"}[b])
    note(f"Recorder: camera {rec.get('camera', 0)}. Wrong camera or black screen? Try another number.")

    section("Thresholds")
    c1, c2 = st.columns(2)
    det = c1.slider("Minimum hand detection", 0.5, 1.0,
                    float(s.get("min_hand_detect_rate", 0.8)), 0.05, format="%.2f", key="det",
                    help="Clips below this are left out of the tests and models.")
    thr = c2.slider("Live test: minimum confidence", 0.3, 0.95, float(s.get("confidence_threshold", 0.85)), 0.05,
                    format="%.2f", key="thr", help="Below this: 'Not sure, try again'.")
    hands = ["auto", "right", "left"]
    cur_hand = s.get("default_hand") if s.get("default_hand") in hands else "auto"
    hand = st.radio("Live test: default hand of the tester", hands, index=hands.index(cur_hand), horizontal=True,
                    key="hand", format_func=lambda h: {"auto": "Auto, detect per sign", "right": "Right",
                                                       "left": "Left"}[h],
                    help="Can be changed per tester, or with H in the live window.")
    changed = (cam != cur_cam or be != cur_be or det != float(s.get("min_hand_detect_rate", 0.8))
               or thr != float(s.get("confidence_threshold", 0.85)) or hand != cur_hand)
    if st.button("Save settings", key="save_settings", type="primary", disabled=not changed):
        cfg.save_settings({"camera": int(cam), "backend": be, "min_hand_detect_rate": float(det),
                           "confidence_threshold": float(thr), "default_hand": hand})
        st.rerun()

    section("Checks")
    if cfg.MODEL_PATH.exists():
        ok = _sha256(str(cfg.MODEL_PATH), cfg.MODEL_PATH.stat().st_mtime) == cfg.MODEL_SHA256
        banner("success" if ok else "error",
               "MediaPipe hand model: OK" if ok else "MediaPipe hand model: checksum does not match",
               "" if ok else "Run 'SETUP (run once).bat' again.")
    else:
        banner("error", "MediaPipe hand model is missing.", "Run 'SETUP (run once).bat' again.")
    hc = recorder.load_common(chosen if not problems else None)
    if getattr(hc, "imported_from", None):
        banner("success", "Trim and handedness rules: imported from the recorder's hudyat_common.py.")
    else:
        banner("warning", "Trim and handedness rules: using the trainer's own copies.",
               "They are tested to give identical results. Choose the data-collection folder to use the recorder's file.")
    with st.expander("Details"):
        note(f"Work folder with all outputs: {cfg.Paths(s).work}<br>Settings file: {cfg.settings_path()}")
