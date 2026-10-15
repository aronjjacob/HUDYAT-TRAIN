"""HUDYAT Trainer app.  Start: double-click 'START HUDYAT TRAINER.bat'  (or: streamlit run trainer_app.py)."""

import sys
import traceback

import streamlit as st

st.set_page_config(page_title="HUDYAT Trainer", layout="wide", initial_sidebar_state="expanded")

from trainer import jobs  # noqa: E402
from ui import (common, data_source, p_dataset, p_evaluate, p_export, p_live, p_preprocess, p_settings, p_split,  # noqa: E402
                p_train, p_validation, state)

RENDER = {"dataset": p_dataset.render, "preprocess": p_preprocess.render, "validation": p_validation.render,
          "split": p_split.render, "train": p_train.render, "evaluate": p_evaluate.render, "export": p_export.render,
          "live": p_live.render}
ICONS = {state.DONE: ":material/check_circle:", state.READY: ":material/radio_button_unchecked:",
         state.LOCKED: ":material/lock:"}
STATE_WORD = {state.DONE: "done", state.READY: "ready", state.LOCKED: "locked"}


def guarded(fn, *args):
    """Show any unexpected error as a plain banner with the details folded away."""
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("RerunException", "StopException"):    # st.rerun / st.stop: not errors
            raise
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        common.banner("error", "Something went wrong on this page.",
                      "Try again. If it keeps happening, send the details below to the developers.")
        with st.expander("Details"):
            st.code(traceback.format_exc(), language=None)


common.inject_css()
data_source.auto_detect()
snap = state.snapshot()


def make_page(key):
    def page():
        guarded(RENDER[key], snap)
    page.__name__ = f"page_{key}"
    return page


def settings_page():
    guarded(p_settings.render)


pages = {}
for i, (key, title) in enumerate(state.STEPS):
    pages[key] = st.Page(make_page(key), title=f"{title}", url_path=key, default=(key == "dataset"))
pages["settings"] = st.Page(settings_page, title="Settings", url_path="settings")
nav = st.navigation(list(pages.values()), position="hidden")

with st.sidebar:
    st.html('<div class="sb-title">HUDYAT Trainer</div>'
            '<div class="sb-sub">FSL clips → landmarks → models</div>')
    data_source.switcher("ds_pick")
    if snap.ok and len(snap.model_cats) > 1:
        opts = [state.ALL_CATS] + snap.model_cats
        if "focus_sel" not in st.session_state:
            saved = snap.settings.get("focus_category")
            st.session_state.focus_sel = saved if saved in opts else state.ALL_CATS
        elif st.session_state.focus_sel not in opts:
            st.session_state.focus_sel = state.ALL_CATS
        f = st.selectbox("Working on", opts, key="focus_sel",
                         format_func=lambda c: "All categories" if c == state.ALL_CATS else c,
                         help="Pick one category to do every step for it alone: extract, split, train, evaluate, "
                              "export and live-test one model at a time.")
        if f != (snap.settings.get("focus_category") or state.ALL_CATS):
            from trainer import settings as _cfg
            _cfg.save_settings({"focus_category": None if f == state.ALL_CATS else f})
    states = state.step_states(snap, state.focus(snap))
    for i, (key, title) in enumerate(state.STEPS):
        s, reason = states[key]
        with st.container(key=f"step_{s}_{key}"):
            st.page_link(pages[key], label=f"{i + 1}. {title}", icon=ICONS[s],
                         help=reason or STATE_WORD[s].capitalize())
    st.html("<style>"
            '[class*="st-key-step_done"] [data-testid="stIconMaterial"]{color:#2E7D32}'
            '[class*="st-key-step_ready"] [data-testid="stIconMaterial"]{color:#1F4E99}'
            '[class*="st-key-step_locked"] [data-testid="stIconMaterial"]{color:#B5B5B0}'
            '[class*="st-key-step_locked"] p{color:#6B6B6B}'
            "</style>")
    st.html('<div class="sb-legend">✓ done · ○ ready · locked = an earlier step is needed</div>')
    nxt = state.next_step(states)
    if not snap.ok:
        st.html('<div class="sb-next">Start here: <b>choose the dataset</b> (a folder or a .zip upload) on '
                'any page.</div>')
    elif nxt:
        n = [k for k, _ in state.STEPS].index(nxt) + 1
        st.html(f'<div class="sb-next">Next: <b>{n}. {dict(state.STEPS)[nxt]}</b></div>')
    running = jobs.busy(snap.paths)
    if running:
        st.html(f'<div class="sb-job">{jobs.JOB_TITLES.get(running, running)} is running…</div>')
    st.divider()
    st.page_link(pages["settings"], label="Settings", icon=":material/settings:")

nav.run()
