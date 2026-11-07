"""HUDYAT Trainer app.  Start: double-click 'START HUDYAT TRAINER.bat'  (or: streamlit run trainer_app.py)."""

import sys
import traceback

import streamlit as st

st.set_page_config(page_title="HUDYAT Trainer", layout="wide", initial_sidebar_state="expanded")

from trainer import jobs  # noqa: E402
from ui import common, data_source, p_data, p_final, p_live, p_paper, p_preprocess, p_settings, p_test, state  # noqa: E402

RENDER = {"data": p_data.render, "preprocess": p_preprocess.render, "test": p_test.render, "final": p_final.render,
          "live": p_live.render, "paper": p_paper.render}


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
        guarded(common.next_link, snap, key)
    page.__name__ = f"page_{key}"
    return page


def settings_page():
    guarded(p_settings.render)


pages = {key: st.Page(make_page(key), title=title, url_path=key, default=(key == "data"))
         for key, title, _ in state.PAGES}
pages["settings"] = st.Page(settings_page, title="Settings", url_path="settings")
state.PAGE_OBJS.update(pages)
nav = st.navigation(list(pages.values()), position="hidden")

with st.sidebar:
    st.html('<div class="sb-title">HUDYAT · Trainer</div>')
    # The usual order, numbered. Every page can still be opened at any time.
    for i, (key, title, _) in enumerate(state.PAGES, 1):
        st.page_link(pages[key], label=f"{i:02d}   {title}")
    nxt = state.next_step(snap) if snap.ok else "data"
    title = next((t for k, t, _ in state.PAGES if k == nxt), None)
    running = jobs.busy(snap.paths)
    st.html('<div class="sb-foot">'
            + (f'<div class="sb-next">Next: <b>{common.esc(title)}</b></div>' if title else "")
            + (f'<div class="sb-job">{jobs.JOB_TITLES.get(running, running)} is running…</div>' if running else "")
            + '</div>')
    data_source.switcher("ds_pick")
    st.page_link(pages["settings"], label="Settings")

nav.run()
