"""Step 7: Export to TF.js (layers format) for the browser app."""

from __future__ import annotations

import json

import streamlit as st

from trainer.tfjs_export import exported_files

from . import data_source, state
from .common import banner, files_table, job_button, job_monitor, note, page_header, section

ALL = "All trained models"


def render(snap: state.Snapshot) -> None:
    page_header("Step 7", "Export",
                "Writes each trained model in TensorFlow.js format (model.json + weights + labels.json) for the "
                "browser app, and checks that the exported files predict exactly like the trained model.")
    if not snap.ok:
        data_source.missing(snap)
        return
    if not snap.trained:
        banner("info", "No trained model yet.", "Train a model first (step 5).")
        st.button("Export to TF.js", key="ex_run_locked", type="primary", disabled=True)
        return
    f = state.focus(snap)
    if f != state.ALL_CATS and f not in snap.trained:
        banner("info", f"{f} has no trained model yet.", "Train it first (step 5). This step unlocks afterwards.")
        return
    cat = state.page_category(snap, "Model", [ALL] + snap.trained, key="ex_cat")
    args = ["--category", *(["all"] if cat == ALL else [cat])]
    job_button("export", "Export to TF.js", args, None, key="ex_run")
    job_monitor("export")

    section("Exported files", "work/models/<category>/tfjs/")
    for c in snap.trained:
        files = exported_files(snap.paths, c)
        st.html(f'<div style="font-weight:600;margin:14px 0 2px 0">{c}</div>')
        if not files:
            note("Not exported yet.")
            continue
        files_table(files, snap.paths.work)
        v = snap.paths.model_dir(c) / "tfjs" / "verify.json"
        if v.exists():
            chk = json.loads(v.read_text(encoding="utf-8"))
            if chk.get("ok"):
                note(f"Check passed: on {chk['n_samples']} samples the exported model gives the same top prediction "
                     f"as Keras ({chk['same_top1']:.0%}); largest probability difference {chk['max_abs_diff']:.1e}.")
            else:
                banner("error", "Export check failed: predictions differ from the Keras model.",
                       "Export again; if it still fails, send the log to the developers.")
    note("Copy the whole tfjs folder of a category into the browser app. Its input is the 32 × 128 sample built "
         "exactly as in README section 6; labels.json gives the class order of the output.")
