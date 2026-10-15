"""Step 6: Evaluate on the test set (signers S6 + S7, or the test part of an upload), once."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer.evaluate import PASS_METRIC, PASS_PER_SIGN, load_report

from . import data_source, state
from .common import banner, job_button, job_monitor, note, page_header, pct, section, table

ALL = "All trained models"


def render(snap: state.Snapshot) -> None:
    page_header("Step 6", "Evaluate",
                f"Tests a trained model once on data it never saw: signers S6 and S7 for the recordings, or the 15 % test "
                f"part of an uploaded dataset. "
                f"PASS = accuracy and macro precision / recall / F1 ≥ {PASS_METRIC:.0%} and every sign's "
                f"recall ≥ {PASS_PER_SIGN:.0%}.")
    if not snap.ok:
        data_source.missing(snap)
        return
    if not snap.trained:
        banner("info", "No trained model yet.", "Train a model first (step 5). This step unlocks afterwards.")
        st.button("Evaluate on the test set", key="ev_run_locked", type="primary", disabled=True)
        return
    f = state.focus(snap)
    if f != state.ALL_CATS and f not in snap.trained:
        banner("info", f"{f} has no trained model yet.", "Train it first (step 5). This step unlocks afterwards.")
        return
    cat = state.page_category(snap, "Model", [ALL] + snap.trained, key="ev_cat")
    cats = snap.trained if cat == ALL else [cat]
    done = [c for c in cats if c in snap.evaluated]
    if done:
        banner("warning", f"Already evaluated: {', '.join(done)}.",
               "The test set should be used once. Evaluate again only if the model was retrained for a reason "
               "other than its test results.")
    confirm = st.checkbox("Tuning is finished; evaluate on the test set once", key="ev_confirm")
    reason = None if confirm else "Tick the box above first."
    args = ["--category", *(["all"] if cat == ALL else [cat]), "--confirm"]
    job_button("evaluate", "Evaluate on the test set", args, reason, key="ev_run")
    job_monitor("evaluate")

    if not snap.evaluated:
        return
    section("Summary", "work/reports/summary.csv")
    rows = []
    classes = {}
    for i, c in enumerate(snap.evaluated):
        r = load_report(snap.paths, c)
        a = r["average"]
        rec = {k: v for k, v in r["pooled"]["per_sign_recall"].items() if v is not None}
        rows.append({"Model": c, "Test clips": r["pooled"]["n"], "Accuracy": pct(a["accuracy"]),
                     "Precision": pct(a["macro_precision"]), "Recall": pct(a["macro_recall"]), "F1": pct(a["macro_f1"]),
                     "Lowest sign recall": pct(min(rec.values())) if rec else "—",
                     "ms / sample": f"{r['ms_per_sample']:.1f}", "Result": "PASS" if r["pass"] else "FAIL"})
        classes[(i, "Result")] = "ok" if r["pass"] else "err"
    table(pd.DataFrame(rows), numeric=["Test clips", "Accuracy", "Precision", "Recall", "F1", "Lowest sign recall",
                                       "ms / sample"], classes=classes)

    show = st.selectbox("Details for", snap.evaluated, key="ev_show",
                        index=snap.evaluated.index(cat) if cat in snap.evaluated else 0)
    r = load_report(snap.paths, show)
    section(f"{show}: test result", r["date"].replace("T", " "))
    ok = r["pass"]
    st.html(f'<div class="verdict {"ok" if ok else "err"}">{"PASS" if ok else "FAIL"}</div>')
    for why in r["reasons"]:
        banner("error", why)
    keys = (("accuracy", "Accuracy"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"),
            ("macro_f1", "Macro F1"))
    rows, classes = [], {}
    for sg, m in r["per_signer"].items():
        rows.append({"Test set": sg, "Clips": m["n"], **{n: pct(m[k]) for k, n in keys}})
    rows.append({"Test set": "Average", "Clips": r["pooled"]["n"], **{n: pct(r["average"][k]) for k, n in keys}})
    names = {s.sign_id: s.sign_label for s in snap.signs}
    for k, n in keys:
        classes[(len(rows) - 1, n)] = "ok" if r["average"][k] >= PASS_METRIC else "err"
    table(pd.DataFrame(rows), numeric=["Clips"] + [n for _, n in keys], classes=classes, total_row=True)
    note(f"Mean inference time: {r['ms_per_sample']:.1f} ms per sample (one sample at a time, CPU).")

    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.html('<div class="sec" style="margin-top:8px">Recall per sign <small>whole test set</small></div>')
        rows, classes = [], {}
        for i, (lab, rec) in enumerate(r["pooled"]["per_sign_recall"].items()):
            rows.append({"Sign": names.get(lab, lab), "Clips": r["pooled"]["support"][lab], "Recall": pct(rec)})
            if rec is not None:
                classes[(i, "Recall")] = "ok" if rec >= PASS_PER_SIGN else "err"
        table(pd.DataFrame(rows), numeric=["Clips", "Recall"], classes=classes)
    with c2:
        st.html('<div class="sec" style="margin-top:8px">Confusion matrix</div>')
        img = snap.paths.report_dir(show) / "confusion_matrix.png"
        if img.exists():
            st.image(str(img))
    note(f"Report: work/reports/{show}/report.md · confusion_matrix.csv (pooled and per signer) · metrics.json")
