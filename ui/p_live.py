"""Step 8: Live test in its own OpenCV window (webcam), with a results summary."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer import recorder
from trainer import settings as cfg
from trainer.train import load_labels

from . import data_source, state
from .common import banner, job_button, job_monitor, note, page_header, pct, section, stats, table

TARGET_MS = 1000


def _load(paths, category: str) -> pd.DataFrame:
    p = paths.live_tests / f"{category}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"tester": str, "target": str, "prediction": str})
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")
    return df


def render(snap: state.Snapshot) -> None:
    page_header("Step 8", "Live Test",
                "Opens a camera window that finds each sign by itself (hands up, sign, hands down) and shows the "
                "model's top 3 guesses. Test mode scores a tester against a target sign.")
    if not snap.ok:
        data_source.missing(snap)
        return
    if not snap.trained:
        banner("info", "No trained model yet.", "Train a model first (step 5).")
        st.button("Start live test", key="lv_start_locked", type="primary", disabled=True)
        return
    s = snap.settings
    rec = recorder.recorder_camera_settings(snap.data_dir)
    cam = s.get("camera") if s.get("camera") is not None else rec.get("camera", 0)
    be = s.get("backend") or rec.get("backend", "msmf")

    c1, c2 = st.columns([2, 3])
    f = state.focus(snap)
    if f != state.ALL_CATS and f not in snap.trained:
        banner("info", f"{f} has no trained model yet.", "Train it first (step 5). This step unlocks afterwards.")
        return
    cat = state.page_category(snap, "Model", snap.trained, key="lv_cat", container=c1)
    mode = c2.radio("Mode", ["Free", "Test"], horizontal=True, key="lv_mode",
                    help="Free: just shows predictions. Test: counts correct attempts for one target sign.")
    labels = load_labels(snap.paths.model_dir(cat))
    args = ["--category", cat]
    reason = None
    if mode == "Test":
        c1, c2, c3 = st.columns([2, 2, 1])
        tester = c1.text_input("Tester name", key="lv_tester")
        target = c2.selectbox("Target sign", labels, key="lv_target")
        trials = c3.number_input("Trials", 1, 200, 25, key="lv_trials")
        if not tester.strip():
            reason = "Type the tester's name first."
        args += ["--mode", "test", "--tester", tester.strip(), "--target", target, "--trials", str(int(trials))]
    args += ["--camera", str(int(cam)), "--backend", be]
    job_button("live", "Start live test", args, reason, key="lv_start")
    thr = float(s.get("confidence_threshold", 0.7))
    note(f"Camera {cam} ({be}) — change it in Settings. A window opens on this computer; press Q or close it to stop. "
         f"An attempt counts as correct when the top guess is the target with at least {thr:.0%} confidence. "
         f"Below that the window says 'Not sure, try again'.")
    job_monitor("live")

    section("Live results", "work/live_tests/<category>.csv, Test mode only")
    df = _load(snap.paths, cat)
    if df.empty:
        note(f"No test-mode attempts for {cat} yet.")
    else:
        n = len(df)
        acc = df["correct"].mean()
        rt = df["response_time_ms"].mean()
        within = (df["response_time_ms"] <= TARGET_MS).mean()
        stats([("Attempts", n, ""), ("Live accuracy", pct(acc), "ok" if acc >= 0.85 else "warn"),
               ("Mean response time", f"{rt:.0f} ms", "ok" if rt <= TARGET_MS else "err"),
               ("Within 1 second", pct(within, 0), ""), ("Testers", df["tester"].nunique(), "")])
        rows, classes = [], {}
        for i, (t, g) in enumerate(df.groupby("target", sort=True)):
            m = g["response_time_ms"].mean()
            rows.append({"Target sign": t, "Attempts": len(g), "Correct": int(g["correct"].sum()),
                         "Accuracy": pct(g["correct"].mean()), "Mean response": f"{m:.0f} ms",
                         "Most common guess": g["prediction"].mode().iat[0]})
            classes[(i, "Mean response")] = "ok" if m <= TARGET_MS else "err"
        table(pd.DataFrame(rows), numeric=["Attempts", "Correct", "Accuracy", "Mean response"], classes=classes)

    section("All categories", f"compare with the SENYAS benchmark's live accuracy; response-time target {TARGET_MS} ms")
    rows = []
    for c in snap.live_cats:
        d = _load(snap.paths, c)
        if d.empty:
            continue
        rows.append({"Category": c, "Attempts": len(d), "Signs tested": d["target"].nunique(),
                     "Live accuracy": pct(d["correct"].mean()), "Mean response": f"{d['response_time_ms'].mean():.0f} ms",
                     "Within 1 s": pct((d["response_time_ms"] <= TARGET_MS).mean(), 0)})
    bench = s.get("senyas_live_accuracy")
    if rows:
        table(pd.DataFrame(rows), numeric=["Attempts", "Signs tested", "Live accuracy", "Mean response", "Within 1 s"])
    else:
        note("No live tests yet.")
    c1, _ = st.columns([1, 2])
    b = c1.number_input("SENYAS benchmark live accuracy (%)", 0.0, 100.0, float(bench or 0.0), 0.1, key="lv_bench",
                        help="Typed in once from the SENYAS paper, for comparison.")
    if b != float(bench or 0.0):
        cfg.save_settings({"senyas_live_accuracy": b})
    if b and rows:
        note(f"SENYAS benchmark: {b:.1f}% live accuracy.")
