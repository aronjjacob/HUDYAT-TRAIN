"""Step 5: Live test with the webcam (a window on this computer), and the results of earlier live tests."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer import live, recorder
from trainer.app_export import list_exports
from trainer.final import load_preprocess
from trainer.train import load_config

from . import data_source, state
from .common import (banner, details_line, headline, job_button, job_monitor, note, page_header, pct, section,
                     stats, table)

TARGET_MS = 1000
HANDS = {"right": "Right", "left": "Left"}
FEATURES = {1: "v1 · hand shape", 2: "v2 · shape + wrist movement", 3: "v3 · shape + face location"}
SETUP = live.SETUPS


def render(snap: state.Snapshot) -> None:
    page_header(state.step_label("live"), "Live test", "Try the model with the camera. Test mode scores a tester.")
    if not snap.ok:
        data_source.missing(snap)
        return
    if not snap.finals:
        banner("info", "No final model yet.", "Train one on the Train model page.")
        return
    tried = [c for c in snap.finals if c in snap.live_cats]
    headline(f"{len(tried)} of {len(snap.finals)} models tried live",
             "Test mode scores a tester; the results are kept per category", len(tried), len(snap.finals))
    cat = state.pick_category(snap.finals, "live")
    t_run, t_res = st.tabs(["Run a live test", "Results"])
    with t_run:
        run = run_live(snap, cat)
    with t_res:
        results(snap, cat, run)


def run_live(snap: state.Snapshot, cat: str) -> str | None:
    """Pick the model and the tester, then Start. Returns the model's run id (its results are shown first)."""
    s = snap.settings
    pre = load_preprocess(snap.paths.model_dir(cat)) or {}
    conf = load_config(snap.paths, cat) or {}
    exports = list_exports(snap.paths, cat)
    # Every model of the category: the trainer's current one and each app export (one per feature version).
    models = {"trainer": {"label": "Trainer model", "feature_version": int(conf.get("feature_version", 1)),
                          "model_run_id": conf.get("model_run_id"), "from_run": conf.get("from_run"),
                          "date": conf.get("date", ""), "when": "trained", "classes": pre.get("labels") or [],
                          "current": True}}
    for e in exports:
        models[e["name"]] = {**e, "label": "HUDYAT app model", "when": "exported"}
    names = {k: f"{m['label']} · v{m['feature_version']} · {m['when']} {_when(m['date'])}"
                + (" · current" if k != "trainer" and m["current"] else "") for k, m in models.items()}
    first = next((k for k, m in models.items() if k != "trainer" and m["current"]), "trainer")
    pick = st.selectbox("Model", list(models), format_func=names.get, index=list(models).index(first),
                        key=f"lv_model_{cat}",
                        help="The app models are exactly what the HUDYAT app runs; each export is kept, so a v1 "
                             "and a v2 model of the same category can be compared.")
    m = models[pick]
    recognizer = "trainer" if pick == "trainer" else "app"
    labels = m["classes"]
    details_line([("Model input", FEATURES[m["feature_version"]]), ("Run", m["model_run_id"] or "—"),
                  (m["when"].capitalize(), _when(m["date"])), ("Settings from", m["from_run"] or "its own settings")])
    if recognizer == "app" and not m["current"]:
        note(f"This export is not the trainer's current {cat} model, v{models['trainer']['feature_version']} trained "
             f"{_when(models['trainer']['date'])}. It is kept so the two can be compared.")
    elif recognizer == "trainer" and not exports:
        note("Export the model on the Train model page to test the HUDYAT app model here.")
    if not pre.get("handedness", {}).get("canonical", True):
        note("This model has no left-hand support.")
    mode = st.radio("Mode", ["free", "test"], horizontal=True, key="lv_mode",
                    format_func=lambda x: {"free": "Free: just show guesses", "test": "Test: score a tester"}[x])
    c1, c2 = st.columns([2, 1.4])
    tester = c1.text_input("Tester name", key="lv_tester", placeholder="e.g. T1 or a name")
    tester_hand = c2.radio("Tester's signing hand", list(HANDS), format_func=HANDS.get, horizontal=True,
                           key="lv_tester_hand", help="Saved with every attempt.")
    default_hand = s.get("default_hand") or "auto"
    capture = "fixed"
    with st.expander("Advanced: hand setting" + (", capture" if recognizer == "app" else "")):
        c1, c2 = st.columns(2)
        hand = c1.radio("Hand setting in the window", ["auto", "same"], horizontal=True, key="lv_hand",
                        index=0 if default_hand == "auto" else 1,
                        format_func=lambda h: {"auto": "Auto, detect per sign", "same": "Use the tester's hand"}[h],
                        help="Press H in the window to change it.")
        if recognizer == "app":
            capture = c2.radio("Capture", list(live.CAPTURES), format_func=live.CAPTURES.get, key="lv_capture",
                               horizontal=True,
                               help="Fixed = the recommended changes for the app: the sign ends when the hands go "
                                    "down, no blending with a missing hand, left hands mirrored. As the app today = "
                                    "its 2-second clip. Run both with the same tester to compare.")
    args = ["--category", cat, "--mode", mode, "--tester-hand", tester_hand,
            "--hand", "auto" if hand == "auto" else tester_hand, "--recognizer", recognizer, "--capture", capture]
    if recognizer == "app":
        args += ["--export", pick]
    reason = None
    if mode == "test":
        c1, c2 = st.columns([2, 1])
        target = c1.selectbox("First target sign", labels, key="lv_target",
                              help="N in the window = next sign.")
        trials = c2.number_input("Trials", 1, 500, 25, key="lv_trials")
        if not tester.strip():
            reason = "Type the tester's name first."
        args += ["--tester", tester.strip(), "--target", target, "--trials", str(int(trials))]
    elif tester.strip():
        args += ["--tester", tester.strip()]
    rec = recorder.recorder_camera_settings(snap.data_dir)
    cam = s.get("camera") if s.get("camera") is not None else rec.get("camera", 0)
    be = s.get("backend") or rec.get("backend", "msmf")
    args += ["--camera", str(int(cam)), "--backend", be]
    job_button("live", "Start", args, reason, key="lv_start")
    thr = float(s.get("confidence_threshold", 0.85))
    rule = (f"at least {live.APP_MIN_SCORE:.0%} sure and {live.APP_MIN_MARGIN * 100:.0f} points ahead of the second "
            f"guess, the app's rule" if recognizer == "app" else f"at least {thr:.0%} sure")
    note(f"Keys: Space start/pause · N next sign · H hand · M mode · Q quit. Correct = right sign and {rule}.")
    job_monitor("live")
    return m["model_run_id"]


def _when(date: str | None) -> str:
    return str(date or "")[:16].replace("T", " ") or "—"


def results(snap: state.Snapshot, cat: str, run: str | None = None) -> None:
    df = live.read_attempts(snap.paths.live_tests / f"{cat}.csv")
    if df.empty:
        note(f"No Test-mode attempts for {cat} yet.")
        return
    per_model = live.by_model(df)
    if len(per_model) > 1:
        version = lambda r: "—" if r["feature_version"] is None else f"v{r['feature_version']}"
        table(pd.DataFrame([{"Model run": r["model_run_id"], "Model input": version(r), "Attempts": r["attempts"],
                             "Accuracy": pct(r["accuracy"])} for r in per_model]),
              numeric=["Attempts", "Accuracy"])
        runs = ["all"] + [r["model_run_id"] for r in per_model]
        show = st.selectbox("Show the results of", runs, index=runs.index(run) if run in runs else 0,
                            key=f"lv_show_{cat}", format_func=lambda x: "every model" if x == "all" else x)
        if show != "all":
            df = df[df["model_run_id"].replace("", "not recorded") == show]
    acc = df["correct"].mean()
    rt = df["response_time_ms"].mean()
    within = (df["response_time_ms"] <= TARGET_MS).mean()
    stats([("Attempts", len(df), ""), ("Live accuracy", pct(acc), "ok" if acc >= 0.85 else "warn"),
           ("Mean response time", f"{rt:.0f} ms", "ok" if rt <= TARGET_MS else "err"),
           ("Within 1 second", pct(within, 0), ""), ("Testers", df["tester"].nunique(), "")])
    rows, classes = [], {}
    for i, ((rec, capt), g) in enumerate(df.groupby(["recognizer", "capture"], sort=True)):
        m, fh = g["response_time_ms"].mean(), g["time_from_hand_ms"].mean()
        rows.append({"Setup": SETUP.get((rec, capt), f"{rec} · {capt}"), "Attempts": len(g),
                     "Accuracy": pct(g["correct"].mean()), "After the sign ends": f"{m:.0f} ms",
                     "After the hand appears": "—" if pd.isna(fh) else f"{fh:.0f} ms",
                     "Within 1 s of the hand": "—" if pd.isna(fh) else pct((g["time_from_hand_ms"] <= TARGET_MS).mean(), 0)})
        classes[(i, "After the sign ends")] = "ok" if m <= TARGET_MS else "err"
    if len(rows) > 1 or rows[0]["Setup"] != SETUP[("trainer", "fixed")]:
        section("By setup", "compare the app's capture today with the fixed one, same tester and signs")
        table(pd.DataFrame(rows), numeric=["Attempts", "Accuracy", "After the sign ends", "After the hand appears"],
              classes=classes)
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        rows, classes = [], {}
        for i, (t, g) in enumerate(df.groupby("target", sort=True)):
            m = g["response_time_ms"].mean()
            rows.append({"Sign": t, "Attempts": len(g), "Accuracy": pct(g["correct"].mean()),
                         "Mean response": f"{m:.0f} ms", "Most common guess": g["prediction"].mode().iat[0]})
            classes[(i, "Mean response")] = "ok" if m <= TARGET_MS else "err"
        table(pd.DataFrame(rows), numeric=["Attempts", "Accuracy", "Mean response"], classes=classes)
    with c2:
        rows = []
        for h, g in df.groupby(df["tester_hand"].replace("", "not recorded"), sort=True):
            rows.append({"Tester's hand": h.capitalize(), "Testers": g["tester"].nunique(), "Attempts": len(g),
                         "Accuracy": pct(g["correct"].mean())})
        table(pd.DataFrame(rows), numeric=["Testers", "Attempts", "Accuracy"])
        if not (df["tester_hand"] == "left").any():
            note("No left-handed testers yet.")
