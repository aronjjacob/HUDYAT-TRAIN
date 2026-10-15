"""Step 2: Preprocess: extraction per category, adjustable settings, trim summary, clip viewer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from trainer import dataset as ds
from trainer.landmarks import IMAGE_EXT
from trainer import recorder
from trainer import settings as cfg

from . import data_source, media, state
from .common import banner, job_button, job_monitor, note, page_header, section, stats, table


ALL = "__all__"
PARAM_LABELS = {"trim_margin_s": "Trim margin", "trim_min_frames": "Minimum window",
                "min_hand_detection_confidence": "Detection confidence",
                "min_hand_presence_confidence": "Presence confidence", "min_tracking_confidence": "Tracking confidence"}


def category_status(snap: state.Snapshot, current_id: str) -> pd.DataFrame:
    """Per category: kept clips, extracted, to do, made with other settings."""
    kept = state.kept_rows(snap.data_dir, include_static=True)
    idx = snap.index
    done, pids = set(), {}
    if not idx.empty:
        okx = idx[idx["status"] != ds.STATUS_FAILED]
        done = {f for f in okx["file"] if (snap.paths.work / f).exists()}
        col = okx["preprocess"] if "preprocess" in okx.columns else pd.Series([""] * len(okx), index=okx.index)
        pids = dict(zip(okx["file"], col.fillna("").astype(str)))
    default_id = recorder.params_id(recorder.default_params(snap.data_dir))
    rows = []
    for c in ds.extract_categories(snap.signs, include_static=True):
        k = kept[kept["category"] == c] if not kept.empty else kept
        files = [ds.npy_rel_path(r) for _, r in k.iterrows()]
        n_done = sum(f in done for f in files)
        other = sum(f in done and (pids.get(f) or default_id) != current_id for f in files)
        rows.append({"category": c, "kept": len(files), "done": n_done, "todo": len(files) - n_done, "other": other})
    return pd.DataFrame(rows, columns=["category", "kept", "done", "todo", "other"])


def _cat_label(c: str) -> str:
    if c == cfg.NONE_CATEGORY:
        return "none (added to every model)"
    if c in cfg.EXCLUDED_CATEGORIES:
        return f"{c} (static A–Y test set)"
    return c


def render(snap: state.Snapshot) -> None:
    page_header("Step 2", "Preprocess",
                "Runs MediaPipe on the kept clips, cuts each one to its sign window and saves a 32 × 128 landmark "
                "sample per clip (README section 6). Do all categories at once, or one at a time.")
    if not snap.ok:
        data_source.missing(snap)
        return
    params = recorder.preprocess_params(snap.settings, snap.data_dir)
    current_id = recorder.params_id(params)
    status = category_status(snap, current_id)
    by_cat = status.set_index("category")
    normal = status[~status["category"].isin(cfg.EXCLUDED_CATEGORIES)]

    options = [ALL] + list(status["category"])

    def fmt(o):
        if o == ALL:
            return f"All categories (except the static test set) — {int(normal['todo'].sum())} to do"
        r = by_cat.loc[o]
        return f"{_cat_label(o)} — {int(r['kept'])} clips, {int(r['todo'])} to do"

    focus = state.focus(snap)
    if focus != state.ALL_CATS:
        # working on one category: its clips + none (every model needs the none class)
        choice = focus
        state.page_category(snap, "", [], "pp_choice")
        with_none = cfg.NONE_CATEGORY in set(status["category"])
        cats = [focus] + ([cfg.NONE_CATEGORY] if with_none else [])
        sel = status[status["category"].isin(cats)]
        cat_args = ["--category", *cats]
        what = f"{focus} + none" if with_none else focus
    else:
        choice = st.selectbox("What to preprocess", options, format_func=fmt, key="pp_choice")
        sel = normal if choice == ALL else status[status["category"] == choice]
        cat_args = [] if choice == ALL else ["--category", choice]
        what = "all categories" if choice == ALL else choice
    n_kept, n_todo, n_other = int(sel["kept"].sum()), int(sel["todo"].sum()), int(sel["other"].sum())
    reason = None if n_kept else "No kept S1–S7 clips here yet."
    c1, c2, _ = st.columns([2, 2, 2])
    with c1:
        job_button("extract", f"Extract new clips ({n_todo})", cat_args,
                   reason or (None if n_todo else "Every clip here is already extracted."), key="extract_new")
    with c2:
        job_button("extract", "Re-extract all" if choice == ALL else f"Re-extract {what}", ["--all", *cat_args],
                   reason, key="extract_all", kind="secondary",
                   help="Extract these clips again, e.g. after changing the preprocessing settings below.")
    note(f"{n_kept} kept clip(s) in {what}, {n_kept - n_todo} already extracted. "
         "Samples go to work/landmarks/, the list to work/landmarks_index.csv.")
    if n_other:
        banner("warning", f"{n_other} extracted sample(s) here were made with other preprocessing settings.",
               "Re-extract them so every sample uses the current settings (a model needs one setting).")
    if choice in cfg.EXCLUDED_CATEGORIES:
        note("The static A–Y test set is extracted for the separate static photo model; no category model is "
             "trained from it here.")
    job_monitor("extract")
    settings_box(snap, params)

    section("Categories")
    rows, classes = [], {}
    for i, r in enumerate(status.itertuples()):
        rows.append({"Category": _cat_label(r.category), "Kept clips": r.kept, "Extracted": r.done, "To do": r.todo,
                     "Other settings": r.other})
        if r.todo:
            classes[(i, "To do")] = "warn"
        if r.other:
            classes[(i, "Other settings")] = "warn"
        if r.kept and not r.todo:
            classes[(i, "Extracted")] = "ok"
    table(pd.DataFrame(rows), numeric=["Kept clips", "Extracted", "To do", "Other settings"], classes=classes)
    results(snap)


def settings_box(snap: state.Snapshot, params: dict) -> None:
    if st.session_state.pop("pp_reset_pending", False):          # Reset: clear the boxes before they are drawn
        for k in ("pp_margin", "pp_minf", "pp_det", "pp_pres", "pp_trk"):
            st.session_state.pop(k, None)
    defaults = recorder.default_params(snap.data_dir)
    adjusted = not recorder.is_default(params, snap.data_dir)
    title = "Adjust preprocessing" + ("  ·  ADJUSTED" if adjusted else "")
    with st.expander(title, expanded=adjusted):
        thr = st.slider("Minimum hand detection inside the sign window (clips below it are not used for training, "
                        "except none)", 0.5, 1.0, float(snap.settings.get("min_hand_detect_rate", 0.8)), 0.05,
                        format="%.2f", key="pp_thr")
        note("Safe to change at any time: it only decides which clips are used. Run Split again afterwards.")
        st.html('<div class="sec" style="margin-top:14px">Advanced <small>changes the landmark format</small></div>')
        banner("warning", "Changing these makes the samples differ from the recorder's rules (README section 6).",
               "The browser app must then use the same values (Export writes them to preprocess.json), "
               "and the clips must be re-extracted.")
        c1, c2 = st.columns(2)
        margin = c1.number_input("Trim margin before / after the hands (seconds)", 0.0, 1.0,
                                 float(params["trim_margin_s"]), 0.05, format="%.2f", key="pp_margin",
                                 help=f"Recorder: {defaults['trim_margin_s']:.2f} s")
        minf = c2.number_input("Minimum sign window (frames)", 1, 32, int(params["trim_min_frames"]), 1,
                               key="pp_minf", help=f"Recorder: {defaults['trim_min_frames']}")
        c1, c2, c3 = st.columns(3)
        det = c1.number_input("Detection confidence", 0.1, 0.95, float(params["min_hand_detection_confidence"]),
                              0.05, format="%.2f", key="pp_det", help="MediaPipe; recorder 0.50")
        pres = c2.number_input("Presence confidence", 0.1, 0.95, float(params["min_hand_presence_confidence"]),
                               0.05, format="%.2f", key="pp_pres", help="MediaPipe; recorder 0.50")
        trk = c3.number_input("Tracking confidence", 0.1, 0.95, float(params["min_tracking_confidence"]), 0.05,
                              format="%.2f", key="pp_trk", help="MediaPipe; recorder 0.50")
        new = {"trim_margin_s": round(float(margin), 3), "trim_min_frames": int(minf),
               "min_hand_detection_confidence": round(float(det), 3),
               "min_hand_presence_confidence": round(float(pres), 3), "min_tracking_confidence": round(float(trk), 3)}
        changed = (recorder.params_id(new) != recorder.params_id(params)
                   or float(thr) != float(snap.settings.get("min_hand_detect_rate", 0.8)))
        b1, b2, _ = st.columns([1.2, 1.6, 3])
        if b1.button("Save settings", key="pp_save", type="primary", disabled=not changed):
            cfg.save_settings({"min_hand_detect_rate": float(thr),
                               "preprocess": {k: v for k, v in new.items() if v != defaults[k]}})
            st.rerun()
        if b2.button("Reset to recorder values", key="pp_reset", disabled=not adjusted):
            cfg.save_settings({"preprocess": {}})
            st.session_state.pp_reset_pending = True
            st.rerun()
        if adjusted:
            diff = ", ".join(f"{PARAM_LABELS[k]} {params[k]} (recorder {defaults[k]})"
                             for k in params if params[k] != defaults[k])
            note(f"In use: {diff}.")
        else:
            note("In use: the recorder's values (trim margin 0.1 s, minimum 8 frames, MediaPipe confidences 0.5).")


def results(snap: state.Snapshot) -> None:
    idx = snap.index
    if idx.empty:
        return
    usable = ds.usable_samples(snap.paths, snap.data_dir, snap.settings)
    ok = usable[usable["status"] != ds.STATUS_FAILED].copy()
    for c in ("n_frames", "frames_kept", "trim_start", "trim_end"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    photos = not ok.empty and bool(ok["source_file"].str.lower().str.endswith(IMAGE_EXT).all())
    if photos:
        section("Photo summary", "each photo becomes a held-still 32-frame sample")
        n_none = int((usable["hand_detect_rate"] < 0.5).sum())
        n_fail = int((usable["status"] == ds.STATUS_FAILED).sum())
        stats([("Photos", len(usable), ""), ("Hand found", len(usable) - n_none - n_fail, "ok"),
               ("No hand found", n_none, "warn" if n_none else ""), ("Failed", n_fail, "err" if n_fail else "")])
        note("Photos without a hand are left out of training. Every hand is stored as a right hand "
             "(photos of a left hand are mirrored), like the signers' dominant hand in the live test.")
    else:
        section("Trim summary", "the sign window keeps the frames with a hand, plus the trim margin on each side")
    if not ok.empty and not photos:
        kept_s = (ok["frames_kept"] / ok["fps"]).mean()
        clip_s = (ok["n_frames"] / ok["fps"]).mean()
        thr = float(snap.settings.get("min_hand_detect_rate", 0.8))
        n_low = int(usable["excluded_reason"].str.startswith("hand detection").sum())
        n_fail = int((usable["status"] == ds.STATUS_FAILED).sum())
        stats([("Extracted clips", len(ok), ""), ("Mean clip length", f"{clip_s:.2f} s", ""),
               ("Mean sign window", f"{kept_s:.2f} s", ""),
               ("Mean hand detection", f"{ok['hand_detect_rate'].mean():.0%}", ""),
               (f"Below {thr:.0%} detection", n_low, "warn" if n_low else ""),
               ("Failed", n_fail, "err" if n_fail else "")])
        g = ok.groupby("category").agg(Clips=("file", "size"), clip=("n_frames", "mean"), win=("frames_kept", "mean"),
                                       det=("hand_detect_rate", "mean"), both=("both_hands_rate", "mean"))
        g = g.reset_index().rename(columns={"category": "Category"})
        g["Mean frames"] = g["clip"].round(0).astype(int)
        g["Mean window (frames)"] = g["win"].round(0).astype(int)
        g["Kept"] = (g["win"] / g["clip"]).map(lambda v: f"{v:.0%}")
        g["Hand detection"] = g["det"].map(lambda v: f"{v:.0%}")
        g["Both hands"] = g["both"].map(lambda v: f"{v:.0%}")
        table(g[["Category", "Clips", "Mean frames", "Mean window (frames)", "Kept", "Hand detection", "Both hands"]],
              numeric=["Clips", "Mean frames", "Mean window (frames)", "Kept", "Hand detection", "Both hands"])

    low = usable[usable["excluded_reason"].str.startswith("hand detection") | (usable["status"] == ds.STATUS_FAILED)]
    section("Low-detection and failed clips", "left out of training (the none class is never left out for this)")
    if low.empty:
        note("None.")
    else:
        show = low[["category", "sign_id", "signer", "take", "hand_detect_rate", "status", "source_file"]].copy()
        show["hand_detect_rate"] = show["hand_detect_rate"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
        st.dataframe(show.rename(columns={"hand_detect_rate": "detection", "source_file": "file"}),
                     hide_index=True, use_container_width=True, height=min(320, 38 + 35 * len(show)))

    if photos:
        section("Look at a photo", "the photo next to the hand MediaPipe found in it")
    else:
        section("Look at a clip", "the video as recorded (unmirrored) next to its 32-frame sample")
    clip_viewer(ok, {s.sign_id: s.sign_label for s in snap.signs})


def clip_viewer(ok: pd.DataFrame, names: dict | None = None) -> None:
    if ok.empty:
        return
    names = names or {}
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category", sorted(ok["category"].unique()), key="pp_cat")
    sub = ok[ok["category"] == cat]
    sign = c2.selectbox("Sign", sorted(sub["sign_id"].unique()), key="pp_sign", format_func=lambda s: names.get(s, s))
    sub = sub[sub["sign_id"] == sign].sort_values(["signer", "take"])
    is_photo = sub["source_file"].str.lower().str.endswith(IMAGE_EXT)
    labels = {r["file"]: (f"{r['source_file'].rsplit('/', 1)[-1]} · "
                          f"{'hand found' if r['hand_detect_rate'] >= 0.5 else 'no hand'}" if ph else
                          f"{r['signer']} · take {str(r['take']).zfill(2)} · detection {r['hand_detect_rate']:.0%}")
              for (_, r), ph in zip(sub.iterrows(), is_photo)}
    f = c3.selectbox("Photo" if is_photo.all() else "Clip", list(labels), format_func=labels.get, key="pp_clip")
    row = sub[sub["file"] == f].iloc[0]
    photo = str(row["source_file"]).lower().endswith(IMAGE_EXT)
    paths = cfg.Paths()
    v1, v2 = st.columns(2, gap="medium")
    with v1:
        media.show_video(cfg.data_dir() / row["source_file"])
        if photo:
            note(f"{row['source_file']} · {int(row['width'])} × {int(row['height'])} px")
        else:
            note(f"{row['source_file']}<br>{int(row['n_frames'])} frames at {row['fps']:.0f} fps · "
                 f"sign window frames {int(row['trim_start'])}–{int(row['trim_end'])}")
    with v2:
        try:
            sample = np.load(paths.work / row["file"])
            media.skeleton(sample, int(row["n_frames"]), int(row["trim_start"]), int(row["trim_end"]), float(row["fps"]))
        except OSError:
            banner("warning", "The sample file is missing.", "Run 'Extract new clips' again.")
