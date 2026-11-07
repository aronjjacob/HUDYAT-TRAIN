"""Step 2: Preprocess: one card per category with its own Extract / Re-extract / Export buttons (nothing to pick),
the settings, and the results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer import dataset as ds
from trainer import recorder
from trainer import settings as cfg
from trainer.landmarks import IMAGE_EXT

from . import common, data_source, state
from .common import banner, esc, job_button, job_monitor, note, page_header, pct, plural, section, stats, table

PARAM_LABELS = {"trim_margin_s": "Trim margin", "trim_min_frames": "Minimum window",
                "min_hand_detection_confidence": "Detection", "min_hand_presence_confidence": "Presence",
                "min_tracking_confidence": "Tracking"}
DEFAULT_DETECT_RATE = cfg.DEFAULTS["min_hand_detect_rate"]
TABS = ["Categories", "Check settings", "Check the results"]


def render(snap: state.Snapshot) -> None:
    page_header(state.step_label("preprocess"), "Preprocess", "Finds the hands in each clip and saves them for training.")
    if not snap.ok:
        data_source.missing(snap)
        return
    headline(snap)
    lm_only = cfg.landmarks_only(snap.data_dir)          # an exported dataset: its samples came with it
    t_cats, t_settings, t_results = st.tabs(TABS)
    with t_cats:
        categories(snap, status(snap), lm_only)
    with t_settings:
        if lm_only:
            note("These samples came with the dataset; the settings they were made with are in its dataset.json.")
        else:
            note("The defaults are the recorder's values. Change them only if many clips are left out.")
            settings(snap, recorder.preprocess_params(snap.settings, snap.data_dir))
            job_button("extract", "Re-extract every category", ["--all"], None, key="pp_extract_all",
                       kind="secondary", help="Do every clip again, e.g. after changing the settings.")
    with t_results:
        if snap.index.empty:
            note("Nothing extracted yet. Extract first.")
        else:
            results(snap)


def headline(snap: state.Snapshot) -> None:
    """How many clips are extracted with the current settings, in one line and one bar."""
    kept = sum(k for k, _ in snap.by_cat.values())
    todo = sum(t for _, t in snap.by_cat.values())
    done = kept - todo
    common.headline(f"{done} of {kept} clips up to date",
                    f"{len(snap.by_cat)} categories · "
                    + ("all done" if not todo else f"{plural(todo, 'clip')} still to extract")
                    + (f" · {plural(snap.n_old, 'older clip')} to redo" if snap.n_old else ""), done, kept)


def status(snap: state.Snapshot) -> dict:
    """Per category: kept clips, new (never extracted), old (old format), ready (to export), left out."""
    u = ds.usable_samples(snap.paths, snap.data_dir, snap.settings) if not snap.index.empty else pd.DataFrame()
    out = {}
    for c, (kept, todo) in snap.by_cat.items():
        uc = u[u["category"] == c] if not u.empty else u
        if uc.empty:
            out[c] = {"kept": kept, "new": todo, "old": 0, "ready": 0, "left": 0}
            continue
        ok, cur = uc["excluded_reason"].eq(""), uc["format"].eq(ds.SAMPLE_FORMAT)
        old = int((ok & ~cur).sum())
        out[c] = {"kept": kept, "new": max(0, todo - old), "old": old, "ready": int((ok & cur).sum()),
                  "left": int((~ok).sum())}
    return out


def categories(snap: state.Snapshot, st_: dict, lm_only: bool) -> None:
    """One card per category: its state and its own buttons. Export = the category's landmark dataset (.npy with the
    signs, signers and hands), to share or to train on elsewhere."""
    from trainer import dataset_export
    for c, x in st_.items():
        todo = x["new"] + x["old"]
        with st.container(border=True):
            a, b, d, e = st.columns([2.2, 3, 1.4, 1.2], vertical_alignment="center")
            a.html(f'<div style="font-weight:600">{esc(c)}</div><div class="note">{plural(x["kept"], "clip")}</div>')
            state_txt = ("came with the dataset" if lm_only else "up to date" if not todo else
                         " · ".join(y for y in (f"{x['new']} new" if x["new"] else "",
                                                f"{x['old']} in the old format" if x["old"] else "") if y))
            b.html(f'<div class="note" style="color:{"var(--ok)" if lm_only or not todo else "var(--warn)"}">'
                   f'{esc(state_txt)}</div>' + (f'<div class="note">{x["left"]} left out</div>' if x["left"] else ""))
            if lm_only:
                continue
            with d:
                if todo:
                    job_button("extract", f"Extract {todo}", ["--category", c], None, key=f"pp_ex_{c}")
                else:
                    job_button("extract", "Re-extract", ["--all", "--category", c], None, key=f"pp_rx_{c}",
                               kind="secondary", help="Do this category's clips again.")
            with e:
                job_button("dataset", "Export", ["--category", c], None if x["ready"] else "Nothing ready.",
                           key=f"pp_dsx_{c}", kind="secondary",
                           help="A copy of the samples (.npy) with signs, signers and hands, for training. "
                                "No videos, no model. It can be opened again on the Data page to train on it.")
            ex = dataset_export.list_exports(snap.paths, c)
            if ex and ex[0]["zip"].exists():
                f, g = st.columns([4.6, 1.2], vertical_alignment="center")
                f.html(f'<div class="note">Latest export: {esc(ex[0]["name"])} · {plural(ex[0]["n_samples"], "sample")}</div>')
                g.download_button("Download .zip", ex[0]["zip"].read_bytes(), file_name=ex[0]["zip"].name,
                                  mime="application/zip", key=f"pp_ds_zip_{c}", use_container_width=True)
    if lm_only:
        return
    job_monitor("extract")
    job_monitor("dataset")


def settings(snap: state.Snapshot, params: dict) -> None:
    """The preprocessing settings, right under the categories. Defaults = the recorder's values."""
    if st.session_state.pop("pp_reset_pending", False):
        for k in ("pp_thr", "pp_margin", "pp_minf", "pp_det", "pp_pres", "pp_trk"):
            st.session_state.pop(k, None)
    defaults = recorder.default_params(snap.data_dir)
    thr_now = float(snap.settings.get("min_hand_detect_rate", DEFAULT_DETECT_RATE))
    adjusted = not recorder.is_default(params, snap.data_dir) or thr_now != DEFAULT_DETECT_RATE
    with st.expander("Settings" + ("  ·  changed" if adjusted else "  ·  default"), expanded=adjusted):
        c1, c2, c3 = st.columns(3)
        thr = c1.slider("Minimum hand detection", 0.5, 1.0, thr_now, 0.05, format="%.2f", key="pp_thr",
                        help="Clips where the hand is seen in fewer frames than this are left out. "
                             f"Default {DEFAULT_DETECT_RATE:.2f}.")
        margin = c2.number_input("Trim margin, seconds", 0.0, 1.0, float(params["trim_margin_s"]), 0.05, format="%.2f",
                                 key="pp_margin", help="Extra time kept before and after the hands. "
                                                       f"Default {defaults['trim_margin_s']:.2f}.")
        minf = c3.number_input("Minimum window, frames", 1, 32, int(params["trim_min_frames"]), 1, key="pp_minf",
                               help=f"Shortest sign kept. Default {defaults['trim_min_frames']}.")
        c1, c2, c3 = st.columns(3)
        det = c1.number_input("Detection confidence", 0.1, 0.95, float(params["min_hand_detection_confidence"]),
                              0.05, format="%.2f", key="pp_det",
                              help=f"How sure MediaPipe must be to find a hand. "
                                   f"Default {defaults['min_hand_detection_confidence']:.2f}.")
        pres = c2.number_input("Presence confidence", 0.1, 0.95, float(params["min_hand_presence_confidence"]),
                               0.05, format="%.2f", key="pp_pres",
                               help=f"How sure it must be that the hand is still there. "
                                    f"Default {defaults['min_hand_presence_confidence']:.2f}.")
        trk = c3.number_input("Tracking confidence", 0.1, 0.95, float(params["min_tracking_confidence"]), 0.05,
                              format="%.2f", key="pp_trk",
                              help=f"How sure it must be to follow the hand to the next frame. "
                                   f"Default {defaults['min_tracking_confidence']:.2f}.")
        new = {"trim_margin_s": round(float(margin), 3), "trim_min_frames": int(minf),
               "min_hand_detection_confidence": round(float(det), 3),
               "min_hand_presence_confidence": round(float(pres), 3), "min_tracking_confidence": round(float(trk), 3)}
        format_changed = recorder.params_id(new) != recorder.params_id(params)
        changed = format_changed or float(thr) != thr_now
        if format_changed:
            note("Changing trim or confidence needs <b>Re-extract all</b>, and the model must be trained again.")
        b1, b2, _ = st.columns([1, 1.3, 3])
        if b1.button("Save", key="pp_save", type="primary", disabled=not changed):
            cfg.save_settings({"min_hand_detect_rate": float(thr),
                               "preprocess": {k: v for k, v in new.items() if v != defaults[k]}})
            st.rerun()
        if b2.button("Reset to default", key="pp_reset", disabled=not adjusted):
            cfg.save_settings({"preprocess": {}, "min_hand_detect_rate": DEFAULT_DETECT_RATE})
            st.session_state.pp_reset_pending = True
            st.rerun()


def results(snap: state.Snapshot) -> None:
    idx = snap.index
    if idx.empty:
        return
    usable = ds.usable_samples(snap.paths, snap.data_dir, snap.settings)
    ok = usable[usable["status"] != ds.STATUS_FAILED].copy()
    for c in ("n_frames", "frames_kept", "trim_start", "trim_end"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    photos = not ok.empty and bool(ok["source_file"].str.lower().str.endswith(IMAGE_EXT).all())
    thr = float(snap.settings.get("min_hand_detect_rate", DEFAULT_DETECT_RATE))
    n_low = int(usable["excluded_reason"].str.startswith("hand detection").sum())
    n_fail = int((usable["status"] == ds.STATUS_FAILED).sum())
    if photos:
        n_none = int((ok["hand_detect_rate"] < 0.5).sum())
        stats([("Photos", len(usable), ""), ("Hand found", len(ok) - n_none, "ok"),
               ("No hand", n_none, "warn" if n_none else ""), ("Failed", n_fail, "err" if n_fail else "")])
    elif not ok.empty:
        win_s = (ok["frames_kept"] / pd.to_numeric(ok["fps"], errors="coerce")).mean()
        stats([("Clips", len(ok), ""), ("Average sign length", "—" if pd.isna(win_s) else f"{win_s:.2f} s", ""),
               ("Hand seen", f"{ok['hand_detect_rate'].mean():.0%}", ""),
               ("Left out", n_low, "warn" if n_low else "ok"), ("Failed", n_fail, "err" if n_fail else "")])
        g = ok.groupby("category").agg(Clips=("file", "size"), win=("frames_kept", "mean"),
                                       det=("hand_detect_rate", "mean"), both=("both_hands_rate", "mean"))
        g = g.reset_index().rename(columns={"category": "Category"})
        share = lambda v: "—" if pd.isna(v) else f"{v:.0%}"  # noqa: E731
        g["Sign length, frames"] = g["win"].map(lambda v: "—" if pd.isna(v) else int(round(v)))
        g["Hand seen"] = g["det"].map(share)
        g["Both hands"] = g["both"].map(share)
        cols = ["Category", "Clips", "Sign length, frames", "Hand seen", "Both hands"]
        table(g[cols], numeric=cols[1:])

    low = usable[usable["excluded_reason"].str.startswith("hand detection") | (usable["status"] == ds.STATUS_FAILED)]
    if not low.empty:
        with st.expander(f"Left out: {plural(len(low), 'clip')} with the hand seen in under {thr:.0%} of frames"):
            show = low[["category", "sign_id", "signer", "take", "hand_detect_rate", "status"]].copy()
            show["hand_detect_rate"] = show["hand_detect_rate"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
            show["status"] = show["status"].map(lambda v: "failed" if v == ds.STATUS_FAILED else "low detection")
            st.dataframe(show.rename(columns={"hand_detect_rate": "hand seen", "sign_id": "sign"}), hide_index=True,
                         use_container_width=True, height=min(300, 38 + 35 * len(show)))
            note("Watch them on the Data page; re-record them if needed.")
    if not photos:
        hands_section(snap, ok)


@st.cache_data(show_spinner=False, max_entries=4)
def _hand_stats(work: str, key: tuple, clips: pd.DataFrame):
    from trainer.handstats import hand_stats
    return hand_stats(work, clips)


def hands_section(snap: state.Snapshot, ok: pd.DataFrame) -> None:
    rich = ok[(ok["format"] == ds.SAMPLE_FORMAT) & ok["signer"].isin(cfg.SIGNERS)]
    if rich.empty:
        return
    section("Hands per signer")
    hands = state.clip_hands(snap.data_dir)
    clips = pd.DataFrame({"file": rich["file"].values, "signer": rich["signer"].values,
                          "hand": [hands.get(f, "right") for f in rich["source_file"]],
                          "source_file": rich["source_file"].values})
    key = (len(clips), str(snap.paths.index_csv.stat().st_mtime) if snap.paths.index_csv.exists() else "")
    with st.spinner("Counting hands..."):
        per, bad = _hand_stats(str(snap.paths.work), key, clips)
    rows = [{"Signer": r.signer, "Signing hand": r.hand, "Clips": r.clips, "Left seen": pct(r.left_pct, 0),
             "Right seen": pct(r.right_pct, 0), "Both": pct(r.both_pct, 0)} for r in per.itertuples()]
    table(pd.DataFrame(rows), numeric=["Clips", "Left seen", "Right seen", "Both"])
    if not bad.empty:
        banner("warning", f"{plural(len(bad), 'clip')} signed with the other hand.", "Check them on the Data page.")
        with st.expander("Which clips"):
            st.dataframe(bad.rename(columns={"source_file": "clip", "hand": "signing hand"}),
                         hide_index=True, use_container_width=True)
