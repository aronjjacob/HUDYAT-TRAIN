"""Step 3: Validation (approve / reject clips). Stored in work/validation.csv, never in the recorder's files."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer import dataset as ds
from trainer import settings as cfg

from . import data_source, media, state
from .common import banner, note, page_header, section, table

ANY = "All"


def render(snap: state.Snapshot) -> None:
    page_header("Step 3", "Validation  (optional)",
                "A quality check by a person: watch the recorded clips and reject the bad ones (wrong sign, wrong "
                "handshape, hands out of frame, cut off...). Rejected clips are left out of training. You can skip "
                "this step: every kept clip is then used.")
    if not snap.ok:
        data_source.missing(snap)
        return
    with st.expander("How does Validation work?"):
        st.html('<div class="note" style="font-size:14px;line-height:1.7">'
                "1. Type your name (it is saved with every decision).<br>"
                "2. Pick a category. The list shows the clips nobody has checked yet.<br>"
                "3. Watch the clip. <b>Approve</b> if the sign is correct. If not, type the reason and press "
                "<b>Reject</b>. The next clip opens by itself.<br>"
                "4. After rejecting clips, run <b>4. Split</b> again so the rejected clips are left out.<br><br>"
                "Decisions are saved in this project (work/validation.csv), never in the recorder's files. "
                "Approving is not required; only rejected clips change anything. A later decision replaces an "
                "earlier one, so a rejected clip can be approved again.</div>")
    kept = snap.kept
    if kept.empty:
        banner("info", "No kept S1–S7 clips yet.")
        return
    dec = ds.latest_decisions(snap.paths)
    kept = kept.copy()
    kept["decision"] = kept["file_path"].map(lambda f: dec.get(f, {}).get("decision", "unreviewed"))
    labels = {s.sign_id: s.sign_label for s in snap.signs}

    c1, c2, c3, c4 = st.columns(4)
    cats = [c for c in ds.extract_categories(snap.signs) if c in set(kept["category"])]
    f = state.focus(snap)
    if f != state.ALL_CATS:
        cats = [c for c in (f, cfg.NONE_CATEGORY) if c in cats] or [f]
    cat = c1.selectbox("Category", cats, key="va_cat") if len(cats) > 1 or f == state.ALL_CATS else cats[0]
    if f != state.ALL_CATS and len(cats) == 1:
        c1.html(f'<div class="note" style="margin-top:30px">Category: <b>{cat}</b></div>')
    sub = kept[kept["category"] == cat]
    sign = c2.selectbox("Sign", [ANY] + sorted(sub["sign_id"].unique()), key="va_sign",
                        format_func=lambda s: labels.get(s, s))
    if sign != ANY:
        sub = sub[sub["sign_id"] == sign]
    if snap.fixed_split:                       # uploaded data has no signers: filter by its fixed split instead
        part = c3.selectbox("Part", [ANY, "train", "val", "test"], key="va_part",
                            format_func=lambda p: {ANY: "All", "val": "Validation"}.get(p, p.title()))
        if part != ANY:
            sub = sub[sub["split"] == part]
    else:
        signer = c3.selectbox("Signer", [ANY] + [g for g in cfg.SIGNERS if g in set(sub["signer"])], key="va_signer")
        if signer != ANY:
            sub = sub[sub["signer"] == signer]
    show = c4.selectbox("Show", ["unreviewed", ANY, "approved", "rejected"], key="va_show",
                        format_func=lambda s: {"unreviewed": "Not reviewed yet", ANY: "All clips"}.get(s, s.title()))
    if show != ANY:
        sub = sub[sub["decision"] == show]
    sub = sub.sort_values(["sign_id", "signer", "take"])

    st.text_input("Reviewer name", key="va_reviewer", placeholder="Your name (saved with every decision)")
    if sub.empty:
        banner("success" if show == "unreviewed" else "info",
               "Every clip in this selection has been reviewed." if show == "unreviewed" else "No clips in this selection.")
    else:
        files = list(sub["file_path"])
        if snap.fixed_split:
            fmt = {r["file_path"]: f"{labels.get(r['sign_id'], r['sign_id'])} · {r['file_path'].rsplit('/', 1)[-1]}"
                                   f" · {r['split']} · {r['decision']}" for _, r in sub.iterrows()}
        else:
            fmt = {r["file_path"]: f"{labels.get(r['sign_id'], r['sign_id'])} · {r['signer']} · take "
                                   f"{str(r['take']).zfill(2)} · {r['decision']}" for _, r in sub.iterrows()}
        nxt = st.session_state.pop("va_next", None)
        if nxt in files:
            st.session_state.va_clip = nxt
        if st.session_state.get("va_clip") not in files:
            st.session_state.va_clip = files[0]
        f = st.selectbox(f"Clip ({len(files)} in this selection)", files, format_func=fmt.get, key="va_clip")
        row = sub[sub["file_path"] == f].iloc[0]
        v1, v2 = st.columns([3, 2], gap="medium")
        with v1:
            media.show_video(snap.data_dir / f)
        with v2:
            d = dec.get(f)
            if d:
                kind = "success" if d["decision"] == "approved" else "error"
                banner(kind, f"{d['decision'].title()} by {d.get('reviewer') or '—'}",
                       f"{d.get('time', '').replace('T', ' ')}" + (f" · {d['reason']}" if d.get("reason") else ""))
            else:
                note("Not reviewed yet.")
            note(f"{f}<br>Recorder detection rate: {row.get('hand_detect_rate') or '—'}")
            reason = st.text_input("Reason (needed to reject)", key=f"va_reason_{f}",
                                   placeholder="e.g. wrong handshape, hand out of frame")
            reviewer = st.session_state.get("va_reviewer", "").strip()
            b1, b2 = st.columns(2)
            ap = b1.button("Approve", key="va_approve", type="primary", disabled=not reviewer, use_container_width=True)
            rj = b2.button("Reject", key="va_reject", type="secondary", disabled=not (reviewer and reason.strip()),
                           use_container_width=True)
            if not reviewer:
                note("Type your name above first.")
            if ap or rj:
                ds.add_validation(snap.paths, f, "approved" if ap else "rejected", reason.strip(), reviewer)
                nxt = files[files.index(f) + 1] if files.index(f) + 1 < len(files) else None
                if nxt:
                    st.session_state.va_next = nxt
                st.rerun()

    section("Progress per sign", cat)
    allc = kept[kept["category"] == cat]
    rows = []
    for sid, g in allc.groupby("sign_id", sort=False):
        n = len(g)
        a = int((g["decision"] == "approved").sum())
        r = int((g["decision"] == "rejected").sum())
        rows.append({"Sign": labels.get(sid, sid), "Clips": n, "Approved": a, "Rejected": r,
                     "Not reviewed": n - a - r, "Reviewed": f"{(a + r) / n:.0%}"})
    if rows:
        df = pd.DataFrame(rows)
        classes = {(i, "Rejected"): "err" for i, r in enumerate(rows) if r["Rejected"]}
        table(df, numeric=["Clips", "Approved", "Rejected", "Not reviewed", "Reviewed"], classes=classes)
    note("Rejected clips are left out the next time you run Split (step 4). Approving is optional.")
