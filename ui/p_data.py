"""Step 1: Data: the dataset in use, what is in it, warnings, and the clip viewer (Reject / Restore)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from trainer import dataset as ds
from trainer import settings as cfg
from trainer.landmarks import IMAGE_EXT

from . import common, data_source, media, state
from .common import banner, esc, note, page_header, plural, section, stats, table


def render(snap: state.Snapshot) -> None:
    page_header(state.step_label("data"), "Data", "Check what was recorded.")
    if not snap.ok:
        data_source.missing(snap)
        return
    data_source.flash()
    if snap.kept.empty:
        in_use(snap)
        banner("info", "Nothing has been recorded yet.", "Record clips with the HUDYAT Hand Tracker first.")
        add_panel(snap)
        return
    # Fixed tab labels: a label that changes (e.g. a count) makes Streamlit jump back to the first tab.
    cat = None
    if snap.photos:
        in_use(snap)
        tabs = st.tabs(["Files per sign", "Review clips", "Add or combine datasets"])
        with tabs[0]:
            photo_overview(snap)
    else:
        f = facts(snap)
        headline(snap, f)
        cat = state.pick_category(snap.model_cats, "data")
        tabs = st.tabs(["To do", "Progress per sign", "Review clips", "Add or combine datasets"])
        with tabs[0]:
            todo(f)
        with tabs[1]:
            progress(snap, f, cat)
    with tabs[-2]:
        viewer(snap, cat)
    with tabs[-1]:
        add_panel(snap)


def in_use(snap: state.Snapshot) -> None:
    st.html(f'<div class="note" style="font-size:14px">In use: <b style="color:#1A1A1A">'
            f'{esc(data_source.label_of(snap.data_dir))}</b></div>')


def add_panel(snap: state.Snapshot) -> None:
    section("Add a dataset", "upload a .zip or a folder")
    note("To switch between datasets, use <b>Dataset</b> in the sidebar.")
    data_source.add_panel("dt_add")
    section("Combine datasets", "e.g. A–Y photos + J and Z videos into one model")
    combine_panel(snap)


# ----------------------------------------------------------------------------
def facts(snap: state.Snapshot) -> dict:
    """Signers and their hands, the clip counts per sign and signer, and the warnings."""
    meta = state.metadata(snap.data_dir)
    ours = meta[meta["signer"].isin(cfg.SIGNERS)]
    kept = snap.kept
    hands = state.clip_hands(snap.data_dir)
    signers = snap.signers
    by_signer = {g: sorted({hands.get(f, "right") for f in kept[kept["signer"] == g]["file_path"]}) for g in signers}
    lefties = [g for g, h in by_signer.items() if "left" in h]
    model_signs = [s for s in snap.signs if s.category in snap.model_cats]
    counts = kept.groupby(["sign_id", "signer"]).size()
    missing = sum(max(0, s.reps - int(counts.get((s.sign_id, g), 0))) for s in model_signs for g in cfg.SIGNERS)
    target = sum(s.reps for s in model_signs) * len(cfg.SIGNERS)

    warnings = []
    if not lefties:
        warnings.append(("Only right-handed signers.",
                         "Left-handed learners are covered by the canonical hand, see Settings for this run on Test the model, "
                         "and checked by the left-hand test on mirrored clips; real left-handed testers are still "
                         "needed on the Live test page."))
    absent = [g for g in cfg.SIGNERS if g not in signers]
    if absent:
        warnings.append((f"No clips yet from {', '.join(absent)}.",
                         f"Test 2 runs one round per signer: {len(signers)} now, up to 7 later."))
    n_rr = int((ours["status"] == "needs_rerecord").sum())
    if n_rr:
        warnings.append((f"{plural(n_rr, 'take')} marked 'needs re-record' in the recorder.", "They are left out."))
    rej = ds.rejected_files(snap.paths) & set(kept["file_path"])
    if rej:
        warnings.append((f"{plural(len(rej), 'clip')} rejected here.", "They are left out. Restore them under Review clips."))
    if not snap.index.empty:
        u = ds.usable_samples(snap.paths, snap.data_dir, snap.settings)
        low = int(u["excluded_reason"].str.startswith("hand detection").sum())
        if low:
            warnings.append((f"{plural(low, 'clip')} with low hand detection, left out.", "See the Preprocess page."))
    return dict(signers=signers, by_signer=by_signer, lefties=lefties, model_signs=model_signs, counts=counts,
                missing=missing, target=target, warnings=warnings,
                other=sorted(set(meta["signer"]) - set(cfg.SIGNERS)))


def headline(snap: state.Snapshot, f: dict) -> None:
    """How far the recording is, in one line and one bar."""
    done = f["target"] - f["missing"]
    hand = "all right-handed" if not f["lefties"] else f"{len(f['lefties'])} left-handed"
    common.headline(f"{done} of {f['target']} clips recorded",
                    f'{esc(data_source.label_of(snap.data_dir))} · {plural(len(f["signers"]), "signer")} ({hand}) · '
                    f'{plural(len(f["model_signs"]), "sign")}', done, f["target"])


def todo(f: dict) -> None:
    """The warnings as a to-do list, then who signed with which hand."""
    if f["warnings"]:
        st.html("".join(f'<div class="dt-todo"><span>○</span><div><b>{esc(w)}</b><div class="note">{esc(h)}</div>'
                        f'</div></div>' for w, h in f["warnings"]))
    else:
        note("Nothing to do.")
    note("Signers: " + " · ".join(f"<b style='color:#1A1A1A'>{g}</b> {'/'.join(h)}-handed"
                                  for g, h in f["by_signer"].items())
         + ". The hand comes from signing_hand in the recorder's metadata.")
    if f["other"]:
        note(f"Ignored everywhere: {', '.join(f['other'])} rows, e.g. the imported FSL-105 public clips.")


def progress(snap: state.Snapshot, f: dict, cat: str) -> None:
    """One bar per sign of the category picked above the tabs, the exact counts folded away."""
    cs = [s for s in snap.signs if s.category == cat]
    bars = []
    for s in cs:
        n = sum(min(s.reps, int(f["counts"].get((s.sign_id, g), 0))) for g in cfg.SIGNERS)
        t = s.reps * len(cfg.SIGNERS)
        bars.append(f'<div class="dt-bar"><span>{esc(s.sign_label)}</span><div><div class="{"full" if n >= t else ""}" '
                    f'style="width:{100 * n / t if t else 100:.0f}%"></div></div><span class="muted">{n}/{t}</span></div>')
    st.html("".join(bars))
    rows, classes = [], {}
    for i, s in enumerate(cs):
        r = {"Sign": s.sign_label}
        tot = 0
        for g in cfg.SIGNERS:
            n = int(f["counts"].get((s.sign_id, g), 0))
            r[g] = n if g in f["signers"] else "—"
            tot += n
            classes[(i, g)] = "zero" if n == 0 else ("full" if n >= s.reps else "")
        r["Total"] = tot
        r["Target"] = s.reps * len(cfg.SIGNERS)
        rows.append(r)
    if rows:
        with st.expander("Exact counts per signer"):
            table(pd.DataFrame(rows), numeric=list(cfg.SIGNERS) + ["Total", "Target"], classes=classes)
            note(f"Green = target reached: {cs[0].reps} takes per signer. — = no clips from this signer yet.")


def photo_overview(snap: state.Snapshot) -> None:
    kept = snap.kept
    labels = {s.sign_id: s.sign_label for s in snap.signs}
    n_img = int(kept["file_path"].str.lower().str.endswith(IMAGE_EXT).sum())
    stats([("Signs", kept["sign_id"].nunique(), ""), ("Photos", n_img, ""), ("Videos", len(kept) - n_img, ""),
           ("Signers", "none, photos", "")])
    banner("info", "Photo dataset: only Test 1 is possible.",
           "Photos have no signer IDs, so Test 2, leave-one-signer-out, can't be run.")
    section("Files per sign")
    df = kept.groupby("sign_id").size().rename("Files").reset_index()
    df["Sign"] = df["sign_id"].map(lambda x: labels.get(x, x))
    tot = pd.DataFrame([{"Sign": "Total", "Files": int(df["Files"].sum())}])
    table(pd.concat([df[["Sign", "Files"]], tot], ignore_index=True), numeric=["Files"], total_row=True)


# ----------------------------------------------------------------------------
def viewer(snap: state.Snapshot, cat: str | None = None) -> None:
    """The clip viewer, for `cat` (the category picked above it), or with its own picker when none was."""
    kept = snap.kept
    names = {s.sign_id: s.sign_label for s in snap.signs}
    dec = ds.latest_decisions(snap.paths)
    cats = [c for c in snap.model_cats if (kept["category"] == c).any()]
    if not cats:
        return
    section("Look at a clip", "reject clips that are wrong")
    if cat is not None:
        if cat not in cats:
            note(f"No clips of {esc(cat)} yet.")
            return
        c2, c3, c4 = st.columns(3)
    else:
        c1, c2, c3, c4 = st.columns(4)
        cat = state.pick_category(cats, "data", box=c1)
    sub = kept[kept["category"] == cat]
    sign = c2.selectbox("Sign", sorted(sub["sign_id"].unique()), key="dt_v_sign", format_func=lambda s: names.get(s, s))
    sub = sub[sub["sign_id"] == sign]
    if not snap.photos:
        signer = c3.selectbox("Signer", sorted(sub["signer"].unique()), key="dt_v_signer")
        sub = sub[sub["signer"] == signer]
    show = c4.selectbox("Show", ["all", "rejected"], key="dt_v_show",
                        format_func=lambda x: {"all": "All clips", "rejected": "Rejected only"}[x])
    if show == "rejected":
        sub = sub[sub["file_path"].map(lambda f: dec.get(f, {}).get("decision") == "rejected")]
    if sub.empty:
        note("No clips in this selection.")
        return
    sub = sub.sort_values("file_path")
    files = list(sub["file_path"])

    def label(f):
        r = sub[sub["file_path"] == f].iloc[0]
        tag = " · rejected" if dec.get(f, {}).get("decision") == "rejected" else ""
        return (f"{f.rsplit('/', 1)[-1]}{tag}" if snap.photos else f"take {str(r.get('take', '')).zfill(2)}{tag}")
    f = st.selectbox(f"Clip · {len(files)}", files, format_func=label, key="dt_v_clip")
    idx = snap.index.set_index("file") if not snap.index.empty else None
    row = sub[sub["file_path"] == f].iloc[0]
    rel = ds.npy_rel_path(row)
    v1, v2 = st.columns(2, gap="medium")
    with v1:
        media.show_video(snap.data_dir / f)
    with v2:
        if idx is not None and rel in idx.index and (snap.paths.work / rel).exists():
            r = idx.loc[rel]
            try:
                sample = np.load(snap.paths.work / rel)
                media.skeleton(sample, int(float(r["n_frames"] or 1)), int(float(r["trim_start"] or 0)),
                               int(float(r["trim_end"] or 0)), float(r["fps"] or 30))
            except (OSError, ValueError):
                note("The landmark sample could not be read. Extract it again on the Preprocess page.")
        else:
            note("Not extracted yet: the skeleton appears after extracting it on the Preprocess page.")
    d = dec.get(f)
    rejected = bool(d and d.get("decision") == "rejected")
    if rejected:
        banner("error", "Rejected: left out of every test and model.",
               f"Reason: {d.get('reason') or '—'}" + (f" · by {d['reviewer']}" if d.get("reviewer") else "")
               + f" · {d.get('time', '').replace('T', ' ')}")
    c1, c2, c3 = st.columns([3, 2, 1.2], vertical_alignment="bottom")
    reviewer = c2.text_input("Your name, optional", key="dt_v_reviewer")
    if rejected:
        if c3.button("Restore", key="dt_v_restore", type="primary", use_container_width=True):
            ds.add_validation(snap.paths, f, "restored", "", reviewer.strip())
            st.rerun()
    else:
        reason = c1.text_input("Reason", key=f"dt_v_reason_{f}", placeholder="e.g. wrong handshape, hand out of frame")
        if c3.button("Reject", key="dt_v_reject", disabled=not reason.strip(), use_container_width=True,
                     help="Type a reason first. Rejected clips are left out of every test and model."):
            ds.add_validation(snap.paths, f, "rejected", reason.strip(), reviewer.strip())
            st.rerun()
    with st.expander("Details"):
        note(f"File: {esc(f)}")


# ----------------------------------------------------------------------------
def combine_panel(snap: state.Snapshot) -> None:
    """Make one dataset (one model) out of signs from several datasets, e.g. A-Y photos + J/Z videos."""
    from trainer.combine import Part, combine, default_leave_out, items_of
    note("Pick the parts to put together, e.g. the uploaded <b>alphabet photos</b> and the recorded "
         "<b>alphabet_dynamic</b>: J and Z. Signs with the same label become one class. Where a sign has both photos "
         "and videos, like J and Z, the photos are left out by default because those signs are movements. Samples that "
         "are already extracted are reused.")
    parts, labels = [], {}
    for ds_label, path in data_source.available():
        try:
            signs = ds.load_signs(path)
        except Exception:  # noqa: BLE001
            continue
        for c in ds.model_categories(signs):
            n = sum(1 for s in signs if s.category == c)
            p = Part(path, c, ds_label)
            parts.append(p)
            labels[p] = f"{ds_label} · {c} · {plural(n, 'sign')}"
    chosen = st.multiselect("Combine these", parts, format_func=labels.get, key="dt_cb_parts",
                            placeholder="e.g. alphabet_static · uploaded, and recordings · alphabet_dynamic")
    if not chosen:
        return
    items = [it for p in chosen for it in items_of(p)]
    leave = default_leave_out(items)
    by_id = {it.id: it for it in items}
    default = [it.id for it in items if it.id not in leave and it.n_files]
    keep = st.multiselect("Signs to include", list(by_id), default=default, key=f"dt_cb_signs_{len(chosen)}",
                          format_func=lambda i: f"{by_id[i].label} · {by_id[i].kind} · {by_id[i].part.dataset_label} "
                                                f"· {by_id[i].n_files}")
    left = [by_id[i] for i in by_id if i not in keep]
    if left:
        note("Left out: " + ", ".join(f"{it.label} · {it.kind} · {it.part.dataset_label}" for it in left) + ".")
    name =st.text_input("Name of the combined dataset", value="alphabet", key="dt_cb_name")
    chosen_items = [by_id[i] for i in keep]
    if st.button("Create combined dataset", key="dt_cb_go", type="primary", disabled=len(chosen_items) < 2):
        with st.spinner("Combining, reusing extracted samples..."):
            try:
                res = combine(name or "combined", chosen_items)
            except ds.PlainError as e:
                banner("error", e.message, e.hint)
                return
        msg = (f"Created '{res.name}': {plural(res.n_files, 'file')}, {plural(len(res.signs), 'sign')}; "
               f"{plural(res.n_reused, 'sample')} reused. ")
        msg += " ".join(res.notes)
        st.session_state.pop("dt_cb_parts", None)
        data_source.use(res.root, msg)
