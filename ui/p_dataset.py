"""Step 1: Datasets: which one is in use, add one (upload / folder), and what is in it."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer import dataset as ds
from trainer import settings as cfg

from . import data_source, state
from .common import banner, note, page_header, section, stats, table


def render(snap: state.Snapshot) -> None:
    page_header("Step 1", "Dataset", "Choose the data to train on (your recordings or an uploaded dataset), add a new "
                                     "one, and see what is in it. Nothing here changes the original files.")
    if not snap.ok:
        data_source.missing(snap)
        return
    data_source.flash()
    overview(snap)
    busy_adding = (st.session_state.get("dsp_zip") is not None
                   or bool((st.session_state.get("dsp_folder") or "").strip()))
    with st.expander("Add a dataset (upload a .zip or a folder)", expanded=busy_adding):
        data_source.add_panel("dsp")
    with st.expander("Combine datasets (e.g. A–Y photos + J and Z videos into one model)",
                     expanded=bool(st.session_state.get("cb_parts"))):
        combine_panel(snap)
    if snap.fixed_split:
        uploaded_details(snap)
        return
    meta = state.metadata(snap.data_dir)
    if meta.empty:
        banner("info", "Nothing has been recorded yet.", "Record clips with the HUDYAT Hand Tracker first.")
        return
    ours = meta[meta["signer"].isin(cfg.SIGNERS)]
    n_kept = int((ours["status"] == "kept").sum())
    n_rr = int((ours["status"] == "needs_rerecord").sum())
    n_exp = int((ours.get("expert_status", pd.Series(dtype=str)) == "rejected").sum())
    n_val = len(ds.rejected_files(snap.paths) & set(ours["file_path"]))
    if snap.index.empty:
        low_txt, low_tone = "not extracted", ""
    else:
        u = ds.usable_samples(snap.paths, snap.data_dir, snap.settings)
        low_txt = int(u["excluded_reason"].str.startswith("hand detection").sum())
        low_tone = "warn" if low_txt else ""
    stats([("Kept clips (S1–S7)", n_kept, ""), ("Needs re-record", n_rr, "warn" if n_rr else ""),
           ("Rejected by expert", n_exp, "err" if n_exp else ""), ("Rejected in Validation", n_val, "err" if n_val else ""),
           ("Low hand detection", low_txt, low_tone)])
    other = sorted(set(meta["signer"]) - set(cfg.SIGNERS))
    if other:
        n_other = int(meta["signer"].isin(other).sum())
        note(f"Ignored here and in training: signer{'s' if len(other) > 1 else ''} {', '.join(other)} "
             f"({n_other} row{'s' if n_other != 1 else ''}, e.g. the imported public clips).")

    signs = snap.signs
    kept = snap.kept
    cats = [c for c in ds.categories(signs)]
    section("Recording progress per category", "kept clips vs. target (reps × 7 signers)")
    rows = []
    for c in cats:
        cs = [s for s in signs if s.category == c]
        target = sum(s.reps for s in cs) * len(cfg.SIGNERS)
        if c in cfg.EXCLUDED_CATEGORIES:
            n = int(((ours["category"] == c) & (ours["status"] == "kept")).sum())
        else:
            n = int((kept["category"] == c).sum())
        rows.append({"Category": c + (" (not trained here)" if c in cfg.EXCLUDED_CATEGORIES else ""),
                     "Signs": len(cs), "Kept": n, "Target": target, "Progress": f"{min(1, n / target):.0%}" if target else "—"})
    df = pd.DataFrame(rows)
    tot = {"Category": "Total", "Signs": int(df["Signs"].sum()), "Kept": int(df["Kept"].sum()),
           "Target": int(df["Target"].sum())}
    tot["Progress"] = f"{min(1, tot['Kept'] / tot['Target']):.0%}" if tot["Target"] else "—"
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    table(df, numeric=["Signs", "Kept", "Target", "Progress"], total_row=True)

    section("Clips per sign and signer")
    cat = state.page_category(snap, "Category", cats, key="ds_cat")
    cs = [s for s in signs if s.category == cat]
    src = ours[(ours["category"] == cat) & (ours["status"] == "kept")] if cat in cfg.EXCLUDED_CATEGORIES else \
        kept[kept["category"] == cat]
    counts = src.groupby(["sign_id", "signer"]).size()
    rows, classes = [], {}
    for i, s in enumerate(cs):
        r = {"Sign": s.sign_label}
        tot = 0
        for g in cfg.SIGNERS:
            n = int(counts.get((s.sign_id, g), 0))
            r[g] = n
            tot += n
            classes[(i, g)] = "zero" if n == 0 else ("full" if n >= s.reps else "")
        r["Total"] = tot
        r["Target"] = s.reps * len(cfg.SIGNERS)
        rows.append(r)
    table(pd.DataFrame(rows), numeric=list(cfg.SIGNERS) + ["Total", "Target"], classes=classes)
    note(f"Green = the signer has reached the target for that sign ({cs[0].reps if cs else 15} takes). "
         "Only clips with status kept are counted; expert-rejected clips are left out.")

    rr = ours[(ours["category"] == cat) & (ours["status"] == "needs_rerecord")]
    if not rr.empty:
        with st.expander(f"{len(rr)} take(s) flagged 'needs re-record' in this category"):
            st.dataframe(rr[["signer", "sign_id", "take", "file_path"]], hide_index=True, use_container_width=True)


def overview(snap: state.Snapshot) -> None:
    """All datasets, the one in use highlighted, with a button to switch."""
    section("Your datasets")
    current = snap.data_dir
    for i, (label, path) in enumerate(data_source.available()):
        in_use = current is not None and path == current
        c1, c2 = st.columns([5, 1.2], vertical_alignment="center")
        kind = "recordings, read in place" if not cfg.import_name(path) else "uploaded copy"
        c1.html(f'<div style="padding:4px 0"><b style="color:{"#1F4E99" if in_use else "#1A1A1A"}">{label}</b>'
                f'{"  · in use" if in_use else ""}<div class="note">{kind} · {path}</div></div>')
        if not in_use and c2.button("Use", key=f"ds_use_{i}", use_container_width=True):
            data_source.use(path, f"Switched to {label}.")


def uploaded_details(snap: state.Snapshot) -> None:
    """An uploaded class-folder dataset: files per sign and split, and a few example photos."""
    kept = snap.kept
    labels = {s.sign_id: s.sign_label for s in snap.signs}
    n_img = int((kept.get("kind", pd.Series(dtype=str)) == "image").sum())
    stats([("Files", len(kept), ""), ("Signs", kept["sign_id"].nunique(), ""), ("Photos", n_img, ""),
           ("Videos", len(kept) - n_img, "")])
    note(f"Trains one model: <b>{snap.model_cats[0] if snap.model_cats else '—'}</b>. Split fixed at upload: 70 % train, 15 % validation, 15 % test per sign; copies of the same photo are "
         "always in the same part, so the test result is honest.")
    section("Files per sign")
    rows = []
    for sid, g in kept.groupby("sign_id", sort=True):
        c = g["split"].value_counts()
        rows.append({"Sign": labels.get(sid, sid), "Train": int(c.get("train", 0)), "Validation": int(c.get("val", 0)),
                     "Test": int(c.get("test", 0)), "Total": len(g)})
    df = pd.DataFrame(rows)
    tot = {"Sign": "Total", **{k: int(df[k].sum()) for k in ("Train", "Validation", "Test", "Total")}}
    table(pd.concat([df, pd.DataFrame([tot])], ignore_index=True),
          numeric=["Train", "Validation", "Test", "Total"], total_row=True)
    imgs = kept[kept.get("kind", pd.Series(dtype=str)) == "image"]
    if not imgs.empty:
        section("Examples")
        pick = imgs.groupby("sign_id", sort=True).head(1).head(12)
        cols = st.columns(6)
        for i, (_, r) in enumerate(pick.iterrows()):
            with cols[i % 6]:
                st.image(str(snap.data_dir / r["file_path"]), caption=labels.get(r["sign_id"], r["sign_id"]),
                         use_container_width=True)


def combine_panel(snap: state.Snapshot) -> None:
    """Make one dataset (one model) out of signs from several datasets, e.g. A-Y photos + J/Z videos."""
    from trainer.combine import Part, combine, default_leave_out, items_of
    st.html('<div class="note" style="font-size:13.5px;line-height:1.7">Pick the parts to put together, e.g. the '
            "uploaded <b>alphabet photos</b> and the recorded <b>alphabet_dynamic</b> (J, Z). Signs with the same "
            "label become one class. Where a sign has both photos and videos (J, Z), the photos are left out by "
            "default because those signs are movements. Already-extracted samples are reused; each part keeps "
            "its split (recordings by signer, uploads their fixed split).</div>")
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
            labels[p] = f"{ds_label} · {c} ({n} sign{'s' if n != 1 else ''})"
    chosen = st.multiselect("Combine these", parts, format_func=labels.get, key="cb_parts",
                            placeholder="e.g. alphabet_static (uploaded) and recordings · alphabet_dynamic")
    if not chosen:
        return
    items = [it for p in chosen for it in items_of(p)]
    leave = default_leave_out(items)
    by_id = {it.id: it for it in items}
    default = [it.id for it in items if it.id not in leave and it.n_files]
    keep = st.multiselect("Signs to include", list(by_id), default=default, key=f"cb_signs_{len(chosen)}",
                          format_func=lambda i: f"{by_id[i].label} · {by_id[i].kind} · {by_id[i].part.dataset_label} "
                                                f"({by_id[i].n_files})")
    left = [by_id[i] for i in by_id if i not in keep]
    if left:
        note("Left out: " + ", ".join(f"{it.label} ({it.kind}, {it.part.dataset_label})" for it in left) + ".")
    nones = []
    for ds_label, path in data_source.available():
        try:
            k = ds.kept_rows(path)
        except Exception:  # noqa: BLE001
            continue
        n = int((k["category"] == cfg.NONE_CATEGORY).sum()) if not k.empty else 0
        if n:
            nones.append((path, f"{ds_label} ({n} clips)"))
    opts = [None] + [p for p, _ in nones]
    none_lab = dict(nones)
    none_from = st.selectbox("Add the 'none' class (hands idle / no sign) from", opts, key="cb_none",
                             format_func=lambda p: "no none class" if p is None else none_lab[p])
    if not nones:
        note("No dataset has 'none' clips yet, so the combined model has no none class. Record 'none' takes with "
             "the HAND TRACKER and combine again to add it (recommended for the live test).")
    name = st.text_input("Name of the combined dataset", value="alphabet", key="cb_name")
    chosen_items = [by_id[i] for i in keep]
    if st.button("Create combined dataset", key="cb_go", type="primary", disabled=len(chosen_items) < 2):
        with st.spinner("Combining (reusing extracted samples)..."):
            try:
                res = combine(name or "combined", chosen_items, none_from)
            except ds.PlainError as e:
                banner("error", e.message, e.hint)
                return
        msg = (f"Created '{res.name}': {res.n_files} file(s), {len(res.signs)} sign(s); {res.n_reused} sample(s) "
               f"reused. ")
        msg += " ".join(res.notes)
        st.session_state.pop("cb_parts", None)
        data_source.use(res.root, msg)
