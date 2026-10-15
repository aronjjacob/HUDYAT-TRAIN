"""Step 4: Split into train / validation / test (by signer, or the fixed split of an upload)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer import settings as cfg
from trainer.split import load_split

from . import data_source, state
from .common import banner, job_button, job_monitor, note, page_header, section, table

ALL = "All categories"


def render(snap: state.Snapshot) -> None:
    page_header("Step 4", "Split",
                "Puts every usable sample into training, validation or test: by signer for the recordings (S1–S4 / S5 / S6–S7), or the fixed 70 / 15 / 15 % split of an uploaded dataset. The none class "
                "is added to every category's model.")
    if not snap.ok:
        data_source.missing(snap)
        return
    reason = "Extract landmarks first (step 2)." if snap.index.empty else None
    cat = state.page_category(snap, "Category", [ALL] + snap.model_cats, key="sp_cat")
    args = [] if cat == ALL else ["--category", cat]
    job_button("split", "Run split" + ("" if cat == ALL else f" for {cat}"), args, reason, key="split_run")
    note("Run it again after extracting new clips or rejecting clips in Validation. "
         "Clips rejected, low in hand detection, or not kept are left out.")
    job_monitor("split")

    if not snap.split_cats:
        return
    section("Samples per sign and split")
    show = st.selectbox("Show category", snap.split_cats, key="sp_show",
                        index=snap.split_cats.index(cat) if cat in snap.split_cats else 0)
    sp = load_split(snap.paths, show)
    by_signer = sp.get("split_by", "signer") == "signer"
    TR, VA, TE = (("Train (S1–S4)", "Validation (S5)", "Test (S6–S7)") if by_signer
                  else ("Train (70 %)", "Validation (15 %)", "Test (15 %)"))
    labels = {s.sign_id: s.sign_label for s in snap.signs}
    rows, classes = [], {}
    for i, lab in enumerate(sp["labels"]):
        c = sp["counts"][lab]
        rows.append({"Sign": labels.get(lab, lab), TR: c["train"], VA: c["val"],
                     TE: c["test"], "Mirror allowed": "yes" if sp["mirror_ok"].get(lab) else "no"})
        for k, col in (("train", TR), ("val", VA), ("test", TE)):
            if c[k] == 0:
                classes[(i, col)] = "err"
    df = pd.DataFrame(rows)
    tot = {"Sign": "Total", TR: len(sp["train"]), VA: len(sp["val"]),
           TE: len(sp["test"]), "Mirror allowed": ""}
    table(pd.concat([df, pd.DataFrame([tot])], ignore_index=True),
          numeric=[TR, VA, TE], classes=classes, total_row=True)
    how = (f"Signers: train {', '.join(cfg.SPLIT_SIGNERS['train'])}, validation "
           f"{', '.join(cfg.SPLIT_SIGNERS['val'])}, test {', '.join(cfg.SPLIT_SIGNERS['test'])}." if by_signer
           else "Split fixed when the dataset was uploaded (copies of the same photo are kept together).")
    note(f"Saved {sp['created'].replace('T', ' ')} in work/splits/{show}.json. {how}")
    if sp["warnings"]:
        banner("warning", f"{len(sp['warnings'])} sign(s) are missing from a split.",
               "Training needs train and validation clips; evaluation needs test clips for every sign.")
        with st.expander("Details"):
            for w in sp["warnings"]:
                st.html(f'<div class="note">{w}</div>')
    missing = [c for c in snap.model_cats if c not in snap.split_cats]
    if missing:
        note(f"Not split yet: {', '.join(missing)}.")
