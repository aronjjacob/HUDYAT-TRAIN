"""Step 6: Results for the paper: Generate figures, a gallery with downloads, tables and the summary."""

from __future__ import annotations

import json
import re

import streamlit as st

from trainer import paper

from . import data_source, state
from .common import banner, esc, headline, job_button, job_monitor, note, page_header, section, table


def _captions(text: str) -> dict[str, str]:
    return {m.group(2): m.group(3).strip() for m in re.finditer(r"\*\*Figure (\d+)\*\* \(`([^`]+)`\)\. (.*)", text)}


def render(snap: state.Snapshot) -> None:
    page_header(state.step_label("paper"), "Results for the paper", "Figures, tables and captions for one category.")
    if not snap.ok:
        data_source.missing(snap)
        return
    cats = sorted({r["category"] for r in snap.runs if r["complete"]}) or snap.model_cats
    if not cats:
        banner("info", "Nothing to show yet.", "Run a test on the Test the model page first.")
        return
    made = [c for c in cats if (paper.out_dir(snap.paths, c) / "paper.json").exists()]
    headline(f"{len(made)} of {len(cats)} categories with figures", "Figures, tables and captions, per category",
             len(made), len(cats))
    cat = state.pick_category(cats, "paper")
    out = paper.out_dir(snap.paths, cat)
    meta_p = out / "paper.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else None
    tabs = st.tabs(["Generate", "Tables", "Figures", "Summary"])
    with tabs[0]:
        generate(snap, cat, out, meta)
    if not meta:
        for t in tabs[1:]:
            with t:
                note("Nothing yet. Generate the figures first.")
        return
    with tabs[1]:
        tables(out)
    with tabs[2]:
        figures(out, meta.get("figures", []))
    with tabs[3]:
        rs = out / "results_summary.md"
        if rs.exists():
            st.markdown(rs.read_text(encoding="utf-8"))
        note(f"SENYAS benchmark values: {esc(paper.SENYAS['source'])}. Constants in trainer/paper.py.")


def generate(snap: state.Snapshot, cat: str, out, meta: dict | None) -> None:
    """Pick the Test 1 / Test 2 runs, Generate figures, then the .zip with everything."""
    args = ["--category", cat]
    try:
        auto = paper.pick_runs(snap.paths, cat)
    except Exception:                                        # noqa: BLE001  (fall back to choosing automatically)
        auto = {}
    c1, c2 = st.columns(2)
    for col, test, key in ((c1, "random", "test1"), (c2, "loso", "test2")):
        runs = [r for r in snap.runs if r["category"] == cat and r["test"] == test and r["complete"]]
        if not runs:
            continue
        ids = [r["run_id"] for r in runs]
        lab = {r["run_id"]: f"{r['run_id']} · accuracy {(r['accuracy_mean'] or 0) * 100:.2f} %"
               for r in runs}
        default = (auto.get(key) or {}).get("run_id")
        pick = col.selectbox(f"{'Test 1: random split' if key == 'test1' else 'Test 2: leave-one-signer-out'} run",
                             ids, index=ids.index(default) if default in ids else 0, format_func=lab.get,
                             key=f"pa_{key}_{cat}")
        args += [f"--{key}", pick]
    job_button("paper", "Generate figures", args, None, key="pa_generate")
    job_monitor("paper")
    if not meta:
        note("Press Generate figures. They are also made by themselves each time you train this category's model.")
        return
    note(f"Made {esc(meta.get('created', '').replace('T', ' ')[:16])}.")
    z = out / "paper.zip"
    if z.exists():
        st.download_button("Download everything as .zip", z.read_bytes(), file_name=f"hudyat_paper_{meta['category']}.zip",
                           mime="application/zip", key="pa_zip", type="primary")
    for m in meta.get("missing", []):
        banner("warning", m)
    with st.expander("Details"):
        note(f"Folder: {esc(out)} · runs used: " + ", ".join(f"{k} {v}" for k, v in meta["runs"].items() if v))


def tables(out) -> None:
    for name, title in (("table6_random", "Table 6: random split"), ("table7_loso", "Table 7: leave-one-signer-out"),
                        ("table7_loso_per_fold", "Table 7 per fold"), ("errors_loso", "Errors in Test 2"),
                        ("errors_random", "Errors in Test 1")):
        p = out / "tables" / f"{name}.csv"
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            section(title)
            if df.empty:
                note("None.")
            else:
                table(df)
    md = out / "tables" / "tables.md"
    if md.exists():
        with st.expander("The same tables as Markdown, to copy"):
            st.markdown(md.read_text(encoding="utf-8"))


def figures(out, figs: list[str]) -> None:
    caps = _captions((out / "captions.md").read_text(encoding="utf-8")) if (out / "captions.md").exists() else {}
    for i in range(0, len(figs), 2):
        cols = st.columns(2, gap="medium")
        for col, name in zip(cols, figs[i:i + 2]):
            png, pdf = out / "figures" / f"{name}.png", out / "figures" / f"{name}.pdf"
            if not png.exists():
                continue
            with col.container(border=True):
                st.image(str(png))
                st.html(f'<div class="note"><b style="color:#1A1A1A">Figure {figs.index(name) + 1}</b> '
                        f'· {esc(name)}. {esc(caps.get(name, ""))}</div>')
                c1, c2 = st.columns(2)
                c1.download_button("PNG · 300 dpi", png.read_bytes(), file_name=png.name, mime="image/png",
                                   key=f"pa_png_{name}", use_container_width=True)
                if pdf.exists():
                    c2.download_button("PDF · vector", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf",
                                       key=f"pa_pdf_{name}", use_container_width=True)
