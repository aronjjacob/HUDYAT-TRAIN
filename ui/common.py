"""Shared UI pieces: CSS, page header, banners, tables, stat cards, job buttons and the job monitor."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from trainer import jobs
from trainer import settings as cfg

CSS_FILE = cfg.PROJECT_DIR / "static" / "app.css"
ACCENT, GREY = "#1F4E99", "#8A8A8A"


def inject_css() -> None:
    css = CSS_FILE.read_text(encoding="utf-8").replace('url("fonts/', 'url("/app/static/fonts/')
    st.html(f"<style>{css}</style>")


def esc(x) -> str:
    return html.escape(str(x))


# ----------------------------------------------------------------------------
# Text blocks
# ----------------------------------------------------------------------------
def page_header(step: str | None, title: str, lede: str) -> None:
    step_html = f'<div class="hd-step">{esc(step)}</div>' if step else ""
    st.html(f'{step_html}<div class="hd-title">{esc(title)}</div><div class="hd-lede">{esc(lede)}</div>')


def section(title: str, note: str = "") -> None:
    st.html(f'<div class="sec">{esc(title)}{f"<small>{esc(note)}</small>" if note else ""}</div>')


def note(text: str) -> None:
    st.html(f'<div class="note">{text}</div>')


def banner(kind: str, text: str, hint: str = "") -> None:
    """Slim inline banner. kind: info | success | warning | error."""
    hint_html = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    st.html(f'<div class="hb hb-{kind}"><b>{esc(text)}</b>{hint_html}</div>')


def stats(items: list[tuple[str, object, str]]) -> None:
    """Stat cards: [(label, value, tone)] with tone '' | ok | warn | err."""
    cards = "".join(f'<div class="stat {tone}"><div class="v">{esc(v)}</div><div class="l">{esc(lab)}</div></div>'
                    for lab, v, tone in items)
    st.html(f'<div class="stats">{cards}</div>')


def table(df: pd.DataFrame, numeric: list[str] | None = None, classes: dict | None = None,
          total_row: bool = False, index: bool = False) -> None:
    """A light HTML table (tabular, right-aligned numbers). classes: {(row_i, col): css class}."""
    numeric = set(numeric or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    classes = classes or {}
    cols = list(df.columns)
    head = "".join(f'<th class="{"n" if c in numeric else ""}">{esc(c)}</th>' for c in cols)
    body = []
    for i, (_, row) in enumerate(df.iterrows()):
        tr_cls = ' class="total"' if total_row and i == len(df) - 1 else ""
        cells = []
        for c in cols:
            v = row[c]
            cls = " ".join(x for x in ("n" if c in numeric else "", classes.get((i, c), "")) if x)
            txt = "" if v is None or (isinstance(v, float) and pd.isna(v)) else v
            cells.append(f'<td class="{cls}">{esc(txt)}</td>')
        body.append(f"<tr{tr_cls}>{''.join(cells)}</tr>")
    st.html(f'<div class="tbl-wrap"><table class="t"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def pct(v, digits: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v * 100:.{digits}f}%"


# ----------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------
def busy_reason() -> str | None:
    b = jobs.busy(cfg.Paths())
    return f"{jobs.JOB_TITLES.get(b, b)} is running. Wait until it has finished." if b else None


def job_button(job: str, label: str, args: list[str], reason: str | None, key: str,
               kind: str = "primary", show_window: bool = False, help: str | None = None) -> None:
    """A button that starts `job` in the background. Disabled (with the reason shown) when it can't run."""
    reason = reason or busy_reason()
    clicked = st.button(label, key=key, type=kind, disabled=reason is not None, help=help)
    if reason:
        st.html(f'<div class="note">{esc(reason)}</div>')
    if clicked:
        ok, msg = jobs.start_background(job, args, show_window=show_window)
        if not ok:
            banner("error", msg, "Wait until it has finished, then try again.")
        else:
            st.rerun()


def _fmt_time(iso: str | None) -> str:
    return (iso or "").replace("T", " ")[:16]


def job_monitor(job: str, while_running: Callable[[], None] | None = None,
                after: Callable[[], None] | None = None) -> None:
    """Live progress + log while `job` runs (refreshes every second), else its last result."""
    paths = cfg.Paths()
    running_at_start = jobs.busy(paths)

    @st.fragment(run_every=1.0 if running_at_start else None)
    def _monitor():
        now = jobs.busy(paths)
        if now != running_at_start:
            st.rerun()                     # a job started/finished: redraw the whole page (buttons, results)
        if now == job:
            p = jobs.read_progress(paths)
            st.progress(float(p.get("fraction", 0.0)), text=p.get("text") or "Running...")
            c1, _ = st.columns([1, 5])
            if c1.button("Stop", key=f"stop_{job}", type="secondary"):
                jobs.cancel(paths)
                st.rerun()
            if while_running:
                while_running()
            with st.expander("Log (live)", expanded=False):
                st.code(jobs.read_log(paths, job, tail_lines=40) or "...", language=None)
            return
        stt = jobs.read_status(paths, job)
        if not stt or stt.get("state") == "running":
            return
        when = _fmt_time(stt.get("finished"))
        if stt.get("state") == "done":
            banner("success", stt.get("message") or "Finished.", f"Finished {when}.")
        else:
            banner("error", stt.get("message") or "Failed.", stt.get("hint") or "")
        log = jobs.read_log(paths, job)
        if log:
            with st.expander("Full log", expanded=False):
                st.code(log[-60000:], language=None)
        if after:
            after()

    _monitor()


# ----------------------------------------------------------------------------
# Charts (Altair): white, thin lines, accent = train, grey = validation
# ----------------------------------------------------------------------------
def history_chart(df: pd.DataFrame, height: int = 220) -> None:
    """df columns: epoch, loss, accuracy, val_loss, val_accuracy."""
    import altair as alt
    if df.empty:
        return
    cols = st.columns(2, gap="medium")
    for col, metric, title in ((cols[0], "loss", "Loss"), (cols[1], "accuracy", "Accuracy")):
        d = pd.concat([
            pd.DataFrame({"Epoch": df["epoch"], "value": df[metric], "Set": "Train"}),
            pd.DataFrame({"Epoch": df["epoch"], "value": df[f"val_{metric}"], "Set": "Validation"}),
        ])
        ch = (alt.Chart(d, title=alt.Title(title, anchor="start", fontSize=13, fontWeight=600, color="#1A1A1A"))
              .mark_line(strokeWidth=1.6)
              .encode(x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1, title="Epoch")),
                      y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
                      color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Validation"], range=[ACCENT, GREY]),
                                      legend=alt.Legend(orient="top-right", title=None)),
                      tooltip=["Set", "Epoch", alt.Tooltip("value:Q", format=".4f")])
              .properties(height=height)
              .configure_view(stroke=None)
              .configure_axis(gridColor="#EEEEEB", domainColor="#E6E6E3", tickColor="#E6E6E3",
                              labelColor="#6B6B6B", titleColor="#6B6B6B", labelFont="Poppins", titleFont="Poppins",
                              titleFontWeight=400, labelFontSize=11, titleFontSize=11)
              .configure_legend(labelFont="Poppins", labelColor="#6B6B6B", labelFontSize=11))
        col.altair_chart(ch, use_container_width=True)


def files_table(paths_list: list[Path], root: Path) -> None:
    rows = []
    for p in paths_list:
        if p.is_file():
            size = p.stat().st_size
            rows.append({"File": str(p.relative_to(root)).replace("\\", "/"),
                         "Size": (f"{size} B" if size < 1024 else f"{size / 1024:,.1f} KB" if size < 1024 ** 2
                                  else f"{size / 1024 ** 2:,.2f} MB")})
    if rows:
        table(pd.DataFrame(rows), numeric=["Size"])
