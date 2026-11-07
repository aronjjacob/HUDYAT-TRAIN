"""Train model: train one model per category on the extracted landmarks (every signer), straight away or with the
settings of a test run, then export it for the HUDYAT app. "All categories" trains one model each, one after another."""

from __future__ import annotations


import streamlit as st

from trainer import experiments as E
from trainer.app_export import exported_files, last_export
from trainer.train import load_config

from . import data_source, state
from .common import (banner, esc, files_table, headline, history_chart, job_button, job_monitor, note,
                     page_header, pct, stats, table)
from .p_test import settings_box, version_pick

ALL = "__all__"
OWN, FROM_RUN = "own", "run"


def _live() -> None:
    import pandas as pd
    p = state.cfg.Paths().jobs / "train_live.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            return
        if not df.empty:
            history_chart(df[df["category"] == df["category"].iloc[-1]], height=180)


def render(snap: state.Snapshot) -> None:
    page_header(state.step_label("final"), "Train model", "Train the final model with every signer, then export it for the app.")
    if not snap.ok:
        data_source.missing(snap)
        return
    if not snap.n_rich:
        banner("info", "No landmarks yet.", "Extract them on the Preprocess page.")
        return
    cats = snap.model_cats
    headline(f"{len(snap.finals)} of {len(cats)} categories trained",
             f"{len(snap.exported)} exported for the HUDYAT app", len(snap.finals), len(cats))
    opts = [ALL] + cats if len(cats) > 1 else cats
    cat = state.pick_category(opts, "final", fmt=lambda c: f"All categories: {len(cats)} models, one after another"
                              if c == ALL else c)
    conf = (load_config(snap.paths, cat) or {}) if cat in snap.finals else {}
    t_train, t_model, t_export = st.tabs(["Train", "Trained model", "Export for the app"])
    with t_train:
        train(snap, cat, cats)
    with t_model:
        overview(snap, cats) if cat == ALL else trained(snap, cat, conf)
    with t_export:
        export_all(snap) if cat == ALL else export(snap, cat, conf)


def train(snap: state.Snapshot, cat: str, cats: list[str]) -> None:
    """Every choice (model version, settings of a test run or its own) in one Training settings box, as on Test."""
    with st.expander("Training settings"):
        v = version_pick("fm", label="Model version", horizontal=True)
        runs = [r for r in snap.runs if cat != ALL and r["category"] == cat and r["complete"]
                and r["feature_version"] == v]
        how = OWN
        if runs:
            how = st.radio("Settings", [OWN, FROM_RUN], horizontal=True, key="fm_how",
                           format_func={OWN: "Choose them here", FROM_RUN: "Use the settings of a test run"}.get)
        if how == FROM_RUN:
            lab = {r["run_id"]: f"{E.SHORT[r['test']]} · v{r['feature_version']} · canonical hand "
                                f"{'on' if r['canonical'] else 'off'} · accuracy {pct(r['accuracy_mean'])} · "
                                f"{r['created'].replace('T', ' ')[:16]}" for r in runs}
            pick = st.selectbox("Test run", list(lab), format_func=lab.get, key="fm_run")
            args = ["--from-run", pick, "--feature-version", str(v)]
        else:
            args = ["--feature-version", str(v), *settings_box(snap, key="fm", title=None, show_version=False)]
    targets = cats if cat == ALL else [cat]
    if any(c in snap.finals for c in targets):
        old = {int((load_config(snap.paths, c) or {}).get("feature_version", 1)) for c in targets if c in snap.finals}
        note("Training again replaces the current model." if old <= {v} else
             f"The current model is v{'/v'.join(map(str, sorted(old)))}; training replaces it with v{v}.")
    label = f"Train {len(cats)} models · v{v}" if cat == ALL else f"Train model · v{v}"
    job_button("train", label, ["--category", *(["all"] if cat == ALL else [cat]), *args], None, key="fm_train")
    job_monitor("train", while_running=_live)


def trained(snap: state.Snapshot, cat: str, conf: dict) -> None:
    if not conf:
        note(f"No model for {esc(cat)} yet. Train one first.")
        return
    note(f"Run {esc(conf.get('model_run_id', ''))}")
    stats([("Validation accuracy", pct(conf.get("best_val_accuracy")), ""),
           ("Best epoch", f"{conf.get('best_epoch', '—')} of {conf.get('epochs_run', '—')}", ""),
           ("Clips", f"{conf.get('n_train', 0) + conf.get('n_val', 0)}", ""),
           ("Signers", ", ".join(conf.get("signers", [])), ""),
           ("Features", f"v{conf.get('feature_version')}", "")])
    o = conf.get("opts", {})
    note(f"Settings: {esc(conf.get('from_run') or 'default')} · left-hand support "
         f"{'on' if o.get('canonical') else 'off'} · seed {o.get('seed')}")
    hist = snap.paths.model_dir(cat) / "history.csv"
    if hist.exists():
        import pandas as pd
        with st.expander("Training curves"):
            history_chart(pd.read_csv(hist), height=180)


def export(snap: state.Snapshot, cat: str, conf: dict) -> None:
    if not conf:
        note("Train a model first.")
        return
    note("model.json, weights.bin, parity.json")
    job_button("export", "Export for the app", ["--category", cat], None, key="fm_export")
    chk = last_export(snap.paths, cat)
    job_monitor("export", quiet_success=bool(chk))           # the check below says the same, with the version
    if chk:
        if chk.get("model_run_id") not in (None, conf.get("model_run_id")):
            banner("warning", "This export is of an older model.", "Export again.")
        elif chk.get("ok"):
            banner("success", "Exported and checked.", f"Version {chk['version']}: same answers on "
                                                         f"{chk['same_top1']:.0%} of {chk['n_samples']} test clips.")
        else:
            banner("error", "Export check failed.", "Send the log to the developers.")
        name = chk["export_name"]
        note(f"Saved as <b>{esc(name)}</b>. Its README.txt says where each file goes in the HUDYAT app.")
        with st.expander("Details"):
            files_table(exported_files(snap.paths, cat), snap.paths.work)
            note(f"Folder: {esc(snap.paths.exports / name)}")


def overview(snap: state.Snapshot, cats: list[str]) -> None:
    """All categories: which have a model and an export."""
    rows = []
    for c in cats:
        conf = load_config(snap.paths, c) if c in snap.finals else None
        rows.append({"Category": c, "Model": "trained" if conf else "—",
                     "Validation accuracy": pct(conf.get("best_val_accuracy")) if conf else "—",
                     "Features": f"v{conf.get('feature_version')}" if conf else "—",
                     "Trained": (conf.get("date") or "").replace("T", " ")[:16] if conf else "—",
                     "Exported": "yes" if c in snap.exported else "—"})
    import pandas as pd
    table(pd.DataFrame(rows))


def export_all(snap: state.Snapshot) -> None:
    """All categories: one button to export every picked model."""
    if not snap.finals:
        note("Train a model first.")
        return
    note("model.json, weights.bin, parity.json")
    pick = st.multiselect("Models to export", snap.finals, default=snap.finals, key="fm_export_pick")
    job_button("export", f"Export {len(pick)} for the app", ["--category", *pick],
               None if pick else "Tick at least one model to export.", key="fm_export_all")
    job_monitor("export")
    note(f"Each one is saved as its own folder in {esc(snap.paths.exports)}, named "
         f"&lt;category&gt;-dabest-{esc(snap.paths.dataset)}-v&lt;model version&gt;.")
