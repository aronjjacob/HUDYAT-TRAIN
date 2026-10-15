"""Step 5: Train one model per category."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trainer import jobs
from trainer import settings as cfg
from trainer.train import DEFAULTS, LIVE_CSV, load_config

from . import data_source, state
from .common import banner, files_table, history_chart, job_button, job_monitor, note, page_header, section, stats

ALL = "All categories"


def _live_chart() -> None:
    p = cfg.Paths().jobs / LIVE_CSV
    if not p.exists():
        st.html('<div class="note">Waiting for the first epoch...</div>')
        return
    try:
        df = pd.read_csv(p)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return
    if df.empty:
        return
    cat = df["category"].iloc[-1]
    d = df[df["category"] == cat]
    st.html(f'<div class="note">Now training <b>{cat}</b> · epoch {int(d["epoch"].iloc[-1])}</div>')
    history_chart(d)


def render(snap: state.Snapshot) -> None:
    page_header("Step 5", "Train",
                "Trains one model per category (its signs + none) on the training part with augmentation, using the "
                "validation part to decide "
                "when to stop.")
    if not snap.ok:
        data_source.missing(snap)
        return
    opts = [ALL] + snap.split_cats if snap.split_cats else [ALL]
    c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1, 1, 1])
    cat = state.page_category(snap, "Category", opts, key="tr_cat", container=c1)
    epochs = c2.number_input("Epochs", 1, 1000, DEFAULTS["epochs"], key="tr_epochs",
                             help="Maximum; training stops early when validation loss stops improving (patience 20).")
    batch = c3.number_input("Batch", 4, 512, DEFAULTS["batch_size"], key="tr_batch")
    aug = c4.number_input("Aug. factor", 0, 20, DEFAULTS["aug_factor"], key="tr_aug",
                          help="Augmented copies of every training clip (training set only).")
    seed = c5.number_input("Seed", 0, 10 ** 6, DEFAULTS["seed"], key="tr_seed")
    balance = st.checkbox("Balance rare signs", value=True, key="tr_balance",
                          help="Signs with fewer training clips count more, so a model trained on e.g. 450 photos "
                               "per letter and 30 videos of J still learns J. Turn off only to compare.")
    reason = None if snap.split_cats else "Run step 4 (Split) first."
    if cat != ALL and cat not in snap.split_cats:
        reason = f"Run step 4 (Split) for {cat} first."
    cats = snap.split_cats if cat == ALL else [cat]
    retrain = [c for c in cats if c in snap.trained]
    args = ["--category", *(["all"] if cat == ALL else [cat]), "--epochs", str(int(epochs)), "--batch",
            str(int(batch)), "--aug", str(int(aug)), "--seed", str(int(seed))] + ([] if balance else ["--no-balance"])
    label = "Start training" + ("" if cat == ALL else f": {cat}")
    from trainer.extract import free_memory_gb
    free = free_memory_gb()
    if free < 1.2:
        banner("warning", f"Only {free:.1f} GB of memory is free. Training needs about 1–1.5 GB.",
               "Close other programs first (extra browsers, Spotify, Zoom), otherwise training can be very slow "
               "or stop. Keep this one browser tab open.")
    job_button("train", label, args, reason, key="train_start")
    if retrain and not reason:
        note(f"Already trained: {', '.join(retrain)}. Training again replaces those models.")
    job_monitor("train", while_running=_live_chart)

    if not snap.trained:
        return
    section("Saved models")
    show = st.selectbox("Show model", snap.trained, key="tr_show",
                        index=snap.trained.index(cat) if cat in snap.trained else 0)
    conf = load_config(snap.paths, show) or {}
    stats([("Best validation accuracy", f"{conf.get('best_val_accuracy', 0):.1%}", ""),
           ("Best epoch", f"{conf.get('best_epoch', '—')} of {conf.get('epochs_run', '—')}", ""),
           ("Training samples", f"{conf.get('n_train_samples', '—')}", ""),
           ("Classes", len(conf.get("labels", [])), ""),
           ("Training time", f"{conf.get('train_seconds', 0) / 60:.1f} min", "")])
    hist = snap.paths.model_dir(show) / "history.csv"
    if hist.exists():
        df = pd.read_csv(hist)
        df.insert(0, "epoch", df.pop("epoch"))
        history_chart(df)
    o = conf.get("options", {})
    note(f"Trained {conf.get('date', '').replace('T', ' ')} · epochs ≤ {o.get('epochs')}, batch {o.get('batch_size')}, "
         f"augmentation ×{o.get('aug_factor')}, seed {o.get('seed')} · validation "
         f"({conf.get('n_val', '—')} clips). Evaluation on the test set is step 6.")
    d = snap.paths.model_dir(show)
    files_table([d / f for f in ("model.keras", "labels.json", "config.json", "history.csv", "training_plot.png")],
                snap.paths.work)
    if jobs.busy(snap.paths) is None and (d / "training_plot.png").exists():
        with st.expander("training_plot.png"):
            st.image(str(d / "training_plot.png"))
