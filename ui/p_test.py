"""Step 3: Test the model: Test 1 (random split) and Test 2 (leave-one-signer-out), results, Compare runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from trainer import experiments as E
from trainer import jobs
from trainer.metrics import PASS_METRIC, PASS_PER_SIGN

from . import data_source, state
from .common import (banner, esc, headline, history_chart, job_button, job_monitor, note, page_header, pct,
                     plural, stats, table)

ALL = "__all__"
INFO = {
    "random": {"title": "Test 1: Random split",
               "question": "Did it learn the signs?",
               "detail": "70% train · 15% check · 15% test, from the same people."},
    "loso": {"title": "Test 2: New signer",
             "question": "Does it work for someone new?",
             "detail": "Each signer is tested by a model that never saw them. The main result."},
}


def render(snap: state.Snapshot) -> None:
    page_header(state.step_label("test"), "Test the model", "Train and check how well the model recognises the signs.")
    if not snap.ok:
        data_source.missing(snap)
        return
    if not snap.n_rich:
        banner("info", "No landmarks yet.", "Extract them on the Preprocess page.")
        return
    cats = snap.model_cats
    main = "random" if snap.photos else "loso"
    tested = [c for c in cats if any(r["category"] == c and r["test"] == main and r["complete"] for r in snap.runs)]
    headline(f"{len(tested)} of {len(cats)} categories tested",
             f"{'Test 1' if snap.photos else 'Test 2, new signer,'} finished · "
             f"{plural(sum(r['complete'] for r in snap.runs), 'finished run')}", len(tested), len(cats))
    opts = [ALL] + cats if len(cats) > 1 else cats
    cat = state.pick_category(opts, "test", fmt=lambda c: "All categories" if c == ALL else c)
    runs = [r for r in snap.runs if cat == ALL or r["category"] == cat]
    shown = [r for r in runs if r["rounds_done"]]
    t_run, t_res, t_cmp = st.tabs(["Run a test", "Results", "Compare runs"])
    with t_run:
        run_tests(snap, cat, runs)
    with t_res:
        results(snap, shown)
    with t_cmp:
        compare(shown) if shown else note("No runs yet.")


def run_tests(snap: state.Snapshot, cat: str, runs: list[dict]) -> None:
    args = settings_box(snap)
    c1, c2 = st.columns(2, gap="medium")
    cat_args = ["--category", *(["all"] if cat == ALL else [cat])]
    for col, test in ((c1, "random"), (c2, "loso")):
        with col.container(border=True):
            i = INFO[test]
            st.html(f'<div style="font-weight:600;font-size:15.5px;margin-bottom:2px">{i["title"]}</div>'
                    f'<div class="note" style="font-size:13.5px;color:#1A1A1A">{i["question"]}</div>'
                    f'<div class="note">{i["detail"]}</div>')
            reason = None
            if test == "loso":
                if snap.photos:
                    reason = "Photos have no signers. Use Test 1."
                elif len(snap.signers) < 2:
                    reason = "Needs clips of at least two signers."
            label = "Run Test 1" if test == "random" else "Run Test 2"
            job_button("experiment", label, ["--test", test, *cat_args, *args], reason, key=f"ts_run_{test}",
                       kind="primary" if test == "loso" else "secondary")
    job_monitor("experiment", while_running=_live)
    stopped = [r for r in runs if r["state"] == "stopped"]
    if stopped and not jobs.busy(snap.paths):
        r = stopped[0]
        banner("warning", f"{E.SHORT[r['test']]} on {r['category']} stopped after "
                          f"{r['rounds_done']} of {r['n_rounds']} rounds.", "Resume keeps the finished rounds.")
        job_button("experiment", "Resume", ["--resume", r["run_id"]], None, key="ts_resume", kind="secondary")


def results(snap: state.Snapshot, shown: list[dict]) -> None:
    if not shown:
        note("No results yet. Run a test first.")
        return
    loso_done = [r for r in shown if r["test"] == "loso" and r["complete"]]
    default = (loso_done or shown)[0]
    labels = {r["run_id"]: f"{E.SHORT[r['test']]} · {r['category']} · v{r['feature_version']} · "
                           f"{r['created'].replace('T', ' ')[:16]}" + ("" if r["complete"] else " · unfinished")
              for r in shown}
    ids = [r["run_id"] for r in shown]
    if st.session_state.get("ts_show") not in ids:
        st.session_state.ts_show = default["run_id"]
    rid = st.selectbox("Show run", ids, format_func=labels.get, key="ts_show")
    show_run(snap, next(r for r in shown if r["run_id"] == rid))


FEATURES = {1: "v1 · hand shape", 2: "v2 · hand shape + wrist movement", 3: "v3 · hand shape + face location, for Family"}


FEATURES_HELP = ("v1, hand-relative-v1: the hand's shape in every frame; the wrist is always 0. "
                 "v2, hand-relative-sequence-v1: the same, plus how far the wrist moved since the sign started, the "
                 "format of Russel's trainer. v3, hand-face-location-v1: v1 plus where the hand is against the face, "
                 "e.g. Father at the forehead and Mother at the chin; meant for Family, where the place on the face "
                 "matters. Compare them with the same seed.")


def version_pick(key: str, where=st, label: str = "Model input", horizontal: bool = False) -> int:
    """The feature version (1, 2 or 3) the model is trained on."""
    return where.radio(label, [1, 2, 3], key=f"{key}_features", format_func=FEATURES.get, horizontal=horizontal,
                       help=FEATURES_HELP)


def settings_box(snap: state.Snapshot, key: str = "ts", title: str | None = "Settings for this run",
                 show_version: bool = True) -> list[str]:
    """Training settings as command-line arguments (shared by Test the model and Train model). With
    show_version=False the caller picks the feature version and adds --feature-version itself; with title=None the
    fields go straight into the caller's own expander."""
    with st.expander(title) if title else st.container():
        c1, c2 = st.columns(2)
        features = version_pick(key, c1) if show_version else None
        c2 = c2 if show_version else c1                     # no version here: the checkboxes take the left column
        canonical = c2.checkbox("Left-hand support: mirror", value=True, key=f"{key}_canon",
                                help="Left-handed signing is mirrored to right-handed. Keep on.")
        mirror = c2.checkbox("Mirror augmentation", value=False, key=f"{key}_mirror",
                             help="Also train on mirrored copies. Off until the FSL experts confirm.")
        share = 0.5
        if mirror:
            share = c2.slider("Share of training clips mirrored", 0.1, 1.0, 0.5, 0.1, key=f"{key}_share")
            note("Only signs marked mirror_ok by the FSL experts are mirrored.")
        c1, c2, c3, c4 = st.columns(4)
        epochs = c1.number_input("Max epochs", 1, 1000, E.TRAIN_DEFAULTS["epochs"], key=f"{key}_epochs",
                                 help="Stops early when it stops improving.")
        seed = c2.number_input("Seed", 0, 10 ** 6, E.TRAIN_DEFAULTS["seed"], key=f"{key}_seed",
                               help="Same seed = same result.")
        batch = c3.number_input("Batch", 4, 512, E.TRAIN_DEFAULTS["batch_size"], key=f"{key}_batch")
        aug = c4.number_input("Extra copies", 0, 20, E.TRAIN_DEFAULTS["aug_factor"], key=f"{key}_aug",
                              help="Slightly changed copies of each training clip.")
    args = ["--epochs", str(int(epochs)), "--seed", str(int(seed)), "--batch", str(int(batch)), "--aug", str(int(aug))]
    if show_version:
        args = ["--feature-version", str(features), *args]
    if not canonical:
        args.append("--no-canonical")
    if mirror:
        args += ["--mirror-aug", "--mirror-share", str(share)]
    return args


def _live() -> None:
    """While a test runs: its rounds (done / running / waiting) and the current training curve."""
    paths = state.cfg.Paths()
    running = [r for r in E.list_runs(paths) if r["state"] == "running"]
    if running:
        r = running[0]
        conf = E.load_config(Path(r["dir"]))
        prog = jobs.read_progress(paths)
        items = []
        for rd in conf["rounds"]:
            m = E._read(Path(r["dir"]) / f"round_{rd['name']}" / "metrics.json")
            who = "all signers" if rd["name"] == "all" else f"test signer {rd['name']}"
            if m:
                items.append(f'<span style="color:#2E7D32">✓</span> {who}: accuracy {m["accuracy"]:.1%}')
            elif prog.get("round") == rd["name"]:
                items.append(f'<span style="color:#1F4E99">●</span> {who}: training…')
            else:
                items.append(f'<span style="color:#B5B5B0">○</span> {who}: waiting')
        st.html(f'<div class="note" style="font-size:13.5px;line-height:1.8">{E.SHORT[r["test"]]} on '
                f'<b>{esc(r["category"])}</b> · v{r["feature_version"]}<br>' + "<br>".join(items) + "</div>")
    p = paths.jobs / "train_live.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            return
        if not df.empty:
            d = df[df["category"] == df["category"].iloc[-1]]
            history_chart(d, height=180)


def show_run(snap: state.Snapshot, r: dict) -> None:
    """The verdict, the main numbers and (Test 2) the per-round table first; the rest in tabs."""
    d = Path(r["dir"])
    conf = E.load_config(d)
    res = E.load_results(d)
    i = INFO[r["test"]]
    if res is None:
        note("No round has finished yet.")
        return
    st.html(f'<div style="font-weight:600;font-size:16px;margin-top:4px">{i["title"]} · {esc(r["category"])}</div>'
            f'<div class="note">features v{r["feature_version"]} · left-hand support '
            f'{"on" if r["canonical"] else "off"} · seed {r["seed"]}</div>')
    if not res["complete"]:
        banner("warning", f"Unfinished: {len(res['rounds'])} of {len(conf['rounds'])} rounds.")
    s = res["summary"]
    ok = res["pass"]
    why = "".join(f"<li>{esc(x)}</li>" for x in res["reasons"])
    st.html(f'<div class="verdict-box {"ok" if ok else "err"}"><span class="verdict {"ok" if ok else "err"}">'
            f'{"PASS" if ok else "FAIL"}</span><span class="note">Pass = {PASS_METRIC:.0%} overall and '
            f'{PASS_PER_SIGN:.0%} for every sign.</span>' + (f'<ul>{why}</ul>' if why else "") + "</div>")

    def ms(k):
        m, sd = s[k]["mean"], s[k]["sd"]
        return pct(m) + (f" ± {sd * 100:.1f}" if sd is not None else "")
    h = res["handedness"]
    cards = [("Accuracy", ms("accuracy"), "ok" if s["accuracy"]["mean"] >= PASS_METRIC else "err"),
             ("Macro precision", ms("macro_precision"), ""), ("Macro recall", ms("macro_recall"), ""),
             ("Macro F1", ms("macro_f1"), ""),
             ("Left-handed, simulated", pct(h["mirrored"]["auto"]["accuracy"]["mean"]), "")]
    if r["test"] == "loso":
        t1 = [x for x in snap.runs if x["test"] == "random" and x["category"] == r["category"] and x["complete"]
              and x["feature_version"] == r["feature_version"]]
        if t1:
            cards.append(("Test 1, same people", pct(t1[0]["accuracy_mean"]), ""))
    stats(cards)

    figs = d / "figures"
    if r["test"] == "loso":
        st.html('<div class="sec" style="margin-top:12px">Per round</div>')
        rows = [{"Test signer": x["held_out"], "Test clips": x["n"], "Accuracy": pct(x["accuracy"]),
                 "Macro precision": pct(x["macro_precision"]), "Macro recall": pct(x["macro_recall"]),
                 "Macro F1": pct(x["macro_f1"])} for x in res["rounds"]]
        rows.append({"Test signer": "Mean ± SD", "Test clips": sum(x["n"] for x in res["rounds"]),
                     "Accuracy": ms("accuracy"), "Macro precision": ms("macro_precision"),
                     "Macro recall": ms("macro_recall"), "Macro F1": ms("macro_f1")})
        table(pd.DataFrame(rows), numeric=["Test clips", "Accuracy", "Macro precision", "Macro recall", "Macro F1"],
              total_row=True)
    st.html('<div style="height:8px"></div>')
    names = (["Per round chart"] if r["test"] == "loso" else []) + ["Per sign", "Confusion matrix", "Training curves",
                                                                   "Left-hand check", "Details"]
    tabs = dict(zip(names, st.tabs(names)))
    if "Per round chart" in tabs:
        with tabs["Per round chart"]:
            if (figs / "round_accuracy.png").exists():
                st.image(str(figs / "round_accuracy.png"), width=420)
    with tabs["Per sign"]:
        if (figs / "per_sign_recall.png").exists():
            st.image(str(figs / "per_sign_recall.png"), width=820)
    with tabs["Confusion matrix"]:
        note("Row = shown, column = guessed.")
        c1, c2 = st.columns([2, 1])
        which = ["summed"] + ([x["name"] for x in res["rounds"]] if len(res["rounds"]) > 1 else [])
        w = c1.selectbox("Rounds", which, key="ts_cm_round",
                         format_func=lambda x: "All rounds, summed" if x == "summed" else f"Test signer {x}")
        kind = c2.radio("Show", ["counts", "% of each sign"], horizontal=True, key="ts_cm_kind")
        name = f"confusion_{w}" + ("_normalized" if kind != "counts" else "")
        if (figs / f"{name}.png").exists():
            st.image(str(figs / f"{name}.png"), width=min(620, 260 + 30 * len(res["labels"])))
    with tabs["Training curves"]:
        if (figs / "curves.png").exists():
            st.image(str(figs / "curves.png"), width=820)
    with tabs["Left-hand check"]:
        handedness(res, figs)
    with tabs["Details"]:
        counts = []
        for rd in conf["rounds"]:
            counts.append({"Round": "all" if rd["name"] == "all" else f"test {rd['name']}",
                           "Training clips": len(rd["train"]), "Validation clips": len(rd["val"]),
                           "Test clips": len(rd["test"])})
        table(pd.DataFrame(counts), numeric=["Training clips", "Validation clips", "Test clips"])
        o = conf["opts"]
        note(f"Run {esc(conf['run_id'])} · max {o['epochs']} epochs · batch {o['batch_size']} · "
             f"{o['aug_factor']} extra copies · {res['ms_per_sample']:.1f} ms per sign<br>{esc(d)}")
        if conf.get("missing_signs"):
            note("Signs without clips: " + ", ".join(conf["missing_signs"]))


def handedness(res: dict, figs: Path) -> None:
    note("Mirrored clips = a pretend left-handed signer.")
    h = res["handedness"]

    def ms(a, b):
        m, sd = h[a][b]["accuracy"]["mean"], h[a][b]["accuracy"]["sd"]
        return pct(m) + (f" ± {sd * 100:.1f}" if sd is not None else "")
    table(pd.DataFrame([{"Clips": "Right-handed", "No fix": ms("recorded", "none"),
                         "Auto mirror": ms("recorded", "auto")},
                        {"Clips": "Left-handed, mirrored", "No fix": ms("mirrored", "none"),
                         "Auto mirror": ms("mirrored", "auto")}]),
          numeric=["No fix", "Auto mirror"])
    note(f"Auto found the signing hand in {pct(h['auto_correct']['recorded'])} of clips.")
    if (figs / "handedness.png").exists():
        st.image(str(figs / "handedness.png"), width=520)


def compare(runs: list[dict]) -> None:
    rows = []
    for r in runs:
        acc = (pct(r["accuracy_mean"]) + (f" ± {r['accuracy_sd'] * 100:.1f}" if r["accuracy_sd"] is not None else "")
               if r["accuracy_mean"] is not None else "—")
        f1 = (pct(r["macro_f1_mean"]) + (f" ± {r['macro_f1_sd'] * 100:.1f}" if r["macro_f1_sd"] is not None else "")
              if r["macro_f1_mean"] is not None else "—")
        rows.append({"Test": E.SHORT[r["test"]], "Category": r["category"], "Features": f"v{r['feature_version']}",
                     "Left-hand": "on" if r["canonical"] else "off", "Seed": r["seed"],
                     "Accuracy": acc, "Macro F1": f1, "Left-handed": pct(r["mirrored_auto_mean"]),
                     "Result": ("PASS" if r["pass"] else "FAIL") if r["complete"] else r["state"],
                     "Run": r["created"].replace("T", " ")[:16]})
    classes = {(i, "Result"): "ok" if x["Result"] == "PASS" else "err" if x["Result"] == "FAIL" else "warn"
               for i, x in enumerate(rows)}
    table(pd.DataFrame(rows), numeric=["Seed", "Accuracy", "Macro F1", "Left-handed"], classes=classes)
