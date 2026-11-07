"""Step 6: publication-ready results of one category in work/paper/<category>/ (each category keeps its own).

  figures/   300 dpi PNG + vector PDF, one style (trainer/figures.py), no titles (captions go in the paper)
  tables/    CSV + tables.md (Markdown, ready to paste into Word). First the paper's tables of this category
             (trainer/paper_tables.py): table6_random, table7_loso, then their details (per sign and per fold with
             Mean rows, 95 % CI, errors with the signer they came from) and confusion matrices in counts.
  captions.md          a suggested caption per figure
  results_summary.md   every key number, each with the run ID it came from
  paper.json           which runs were used
  paper.zip            all of the above

Which runs: the Test 1 and Test 2 runs chosen on the page (test1= / test2=), else: Test 2 = the run the final model was
trained from if that is a finished Test 2, else the newest finished Test 2 (canonical hand on preferred). Test 1 = the
newest finished Test 1.
v1 vs v2 = the newest finished Test 2 of each version with the same seed and hand setting as the Test 2 run.
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset as ds
from . import experiments as E
from . import figures as G
from . import live
from . import paper_tables as T
from . import settings as cfg

# The benchmark (editable). Source: Oropesa, Felicen & de Guzman (2024), SENYAS: Filipino Sign Language recognition
# (testing accuracy 92 %, live accuracy 86 %).
SENYAS = {"test_accuracy": 0.92, "live_accuracy": 0.86,
          "source": "Oropesa, Felicen & de Guzman (2024), SENYAS: 92 % testing accuracy, 86 % live accuracy"}
DPI = 300
TARGET_MS = 1000


def _pct(v, d: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{d}f} %"


def _ms(m: dict, d: int = 1) -> str:
    if m.get("mean") is None:
        return "—"
    return _pct(m["mean"], d) + (f" ± {m['sd'] * 100:.{d}f}" if m.get("sd") is not None else "")


def pick_runs(paths: cfg.Paths, category: str, test1: str | None = None,
              test2: str | None = None) -> dict[str, dict | None]:
    runs = [r for r in E.list_runs(paths, category=category) if r["complete"]]
    loso = [r for r in runs if r["test"] == "loso"]
    rand = [r for r in runs if r["test"] == "random"]
    for rid, pool in ((test1, rand), (test2, loso)):
        if rid and not any(r["run_id"] == rid for r in pool):
            raise ds.PlainError(f"Run {rid} is not a finished test run of {category}.", "Choose another run.")
    t2 = next((r for r in loso if r["run_id"] == test2), None) if test2 else None
    conf = paths.model_dir(category) / "config.json"
    if t2 is None and conf.exists():
        src = json.loads(conf.read_text(encoding="utf-8")).get("from_run")
        t2 = next((r for r in loso if r["run_id"] == src), None)
    t2 = t2 or next((r for r in loso if r["canonical"]), None) or (loso[0] if loso else None)
    t1 = next((r for r in rand if r["run_id"] == test1), None) if test1 else None
    t1 = t1 or (rand[0] if rand else None)
    return {"test1": t1, "test2": t2}


def out_dir(paths: cfg.Paths, category: str) -> Path:
    return paths.paper / category


# the paper's own tables come first in tables.md
PAPER_TABLES = ("table6_random", "table7_loso", "table7_loso_per_fold", "table6_random_per_sign",
                "table7_loso_per_sign", "errors_random", "errors_loso")


def _table_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def _errors_line(err: pd.DataFrame, who: str) -> str:
    if err.empty:
        return "- Errors: none."
    return (f"- Errors: {int(err['Count'].sum())} (" + "; ".join(
        f"{r['True sign']} → {r['Predicted as']} ×{r['Count']}, {who} {r['Signer of the clips']}"
        for _, r in err.iterrows()) + ").")


def _folds_line(rounds: list[dict]) -> str:
    bad = [(x["held_out"], T.fold_check(x)[1]) for x in rounds if not T.fold_check(x)[0]]
    return "yes." if not bad else "no (" + "; ".join(f"test {g}: {', '.join(w)}" for g, w in bad) + ")."


def generate(ctx, category: str, test1: str | None = None, test2: str | None = None) -> dict:
    s = cfg.load_settings()
    paths = cfg.Paths(s)
    data_dir = cfg.data_dir(s)
    out = out_dir(paths, category)
    if out.exists():
        shutil.rmtree(out)
    figs, tabs = out / "figures", out / "tables"
    figs.mkdir(parents=True)
    tabs.mkdir(parents=True)
    kw = {"dpi": DPI, "pdf": True}
    made: list[str] = []
    missing: list[str] = []
    captions: dict[str, str] = {}
    tables: dict[str, pd.DataFrame] = {}
    summary: list[str] = [f"# HUDYAT results: {category}", "",
                          f"Generated {datetime.now():%Y-%m-%d %H:%M} by the HUDYAT Trainer. Every number names the run "
                          "it comes from (work/experiments/<test>/<category>/<run ID>/).", ""]
    runs = pick_runs(paths, category, test1, test2)
    ctx.progress(0.05, "Dataset figures")

    # --- 1. dataset ---------------------------------------------------------------------------------
    signs = [x for x in ds.load_signs(data_dir) if x.category == category]
    sign_name = {x.sign_id: x.sign_label for x in signs}
    kept = ds.kept_rows(data_dir)
    kc = kept[kept["category"] == category]
    if not kc.empty:
        from .splits import signer_order
        signers = signer_order(g for g in kc["signer"] if g in cfg.SIGNERS) or sorted(set(kc["signer"]))
        counts = kc.groupby(["sign_id", "signer"]).size()
        ids = [x.sign_id for x in signs]
        names = [x.sign_label for x in signs]
        G.stacked_bars(names, {g: [int(counts.get((i, g), 0)) for i in ids] for g in signers},
                       figs / "dataset_clips", "Kept clips (count)", "Sign", **kw)
        made.append("dataset_clips")
        captions["dataset_clips"] = (f"Kept clips per sign of the {category} category, stacked by signer "
                                     f"({', '.join(signers)}).")
        rows = [{"Sign": n, **{g: int(counts.get((i, g), 0)) for g in signers},
                 "Total": int(sum(counts.get((i, g), 0) for g in signers))} for i, n in zip(ids, names)]
        tables["dataset_summary"] = pd.DataFrame(rows)
        hands = ds.clip_hands(data_dir)
        summary += ["## Dataset", "", f"- Category: {category}; {len(signs)} signs; {len(kc)} kept clips from "
                    f"{len(signers)} signers ({', '.join(f'{g} {hands.get(f, 'right')}' for g, f in ((g, kc[kc['signer'] == g]['file_path'].iloc[0]) for g in signers))}).",
                    f"- Model outputs: {len(signs)} classes, one per sign (no none class).", ""]
        idx = ds.read_index(paths)
        det = idx[(idx["category"] == category) & (idx["status"] != ds.STATUS_FAILED)]["hand_detect_rate"].dropna()
        if len(det):
            G.histogram(det.values, figs / "detection_rate", "Hand detection rate inside the sign window (%)",
                        line=float(s.get("min_hand_detect_rate", 0.8)), line_label="Inclusion threshold",
                        ylabel="Clips (count)", percent_x=True, **kw)
            made.append("detection_rate")
            captions["detection_rate"] = ("Distribution of the hand detection rate inside the sign window per clip; "
                                          "clips left of the dashed line are excluded.")
        else:
            missing.append("Hand detection rates: extract the landmarks first (Preprocess page).")
    else:
        missing.append(f"Dataset figures: no kept clips of {category}.")

    # --- 2. Test 1 ------------------------------------------------------------------------------------
    ctx.progress(0.25, "Test 1 figures")
    t1 = runs["test1"]
    r1 = E.load_results(Path(t1["dir"])) if t1 else None
    if r1:
        G.curves(r1["histories"], figs / "test1_curves", **kw)
        G.confusion(r1["pooled"]["confusion"], [sign_name.get(x, x) for x in r1["labels"]], figs / "test1_confusion",
                    both=True, **kw)
        G.per_sign_recall(r1["pooled"]["per_sign_recall"], figs / "test1_recall", **kw)
        made += ["test1_curves", "test1_confusion", "test1_recall"]
        captions.update({
            "test1_curves": "Test 1 (random 70/15/15 split): training and validation accuracy and loss per epoch.",
            "test1_confusion": "Test 1: confusion matrix (count of test clips, with the % of each true sign). Test "
                               "clips come from the same signers as the training clips.",
            "test1_recall": "Test 1: recall per sign; the dashed line marks the 70 % target."})
        sm = r1["summary"]
        tables["table6_random"] = T.table6(r1, category, sign_name, SENYAS["test_accuracy"])
        tables["table6_random_per_sign"] = T.per_sign_with_mean(r1, sign_name)
        tables["errors_random"] = T.errors(T.read_predictions(Path(t1["dir"]), r1["rounds"]), sign_name)
        T.confusion_counts_csv(tabs / "confusion_random_counts.csv", r1["pooled"]["confusion"], r1["labels"], sign_name)
        summary += ["## Test 1: random split (same signers)", "",
                    f"- Accuracy {_pct(sm['accuracy']['mean'], 2)} (95 % CI {T.ci_text(r1['pooled']['confusion'])} %, "
                    f"exact Clopper–Pearson, {r1['pooled']['n']} test clips), macro precision "
                    f"{_pct(sm['macro_precision']['mean'], 2)}, macro recall {_pct(sm['macro_recall']['mean'], 2)}, "
                    f"macro F1 {_pct(sm['macro_f1']['mean'], 2)} (run {t1['run_id']}, feature v{t1['feature_version']}).",
                    f"- Lowest recall: {T.lowest(r1['pooled']['per_sign_recall'], sign_name)} %.",
                    _errors_line(tables["errors_random"], "signer"),
                    "- Limitation: the test clips come from the same people the model trained on.", ""]
    else:
        missing.append("Test 1 figures: run Test 1 (Test the model page).")

    # --- 3. Test 2 ------------------------------------------------------------------------------------
    ctx.progress(0.45, "Test 2 figures")
    t2 = runs["test2"]
    r2 = E.load_results(Path(t2["dir"])) if t2 else None
    if r2:
        sm = r2["summary"]
        G.round_accuracy([x["held_out"] for x in r2["rounds"]], [x["accuracy"] for x in r2["rounds"]],
                         sm["accuracy"]["mean"], sm["accuracy"]["sd"], figs / "test2_round_accuracy", **kw)
        G.per_sign_recall(r2["pooled"]["per_sign_recall"], figs / "test2_recall",
                          {x["held_out"]: x["per_sign_recall"] for x in r2["rounds"]}, **kw)
        shown = [sign_name.get(x, x) for x in r2["labels"]]
        G.confusion(r2["pooled"]["confusion"], shown, figs / "test2_confusion", both=True, **kw)
        G.curves(r2["histories"], figs / "test2_curves", **kw)
        made += ["test2_round_accuracy", "test2_recall", "test2_confusion", "test2_curves"]
        for x in r2["rounds"]:
            g = x["held_out"]
            G.confusion(x["confusion"], shown, figs / f"test2_confusion_{g}", both=True, **kw)
            made.append(f"test2_confusion_{g}")
            captions[f"test2_confusion_{g}"] = (f"Test 2, fold with {g} as the test signer: confusion matrix (count of "
                                                "test clips, with the % of each true sign).")
            T.confusion_counts_csv(tabs / f"confusion_loso_test{g}_counts.csv", x["confusion"], r2["labels"], sign_name)
        T.confusion_counts_csv(tabs / "confusion_loso_counts.csv", r2["pooled"]["confusion"], r2["labels"], sign_name)
        tables["table7_loso"] = T.table7(r2, category, sign_name)
        tables["table7_loso_per_fold"] = T.table7_folds(r2, sign_name)
        tables["table7_loso_per_sign"] = T.per_sign_with_mean(r2, sign_name)
        tables["errors_loso"] = T.errors(T.read_predictions(Path(t2["dir"]), r2["rounds"]), sign_name)
        n = len(r2["rounds"])
        captions.update({
            "test2_round_accuracy": f"Test 2 (leave-one-signer-out, {n} rounds): test accuracy per held-out signer; "
                                    "the line and band show the mean ± SD.",
            "test2_recall": "Test 2: recall per sign over all rounds (bars) and per round (dots); the dashed line marks "
                            "the 70 % target.",
            "test2_confusion": "Test 2: confusion matrix summed over all rounds (count of test clips, with the % of "
                               "each true sign).",
            "test2_curves": "Test 2: training and validation accuracy and loss per epoch; thin lines are single rounds, "
                            "the band is the mean ± SD across rounds."})
        tables["per_round"] = pd.DataFrame(
            [{"Test signer": x["held_out"], "Test clips": x["n"], "Accuracy": _pct(x["accuracy"]),
              "Macro precision": _pct(x["macro_precision"]), "Macro recall": _pct(x["macro_recall"]),
              "Macro F1": _pct(x["macro_f1"])} for x in r2["rounds"]]
            + [{"Test signer": "Mean ± SD", "Test clips": sum(x["n"] for x in r2["rounds"]),
                "Accuracy": _ms(sm["accuracy"]), "Macro precision": _ms(sm["macro_precision"]),
                "Macro recall": _ms(sm["macro_recall"]), "Macro F1": _ms(sm["macro_f1"])}])
        srows = [{"Test": "Test 2: leave-one-signer-out", "Run": t2["run_id"], "Features": f"v{t2['feature_version']}",
                  "Accuracy": _ms(sm["accuracy"]), "Macro precision": _ms(sm["macro_precision"]),
                  "Macro recall": _ms(sm["macro_recall"]), "Macro F1": _ms(sm["macro_f1"]),
                  "Result": "PASS" if r2["pass"] else "FAIL"}]
        if r1:
            s1 = r1["summary"]
            srows.append({"Test": "Test 1: random split", "Run": t1["run_id"], "Features": f"v{t1['feature_version']}",
                          "Accuracy": _ms(s1["accuracy"]), "Macro precision": _ms(s1["macro_precision"]),
                          "Macro recall": _ms(s1["macro_recall"]), "Macro F1": _ms(s1["macro_f1"]),
                          "Result": "PASS" if r1["pass"] else "FAIL"})
        tables["summary_mean_sd"] = pd.DataFrame(srows)
        tables["per_sign"] = pd.DataFrame([{"Sign": p["sign"], "Precision": _pct(p["precision"]),
                                            "Recall": _pct(p["recall"]), "F1": _pct(p["f1"]), "Test clips": p["support"]}
                                           for p in r2["per_sign"]])
        worst = min((v, k) for k, v in r2["pooled"]["per_sign_recall"].items() if v is not None)
        summary += ["## Test 2: leave-one-signer-out (new people, the headline)", "",
                    f"- {n} rounds (signers {', '.join(x['held_out'] for x in r2['rounds'])}); run {t2['run_id']}, "
                    f"feature v{t2['feature_version']}, canonical hand {'on' if t2['canonical'] else 'off'}, "
                    f"mirror augmentation {'on' if t2['mirror_aug'] else 'off'}, seed {t2['seed']}.",
                    f"- Accuracy {_ms(sm['accuracy'])}; macro precision {_ms(sm['macro_precision'])}; macro recall "
                    f"{_ms(sm['macro_recall'])}; macro F1 {_ms(sm['macro_f1'])} (mean ± SD across rounds).",
                    f"- Lowest per-sign recall: {worst[1]} {_pct(worst[0])}.",
                    "- Every fold passes the benchmark by itself: " + _folds_line(r2["rounds"]),
                    _errors_line(tables["errors_loso"], "test signer"),
                    f"- Targets (≥ 85 % accuracy and macro P/R/F1, every sign ≥ 70 %): "
                    f"{'PASS' if r2['pass'] else 'FAIL'}" + (f" ({'; '.join(r2['reasons'])})" if r2["reasons"] else "")
                    + ".", f"- Inference: {r2['ms_per_sample']:.1f} ms per sample on the training laptop's CPU.", ""]
        # --- 4. handedness
        h = r2["handedness"]
        G.handedness(h, figs / "handedness", **kw)
        made.append("handedness")
        captions["handedness"] = ("Left-hand check (Test 2): accuracy on the test clips as recorded and mirrored "
                                  "(simulated left-handed signers), without and with the canonical mirror (hand "
                                  "detected automatically). Mirrored clips only simulate left-handed signers.")
        tables["handedness"] = pd.DataFrame(
            [{"Test clips": "As recorded", "Without canonical mirror": _ms(h["recorded"]["none"]["accuracy"]),
              "With canonical mirror (Auto)": _ms(h["recorded"]["auto"]["accuracy"])},
             {"Test clips": "Mirrored (simulated left-handed)",
              "Without canonical mirror": _ms(h["mirrored"]["none"]["accuracy"]),
              "With canonical mirror (Auto)": _ms(h["mirrored"]["auto"]["accuracy"])}])
        summary += ["## Handedness (simulated with mirrored test clips)", "",
                    f"- As recorded: {_ms(h['recorded']['none']['accuracy'])} without, "
                    f"{_ms(h['recorded']['auto']['accuracy'])} with the canonical mirror (Auto) (run {t2['run_id']}).",
                    f"- Mirrored: {_ms(h['mirrored']['none']['accuracy'])} without, "
                    f"{_ms(h['mirrored']['auto']['accuracy'])} with the canonical mirror (Auto).",
                    f"- Auto found the signing hand for {_pct(h['auto_correct']['recorded'])} (recorded) and "
                    f"{_pct(h['auto_correct']['mirrored'])} (mirrored) of the test clips.",
                    "- Mirrored clips only simulate left-handed signers; real left-handed testers are still needed.", ""]
    else:
        missing.append("Test 2 figures and tables: run Test 2 (Test the model page).")
        missing.append("Handedness figure: run Test 2 (Test the model page).")

    # --- 6. live test -----------------------------------------------------------------------------------
    ctx.progress(0.75, "Live test")
    thr = float(s.get("confidence_threshold", 0.85))
    att = live.read_attempts(paths.live_tests / f"{category}.csv")
    live_acc, live_setup = None, None
    if not att.empty:
        # Each setup (recognizer + capture) is reported on its own; the figures and the benchmark use the first
        # setup in live.SETUPS that has attempts (the app model with the fixed capture, when it was tested).
        att = live.score_attempts(att, thr)
        groups = {k: g for k, g in att.groupby(["recognizer", "capture"], sort=False)}
        order = [k for k in live.SETUPS if k in groups] + sorted(k for k in groups if k not in live.SETUPS)
        live_setup = order[0]
        main, name = groups[live_setup], live.setup_label(*live_setup)
        rule = live.setup_rule(live_setup[0], thr)
        g = main.groupby("target")["correct"].mean()
        G.bars(list(g.index), list(g.values), figs / "live_accuracy", "Live accuracy (%)", "Target sign", line=0.85,
               line_label="85 % target", **kw)
        G.histogram(main["response_time_ms"].dropna().values, figs / "response_time", "Response time (ms)",
                    line=TARGET_MS, line_label="1-second target", ylabel="Attempts (count)", **kw)
        hand = main.assign(h=main["tester_hand"].replace("", "not recorded")).groupby("h")["correct"].mean()
        G.bars([f"{k.capitalize()}-handed" if k in ("right", "left") else k for k in hand.index], list(hand.values),
               figs / "tester_hand", "Live accuracy (%)", "Tester", **kw)
        made += ["live_accuracy", "response_time", "tester_hand"]
        captions.update({
            "live_accuracy": f"Live test ({name}): accuracy per target sign (an attempt is correct when {rule}); "
                             "dashed line = 85 % target.",
            "response_time": f"Live test ({name}): response time from the end of the sign window to the prediction; "
                             "dashed line = 1 second.",
            "tester_hand": f"Live test ({name}): accuracy of right- and left-handed testers."})
        if len(order) > 1:
            G.bars([live.setup_label(*k).replace(" · ", "\n") for k in order],
                   [groups[k]["correct"].mean() for k in order], figs / "live_setups", "Live accuracy (%)",
                   highlight=[k == live_setup for k in order], **kw)
            made.append("live_setups")
            captions["live_setups"] = ("Live test: accuracy of each setup (same testers and signs). App model: correct "
                                       f"when {live.setup_rule('app', thr)}. Trainer model: correct when "
                                       f"{live.setup_rule('trainer', thr)}.")
        live_acc = float(main["correct"].mean())
        rows = []
        for k in order:
            x = groups[k]
            for t, y in [*x.groupby("target"), ("All", x)]:
                fh = y["time_from_hand_ms"].mean()
                rows.append({"Setup": live.setup_label(*k), "Sign": t, "Attempts": len(y),
                             "Accuracy": _pct(y["correct"].mean()),
                             "Mean response after the sign (ms)": f"{y['response_time_ms'].mean():.0f}",
                             "Mean response after the hand appears (ms)": "—" if pd.isna(fh) else f"{fh:.0f}"})
        tables["live_summary"] = pd.DataFrame(rows)
        summary += ["## Live test", ""]
        for k in order:
            x = groups[k]
            fh = x["time_from_hand_ms"].mean()
            summary.append(
                f"- {live.setup_label(*k)}{' (reported)' if k == live_setup else ''}: Live accuracy "
                f"{_pct(x['correct'].mean())} over {len(x)} attempts by {x['tester'].nunique()} tester(s), correct when "
                f"{live.setup_rule(k[0], thr)}; mean response {x['response_time_ms'].mean():.0f} ms after the sign "
                f"ends, {_pct((x['response_time_ms'] <= TARGET_MS).mean(), 0)} within 1 second"
                + ("" if pd.isna(fh) else f"; {fh:.0f} ms after the hand appears") + " (models: "
                f"{', '.join(sorted(set(x['model_run_id']) - {''})) or 'not recorded'}).")
        summary += ["- By tester hand (" + name + "): "
                    + ", ".join(f"{k or 'not recorded'} {_pct(v)}" for k, v in hand.items()) + ".", ""]
    else:
        missing.append("Live test figures: run the live test in Test mode (Live test page).")

    # --- 7. benchmark -------------------------------------------------------------------------------------
    if r2:
        names = ["HUDYAT\nTest 2", f"SENYAS\ntesting"]
        vals = [r2["summary"]["accuracy"]["mean"], SENYAS["test_accuracy"]]
        errs = [r2["summary"]["accuracy"]["sd"], None]
        hl = [True, False]
        if live_acc is not None:
            names += ["HUDYAT\nlive", "SENYAS\nlive"]
            vals += [live_acc, SENYAS["live_accuracy"]]
            errs += [None, None]
            hl += [True, False]
        G.bars(names, vals, figs / "benchmark", "Accuracy (%)", errors=errs, highlight=hl, **kw)
        made.append("benchmark")
        captions["benchmark"] = (f"HUDYAT compared with the SENYAS study ({SENYAS['source']}). HUDYAT's testing number "
                                 "is the leave-one-signer-out mean ± SD (new people)"
                                 + (f"; its live number is the live test with the {live.setup_label(*live_setup)}"
                                    if live_setup else "") + "; the studies' test protocols differ.")
        summary += ["## Benchmark", "", f"- SENYAS: {SENYAS['source']}.",
                    f"- HUDYAT Test 2 accuracy {_ms(r2['summary']['accuracy'])}"
                    + (f"; live accuracy {_pct(live_acc)} ({live.setup_label(*live_setup)})." if live_acc is not None
                       else "; no live test yet."), ""]

    # --- files -----------------------------------------------------------------------------------------
    ctx.progress(0.9, "Tables and summary")
    md = [f"# Tables: {category}", ""]
    for name in [t for t in PAPER_TABLES if t in tables] + [t for t in tables if t not in PAPER_TABLES]:
        df = tables[name]
        df.to_csv(tabs / f"{name}.csv", index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
        md += [f"## {name.replace('_', ' ').capitalize()}", "", _table_md(df), ""]
    (tabs / "tables.md").write_text("\n".join(md), encoding="utf-8")
    cap = ["# Suggested figure captions", ""]
    for i, name in enumerate(made, 1):
        cap += [f"**Figure {i}** (`{name}`). {captions.get(name, '')}", ""]
    (out / "captions.md").write_text("\n".join(cap), encoding="utf-8")
    if missing:
        summary += ["## Not made yet", ""] + [f"- {m}" for m in missing] + [""]
    summary += ["## Still needs real data", "", "- Signers S4–S7 (more Test 2 rounds).",
                "- Real left-handed testers (the mirrored clips only simulate them).", ""]
    (out / "results_summary.md").write_text("\n".join(summary), encoding="utf-8")
    meta = {"category": category, "created": datetime.now().isoformat(timespec="seconds"),
            "runs": {k: (v["run_id"] if v else None) for k, v in runs.items()}, "figures": made, "missing": missing,
            "senyas": SENYAS}
    (out / "paper.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    zp = out.parent / f"{category}_paper.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    final_zip = out / "paper.zip"
    shutil.move(str(zp), final_zip)
    msg = f"Made {ds.plural(len(made), 'figure')} and {ds.plural(len(tables), 'table')} for {category}."
    if missing:
        msg += f" {ds.plural(len(missing), 'item')} need more data; see the page."
    ctx.log(msg)
    return {"message": msg, "figures": made, "missing": missing, "zip": str(final_zip)}


def run(ctx, category: str, test1: str | None = None, test2: str | None = None) -> dict:
    return generate(ctx, category, test1, test2)
