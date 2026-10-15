"""Step 6: evaluate on the test signers S6 and S7, once, without augmentation."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime

import numpy as np

from . import dataset as ds
from . import settings as cfg
from .model import fast_predict, load_samples
from .split import load_split
from .train import load_labels, model_exists

PASS_METRIC = 0.85        # accuracy and macro precision / recall / F1
PASS_PER_SIGN = 0.70      # every per-sign recall
SUMMARY_FIELDS = ["category", "date", "n_test", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                  "min_sign_recall", "worst_sign", "ms_per_sample", "result"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    idx = list(range(len(labels)))
    present = sorted(set(y_true.tolist()))          # macro over classes that occur in this test set
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    _, rec, _, sup = precision_recall_fscore_support(y_true, y_pred, labels=idx, average=None, zero_division=0)
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
            "per_sign_recall": {labels[i]: (float(rec[i]) if sup[i] else None) for i in idx},
            "support": {labels[i]: int(sup[i]) for i in idx},
            "confusion": confusion_matrix(y_true, y_pred, labels=idx).tolist()}


def verdict(avg: dict, pooled: dict) -> tuple[bool, list[str]]:
    reasons = []
    for k, name in (("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
                    ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")):
        if avg[k] < PASS_METRIC:
            reasons.append(f"{name} {avg[k]:.1%} < {PASS_METRIC:.0%}")
    for lab, r in pooled["per_sign_recall"].items():
        if r is not None and r < PASS_PER_SIGN:
            reasons.append(f"Recall of '{lab}' {r:.1%} < {PASS_PER_SIGN:.0%}")
    missing = [lab for lab, n in pooled["support"].items() if n == 0]
    if missing:
        reasons.append(f"No test clips for: {', '.join(missing)}")
    return not reasons, reasons


def _write_confusion_csv(path, cm, labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + labels)
        for lab, row in zip(labels, cm):
            w.writerow([lab] + list(row))


def evaluate_category(ctx, category: str) -> dict:
    import keras
    paths = cfg.Paths()
    if not model_exists(paths, category):
        raise ds.PlainError(f"No trained model for '{category}'.", "Run step 5 (Train) first.")
    sp = load_split(paths, category)
    if sp is None or not sp["test"]:
        raise ds.PlainError(f"'{category}' has no test clips (S6, S7, or the test split of an upload).",
                            "Record and extract S6 and S7, run Split again, then evaluate.")
    mdir = paths.model_dir(category)
    labels = load_labels(mdir)
    model = keras.models.load_model(mdir / "model.keras")
    X, y = load_samples(paths.work, sp["test"], labels)
    signers = np.array([it["signer"] for it in sp["test"]])
    prob = model.predict(X, verbose=0, batch_size=64)
    pred = prob.argmax(1)
    # mean single-sample inference time (as in the live test), after a warm-up call
    predict = fast_predict(model)
    k = min(len(X), 50)
    t0 = time.perf_counter()
    for i in range(k):
        predict(X[i:i + 1])
    ms = (time.perf_counter() - t0) * 1000 / k

    per_signer = {}
    for sg in sorted(set(signers.tolist())):          # S6, S7 for the recordings; one group for an upload
        m = signers == sg
        per_signer[sg if sg in cfg.SIGNERS else "test split"] = metrics(y[m], pred[m], labels)
    pooled = metrics(y, pred, labels)
    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    avg = {k2: float(np.mean([per_signer[s][k2] for s in per_signer])) for k2 in keys}
    ok, reasons = verdict(avg, pooled)

    rdir = paths.report_dir(category)
    rdir.mkdir(parents=True, exist_ok=True)
    from .charts import confusion_plot
    cm = np.array(pooled["confusion"])
    confusion_plot(cm, labels, rdir / "confusion_matrix.png",
                   f"{category}: test set ({' + '.join(per_signer)})")
    _write_confusion_csv(rdir / "confusion_matrix.csv", cm, labels)
    for sg, m in per_signer.items():
        if sg in cfg.SIGNERS:
            _write_confusion_csv(rdir / f"confusion_matrix_{sg}.csv", m["confusion"], labels)
    date = datetime.now().isoformat(timespec="seconds")
    result = {"category": category, "date": date, "labels": labels, "per_signer": per_signer, "pooled": pooled,
              "average": avg, "ms_per_sample": ms, "pass": ok, "reasons": reasons,
              "thresholds": {"metric": PASS_METRIC, "per_sign_recall": PASS_PER_SIGN}}
    (rdir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (rdir / "report.md").write_text(report_md(result), encoding="utf-8")
    _update_summary(paths, result)
    ctx.log(f"[{category}] {'PASS' if ok else 'FAIL'}: accuracy {avg['accuracy']:.1%}, "
            f"macro F1 {avg['macro_f1']:.1%}, {ms:.1f} ms/sample")
    for r in reasons:
        ctx.log(f"   {r}")
    return result


def report_md(r: dict) -> str:
    pct = lambda v: "—" if v is None else f"{v:.1%}"  # noqa: E731
    lines = [f"# Test report: {r['category']}", "",
             f"Date: {r['date']}  ", f"Test set: {', '.join(r['per_signer'])} (no augmentation)  ",
             f"Result: **{'PASS' if r['pass'] else 'FAIL'}**", "",
             f"PASS requires accuracy and macro precision / recall / F1 >= {PASS_METRIC:.0%} "
             f"(average of the test signers) and every per-sign recall >= {PASS_PER_SIGN:.0%} (S6 + S7 pooled).", "",
             "## Metrics", "", "| | n | Accuracy | Macro precision | Macro recall | Macro F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for sg, m in r["per_signer"].items():
        lines.append(f"| {sg} | {m['n']} | {pct(m['accuracy'])} | {pct(m['macro_precision'])} | "
                     f"{pct(m['macro_recall'])} | {pct(m['macro_f1'])} |")
    a = r["average"]
    lines.append(f"| **Average** | {r['pooled']['n']} | **{pct(a['accuracy'])}** | **{pct(a['macro_precision'])}** | "
                 f"**{pct(a['macro_recall'])}** | **{pct(a['macro_f1'])}** |")
    lines += ["", f"Mean inference time: {r['ms_per_sample']:.1f} ms per sample (single sample, CPU).", "",
              "## Per-sign recall (whole test set)", "", "| Sign | Test clips | Recall |", "|---|---:|---:|"]
    for lab, rec in r["pooled"]["per_sign_recall"].items():
        lines.append(f"| {lab} | {r['pooled']['support'][lab]} | {pct(rec)} |")
    if r["reasons"]:
        lines += ["", "## Why it did not pass", ""] + [f"- {x}" for x in r["reasons"]]
    lines += ["", "## Confusion matrix", "", "![Confusion matrix](confusion_matrix.png)", "",
              "CSV: `confusion_matrix.csv` (whole test set) and one per test signer when the test set has signers.",
              ""]
    return "\n".join(lines)


def _update_summary(paths: cfg.Paths, r: dict) -> None:
    rows = {}
    if paths.summary_csv.exists():
        with open(paths.summary_csv, newline="", encoding="utf-8") as f:
            rows = {row["category"]: row for row in csv.DictReader(f)}
    recalls = {k: v for k, v in r["pooled"]["per_sign_recall"].items() if v is not None}
    worst = min(recalls, key=recalls.get) if recalls else ""
    a = r["average"]
    rows[r["category"]] = {"category": r["category"], "date": r["date"], "n_test": r["pooled"]["n"],
                           "accuracy": f"{a['accuracy']:.4f}", "macro_precision": f"{a['macro_precision']:.4f}",
                           "macro_recall": f"{a['macro_recall']:.4f}", "macro_f1": f"{a['macro_f1']:.4f}",
                           "min_sign_recall": f"{recalls[worst]:.4f}" if worst else "", "worst_sign": worst,
                           "ms_per_sample": f"{r['ms_per_sample']:.2f}", "result": "PASS" if r["pass"] else "FAIL"}
    paths.reports.mkdir(parents=True, exist_ok=True)
    with open(paths.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows[c] for c in sorted(rows))


def run(ctx, categories: list[str]) -> dict:
    res = []
    for i, c in enumerate(categories):
        ctx.progress(i / len(categories), f"Evaluating {c}")
        res.append(evaluate_category(ctx, c))
    passed = sum(r["pass"] for r in res)
    return {"message": f"Evaluated {len(res)} model(s): {passed} PASS, {len(res) - passed} FAIL."}


def load_report(paths: cfg.Paths, category: str) -> dict | None:
    p = paths.report_dir(category) / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
