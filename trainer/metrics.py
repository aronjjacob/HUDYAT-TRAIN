"""Metrics of one test set, PASS / FAIL against the HUDYAT targets."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PASS_METRIC = 0.85        # accuracy and macro precision / recall / F1
PASS_PER_SIGN = 0.70      # every per-sign recall
KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
NAMES = {"accuracy": "Accuracy", "macro_precision": "Macro precision", "macro_recall": "Macro recall",
         "macro_f1": "Macro F1"}


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Accuracy, macro P / R / F1 (over the signs that occur in this test set), recall and support per sign,
    and the confusion matrix (rows = true, columns = predicted)."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    idx = list(range(len(labels)))
    if len(y_true) == 0:
        return {"n": 0, **{k: None for k in KEYS}, "per_sign_recall": {lab: None for lab in labels},
                "support": {lab: 0 for lab in labels}, "confusion": np.zeros((len(labels),) * 2, int).tolist()}
    present = sorted(set(y_true.tolist()))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    _, rec, _, sup = precision_recall_fscore_support(y_true, y_pred, labels=idx, average=None, zero_division=0)
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
            "per_sign_recall": {labels[i]: (float(rec[i]) if sup[i] else None) for i in idx},
            "support": {labels[i]: int(sup[i]) for i in idx},
            "confusion": confusion_matrix(y_true, y_pred, labels=idx).tolist()}


def per_sign_table(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[dict]:
    """Precision, recall, F1 and support per sign (for the paper's per-sign table)."""
    from sklearn.metrics import precision_recall_fscore_support
    idx = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(np.asarray(y_true, int), np.asarray(y_pred, int), labels=idx,
                                                 average=None, zero_division=0)
    return [{"sign": labels[i], "precision": float(p[i]), "recall": float(r[i]) if s[i] else None,
             "f1": float(f[i]), "support": int(s[i])} for i in idx]


def verdict(avg: dict, pooled: dict) -> tuple[bool, list[str]]:
    """PASS = every averaged metric >= 85 % and every sign's (pooled) recall >= 70 %, with test clips for every sign."""
    reasons = []
    for k in KEYS:
        if avg.get(k) is None or avg[k] < PASS_METRIC:
            reasons.append(f"{NAMES[k]} {avg[k]:.1%} < {PASS_METRIC:.0%}" if avg.get(k) is not None
                           else f"{NAMES[k]}: no result")
    for lab, r in pooled["per_sign_recall"].items():
        if r is not None and r < PASS_PER_SIGN:
            reasons.append(f"Recall of '{lab}' {r:.1%} < {PASS_PER_SIGN:.0%}")
    missing = [lab for lab, n in pooled["support"].items() if n == 0]
    if missing:
        reasons.append(f"No test clips for: {', '.join(missing)}")
    return not reasons, reasons


def row_normalized(cm) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)


def write_confusion_csv(path: Path, cm, labels: list[str], normalized: bool = False) -> None:
    m = row_normalized(cm) if normalized else np.asarray(cm)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + labels)
        for lab, row in zip(labels, m):
            w.writerow([lab] + [f"{v:.4f}" if normalized else int(v) for v in row])
