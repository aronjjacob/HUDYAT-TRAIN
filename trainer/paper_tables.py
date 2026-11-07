"""The paper's results tables of one category, in the layout of the paper (Table 6: random split, Table 7:
leave-one-signer-out), plus the details behind them: per-sign and per-fold rows with Mean rows, the exact 95 %
confidence interval of the accuracy, the lowest recall of each test, the benchmark checked in every fold, and every
error with the signer it came from. Numbers are percentages with two decimals, as in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M

BENCH = {"metric": f"≥ {M.PASS_METRIC * 100:.0f}", "recall": f"≥ {M.PASS_PER_SIGN * 100:.0f}"}


def clopper_pearson(k: int, n: int) -> tuple[float | None, float | None]:
    """Exact (Clopper–Pearson) 95 % confidence interval of k correct out of n."""
    from scipy.stats import beta
    if n <= 0:
        return None, None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def p2(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.2f}"


def ms2(m: dict) -> str:
    if m.get("mean") is None:
        return "—"
    return p2(m["mean"]) + (f" ± {m['sd'] * 100:.2f}" if m.get("sd") is not None else "")


def ci_text(cm) -> str:
    cm = np.asarray(cm)
    lo, hi = clopper_pearson(int(np.trace(cm)), int(cm.sum()))
    return "—" if lo is None else f"{p2(lo)}–{p2(hi)}"


def lowest(recall: dict, names: dict) -> str:
    """'92.50 (J)', or '100.00 (all signs tie)' when every sign has the same recall."""
    vals = {k: v for k, v in recall.items() if v is not None}
    if not vals:
        return "—"
    low = min(vals.values())
    signs = [names.get(k, k) for k, v in vals.items() if v == low]
    return f"{p2(low)} ({'all signs tie' if len(signs) == len(vals) else ', '.join(signs)})"


def fold_check(m: dict) -> tuple[bool, list[str]]:
    """The benchmark on one fold by itself: accuracy and macro P / R / F1 ≥ 85 %, every sign's recall ≥ 70 %."""
    return M.verdict(m, m)


def read_predictions(run_dir: Path, rounds: list[dict]) -> pd.DataFrame:
    parts = []
    for r in rounds:
        p = Path(run_dir) / f"round_{r['name']}" / "predictions.csv"
        if p.exists():
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            parts.append(df.assign(round=r["name"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["label", "prediction", "signer"])


def errors(preds: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Every wrong prediction, grouped by (true, predicted), most frequent first, with the signers of those clips
    (in Test 2 that is the test signer of the fold)."""
    wrong = preds[preds["label"] != preds["prediction"]]
    rows = []
    for (t, p), g in wrong.groupby(["label", "prediction"]):
        by = g["signer"].value_counts()
        rows.append({"True sign": names.get(t, t), "Predicted as": names.get(p, p), "Count": int(len(g)),
                     "Signer of the clips": ", ".join(f"{s}: {n}" for s, n in sorted(by.items()))})
    df = pd.DataFrame(rows, columns=["True sign", "Predicted as", "Count", "Signer of the clips"])
    return df.sort_values(["Count", "True sign"], ascending=[False, True], ignore_index=True)


def per_sign_with_mean(res: dict, names: dict) -> pd.DataFrame:
    """Precision / recall / F1 per sign, then the Mean row (= the macro averages of the table above it)."""
    ps = res["per_sign"]
    rows = [{"Sign": names.get(p["sign"], p["sign"]), "Precision (%)": p2(p["precision"]),
             "Recall (%)": p2(p["recall"]), "F1 (%)": p2(p["f1"]), "Test clips": p["support"]} for p in ps]
    pooled = res["pooled"]
    rows.append({"Sign": "Mean (macro)", "Precision (%)": p2(pooled["macro_precision"]),
                 "Recall (%)": p2(pooled["macro_recall"]), "F1 (%)": p2(pooled["macro_f1"]),
                 "Test clips": sum(p["support"] for p in ps)})
    return pd.DataFrame(rows)


def table6(res: dict, category: str, names: dict, senyas_accuracy: float) -> pd.DataFrame:
    """Table 6 row of this category (random split), the benchmark row and the SENYAS row."""
    m = res["pooled"]
    ok = res["pass"]
    cols = ["Model", "Accuracy (%)", "95% CI (%)", "Precision (%)", "Recall (%)", "F1 (%)", "Lowest recall (%)",
            "Test clips", "Benchmark"]
    return pd.DataFrame([
        [category.capitalize(), p2(m["accuracy"]), ci_text(m["confusion"]), p2(m["macro_precision"]),
         p2(m["macro_recall"]), p2(m["macro_f1"]), lowest(m["per_sign_recall"], names), m["n"],
         "PASS" if ok else "FAIL"],
        ["Benchmark", BENCH["metric"], "", BENCH["metric"], BENCH["metric"], BENCH["metric"], BENCH["recall"], "", ""],
        ["SENYAS (Oropesa et al., 2024)", p2(senyas_accuracy), "", "—", "—", "—", "—", "", ""]], columns=cols)


def table7(res: dict, category: str, names: dict) -> pd.DataFrame:
    """Table 7 row of this category (leave-one-signer-out): mean ± sample SD over the folds, accuracy per test
    signer, lowest recall pooled over the folds, and whether every fold passes the benchmark by itself."""
    sm, rounds = res["summary"], res["rounds"]
    every = all(fold_check(r)[0] for r in rounds)
    cols = ["Model", "Accuracy (%)", "F1 (%)", f"Test {' / '.join(r['held_out'] for r in rounds)} (%)",
            "Lowest recall (%)", "Every fold passes", "Benchmark"]
    return pd.DataFrame([
        [category.capitalize(), ms2(sm["accuracy"]), ms2(sm["macro_f1"]), " / ".join(p2(r["accuracy"]) for r in rounds),
         lowest(res["pooled"]["per_sign_recall"], names), "yes" if every else "no", "PASS" if res["pass"] else "FAIL"],
        ["Benchmark", BENCH["metric"], BENCH["metric"], BENCH["metric"], BENCH["recall"], "", ""]], columns=cols)


def table7_folds(res: dict, names: dict) -> pd.DataFrame:
    """One row per fold (each checked against the benchmark by itself), the Mean ± SD row and the pooled row."""
    rows = []
    for r in res["rounds"]:
        ok, why = fold_check(r)
        rows.append({"Test signer": r["held_out"], "Test clips": r["n"], "Accuracy (%)": p2(r["accuracy"]),
                     "95% CI (%)": ci_text(r["confusion"]), "Precision (%)": p2(r["macro_precision"]),
                     "Recall (%)": p2(r["macro_recall"]), "F1 (%)": p2(r["macro_f1"]),
                     "Lowest recall (%)": lowest(r["per_sign_recall"], names),
                     "Benchmark": "PASS" if ok else "FAIL: " + "; ".join(why)})
    sm, pooled = res["summary"], res["pooled"]
    rows.append({"Test signer": "Mean ± SD", "Test clips": sum(r["n"] for r in res["rounds"]),
                 "Accuracy (%)": ms2(sm["accuracy"]), "95% CI (%)": "", "Precision (%)": ms2(sm["macro_precision"]),
                 "Recall (%)": ms2(sm["macro_recall"]), "F1 (%)": ms2(sm["macro_f1"]), "Lowest recall (%)": "",
                 "Benchmark": "PASS" if res["pass"] else "FAIL"})
    rows.append({"Test signer": "All folds pooled", "Test clips": pooled["n"], "Accuracy (%)": p2(pooled["accuracy"]),
                 "95% CI (%)": ci_text(pooled["confusion"]), "Precision (%)": p2(pooled["macro_precision"]),
                 "Recall (%)": p2(pooled["macro_recall"]), "F1 (%)": p2(pooled["macro_f1"]),
                 "Lowest recall (%)": lowest(pooled["per_sign_recall"], names), "Benchmark": ""})
    return pd.DataFrame(rows)


def confusion_counts_csv(path: Path, cm, labels: list[str], names: dict) -> None:
    M.write_confusion_csv(path, cm, [names.get(x, x) for x in labels])
