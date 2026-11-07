"""Step 3: the two tests of a category model.

  Test 1 (random): one round, a seeded, stratified 70 / 15 / 15 split of every kept clip of every signer.
      Answers "did the model learn the signs?" (the test clips come from people it trained on).
  Test 2 (loso):   one round per signer; the held-out signer is never in training or validation.
      Answers "will it work for someone it has never seen?" (the headline number).

A run lives in work/experiments/<test>/<category>/<run_id>/:
  config.json   seed, settings, versions, signers, the exact clips of every round and their counts
  state.json    running / stopped / done
  round_<name>/ history.csv, predictions.csv, confusion_matrix(_normalized).csv, metrics.json (written last:
                a round with metrics.json is finished, so Resume skips it)
  results.json  every round, mean +- SD, pooled per-sign results, summed confusion, PASS/FAIL, handedness check

No leakage: augmentation and mirror augmentation happen inside `fit`, on the training clips only; validation is
used only for early stopping; the mirrored test clips of the handedness check are made from the test set only.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import dataset as ds
from . import features as F
from . import metrics as M
from . import settings as cfg
from . import splits as S
from .splits import SplitError

TESTS = {"random": "Test 1: Random split (70 / 15 / 15)", "loso": "Test 2: Leave-one-signer-out"}
SHORT = {"random": "Test 1", "loso": "Test 2"}
TRAIN_DEFAULTS = {"epochs": 150, "batch_size": 32, "aug_factor": 5, "seed": 42, "lr": 1e-3, "balance": True,
                  "patience": 20, "reduce_lr_patience": 8, "reduce_lr_factor": 0.5, "min_lr": 1e-5}
DEFAULT_OPTS = {**TRAIN_DEFAULTS, "feature_version": 1, "canonical": True, "mirror_aug": False,
                "mirror_share": 0.5}
CONDITIONS = (("recorded", "none"), ("recorded", "auto"), ("mirrored", "none"), ("mirrored", "auto"))
IMPORT_SIGNER = "IMP"          # photo datasets (no signer IDs)


@dataclass
class Fitted:
    predict: Callable[[np.ndarray], np.ndarray]     # derived samples (N, 32, 128) -> probabilities
    history: dict                                    # loss, accuracy, val_loss, val_accuracy per epoch
    model: object = None                             # the Keras model (None for the test stand-in)


# ----------------------------------------------------------------------------
# Clips of a category
# ----------------------------------------------------------------------------
def clip_table(category: str, settings: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Usable clips of `category`: file, label, signer, hand, group, source_file; and the labels (the category's signs
    that have clips, nothing else: no none class)."""
    s = settings or cfg.load_settings()
    data_dir = cfg.data_dir(s)
    problems = cfg.check_data_dir(data_dir)
    if problems:
        raise ds.PlainError(problems[0], "Choose the dataset on the Data page.")
    paths = cfg.Paths(s)
    signs = ds.load_signs(data_dir)
    cat_signs = [x.sign_id for x in signs if x.category == category]
    if not cat_signs or category == cfg.NONE_CATEGORY:
        raise ds.PlainError(f"Unknown category: {category}.", "Pick a category from the list.")
    usable = ds.usable_samples(paths, data_dir, s)
    if usable.empty:
        raise ds.PlainError("No landmarks extracted yet.", "Run the Preprocess page first.")
    u = usable[usable["excluded_reason"] == ""]
    rows = u[(u["category"] == category) & u["sign_id"].isin(cat_signs)]
    if rows.empty:
        raise ds.PlainError(f"No usable clips of {category} yet.", "Extract them on the Preprocess page.")
    old = rows[rows["format"].fillna("") != ds.SAMPLE_FORMAT]
    if not old.empty:
        raise ds.PlainError(f"{ds.plural(len(old), 'sample')} of {category} made before the new sample format.",
                            "Open the Preprocess page and press 'Extract landmarks' once; they are extracted again.")
    hands = ds.clip_hands(data_dir)
    from .upload import group_key
    out = pd.DataFrame({
        "file": rows["file"].values,
        "label": rows["sign_id"].values,
        "signer": rows["signer"].values,
        "hand": [hands.get(f, "right") for f in rows["source_file"]],
        "source_file": rows["source_file"].values,
    })
    out["group"] = [f"{lab}|{group_key(Path(src).stem)}" if g == IMPORT_SIGNER else f
                    for f, lab, g, src in zip(out["file"], out["label"], out["signer"], out["source_file"])]
    out = out.sort_values("file").reset_index(drop=True)
    present = set(out["label"])
    labels = [x for x in cat_signs if x in present]
    return out, labels


def missing_signs(category: str, labels: list[str], settings: dict | None = None) -> list[str]:
    s = settings or cfg.load_settings()
    return [x.sign_id for x in ds.load_signs(cfg.data_dir(s)) if x.category == category and x.sign_id not in labels]


def real_signers(clips: pd.DataFrame) -> list[str]:
    return S.signer_order(g for g in clips["signer"] if g in cfg.SIGNERS)


def check_test_possible(clips: pd.DataFrame, test: str) -> None:
    if test not in TESTS:
        raise SplitError(f"Unknown test {test!r}.")
    if test == "loso" and len(real_signers(clips)) < 2:
        if (clips["signer"] == IMPORT_SIGNER).any():
            raise SplitError("Test 2 needs signer IDs. Photo datasets have none, so only Test 1 is possible here.")
        raise SplitError("Test 2 needs clips of at least two signers.")


def mirror_allowed(labels: list[str], signs: list[ds.Sign], on: bool) -> dict[str, int]:
    """Which signs mirror augmentation may flip. Off: none. On: the signs marked mirror_ok = 1 by the experts, or
    every sign while no sign is marked yet."""
    if not on:
        return {lab: 0 for lab in labels}
    marked = {x.sign_id for x in signs if x.mirror_ok}
    return {lab: int(lab in marked) if marked else 1 for lab in labels}


def load_rich(work: Path, items: list[dict], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {lab: i for i, lab in enumerate(labels)}
    X = np.zeros((len(items), 32, F.RICH_DIM), np.float32)
    y = np.zeros(len(items), np.int64)
    for i, it in enumerate(items):
        a = np.load(Path(work) / it["file"])
        if a.shape != (32, F.RICH_DIM):
            raise ds.PlainError(f"{it['file']} has the old format {a.shape}.",
                                "Open the Preprocess page and press 'Extract landmarks'.")
        X[i] = a
        y[i] = index[it["label"]]
    return X, y


def to_canonical(X: np.ndarray, hands: list[str]) -> np.ndarray:
    return np.stack([F.canonical(x, h) for x, h in zip(X, hands)]) if len(X) else X


def auto_canonical(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    found = [F.auto_hand(x) for x in X]
    return to_canonical(X, found), found


def preprocess_of(paths: cfg.Paths, files: list[str], data_dir: Path | None = None) -> dict:
    """The one preprocessing setting all these samples were made with (README 6.4 + MediaPipe values)."""
    from . import recorder
    idx = ds.read_index(paths)
    col = idx.set_index("file")["preprocess"] if "preprocess" in idx.columns else pd.Series(dtype=str)
    ids = {recorder.params_id(recorder.parse_params_id(str(col.get(f, "") or ""), data_dir)) for f in files}
    if len(ids) > 1:
        raise ds.PlainError(f"These samples were made with different preprocessing settings: {', '.join(sorted(ids))}.",
                            "On the Preprocess page use 'More' > 'Re-extract all', then run the test again.")
    return recorder.parse_params_id(next(iter(ids)) if ids else "", data_dir)


# ----------------------------------------------------------------------------
# Runs on disk
# ----------------------------------------------------------------------------
def run_dir(paths: cfg.Paths, test: str, category: str, run_id: str) -> Path:
    return paths.experiments / test / category / run_id


def new_run_id(paths: cfg.Paths, test: str, category: str, version: int) -> str:
    base = f"{datetime.now():%Y%m%d-%H%M%S}-v{int(version)}"
    rid, n = base, 2
    while run_dir(paths, test, category, rid).exists():
        rid, n = f"{base}-{n}", n + 1
    return rid


def find_run(paths: cfg.Paths, run_id: str) -> Path:
    for p in paths.experiments.glob(f"*/*/{run_id}"):
        if (p / "config.json").exists():
            return p
    raise ds.PlainError(f"Run {run_id} was not found.", "Pick a run from the list.")


def _json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_config(d: Path) -> dict | None:
    return _read(Path(d) / "config.json")


def load_results(d: Path) -> dict | None:
    return _read(Path(d) / "results.json")


def run_state(d: Path, paths: cfg.Paths | None = None) -> str:
    """done / running / stopped."""
    if (Path(d) / "results.json").exists() and (_read(Path(d) / "state.json") or {}).get("state") == "done":
        return "done"
    st = (_read(Path(d) / "state.json") or {}).get("state", "stopped")
    if st == "running":
        from . import jobs
        lock = jobs.read_lock(paths or cfg.Paths())
        if not lock or lock.get("job") != "experiment":
            return "stopped"
    return st


def list_runs(paths: cfg.Paths, category: str | None = None, test: str | None = None) -> list[dict]:
    """Every run (newest first) with its settings and headline numbers, for Compare runs."""
    out = []
    if not paths.experiments.exists():
        return out
    for conf_p in paths.experiments.glob("*/*/*/config.json"):
        conf = _read(conf_p)
        if not conf or (category and conf["category"] != category) or (test and conf["test"] != test):
            continue
        d = conf_p.parent
        res = load_results(d) or {}
        o = conf["opts"]
        summ = res.get("summary", {})
        done = sum((d / f"round_{r['name']}" / "metrics.json").exists() for r in conf["rounds"])
        h = res.get("handedness", {})
        out.append({"run_id": conf["run_id"], "test": conf["test"], "category": conf["category"],
                    "created": conf["created"], "feature_version": int(o["feature_version"]),
                    "canonical": bool(o["canonical"]), "mirror_aug": bool(o["mirror_aug"]),
                    "epochs": int(o["epochs"]), "seed": int(o["seed"]), "n_rounds": len(conf["rounds"]),
                    "rounds_done": done, "state": run_state(d, paths),
                    "accuracy_mean": summ.get("accuracy", {}).get("mean"), "accuracy_sd": summ.get("accuracy", {}).get("sd"),
                    "macro_f1_mean": summ.get("macro_f1", {}).get("mean"), "macro_f1_sd": summ.get("macro_f1", {}).get("sd"),
                    "mirrored_auto_mean": h.get("mirrored", {}).get("auto", {}).get("accuracy", {}).get("mean"),
                    "pass": res.get("pass"), "complete": bool(res.get("complete")), "dir": str(d)})
    return sorted(out, key=lambda r: (r["created"], r["run_id"]), reverse=True)


# ----------------------------------------------------------------------------
# Running a test
# ----------------------------------------------------------------------------
class RoundCtx:
    """Progress of one round as a share of the whole run, with an ETA from the time spent so far."""

    def __init__(self, ctx, i: int, n: int, name: str, started: float, start_fraction: float) -> None:
        self.ctx, self.i, self.n, self.name = ctx, i, n, name
        self.started, self.start_fraction = started, start_fraction

    def progress(self, fraction: float, text: str = "", **extra) -> None:
        total = (self.i + max(0.0, min(1.0, fraction))) / self.n
        done = total - self.start_fraction
        eta = ""
        if done > 0.01:
            left = (time.time() - self.started) / done * (1 - total)
            eta = f"  ·  about {max(1, round(left / 60))} min left" if left >= 60 else "  ·  less than a minute left"
        who = "all signers" if self.name == "all" else f"test signer {self.name}"
        self.ctx.progress(total, f"Round {self.i + 1} / {self.n} · {who}: {text}{eta}", round=self.name, **extra)

    def log(self, msg: str) -> None:
        self.ctx.log(msg)


def _items_with(clips: pd.DataFrame, part: list[dict]) -> list[dict]:
    keep = ("file", "label", "signer", "hand")
    return [{k: it[k] for k in keep} for it in part]


def _counts(items: list[dict], labels: list[str]) -> dict[str, int]:
    return {lab: sum(it["label"] == lab for it in items) for lab in labels}


def run(ctx, test: str | None = None, category: str | None = None, opts: dict | None = None,
        resume: str | None = None, fit=None) -> dict:
    paths = cfg.Paths()
    if fit is None:
        from .train import fit_model as fit
    if resume:
        d = find_run(paths, resume)
        conf = load_config(d)
        test, category = conf["test"], conf["category"]
        F.require_app_features(conf, f"Run {conf['run_id']}", "Start a new test run instead of resuming this one.")
        ctx.log(f"Resuming {conf['run_id']}: finished rounds are kept.")
    else:
        o = {**DEFAULT_OPTS, **{k: v for k, v in (opts or {}).items() if v is not None}}
        clips, labels = clip_table(category)
        check_test_possible(clips, test)
        seed = int(o["seed"])
        if test == "random":
            sp = S.random_split(clips, seed)
            rounds = [{"name": "all", "held_out": None, **{p: _items_with(clips, sp[p]) for p in S.PARTS}}]
        else:
            real = clips[clips["signer"].isin(cfg.SIGNERS)]
            rounds = [{"name": r["held_out"], "held_out": r["held_out"],
                       **{p: _items_with(real, r[p]) for p in S.PARTS}} for r in S.loso_rounds(real, seed)]
        for r in rounds:
            r["counts"] = {p: _counts(r[p], labels) for p in S.PARTS}
        s = cfg.load_settings()
        from .train import versions
        rid = new_run_id(paths, test, category, o["feature_version"])
        d = run_dir(paths, test, category, rid)
        conf = {"run_id": rid, "test": test, "test_name": TESTS[test], "category": category,
                "created": datetime.now().isoformat(timespec="seconds"), "normalization": F.normalization(o["feature_version"]),
                "opts": o, "labels": labels, "missing_signs": missing_signs(category, labels, s),
                "signers": real_signers(clips) or sorted(set(clips["signer"])),
                "hands": {g: sorted(set(clips[clips["signer"] == g]["hand"])) for g in sorted(set(clips["signer"]))},
                "n_clips": int(len(clips)), "preprocess": preprocess_of(paths, list(clips["file"]), cfg.data_dir(s)),
                "settings": {k: s.get(k) for k in ("data_dir", "min_hand_detect_rate")},
                "dataset": cfg.import_name(cfg.data_dir(s)) or "recordings", "versions": versions(),
                "rounds": rounds}
        _json(d / "config.json", conf)
        ctx.log(f"{TESTS[test]} on {category}: run {rid}, {len(rounds)} round(s), {len(clips)} clips, "
                f"feature v{o['feature_version']}, canonical hand {'on' if o['canonical'] else 'off'}, "
                f"mirror augmentation {'on' if o['mirror_aug'] else 'off'}, seed {seed}.")
    o, labels, rounds = conf["opts"], conf["labels"], conf["rounds"]
    _json(d / "state.json", {"state": "running", "t": time.time()})
    signs = ds.load_signs(cfg.data_dir())
    mirror_ok = mirror_allowed(labels, signs, bool(o["mirror_aug"]))
    started = time.time()
    done_before = sum((d / f"round_{r['name']}" / "metrics.json").exists() for r in rounds)
    try:
        for i, r in enumerate(rounds):
            rdir = d / f"round_{r['name']}"
            if (rdir / "metrics.json").exists():
                ctx.log(f"Round {i + 1} ({r['name']}): already finished, skipped.")
                continue
            rc = RoundCtx(ctx, i, len(rounds), r["name"], started, done_before / len(rounds))
            rc.progress(0.0, "loading clips")
            _run_round(rc, paths, r, labels, o, mirror_ok, rdir, fit)
        results = aggregate(d)
        ctx.progress(1.0, "drawing the figures")
        try:
            from .figures import run_figures
            run_figures(d, results)
        except Exception as e:  # noqa: BLE001 - the numbers are saved; a figure problem must not lose them
            ctx.log(f"Warning: the figures could not be drawn ({type(e).__name__}: {e}). The results are saved.")
        _json(d / "state.json", {"state": "done", "t": time.time()})
    except BaseException:
        _json(d / "state.json", {"state": "stopped", "t": time.time()})
        raise
    m, sd = results["summary"]["accuracy"]["mean"], results["summary"]["accuracy"]["sd"]
    msg = (f"{SHORT[test]} on {category}, features v{o['feature_version']}: {'PASS' if results['pass'] else 'FAIL'}, "
           f"accuracy {m:.1%}" + (f" ± {sd:.1%}" if sd is not None else "") + ".")
    ctx.log(msg)
    return {"message": msg, "run_id": conf["run_id"], "dir": str(d)}


def _run_round(ctx, paths, r: dict, labels: list[str], o: dict, mirror_ok: dict, rdir: Path, fit) -> None:
    version = int(o["feature_version"])
    Xtr, ytr = load_rich(paths.work, r["train"], labels)
    Xva, yva = load_rich(paths.work, r["val"], labels)
    Xte, yte = load_rich(paths.work, r["test"], labels)
    if not len(Xtr) or not len(Xva) or not len(Xte):
        raise ds.PlainError(f"Round {r['name']} has no {'training' if not len(Xtr) else 'validation' if not len(Xva) else 'test'} "
                            "clips.", "Record more clips, then run the test again.")
    if o["canonical"]:
        Xtr = to_canonical(Xtr, [it["hand"] for it in r["train"]])
        Xva = to_canonical(Xva, [it["hand"] for it in r["val"]])
    ctx.log(f"Round {r['name']}: train {len(Xtr)}, validation {len(Xva)}, test {len(Xte)} clips.")
    fitted = fit(ctx, r["train"], r["val"], Xtr, ytr, Xva, yva, labels, o, mirror_ok, rdir, r["name"])
    rdir.mkdir(parents=True, exist_ok=True)
    ev = evaluate_set(fitted.predict, Xte, yte, [it["hand"] for it in r["test"]], labels, version, bool(o["canonical"]))
    _write_history(rdir / "history.csv", fitted.history)
    with open(rdir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "signer", "hand", "label", "prediction", "confidence", "auto_hand_recorded",
                    "pred_recorded_none", "pred_recorded_auto", "pred_mirrored_none", "pred_mirrored_auto"])
        for i, it in enumerate(r["test"]):
            w.writerow([it["file"], it["signer"], it["hand"], it["label"], labels[ev["pred"][i]],
                        f"{ev['conf'][i]:.4f}", ev["auto_recorded"][i]]
                       + [labels[ev["cond_pred"][c][i]] for c in CONDITIONS])
    M.write_confusion_csv(rdir / "confusion_matrix.csv", ev["metrics"]["confusion"], labels)
    M.write_confusion_csv(rdir / "confusion_matrix_normalized.csv", ev["metrics"]["confusion"], labels, normalized=True)
    hist = fitted.history
    best = int(np.argmin(hist["val_loss"])) + 1 if hist.get("val_loss") else None
    _json(rdir / "metrics.json", {"name": r["name"], "held_out": r["held_out"], **ev["metrics"],
                                  "per_sign": M.per_sign_table(yte, ev["pred"], labels),
                                  "handedness": ev["handedness"], "ms_per_sample": ev["ms"],
                                  "epochs_run": len(hist.get("loss", [])), "best_epoch": best,
                                  "finished": datetime.now().isoformat(timespec="seconds")})
    ctx.log(f"Round {r['name']}: accuracy {ev['metrics']['accuracy']:.1%}, macro F1 {ev['metrics']['macro_f1']:.1%}; "
            f"mirrored + Auto {ev['handedness']['mirrored']['auto']['accuracy']:.1%}.")


def evaluate_set(predict, X: np.ndarray, y: np.ndarray, hands: list[str], labels: list[str], version: int,
                 canonical: bool) -> dict:
    """The main result (canonical with the known hand when the run uses it) and the 4-way handedness check."""
    def run_(Xc):
        P = predict(F.derive(Xc, version))
        return np.asarray(P).argmax(1), np.asarray(P).max(1)
    main = to_canonical(X, hands) if canonical else X
    pred, conf = run_(main)
    mX = F.mirror(X)
    rec_auto, found_r = auto_canonical(X)
    mir_auto, found_m = auto_canonical(mX)
    sets = {("recorded", "none"): X, ("recorded", "auto"): rec_auto, ("mirrored", "none"): mX,
            ("mirrored", "auto"): mir_auto}
    cond_pred, hand = {}, {"recorded": {}, "mirrored": {}}
    for c, Xc in sets.items():
        p, _ = run_(Xc)
        cond_pred[c] = p
        m = M.metrics(y, p, labels)
        hand[c[0]][c[1]] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "n": m["n"]}
    opposite = {"right": "left", "left": "right"}
    hand["auto_correct"] = {"recorded": float(np.mean([a == h for a, h in zip(found_r, hands)])) if hands else None,
                            "mirrored": float(np.mean([a == opposite[h] for a, h in zip(found_m, hands)]))
                            if hands else None, "n": len(hands)}
    d = F.derive(main[: min(len(main), 20)], version)
    t0 = time.perf_counter()
    for i in range(len(d)):
        predict(d[i:i + 1])
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(d))
    return {"pred": pred, "conf": conf, "metrics": M.metrics(y, pred, labels), "handedness": hand,
            "cond_pred": cond_pred, "auto_recorded": found_r, "ms": ms}


def _write_history(path: Path, h: dict) -> None:
    keys = [k for k in ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate") if k in h]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for e in range(len(h.get("loss", []))):
            w.writerow([e + 1] + [f"{float(h[k][e]):.6f}" for k in keys])


def read_history(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return {c: df[c].tolist() for c in df.columns}


# ----------------------------------------------------------------------------
# Results across rounds
# ----------------------------------------------------------------------------
def _ms(values) -> dict:
    m, sd = S.mean_sd(values)
    return {"mean": m, "sd": sd}


def aggregate(d: Path) -> dict:
    """results.json from the finished rounds of a run."""
    conf = load_config(d)
    labels = conf["labels"]
    rounds, preds, histories = [], [], {}
    for r in conf["rounds"]:
        rdir = Path(d) / f"round_{r['name']}"
        m = _read(rdir / "metrics.json")
        if m is None:
            continue
        rounds.append(m)
        p = pd.read_csv(rdir / "predictions.csv", dtype=str, keep_default_na=False)
        preds.append(p)
        histories[r["name"]] = read_history(rdir / "history.csv")
    if not rounds:
        raise ds.PlainError("No round of this run has finished yet.", "Resume the run.")
    allp = pd.concat(preds, ignore_index=True)
    index = {lab: i for i, lab in enumerate(labels)}
    yt = allp["label"].map(index).to_numpy()
    yp = allp["prediction"].map(index).to_numpy()
    pooled = M.metrics(yt, yp, labels)
    summary = {k: _ms([r[k] for r in rounds]) for k in M.KEYS}
    avg = {k: summary[k]["mean"] for k in M.KEYS}
    ok, reasons = M.verdict(avg, pooled)
    hand: dict = {"recorded": {}, "mirrored": {}}
    for a, b in CONDITIONS:
        hand[a][b] = {k: _ms([r["handedness"][a][b][k] for r in rounds]) for k in ("accuracy", "macro_f1")}
    n = sum(r["handedness"]["auto_correct"]["n"] for r in rounds)
    hand["auto_correct"] = {w: (sum(r["handedness"]["auto_correct"][w] * r["handedness"]["auto_correct"]["n"]
                                    for r in rounds) / n if n else None) for w in ("recorded", "mirrored")}
    results = {"run_id": conf["run_id"], "test": conf["test"], "category": conf["category"], "labels": labels,
               "feature_version": conf["opts"]["feature_version"], "opts": conf["opts"],
               "complete": len(rounds) == len(conf["rounds"]), "rounds": rounds, "summary": summary,
               "pooled": pooled, "per_sign": M.per_sign_table(yt, yp, labels), "pass": ok, "reasons": reasons,
               "handedness": hand, "histories": histories,
               "ms_per_sample": float(np.mean([r["ms_per_sample"] for r in rounds])),
               "thresholds": {"metric": M.PASS_METRIC, "per_sign_recall": M.PASS_PER_SIGN},
               "created": datetime.now().isoformat(timespec="seconds")}
    _json(Path(d) / "results.json", results)
    return results


def mean_curves(histories: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(epochs, mean, sd) of one history key across rounds; each epoch over the rounds that reached it."""
    series = [np.asarray(h[key], float) for h in histories.values() if h.get(key)]
    if not series:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    n = max(len(s) for s in series)
    mean, sd = np.zeros(n), np.zeros(n)
    for e in range(n):
        v = [s[e] for s in series if len(s) > e]
        mean[e] = np.mean(v)
        sd[e] = np.std(v, ddof=1) if len(v) > 1 else 0.0
    return np.arange(1, n + 1), mean, sd
