"""Step 4: the final model the app will use: every signer's clips (15 % stratified validation for early stopping),
trained with the settings of a chosen test run. Writes work/models/<category>/ including preprocess.json."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime

import numpy as np

from . import dataset as ds
from . import experiments as E
from . import features as F
from . import settings as cfg
from . import splits as S

CONTRACT = "README section 6 of the HUDYAT data collector + docs/FEATURES.md of the HUDYAT Trainer"


def preprocess_json(conf: dict, settings: dict | None = None) -> dict:
    """Every setting the browser app must copy to build the model's input."""
    from . import live, recorder
    s = settings or cfg.load_settings()
    pre = conf["preprocess"]
    v = int(conf["feature_version"])
    return {
        "contract": CONTRACT, "frames": 32, "stored_frame_values": F.RICH_DIM,
        "feature_version": v, "features_per_frame": F.feature_dim(v), "normalization": F.normalization(v),
        **pre, "recorder_values": recorder.is_default(pre),
        "handedness": {"canonical": bool(conf["opts"]["canonical"]), "default_hand": "auto",
                       "hands": ["right", "left", "auto"], "rule": "docs/FEATURES.md section 4"},
        "mirror_augmentation": bool(conf["opts"]["mirror_aug"]),
        "labels": conf["labels"], "confidence_threshold": float(s.get("confidence_threshold", 0.85)),
        "live_window": {"pre_buffer_s": live.PRE_BUFFER_S, "end_absent_s": live.END_ABSENT_S,
                        "max_window_s": live.MAX_WINDOW_S},
        "model_run_id": conf["model_run_id"], "from_experiment": conf["from_run"], "category": conf["category"],
    }


def run(ctx, category: str, from_run: str | None = None, opts: dict | None = None, fit=None) -> dict:
    paths = cfg.Paths()
    s = cfg.load_settings()
    if fit is None:
        from .train import fit_model as fit
    if from_run:
        rc = E.load_config(E.find_run(paths, from_run))
        if rc["category"] != category:
            raise ds.PlainError(f"Run {from_run} is a test of {rc['category']}, not {category}.",
                                "Pick a run of this category.")
        o = dict(rc["opts"])
    else:
        o = dict(E.DEFAULT_OPTS)
    o.update({k: v for k, v in (opts or {}).items() if v is not None})
    clips, labels = E.clip_table(category)
    sp = S.train_val_split(clips, int(o["seed"]))
    train_items = E._items_with(clips, sp["train"])
    val_items = E._items_with(clips, sp["val"])
    Xtr, ytr = E.load_rich(paths.work, train_items, labels)
    Xva, yva = E.load_rich(paths.work, val_items, labels)
    if o["canonical"]:
        Xtr = E.to_canonical(Xtr, [it["hand"] for it in train_items])
        Xva = E.to_canonical(Xva, [it["hand"] for it in val_items])
    mirror_ok = E.mirror_allowed(labels, ds.load_signs(cfg.data_dir(s)), bool(o["mirror_aug"]))
    out = paths.model_dir(category)
    out.mkdir(parents=True, exist_ok=True)
    model_run_id = f"final-{category}-{datetime.now():%Y%m%d-%H%M%S}"
    ctx.log(f"Final model for {category} ({model_run_id}): all signers, train {len(Xtr)}, validation {len(Xva)} "
            f"clips, settings of {from_run or 'the defaults'}.")
    fitted = fit(ctx, train_items, val_items, Xtr, ytr, Xva, yva, labels, o, mirror_ok, out, "final")
    for old in ("model.keras", "tfjs"):          # never leave an older model next to the new one ("tfjs": before v2)
        p = out / old
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    if fitted.model is not None:
        fitted.model.save(out / "model.keras")
    h = fitted.history
    E._write_history(out / "history.csv", h)
    if h.get("loss"):
        from .figures import curves
        curves({"final": h}, out / "training_plot")
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    from .train import versions
    vl = h.get("val_loss") or []
    conf = {"category": category, "model_run_id": model_run_id, "from_run": from_run, "labels": labels,
            "date": datetime.now().isoformat(timespec="seconds"), "normalization": F.normalization(o["feature_version"]), "opts": o,
            "feature_version": int(o["feature_version"]), "input_shape": [32, F.feature_dim(o["feature_version"])],
            "preprocess": E.preprocess_of(paths, list(clips["file"]), cfg.data_dir(s)),
            "signers": E.real_signers(clips) or sorted(set(clips["signer"])),
            "counts": {"train": E._counts(train_items, labels), "val": E._counts(val_items, labels)},
            "n_train": len(train_items), "n_val": len(val_items), "mirror_ok": mirror_ok,
            "epochs_run": len(h.get("loss", [])), "best_epoch": int(np.argmin(vl)) + 1 if vl else None,
            "best_val_accuracy": float(np.asarray(h["val_accuracy"])[int(np.argmin(vl))]) if vl else None,
            "train_seconds": h.get("seconds"), "versions": versions(),
            "settings": {k: s.get(k) for k in ("data_dir", "min_hand_detect_rate")}}
    (out / "config.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    (out / "preprocess.json").write_text(json.dumps(preprocess_json(conf, s), indent=2), encoding="utf-8")
    with open(out / "train_files.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "file", "label", "signer", "hand"])
        for name, items in (("train", train_items), ("val", val_items)):
            for it in items:
                w.writerow([name, it["file"], it["label"], it["signer"], it["hand"]])
    acc = conf["best_val_accuracy"]
    msg = f"Model for {category} trained" + (f": validation accuracy {acc:.1%}." if acc is not None else ".")
    ctx.log(msg)
    return {"message": msg, "model_run_id": model_run_id}


def load_preprocess(model_dir) -> dict | None:
    from pathlib import Path
    p = Path(model_dir) / "preprocess.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class _NoProgress:
    """The job's log, without moving its progress bar (the training already finished it)."""

    def __init__(self, ctx) -> None:
        self.ctx = ctx

    def progress(self, *_, **__) -> None:
        pass

    def log(self, msg: str) -> None:
        self.ctx.log(msg)


def paper_after_training(ctx, category: str) -> str:
    """Make the paper's results of `category` (Results for the paper page) right after its model is trained,
    when it has a finished test run. A problem here never undoes the training: it is only logged."""
    from . import paper
    if not any(r["complete"] for r in E.list_runs(cfg.Paths(), category=category)):
        return ""
    try:
        paper.generate(_NoProgress(ctx), category)
        return f"Paper results of {category} updated."
    except Exception as e:                                   # noqa: BLE001  (training stays done)
        ctx.log(f"Paper results of {category} not made: {e}")
        return f"Paper results of {category} not made (see the log)."


def run_many(ctx, categories: list[str], from_run: str | None = None, opts: dict | None = None, fit=None) -> dict:
    """One model per category, one after another. A category without usable landmarks is skipped (with a note),
    so 'All categories' still trains the ones that are ready; nothing trained at all is an error."""
    trained, skipped, notes, msgs = [], [], [], []
    n = len(categories)
    for i, c in enumerate(categories):
        if n > 1:
            ctx.progress(i / n, f"{c} · {i + 1} of {n}")
        try:
            msgs.append(run(ctx, c, from_run=from_run, opts=opts, fit=fit)["message"])
            trained.append(c)
        except ds.PlainError as e:
            if n == 1:
                raise
            skipped.append(c)
            notes.append(f"{c}: {e.message}")
            ctx.log(f"Skipped {c}: {e.message}")
    if not trained:
        raise ds.PlainError("No category could be trained.", " ".join(notes) or "Extract landmarks first.")
    msgs += [paper_after_training(ctx, c) for c in trained]
    msg = " ".join(m for m in msgs if m)
    if skipped:
        msg += f" Skipped, no usable landmarks: {', '.join(skipped)}."
    return {"message": msg, "trained": trained, "skipped": skipped}
