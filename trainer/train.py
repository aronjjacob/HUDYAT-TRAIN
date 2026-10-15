"""Step 5: train one model per category (with the none class)."""

from __future__ import annotations

import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import augment
from . import dataset as ds
from . import settings as cfg
from .model import build_model, load_samples
from .split import load_split

DEFAULTS = {"epochs": 150, "batch_size": 32, "aug_factor": 5, "seed": 42, "lr": 1e-3, "balance": True,
            "patience": 20, "reduce_lr_patience": 8, "reduce_lr_factor": 0.5, "min_lr": 1e-5}
LIVE_CSV = "train_live.csv"


def versions() -> dict:
    import keras
    import sklearn
    import tensorflow as tf
    out = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
           "numpy": np.__version__, "scikit-learn": sklearn.__version__}
    try:
        import mediapipe
        out["mediapipe"] = mediapipe.__version__
    except Exception:  # noqa: BLE001
        pass
    return out


def balanced_weights(y: np.ndarray, n_classes: int) -> dict[int, float] | None:
    """Rarer signs count more (total / (classes x count)); None when the classes are already even."""
    counts = np.bincount(np.asarray(y, int), minlength=n_classes)
    present = counts > 0
    if not present.any() or counts[present].max() <= 1.2 * counts[present].min():
        return None
    w = counts.sum() / (present.sum() * np.maximum(counts, 1))
    return {i: float(w[i]) for i in range(n_classes) if present[i]}


def train_category(ctx, category: str, opts: dict, part: tuple[int, int] = (0, 1)) -> dict:
    import keras
    s = cfg.load_settings()
    paths = cfg.Paths(s)
    sp = load_split(paths, category)
    if sp is None:
        raise ds.PlainError(f"No split for '{category}' yet.", "Run step 4 (Split) first.")
    labels = sp["labels"]
    if not sp["train"]:
        raise ds.PlainError(f"'{category}' has no training clips (S1-S4).",
                            "Record and extract clips for S1-S4, then split again.")
    if not sp["val"]:
        raise ds.PlainError(f"'{category}' has no validation clips (S5).",
                            "Record and extract S5's clips, then split again. Validation is needed "
                            "for early stopping.")
    from . import recorder
    idx = ds.read_index(paths)
    used = {it["file"] for part_ in ("train", "val", "test") for it in sp[part_]}
    col = idx["preprocess"] if "preprocess" in idx.columns else None
    ids = {recorder.params_id(recorder.parse_params_id(v)) for v in
           (col[idx["file"].isin(used)].fillna("") if col is not None else [""])}
    if len(ids) > 1:
        raise ds.PlainError(f"The samples of '{category}' were made with different preprocessing settings "
                            f"({', '.join(sorted(ids))}).",
                            "On the Preprocess page, choose this category (and none) and press Re-extract, "
                            "then run Split again.")
    preprocess = recorder.parse_params_id(next(iter(ids)) if ids else "")
    keras.utils.set_random_seed(int(opts["seed"]))
    Xtr, ytr = load_samples(paths.work, sp["train"], labels)
    Xva, yva = load_samples(paths.work, sp["val"], labels)
    n = len(labels)
    factor = int(opts["aug_factor"])
    n_samples = len(Xtr) * (factor + 1)
    Yva = keras.utils.to_categorical(yva, n)
    ctx.log(f"[{category}] {n} classes; train {len(Xtr)} clips x{int(opts['aug_factor']) + 1} "
            f"= {n_samples} samples per epoch (augmented on the fly); validation {len(Xva)}")

    out = paths.model_dir(category)
    out.mkdir(parents=True, exist_ok=True)
    live = paths.jobs / LIVE_CSV
    epochs = int(opts["epochs"])
    k, total_parts = part

    class Live(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            new = not live.exists()
            with open(live, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["category", "epoch", "loss", "accuracy", "val_loss", "val_accuracy", "lr"])
                w.writerow([category, epoch + 1] + [f"{float(logs.get(m, float('nan'))):.5f}" for m in
                                                     ("loss", "accuracy", "val_loss", "val_accuracy")]
                           + [f"{float(self.model.optimizer.learning_rate.numpy()):.6f}"])
            ctx.log(f"[{category}] epoch {epoch + 1}/{epochs}  loss {logs.get('loss', 0):.4f}  "
                    f"acc {logs.get('accuracy', 0):.3f}  val_loss {logs.get('val_loss', 0):.4f}  "
                    f"val_acc {logs.get('val_accuracy', 0):.3f}")
            ctx.progress((k + (epoch + 1) / epochs) / total_parts,
                         f"{category}: epoch {epoch + 1} / {epochs} (stops early if validation stops improving)",
                         category=category)

    model = build_model(n, float(opts["lr"]))
    ckpt = out / "model.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(opts["patience"]), restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=float(opts["reduce_lr_factor"]),
                                          patience=int(opts["reduce_lr_patience"]), min_lr=float(opts["min_lr"])),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        Live(),
    ]
    class_weight = balanced_weights(ytr, n) if opts.get("balance", True) else None
    gen = augment.OnTheFly(Xtr, ytr, labels, sp.get("mirror_ok", {}), factor, int(opts["seed"]),
                           int(opts["batch_size"]), n, class_weight)

    class Batches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False)

        def __len__(self):
            return len(gen)

        def __getitem__(self, i):
            return gen.batch(i)

        def on_epoch_end(self):
            gen.next_epoch()
    if class_weight:
        ctx.log(f"[{category}] class balancing: weights {', '.join(f'{labels[k]} {v:.2f}' for k, v in class_weight.items())}")
    t0 = time.time()
    hist = model.fit(Batches(), validation_data=(Xva, Yva), epochs=epochs, callbacks=callbacks, verbose=0)
    seconds = time.time() - t0
    model.save(ckpt)                                   # best weights (restored by EarlyStopping)
    h = {k2: [float(v) for v in vals] for k2, vals in hist.history.items()}
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = [k2 for k2 in ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate") if k2 in h]
        w.writerow(["epoch"] + keys)
        for e in range(len(h["loss"])):
            w.writerow([e + 1] + [f"{h[k2][e]:.6f}" for k2 in keys])
    from .charts import training_plot
    training_plot(h, out / "training_plot.png", f"{category}: training history")
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    best_epoch = int(np.argmin(h["val_loss"])) + 1
    val_loss, val_acc = model.evaluate(Xva, Yva, verbose=0)
    config = {
        "category": category, "labels": labels, "date": datetime.now().isoformat(timespec="seconds"),
        "input_shape": [32, 128], "options": opts, "preprocess": preprocess,
        "augmentation": {**augment.DEFAULTS, "factor": int(opts["aug_factor"]),
                         "mirror_ok": sp.get("mirror_ok", {})},
        "split_signers": cfg.SPLIT_SIGNERS, "split_created": sp.get("created"),
        "class_weight": {labels[k]: round(v, 3) for k, v in (class_weight or {}).items()},
        "counts": sp["counts"], "n_train_clips": len(Xtr), "n_train_samples": n_samples, "n_val": len(Xva),
        "epochs_run": len(h["loss"]), "best_epoch": best_epoch, "best_val_loss": float(val_loss),
        "best_val_accuracy": float(val_acc), "train_seconds": round(seconds, 1),
        "settings": {k2: s.get(k2) for k2 in ("data_dir", "min_hand_detect_rate")},
        "versions": versions(),
        "architecture": "Conv1D64k3-BN-MaxPool2-Drop.3-Conv1D128k3-BN-Drop.3-BiLSTM128seq-Drop.3-"
                        "BiLSTM64-Drop.3-Dense64-Dense softmax",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    ctx.log(f"[{category}] done in {seconds:.0f} s: best epoch {best_epoch}, val accuracy {val_acc:.3f}")
    return config


def run(ctx, categories: list[str], **opts) -> dict:
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v is not None}}
    paths = cfg.Paths()
    (paths.jobs / LIVE_CSV).unlink(missing_ok=True)
    results = {}
    for i, c in enumerate(categories):
        results[c] = train_category(ctx, c, o, (i, len(categories)))
    accs = ", ".join(f"{c} {r['best_val_accuracy']:.0%}" for c, r in results.items())
    return {"message": f"Trained {len(results)} model(s). Validation accuracy: {accs}.", "results": list(results)}


def model_exists(paths: cfg.Paths, category: str) -> bool:
    d = paths.model_dir(category)
    return (d / "model.keras").exists() and (d / "labels.json").exists()


def trained_categories(paths: cfg.Paths) -> list[str]:
    if not paths.models.exists():
        return []
    return sorted(p.name for p in paths.models.iterdir() if p.is_dir() and model_exists(paths, p.name))


def load_config(paths: cfg.Paths, category: str) -> dict | None:
    p = paths.model_dir(category) / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_labels(model_dir: Path) -> list[str]:
    return json.loads((Path(model_dir) / "labels.json").read_text(encoding="utf-8"))
