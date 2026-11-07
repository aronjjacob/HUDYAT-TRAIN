"""Training one category model (used by both tests and by the final model)."""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import augment
from . import dataset as ds
from . import settings as cfg
from .experiments import TRAIN_DEFAULTS as DEFAULTS
from .model import build_model, fast_predict

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


def fit_model(ctx, train_items, val_items, Xtr, ytr, Xva, yva, labels, opts, mirror_ok, out_dir, tag):
    """Train one Conv1D-BiLSTM on rich training clips (already in their canonical form when the run uses it).
    Augmentation happens here, on the training clips only; the feature version is derived after it.
    Validation is used only for early stopping. Returns experiments.Fitted."""
    import keras
    from .experiments import Fitted
    from .features import derive, feature_dim
    version = int(opts["feature_version"])
    keras.utils.set_random_seed(int(opts["seed"]))
    n = len(labels)
    Xva_d = derive(Xva, version)
    Yva = keras.utils.to_categorical(yva, n)
    factor = int(opts["aug_factor"])
    epochs = int(opts["epochs"])
    class_weight = balanced_weights(ytr, n) if opts.get("balance", True) else None
    p = {**augment.DEFAULTS, "p_mirror": float(opts.get("mirror_share", 0.5))}
    gen = augment.OnTheFly(Xtr, ytr, labels, mirror_ok, factor, int(opts["seed"]), int(opts["batch_size"]), n,
                           class_weight, p=p, version=version)
    paths = cfg.Paths()
    live = paths.jobs / LIVE_CSV
    ctx.log(f"[{tag}] {n} classes; train {len(Xtr)} clips x{factor + 1} = {gen.total} samples per epoch "
            f"(augmented on the fly{', mirror augmentation on' if any(mirror_ok.values()) else ''}); "
            f"validation {len(Xva)}; feature v{version} ({feature_dim(version)} values per frame)")

    class Live(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            live.parent.mkdir(parents=True, exist_ok=True)
            new = not live.exists()
            with open(live, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["category", "epoch", "loss", "accuracy", "val_loss", "val_accuracy", "lr"])
                w.writerow([tag, epoch + 1] + [f"{float(logs.get(m, float('nan'))):.5f}" for m in
                                                ("loss", "accuracy", "val_loss", "val_accuracy")]
                           + [f"{float(self.model.optimizer.learning_rate.numpy()):.6f}"])
            ctx.log(f"[{tag}] epoch {epoch + 1}/{epochs}  loss {logs.get('loss', 0):.4f}  "
                    f"acc {logs.get('accuracy', 0):.3f}  val_loss {logs.get('val_loss', 0):.4f}  "
                    f"val_acc {logs.get('val_accuracy', 0):.3f}")
            ctx.progress((epoch + 1) / epochs, f"epoch {epoch + 1} / {epochs} · stops early when validation stops "
                                               f"improving")

    class Batches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False)

        def __len__(self):
            return len(gen)

        def __getitem__(self, i):
            return gen.batch(i)

        def on_epoch_end(self):
            gen.next_epoch()

    model = build_model(n, float(opts["lr"]), feature_dim(version))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(opts["patience"]), restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=float(opts["reduce_lr_factor"]),
                                          patience=int(opts["reduce_lr_patience"]), min_lr=float(opts["min_lr"])),
        Live(),
    ]
    if class_weight:
        ctx.log(f"[{tag}] class balancing: " + ", ".join(f"{labels[k]} {v:.2f}" for k, v in class_weight.items()))
    t0 = time.time()
    hist = model.fit(Batches(), validation_data=(Xva_d, Yva), epochs=epochs, callbacks=callbacks, verbose=0)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    h["seconds"] = round(time.time() - t0, 1)
    return Fitted(fast_predict(model), h, model)


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
