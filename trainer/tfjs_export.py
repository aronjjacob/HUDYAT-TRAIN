"""Step 7: TF.js layers-format export, written with numpy (no tensorflowjs package).

The official `tensorflowjs` converter pins packages (tensorflow-decision-forests,
jax, an older TensorFlow) that do not install next to TF 2.19 on Windows, and it
does not read Keras 3 models reliably. So this module writes the same files the
converter would:

  tfjs/model.json              {"format": "layers-model", "modelTopology": Keras-2 style config,
                                "weightsManifest": [...]}
  tfjs/group1-shard1of1.bin    all weights, float32 little-endian, manifest order
  tfjs/labels.json

and then checks the export by running the model a second time from those files
with a small numpy implementation of every layer (forward_numpy) and comparing
the predictions with Keras (verify.json). The end-to-end test also loads the
files with the real @tensorflow/tfjs in Node when Node is installed.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import settings as cfg

KERAS2_VERSION = "2.15.0"     # topology dialect understood by @tensorflow/tfjs-layers 4.x
GLOROT = {"class_name": "GlorotUniform", "config": {"seed": None}}
ORTHO = {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": None}}
ZEROS = {"class_name": "Zeros", "config": {}}
ONES = {"class_name": "Ones", "config": {}}


def _base(layer) -> dict:
    return {"name": layer.name, "trainable": True, "dtype": "float32"}


def _act(fn) -> str:
    name = getattr(fn, "__name__", str(fn))
    return {"relu": "relu", "softmax": "softmax", "tanh": "tanh", "sigmoid": "sigmoid",
            "linear": "linear"}.get(name, name)


def _lstm_config(lstm) -> dict:
    return {**_base(lstm), "units": int(lstm.units), "activation": _act(lstm.activation),
            "recurrent_activation": _act(lstm.recurrent_activation), "use_bias": bool(lstm.use_bias),
            "kernel_initializer": GLOROT, "recurrent_initializer": ORTHO, "bias_initializer": ZEROS,
            "unit_forget_bias": True, "kernel_regularizer": None, "recurrent_regularizer": None,
            "bias_regularizer": None, "activity_regularizer": None, "kernel_constraint": None,
            "recurrent_constraint": None, "bias_constraint": None, "dropout": 0.0, "recurrent_dropout": 0.0,
            "implementation": 2, "return_sequences": bool(lstm.return_sequences), "return_state": False,
            "go_backwards": False, "stateful": False, "unroll": False}


def topology_and_weights(model) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """Keras-2 style Sequential topology + [(tfjs weight name, array)] in manifest order."""
    from keras import layers as L
    cfg_layers = [{"class_name": "InputLayer",
                   "config": {"batch_input_shape": [None, *[int(d) for d in model.input_shape[1:]]],
                              "dtype": "float32", "sparse": False, "name": "landmarks"}}]
    weights: list[tuple[str, np.ndarray]] = []
    for layer in model.layers:
        w = [np.asarray(v, dtype=np.float32) for v in layer.get_weights()]
        if isinstance(layer, L.Conv1D):
            c = {**_base(layer), "filters": int(layer.filters), "kernel_size": [int(layer.kernel_size[0])],
                 "strides": [int(layer.strides[0])], "padding": layer.padding, "data_format": "channels_last",
                 "dilation_rate": [int(layer.dilation_rate[0])], "groups": 1, "activation": _act(layer.activation),
                 "use_bias": bool(layer.use_bias), "kernel_initializer": GLOROT, "bias_initializer": ZEROS,
                 "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None,
                 "kernel_constraint": None, "bias_constraint": None}
            cfg_layers.append({"class_name": "Conv1D", "config": c})
            weights += [(f"{layer.name}/kernel", w[0]), (f"{layer.name}/bias", w[1])]
        elif isinstance(layer, L.BatchNormalization):
            c = {**_base(layer), "axis": -1, "momentum": float(layer.momentum), "epsilon": float(layer.epsilon),
                 "center": True, "scale": True, "beta_initializer": ZEROS, "gamma_initializer": ONES,
                 "moving_mean_initializer": ZEROS, "moving_variance_initializer": ONES,
                 "beta_regularizer": None, "gamma_regularizer": None, "beta_constraint": None,
                 "gamma_constraint": None}
            cfg_layers.append({"class_name": "BatchNormalization", "config": c})
            names = ("gamma", "beta", "moving_mean", "moving_variance")
            weights += [(f"{layer.name}/{n}", a) for n, a in zip(names, w)]
        elif isinstance(layer, L.MaxPooling1D):
            c = {**_base(layer), "pool_size": [int(layer.pool_size[0])], "strides": [int(layer.strides[0])],
                 "padding": layer.padding, "data_format": "channels_last"}
            cfg_layers.append({"class_name": "MaxPooling1D", "config": c})
        elif isinstance(layer, L.Dropout):
            cfg_layers.append({"class_name": "Dropout",
                               "config": {**_base(layer), "rate": float(layer.rate), "noise_shape": None, "seed": None}})
        elif isinstance(layer, L.Bidirectional):
            inner = layer.forward_layer
            c = {**_base(layer), "layer": {"class_name": "LSTM", "config": _lstm_config(inner)},
                 "merge_mode": layer.merge_mode}
            cfg_layers.append({"class_name": "Bidirectional", "config": c})
            fw = [np.asarray(v, np.float32) for v in layer.forward_layer.get_weights()]
            bw = [np.asarray(v, np.float32) for v in layer.backward_layer.get_weights()]
            for d, arrs in (("forward", fw), ("backward", bw)):
                prefix = f"{layer.name}/{d}_{inner.name}/lstm_cell"
                weights += [(f"{prefix}/kernel", arrs[0]), (f"{prefix}/recurrent_kernel", arrs[1]),
                            (f"{prefix}/bias", arrs[2])]
        elif isinstance(layer, L.Dense):
            c = {**_base(layer), "units": int(layer.units), "activation": _act(layer.activation),
                 "use_bias": bool(layer.use_bias), "kernel_initializer": GLOROT, "bias_initializer": ZEROS,
                 "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None,
                 "kernel_constraint": None, "bias_constraint": None}
            cfg_layers.append({"class_name": "Dense", "config": c})
            weights += [(f"{layer.name}/kernel", w[0]), (f"{layer.name}/bias", w[1])]
        else:
            raise ValueError(f"layer type not supported by the exporter: {type(layer).__name__}")
    topology = {"class_name": "Sequential", "config": {"name": model.name, "layers": cfg_layers},
                "keras_version": KERAS2_VERSION, "backend": "tensorflow"}
    return topology, weights


def write_tfjs(model, labels: list[str], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    topology, weights = topology_and_weights(model)
    blob = b"".join(np.ascontiguousarray(a, dtype="<f4").tobytes() for _, a in weights)
    shard = "group1-shard1of1.bin"
    (out_dir / shard).write_bytes(blob)
    manifest = [{"paths": [shard], "weights": [{"name": n, "shape": list(a.shape), "dtype": "float32"}
                                               for n, a in weights]}]
    model_json = {"format": "layers-model", "generatedBy": "keras v3 (exported by HUDYAT Trainer)",
                  "convertedBy": "HUDYAT Trainer numpy TF.js writer", "modelTopology": topology,
                  "weightsManifest": manifest}
    (out_dir / "model.json").write_text(json.dumps(model_json), encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return out_dir / "model.json"


# ----------------------------------------------------------------------------
# Independent numpy forward pass from the exported files (export check)
# ----------------------------------------------------------------------------
def read_tfjs(model_json: Path) -> tuple[dict, dict[str, np.ndarray]]:
    mj = json.loads(Path(model_json).read_text(encoding="utf-8"))
    weights = {}
    for group in mj["weightsManifest"]:
        blob = b"".join((Path(model_json).parent / p).read_bytes() for p in group["paths"])
        off = 0
        for w in group["weights"]:
            n = int(np.prod(w["shape"])) if w["shape"] else 1
            weights[w["name"]] = np.frombuffer(blob, dtype="<f4", count=n, offset=off).reshape(w["shape"])
            off += 4 * n
    return mj["modelTopology"], weights


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _activation(name, x):
    if name == "relu":
        return np.maximum(x, 0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return _sigmoid(x)
    if name == "softmax":
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    return x


def _lstm(x, k, rk, b, units, return_sequences, reverse=False):
    B, T, _ = x.shape
    h = np.zeros((B, units), np.float64)
    c = np.zeros((B, units), np.float64)
    outs = [None] * T
    steps = range(T - 1, -1, -1) if reverse else range(T)
    for t in steps:
        z = x[:, t] @ k + h @ rk + b
        i, f, g, o = np.split(z, 4, axis=-1)
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        outs[t] = h
    return np.stack(outs, 1) if return_sequences else h


def forward_numpy(topology: dict, weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    for layer in topology["config"]["layers"]:
        cls, c = layer["class_name"], layer["config"]
        n = c.get("name")
        if cls in ("InputLayer", "Dropout"):
            continue
        if cls == "Conv1D":
            k, bias = weights[f"{n}/kernel"], weights[f"{n}/bias"]
            ks = k.shape[0]
            pad_l = (ks - 1) // 2
            xp = np.pad(x, ((0, 0), (pad_l, ks - 1 - pad_l), (0, 0)))
            T = x.shape[1]
            x = sum(xp[:, i:i + T] @ k[i] for i in range(ks)) + bias
            x = _activation(c["activation"], x)
        elif cls == "BatchNormalization":
            g, b = weights[f"{n}/gamma"], weights[f"{n}/beta"]
            m, v = weights[f"{n}/moving_mean"], weights[f"{n}/moving_variance"]
            x = g * (x - m) / np.sqrt(v + c["epsilon"]) + b
        elif cls == "MaxPooling1D":
            p = c["pool_size"][0]
            T = (x.shape[1] // p) * p
            x = x[:, :T].reshape(x.shape[0], T // p, p, x.shape[2]).max(2)
        elif cls == "Bidirectional":
            lc = c["layer"]["config"]
            parts = []
            for d, rev in (("forward", False), ("backward", True)):
                pre = f"{n}/{d}_{lc['name']}/lstm_cell"
                parts.append(_lstm(x, weights[f"{pre}/kernel"], weights[f"{pre}/recurrent_kernel"],
                                   weights[f"{pre}/bias"], lc["units"], lc["return_sequences"], rev))
            x = np.concatenate(parts, -1)
        elif cls == "Dense":
            x = _activation(c["activation"], x @ weights[f"{n}/kernel"] + weights[f"{n}/bias"])
        else:
            raise ValueError(cls)
    return x.astype(np.float32)


def export_category(ctx, category: str, n_check: int = 16) -> dict:
    import keras
    from .split import load_split
    from .train import load_labels, model_exists
    from .model import load_samples
    paths = cfg.Paths()
    if not model_exists(paths, category):
        raise ds.PlainError(f"No trained model for '{category}'.", "Run step 5 (Train) first.")
    mdir = paths.model_dir(category)
    labels = load_labels(mdir)
    model = keras.models.load_model(mdir / "model.keras")
    mj = write_tfjs(model, labels, mdir / "tfjs")
    from . import recorder
    from .train import load_config
    pre = (load_config(paths, category) or {}).get("preprocess") or recorder.default_params()
    (mdir / "tfjs" / "preprocess.json").write_text(json.dumps(
        {"frames": 32, "features_per_frame": 128, **pre, "recorder_values": recorder.is_default(pre),
         "note": "Build the 32 x 128 input exactly as in README section 6 of the data-collection folder, "
                 "with these trim and MediaPipe values."}, indent=2), encoding="utf-8")
    # check: real samples if available (test, else val, else train), else random
    sp = load_split(paths, category) or {}
    items = (sp.get("test") or sp.get("val") or sp.get("train") or [])[:n_check]
    if items:
        X, _ = load_samples(paths.work, items, labels)
    else:
        X = np.random.default_rng(0).normal(size=(n_check, 32, 128)).astype(np.float32)
    keras_p = model.predict(X, verbose=0)
    topo, w = read_tfjs(mj)
    np_p = forward_numpy(topo, w, X)
    max_diff = float(np.abs(keras_p - np_p).max())
    same_top1 = float((keras_p.argmax(1) == np_p.argmax(1)).mean())
    ok = max_diff < 1e-4 and same_top1 == 1.0
    check = {"category": category, "date": datetime.now().isoformat(timespec="seconds"), "n_samples": len(X),
             "max_abs_diff": max_diff, "same_top1": same_top1, "ok": ok,
             "method": "numpy forward pass from model.json + weights vs Keras"}
    (mdir / "tfjs" / "verify.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    ctx.log(f"[{category}] exported to {mdir / 'tfjs'}; check: max diff {max_diff:.2e}, "
            f"same top-1 {same_top1:.0%} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise ds.PlainError(f"The TF.js export of '{category}' does not match the Keras model.",
                            "Send the full log to the developers.")
    return check


def run(ctx, categories: list[str]) -> dict:
    for i, c in enumerate(categories):
        ctx.progress(i / len(categories), f"Exporting {c}")
        export_category(ctx, c)
    return {"message": f"Exported {len(categories)} model(s) to TF.js; predictions match Keras."}


def exported_files(paths: cfg.Paths, category: str) -> list[Path]:
    d = paths.model_dir(category) / "tfjs"
    return sorted(d.iterdir()) if d.exists() else []
