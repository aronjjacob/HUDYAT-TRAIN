"""Step 7: export a final model in the format the HUDYAT app loads (HUDYAT/client/src/features/fsl-recognition,
category-runtime.ts + category-core.js). Writes work/exports/<category>-dabest-<dataset>-v1/:

  model.json          {"format": "fsl-conv1d-bilstm-v1", "category", "version", "inputShape": [32, 128], "classes",
                       "preprocessing", "weights": [{"shape", "dtype", "offset", "bytes"}], "weightBytes",
                       "weightSha256"}: the app builds the network itself and reads the weights from this list
  weights.bin         every weight, float32 little-endian, layer by layer in Keras order (what loadWeights expects)
  parity.json         {"inputs", "expected"}: real validation clips in the app's input form and the trained model's
                      answers; the app's model test checks the browser gives the same (max error 0.0002)
  preprocessing.json  the same "preprocessing" block on its own (for reading and for the paper)
  catalog_entry.json  the entry for server/model-assets/catalog.json (hashes and byte counts)
  verify.json         this exporter's own check (numpy forward pass from the written files vs Keras)
  README.txt          where each file goes in the app

The app keeps a frame's hands at other indices than the trainer (Left 0-62, Right 63-125, flags 126 and 127, instead
of Left 0-62 + flag 63, Right 64-126 + flag 127). The model's first layer is the only one that sees those indices,
so its kernel rows are reordered here; the exported model then gives exactly the trained model's answers on the
app's frames. Nothing else in the network changes.

Feature v3 (hand-face-location-v1) adds 13 values after the 128, in the same order in both: Left wrist, thumb tip,
index tip (x, y), the same for Right, then the face flag; measured against the face (README 6.7). Its export also
names the face model (preprocessing.face, catalog faceUrl / faceSha256).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import settings as cfg
from .features import APP_VERSIONS, FACE_CHIN, FACE_FOREHEAD, FACE_NOSE, LOC_DIM, LOC_POINTS, VERSIONS, \
    feature_dim, normalization, require_app_features
from .landmarks import FRAME_DIM, N_FRAMES

FORMAT = "fsl-conv1d-bilstm-v1"
# app index j holds trainer index APP_ORDER[j]
APP_ORDER = np.r_[np.arange(0, 63), np.arange(64, 127), 63, 127]
APP_ORDER_V3 = np.r_[APP_ORDER, np.arange(FRAME_DIM, FRAME_DIM + LOC_DIM)]     # the location values stay in place
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
                 "hand_landmarker.task")
FACE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                 "face_landmarker.task")
PARITY_TOLERANCE = 2e-4           # what the app's category-model.test.mjs allows
# Feature v2 is computed on the 32 resampled frames, like the stored training samples; Russel's trainer computes it on
# every camera frame and resamples after. The wrist path can start up to one 32-frame step apart, so the export says.
WRIST_PATH = "after resampling to 32 frames: the wrist of the first frame with the hand, the median palm length of the 32"
COORDINATES = {
    1: "per hand: landmarks minus the wrist, in pixels (x * width, y * height, z * width), divided by the 2D wrist to "
       "middle-finger MCP (landmark 9) length; a hand whose length is ~0 is absent",
    2: "per hand: landmarks 1-20 minus the wrist, in pixels (x * width, y * height, z * width), divided by the 2D "
       "wrist to middle-finger MCP (landmark 9) length; landmark 0: the wrist minus the wrist of the hand's first "
       "valid frame of the clip, divided by the median of that length over the clip; a hand whose length is ~0 is "
       "absent",
}
COORDINATES[3] = COORDINATES[1] + "; then the 13 values of preprocessing.face"
LOCATION = ("per hand (Left, then Right): wrist, thumb tip and index tip as (x, y) minus the nose (face landmark 1), "
            "x multiplied by width / height first, divided by the 2D forehead (10) to chin (152) length; then the "
            "face flag. A hand that is absent (or whose palm length is ~0), or a frame without a face, gives zeros")


def app_order(width: int) -> np.ndarray:
    """The app's index order for a model input of `width` values (128, or 141 for feature v3)."""
    if width == FRAME_DIM:
        return APP_ORDER
    if width == FRAME_DIM + LOC_DIM:
        return APP_ORDER_V3
    raise ValueError(f"no app layout for {width} values per frame")


def to_app_layout(x: np.ndarray) -> np.ndarray:
    """Trainer frames (..., 128 or 141) -> the app's frame layout."""
    x = np.asarray(x)
    return x[..., app_order(x.shape[-1])]


def model_export_name(paths: cfg.Paths, category: str) -> str:
    """export_name of the category's current final model (its feature version)."""
    from .train import load_config
    return export_name(paths, category, int((load_config(paths, category) or {}).get("feature_version", 1)))


def export_name(paths: cfg.Paths, category: str, feature_version: int) -> str:
    """<category>-dabest-<dataset>-v<feature version>, e.g. colors-dabest-recorder-v1."""
    slug = lambda t: re.sub(r"[^a-z0-9]+", "-", str(t).lower()).strip("-") or "x"
    return f"{slug(category)}-dabest-{slug(paths.dataset)}-v{int(feature_version)}"


def app_weights(model) -> list[np.ndarray]:
    """Every weight in the order the app's loadWeights sets them (layer by layer), first kernel reordered."""
    out = []
    for layer in model.layers:
        ws = [np.asarray(w, dtype=np.float32) for w in layer.get_weights()]
        if layer.name == model.layers[0].name:
            ws[0] = ws[0][:, app_order(ws[0].shape[1]), :]
        out += ws
    return out


def preprocessing_block(pre: dict, settings: dict) -> dict:
    """What the app reads (normalization, extraction.detection_confidence, extraction.swap_hands) plus what the
    model was trained on, from the model's preprocess.json."""
    import mediapipe
    task = cfg.MODEL_PATH
    v = int(pre.get("feature_version", 1))
    layout = ["left_xyz_landmarks_0_to_20", "right_xyz_landmarks_0_to_20", "left_present", "right_present"]
    face = {}
    if v == 3:
        from .landmarks import face_model_path
        ftask = face_model_path(cfg.data_dir(settings))
        layout += ["left_location_wrist_thumb_tip_index_tip_xy", "right_location_wrist_thumb_tip_index_tip_xy",
                   "face_present"]
        face = {"face": {"backend": "MediaPipe FaceLandmarker", "num_faces": 1, "running_mode": "VIDEO",
                         "model_sha256": hashlib.sha256(ftask.read_bytes()).hexdigest(), "model_url": FACE_TASK_URL,
                         "nose": FACE_NOSE, "forehead": FACE_FOREHEAD, "chin": FACE_CHIN,
                         "hand_points": list(LOC_POINTS), "values": LOCATION,
                         "interpolation": "a hand's location values linear only when the hand and the face are in "
                                          "both frames, else from the nearest frame; the face flag from the nearest "
                                          "frame",
                         "mirror": "a left-handed signer: location x negated, Left and Right location swapped, the "
                                   "face flag stays"}}
    return {
        "frames": N_FRAMES, "features": feature_dim(v),
        "layout": layout,
        "normalization": normalization(v), "coordinates": COORDINATES[v],
        **({"wrist_path": WRIST_PATH} if v == 2 else {}),
        "interpolation": "linear coordinates, nearest binary flags, absent coordinates zeroed",
        "sign_window": {"trim_margin_s": pre.get("trim_margin_s"), "trim_min_frames": pre.get("trim_min_frames"),
                        "rule": "training clips are cut to the frames with a hand (plus the margin), then resampled "
                                "to 32 frames"},
        "extraction": {"backend": "MediaPipe HandLandmarker", "mediapipe_version": mediapipe.__version__,
                       "model_sha256": hashlib.sha256(task.read_bytes()).hexdigest(), "model_url": HAND_TASK_URL,
                       "num_hands": 2, "detection_confidence": pre.get("min_hand_detection_confidence", 0.5),
                       "presence_confidence": pre.get("min_hand_presence_confidence", 0.5),
                       "tracking_confidence": pre.get("min_tracking_confidence", 0.5), "swap_hands": False,
                       "image_orientation": "unmirrored camera frames; only the preview is mirrored"},
        "handedness": {"canonical": bool(pre.get("handedness", {}).get("canonical", True)),
                       "rule": "trained on right-dominant signing; a left-handed signer's frames are mirrored first"},
        "confidence_threshold": pre.get("confidence_threshold", settings.get("confidence_threshold")),
        "trainer": {"feature_version": v, "model_run_id": pre.get("model_run_id"),
                    "from_experiment": pre.get("from_experiment")},
        **face,
    }


# ----------------------------------------------------------------------------
# Independent numpy forward pass of the app's network (the export check)
# ----------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x, k, rk, b, units, return_sequences, reverse=False):
    B, T, _ = x.shape
    h = np.zeros((B, units), np.float64)
    c = np.zeros((B, units), np.float64)
    outs = [None] * T
    for t in (range(T - 1, -1, -1) if reverse else range(T)):
        i, f, g, o = np.split(x[:, t] @ k + h @ rk + b, 4, axis=-1)
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        outs[t] = h
    return np.stack(outs, 1) if return_sequences else h


def forward_app(weights: list[np.ndarray], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """The app's buildModel (category-core.js) in numpy, on frames in the app's layout."""
    w = [np.asarray(a, np.float64) for a in weights]
    x = np.asarray(x, np.float64)

    def conv(x, k, b):
        xp = np.pad(x, ((0, 0), (1, 1), (0, 0)))
        T = x.shape[1]
        return np.maximum(sum(xp[:, i:i + T] @ k[i] for i in range(3)) + b, 0)

    def bn(x, g, b, m, v):
        return g * (x - m) / np.sqrt(v + eps) + b

    x = bn(conv(x, w[0], w[1]), *w[2:6])
    x = x[:, : (x.shape[1] // 2) * 2].reshape(x.shape[0], -1, 2, x.shape[2]).max(2)
    x = bn(conv(x, w[6], w[7]), *w[8:12])
    x = np.concatenate([_lstm(x, *w[12:15], 128, True), _lstm(x, *w[15:18], 128, True, True)], -1)
    x = np.concatenate([_lstm(x, *w[18:21], 64, False), _lstm(x, *w[21:24], 64, False, True)], -1)
    x = np.maximum(x @ w[24] + w[25], 0)
    z = x @ w[26] + w[27]
    e = np.exp(z - z.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def read_export(folder: Path) -> tuple[dict, list[np.ndarray]]:
    meta = json.loads((Path(folder) / "model.json").read_text(encoding="utf-8"))
    blob = (Path(folder) / "weights.bin").read_bytes()
    ws = [np.frombuffer(blob, dtype="<f4", count=s["bytes"] // 4, offset=s["offset"]).reshape(s["shape"])
          for s in meta["weights"]]
    return meta, ws


# ----------------------------------------------------------------------------
def _check_clips(paths, mdir: Path, conf: dict, labels: list[str], n: int) -> np.ndarray | None:
    """Up to n validation clips of the final model, in the form the model gets (trainer layout)."""
    from . import experiments as E
    from .features import derive
    tf_csv = mdir / "train_files.csv"
    if not tf_csv.exists():
        return None
    with open(tf_csv, newline="", encoding="utf-8") as f:
        items = [r for r in csv.DictReader(f) if r["set"] == "val"][:n]
    try:
        X, _ = E.load_rich(paths.work, items, labels) if items else (None, None)
    except (ds.PlainError, OSError):
        return None
    if X is None or not len(X):
        return None
    if conf["opts"].get("canonical", True):
        X = E.to_canonical(X, [it["hand"] for it in items])
    return derive(X, int(conf["feature_version"]))


def export_category(ctx, category: str, n_check: int = 8) -> dict:
    """Write the app's files for the final model of `category`, check them, and return the check."""
    import keras
    from .train import load_config, load_labels, model_exists
    s = cfg.load_settings()
    paths = cfg.Paths(s)
    if not model_exists(paths, category):
        raise ds.PlainError(f"No final model for '{category}'.", "Train it first on the Train model page.")
    mdir = paths.model_dir(category)
    conf = load_config(paths, category) or {}
    pre_p = mdir / "preprocess.json"
    require_app_features(conf, f"The model of '{category}'", "Train the model again on the Train model page, then export.")
    v = int(conf.get("feature_version", 0))
    if not pre_p.exists() or v not in APP_VERSIONS or conf.get("input_shape") != [N_FRAMES, feature_dim(v)]:
        raise ds.PlainError(f"The model of '{category}' was not trained with the app's features.",
                            "Train the model again on the Train model page, then export.")
    pre = json.loads(pre_p.read_text(encoding="utf-8"))
    labels = load_labels(mdir)
    model = keras.models.load_model(mdir / "model.keras")

    ws = app_weights(model)
    blob = b"".join(w.astype("<f4").tobytes() for w in ws)
    specs, off = [], 0
    for w in ws:
        specs.append({"shape": list(w.shape), "dtype": "float32", "offset": off, "bytes": int(w.size * 4)})
        off += w.size * 4
    wsha = hashlib.sha256(blob).hexdigest()
    version = wsha[:16]
    prep = preprocessing_block(pre, s)
    meta = {"format": FORMAT, "category": category, "version": version, "inputShape": [N_FRAMES, feature_dim(v)],
            "classes": labels, "preprocessing": prep, "weights": specs, "weightBytes": len(blob),
            "weightSha256": wsha}

    X = _check_clips(paths, mdir, conf, labels, n_check)
    real = X is not None
    if not real:
        X = np.random.default_rng(0).normal(size=(n_check, N_FRAMES, feature_dim(v))).astype(np.float32)
    keras_p = model.predict(X, verbose=0)

    name = export_name(paths, category, int(conf["feature_version"]))
    out = paths.exports / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    (out / "model.json").write_bytes(model_bytes)
    (out / "weights.bin").write_bytes(blob)
    X_app = to_app_layout(X)
    (out / "parity.json").write_text(json.dumps({"inputs": X_app.tolist(), "expected": keras_p.tolist()}),
                                     encoding="utf-8")
    (out / "preprocessing.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")

    _, back = read_export(out)
    np_p = forward_app(back, X_app)
    max_diff = float(np.abs(keras_p - np_p).max())
    same_top1 = float((keras_p.argmax(1) == np_p.argmax(1)).mean())
    ok = max_diff < PARITY_TOLERANCE and same_top1 == 1.0
    task = cfg.MODEL_PATH
    task_sha = prep["extraction"]["model_sha256"]
    entry = {"id": category, "name": category.replace("_", " ").title(), "available": True, "version": version,
             "modelUrl": f"./models/{category}/{version}/model.json",
             "weightsUrl": f"./models/{category}/{version}/weights.bin",
             "modelSha256": hashlib.sha256(model_bytes).hexdigest(), "weightsSha256": wsha,
             "bytes": len(model_bytes) + len(blob), "handUrl": f"./shared/{task_sha}.task", "handSha256": task_sha,
             "handBytes": task.stat().st_size, "classes": len(labels)}
    if v == 3:
        from .landmarks import face_model_path
        ftask, fsha = face_model_path(cfg.data_dir(s)), prep["face"]["model_sha256"]
        entry.update({"faceUrl": f"./shared/{fsha}.task", "faceSha256": fsha, "faceBytes": ftask.stat().st_size})
    (out / "catalog_entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    check = {"category": category, "date": datetime.now().isoformat(timespec="seconds"), "export_name": name,
             "version": version, "n_samples": len(X), "real_clips": real, "max_abs_diff": max_diff,
             "same_top1": same_top1, "ok": ok, "model_run_id": conf.get("model_run_id"),
             "method": "numpy forward pass of the app's network from model.json + weights.bin, on the app's frame "
                       "layout, vs the Keras model on the trainer's"}
    (out / "verify.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        f"HUDYAT app model: {category}, version {version} (trainer run {conf.get('model_run_id')}).\n\n"
        f"1. Copy model.json, weights.bin and parity.json to\n"
        f"   HUDYAT/server/model-assets/models/{category}/{version}/\n"
        f"2. In HUDYAT/server/model-assets/catalog.json, replace the entry with \"id\": \"{category}\" by the one in\n"
        f"   catalog_entry.json (the hand detector in shared/ is the same file the app already has).\n"
        + (f"   This model also needs the face detector: copy face_landmarker.task to\n"
           f"   HUDYAT/server/model-assets/shared/{entry['faceSha256']}.task\n" if v == 3 else "") +
        f"3. In HUDYAT/server run: npm run models:register\n", encoding="utf-8")
    ctx.log(f"[{category}] exported to {out} (version {version}); check: max diff {max_diff:.2e}, "
            f"same top-1 {same_top1:.0%} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise ds.PlainError(f"The app export of '{category}' does not match the Keras model.",
                            "Send the full log to the developers.")
    return check


def run(ctx, categories: list[str]) -> dict:
    names = []
    for i, c in enumerate(categories):
        ctx.progress(i / len(categories), f"Exporting {c}")
        names.append(export_category(ctx, c)["export_name"])
    return {"message": f"Exported {ds.plural(len(categories), 'model')} for the HUDYAT app: {', '.join(names)}. "
                       "Predictions match the trained model.", "exports": names}


def list_exports(paths: cfg.Paths, category: str) -> list[dict]:
    """Every app export of `category` (one folder per feature version), newest first, with what the Live test shows:
    name, feature_version, version (the weights' hash), model_run_id, from_run, date, and current (= the trainer's
    final model now)."""
    from .features import version_of
    from .train import load_config
    current = (load_config(paths, category) or {}).get("model_run_id")
    found = []
    for d in paths.exports.iterdir() if paths.exports.exists() else []:
        model, verify = d / "model.json", d / "verify.json"
        if not (model.exists() and verify.exists()):
            continue
        try:
            meta = json.loads(model.read_text(encoding="utf-8"))
            check = json.loads(verify.read_text(encoding="utf-8"))
            fv = version_of(meta["preprocessing"]["normalization"])
        except (OSError, ValueError, KeyError):
            continue
        if meta.get("category") != category:
            continue
        trainer = meta["preprocessing"].get("trainer") or {}
        found.append({"name": d.name, "feature_version": fv, "version": meta["version"], "classes": meta["classes"],
                      "model_run_id": trainer.get("model_run_id") or check.get("model_run_id"),
                      "from_run": trainer.get("from_experiment"), "date": check.get("date", ""),
                      "current": bool(current) and trainer.get("model_run_id") == current,
                      "_order": verify.stat().st_mtime_ns})
    found.sort(key=lambda e: e.pop("_order"), reverse=True)
    return found


def export_dir(paths: cfg.Paths, category: str, name: str | None) -> Path:
    """The folder of the export `name` of `category` (None: the export of the trainer's current model)."""
    if name is None:
        name = model_export_name(paths, category)
    if name not in {e["name"] for e in list_exports(paths, category)}:
        raise ds.PlainError(f"No app export '{name}' of '{category}'.",
                            "Export the model on the Train model page, then pick it on the Live test page.")
    return paths.exports / name


def last_export(paths: cfg.Paths, category: str) -> dict | None:
    """verify.json of this category's export, if any."""
    p = paths.exports / model_export_name(paths, category) / "verify.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def exported_files(paths: cfg.Paths, category: str) -> list[Path]:
    d = paths.exports / model_export_name(paths, category)
    return sorted(d.iterdir()) if d.exists() else []
