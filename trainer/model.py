"""The category model (input 32 x 128). It is the network the HUDYAT app builds (client/src/features/
fsl-recognition/category-core.js), layer for layer, including batch normalization's epsilon of 0.0001, so the
exported weights give the same answers there."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .landmarks import FRAME_DIM, N_FRAMES


def build_model(n_classes: int, lr: float = 1e-3, input_dim: int = FRAME_DIM):
    import keras
    from keras import layers
    model = keras.Sequential(name="hudyat_category_model", layers=[
        keras.Input(shape=(N_FRAMES, int(input_dim)), name="landmarks"),
        layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d"),
        layers.BatchNormalization(epsilon=1e-4, name="batch_normalization"),
        layers.MaxPooling1D(2, name="max_pooling1d"),
        layers.Dropout(0.3, name="dropout"),
        layers.Conv1D(128, 3, padding="same", activation="relu", name="conv1d_1"),
        layers.BatchNormalization(epsilon=1e-4, name="batch_normalization_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True, name="lstm"), name="bidirectional"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Bidirectional(layers.LSTM(64, name="lstm_1"), name="bidirectional_1"),
        layers.Dropout(0.3, name="dropout_3"),
        layers.Dense(64, activation="relu", name="dense"),
        layers.Dense(n_classes, activation="softmax", name="dense_1"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def fast_predict(model):
    """Compiled single-call prediction (x: (n, 32, features) -> probabilities). About 100x faster per sample
    than calling the Keras model eagerly; this is what the live test and the test timings use."""
    import tensorflow as tf
    dim = int(model.input_shape[-1])

    @tf.function(input_signature=[tf.TensorSpec([None, N_FRAMES, dim], tf.float32)], reduce_retracing=True)
    def _predict(x):
        return model(x, training=False)

    def predict(x) -> np.ndarray:
        return _predict(np.asarray(x, dtype=np.float32)).numpy()

    predict(np.zeros((1, N_FRAMES, dim), np.float32))                # trace once (warm-up)
    return predict


def load_samples(work: Path, items: list[dict], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {lab: i for i, lab in enumerate(labels)}
    X = np.zeros((len(items), N_FRAMES, FRAME_DIM), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int64)
    for i, it in enumerate(items):
        arr = np.load(Path(work) / it["file"])
        if arr.shape != (N_FRAMES, FRAME_DIM):
            raise ValueError(f"{it['file']}: shape {arr.shape}, expected (32, 128)")
        X[i] = arr
        y[i] = index[it["label"]]
    return X, y
