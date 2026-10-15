"""Training-set augmentation (in memory, fixed seed). Never applied to validation or test.

Every augmented copy gets noise, scale and rotation; time warp and frame dropout
each with probability 0.5; mirror with probability 0.5 but ONLY for signs with
mirror_ok == 1. No shift (the wrist is the origin). A hand whose present flag is
0 stays all zeros.
"""

from __future__ import annotations

import numpy as np

from .landmarks import (FRAME_DIM, HAND_DIM, LEFT_FLAG, LEFT_OFFSET, N_FRAMES, RIGHT_FLAG,
                        RIGHT_OFFSET, resample)

DEFAULTS = {"factor": 5, "noise_std": 0.01, "scale": 0.10, "rotation_deg": 15.0,
            "time_warp": (0.10, 0.20), "dropout_frames": (1, 3), "p_time_warp": 0.5,
            "p_dropout": 0.5, "p_mirror": 0.5}

SLOTS = ((LEFT_OFFSET, LEFT_FLAG), (RIGHT_OFFSET, RIGHT_FLAG))


def _hand_view(x: np.ndarray, off: int) -> np.ndarray:
    return x[:, off:off + HAND_DIM].reshape(len(x), 21, 3)


def zero_absent(x: np.ndarray) -> np.ndarray:
    for off, flag in SLOTS:
        absent = x[:, flag] < 0.5
        x[absent, off:off + HAND_DIM] = 0.0
        x[absent, flag] = 0.0
    return x


def add_noise(x, rng, std):
    x = x.copy()
    for off, flag in SLOTS:
        present = x[:, flag] > 0.5
        x[present, off:off + HAND_DIM] += rng.normal(0, std, (present.sum(), HAND_DIM)).astype(np.float32)
    return x


def scale(x, factor):
    x = x.copy()
    for off, _ in SLOTS:
        x[:, off:off + HAND_DIM] *= factor
    return x


def rotate(x, degrees):
    """Rotate (x, y) about the wrist (the origin after normalization)."""
    x = x.copy()
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    for off, _ in SLOTS:
        h = _hand_view(x, off)
        hx, hy = h[..., 0].copy(), h[..., 1].copy()
        h[..., 0] = c * hx - s * hy
        h[..., 1] = s * hx + c * hy
        x[:, off:off + HAND_DIM] = h.reshape(len(x), HAND_DIM)
    return x


def mirror(x):
    """x -> -x for every landmark and swap the Left/Right 64-value blocks."""
    x = x.copy()
    for off, _ in SLOTS:
        h = _hand_view(x, off)
        h[..., 0] *= -1
        x[:, off:off + HAND_DIM] = h.reshape(len(x), HAND_DIM)
    left = x[:, 0:64].copy()
    x[:, 0:64] = x[:, 64:128]
    x[:, 64:128] = left
    return x


def time_warp(x, rng, lo, hi):
    """Monotonic warp keeping first/last frame: t = 31 * (j/31) ** g, g = 1 +- U(lo, hi)."""
    amount = rng.uniform(lo, hi) * rng.choice([-1, 1])
    g = 1.0 + amount
    j = np.arange(N_FRAMES) / (N_FRAMES - 1)
    times = (N_FRAMES - 1) * j ** g
    return resample(x, N_FRAMES, times)


def frame_dropout(x, rng, lo, hi):
    """Drop 1-3 random inner frames, then resample back to 32."""
    k = int(rng.integers(lo, hi + 1))
    drop = rng.choice(np.arange(1, N_FRAMES - 1), size=k, replace=False)
    keep = np.setdiff1d(np.arange(N_FRAMES), drop)
    return resample(x[keep], N_FRAMES)


def augment_one(x: np.ndarray, rng: np.random.Generator, mirror_ok: bool, p: dict = DEFAULTS) -> np.ndarray:
    y = x.astype(np.float32).copy()
    if rng.random() < p["p_time_warp"]:
        y = time_warp(y, rng, *p["time_warp"])
    if rng.random() < p["p_dropout"]:
        y = frame_dropout(y, rng, *p["dropout_frames"])
    if mirror_ok and rng.random() < p["p_mirror"]:
        y = mirror(y)
    y = rotate(y, rng.uniform(-p["rotation_deg"], p["rotation_deg"]))
    y = scale(y, rng.uniform(1 - p["scale"], 1 + p["scale"]))
    y = add_noise(y, rng, p["noise_std"])
    return zero_absent(y)


def augment_set(X: np.ndarray, y: np.ndarray, labels: list[str], mirror_ok: dict[str, int],
                factor: int = 5, seed: int = 42, p: dict = DEFAULTS) -> tuple[np.ndarray, np.ndarray]:
    """Originals + `factor` augmented copies of each (so factor 5 -> 6x the data)."""
    rng = np.random.default_rng(seed)
    if factor <= 0 or len(X) == 0:
        return X, y
    Xs, ys = [X], [y]
    for _ in range(factor):
        aug = np.empty_like(X)
        for i in range(len(X)):
            aug[i] = augment_one(X[i], rng, bool(mirror_ok.get(labels[int(y[i])], 0)), p)
        Xs.append(aug)
        ys.append(y)
    assert Xs[0].shape[1:] == (N_FRAMES, FRAME_DIM)
    return np.concatenate(Xs), np.concatenate(ys)


class OnTheFly:
    """Training batches made on the fly: every epoch shows each original once plus `factor` freshly
    augmented copies of it, shuffled. Only the originals are kept in memory, so a large dataset (e.g.
    thousands of photos) trains on an 8 GB laptop. Deterministic for a given seed.

    batch(i) -> (x, y_onehot, sample_weight)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, labels: list[str], mirror_ok: dict[str, int], factor: int,
                 seed: int, batch_size: int, n_classes: int, class_weight: dict[int, float] | None = None,
                 p: dict = DEFAULTS) -> None:
        self.X, self.y = X, np.asarray(y, int)
        self.mirror = np.array([bool(mirror_ok.get(labels[int(c)], 0)) for c in self.y])
        self.factor, self.seed, self.bs, self.n, self.p = max(0, int(factor)), int(seed), int(batch_size), n_classes, p
        w = class_weight or {}
        self.sw = np.array([w.get(int(c), 1.0) for c in self.y], np.float32)
        self.total = len(X) * (self.factor + 1)
        self.epoch = 0
        self._order()

    def _order(self) -> None:
        self.perm = np.random.default_rng((self.seed, self.epoch)).permutation(self.total)

    def __len__(self) -> int:
        return int(np.ceil(self.total / self.bs))

    def next_epoch(self) -> None:
        self.epoch += 1
        self._order()

    def batch(self, i: int):
        ks = self.perm[i * self.bs:(i + 1) * self.bs]
        src = ks % len(self.X)
        copy = ks // len(self.X)
        xb = np.empty((len(ks), N_FRAMES, FRAME_DIM), np.float32)
        for j, (k, s, c) in enumerate(zip(ks, src, copy)):
            if c == 0:
                xb[j] = self.X[s]
            else:
                rng = np.random.default_rng((self.seed, self.epoch, int(k)))
                xb[j] = augment_one(self.X[s], rng, bool(self.mirror[s]), self.p)
        yb = np.eye(self.n, dtype=np.float32)[self.y[src]]
        return xb, yb, self.sw[src]
