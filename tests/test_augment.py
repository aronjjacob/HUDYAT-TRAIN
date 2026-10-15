"""Augmentation: absent hand stays zero, mirror only when mirror_ok, no shift, fixed seed."""

import numpy as np

from trainer import augment as A


def sample(left=True, right=False, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((32, 128), np.float32)
    for on, off, flag in ((left, 0, 63), (right, 64, 127)):
        if on:
            h = rng.normal(0, 0.5, (32, 21, 3)).astype(np.float32)
            h[:, 0] = 0                                   # wrist at the origin (6.3)
            x[:, off:off + 63] = h.reshape(32, 63)
            x[:, flag] = 1
    return x


def test_absent_hand_stays_zero():
    rng = np.random.default_rng(0)
    x = sample(left=True, right=False)
    x[10:14, 0:64] = 0                                    # left also absent in a few frames
    for _ in range(200):
        y = A.augment_one(x, rng, mirror_ok=False)
        assert np.all(y[:, 64:128] == 0)
        absent = y[:, 63] < 0.5
        assert np.all(y[absent, 0:63] == 0)


def test_flags_stay_binary():
    rng = np.random.default_rng(1)
    x = sample(left=True, right=True)
    for _ in range(100):
        y = A.augment_one(x, rng, mirror_ok=True)
        assert set(np.unique(y[:, [63, 127]])) <= {0.0, 1.0}


def test_mirror_never_used_when_not_mirror_ok():
    rng = np.random.default_rng(2)
    x = sample(left=True, right=False)
    for _ in range(300):
        y = A.augment_one(x, rng, mirror_ok=False)
        assert np.all(y[:, 63] == 1) and np.all(y[:, 127] == 0)     # never moved to the Right slot


def test_mirror_used_when_mirror_ok():
    rng = np.random.default_rng(3)
    x = sample(left=True, right=False)
    moved = sum(bool(np.all(A.augment_one(x, rng, mirror_ok=True)[:, 127] == 1)) for _ in range(200))
    assert 60 < moved < 140                                         # p = 0.5


def test_mirror_function():
    x = sample(left=True, right=True, seed=4)
    y = A.mirror(x)
    lx = x[:, 0:63].reshape(32, 21, 3)
    ry = y[:, 64:127].reshape(32, 21, 3)
    np.testing.assert_allclose(ry[..., 0], -lx[..., 0])
    np.testing.assert_allclose(ry[..., 1:], lx[..., 1:])
    np.testing.assert_array_equal(y[:, 127], x[:, 63])
    np.testing.assert_allclose(A.mirror(y), x)


def test_no_shift_wrist_stays_at_origin():
    rng = np.random.default_rng(5)
    x = sample(left=True, right=True)
    for _ in range(100):
        y = A.augment_one(x, rng, mirror_ok=True, p={**A.DEFAULTS, "noise_std": 0.0})
        for off in (0, 64):
            np.testing.assert_allclose(y[:, off:off + 3], 0, atol=1e-6)


def test_rotation_about_wrist_and_scale():
    x = sample(left=True)
    y = A.rotate(x, 90)
    h, g = x[:, 0:63].reshape(32, 21, 3), y[:, 0:63].reshape(32, 21, 3)
    np.testing.assert_allclose(g[..., 0], -h[..., 1], atol=1e-5)
    np.testing.assert_allclose(g[..., 1], h[..., 0], atol=1e-5)
    np.testing.assert_allclose(g[..., 2], h[..., 2])
    np.testing.assert_allclose(A.scale(x, 1.1)[:, 0:63], x[:, 0:63] * 1.1, rtol=1e-6)


def test_augment_set_fixed_seed_and_size():
    X = np.stack([sample(seed=i) for i in range(4)])
    y = np.array([0, 1, 0, 1])
    a1 = A.augment_set(X, y, ["red", "blue"], {"red": 1, "blue": 0}, factor=5, seed=42)
    a2 = A.augment_set(X, y, ["red", "blue"], {"red": 1, "blue": 0}, factor=5, seed=42)
    assert a1[0].shape == (24, 32, 128)
    np.testing.assert_array_equal(a1[0], a2[0])
    np.testing.assert_array_equal(a1[0][:4], X)                     # originals first, untouched
    blue = a1[0][a1[1] == 1]
    assert np.all(blue[:, :, 127] == 0)                              # blue (mirror_ok 0) never mirrored


def test_balanced_class_weights():
    from trainer.train import balanced_weights
    y = np.array([0] * 300 + [1] * 300 + [2] * 30)
    w = balanced_weights(y, 3)
    assert w[2] > 5 * w[0] and abs(w[0] - w[1]) < 1e-9
    assert balanced_weights(np.array([0] * 10 + [1] * 11), 2) is None          # already even


def test_on_the_fly_batches():
    X = np.stack([sample(left=True, seed=i) for i in range(10)])
    y = np.array([0] * 8 + [1] * 2)
    g = A.OnTheFly(X, y, ["red", "blue"], {"red": 0, "blue": 0}, factor=3, seed=7, batch_size=8, n_classes=2,
                   class_weight={0: 0.5, 1: 2.0})
    assert len(g) == 5 and g.total == 40
    seen = []
    for i in range(len(g)):
        xb, yb, sw = g.batch(i)
        assert xb.shape[1:] == (32, 128) and yb.shape[1] == 2
        assert np.all(xb[:, :, 64:] == 0)                       # absent right hand stays zero
        assert set(np.round(sw, 3)) <= {0.5, 2.0}
        seen.append(xb)
    allx = np.concatenate(seen)
    assert len(allx) == 40
    assert sum(np.any(np.all(allx == x, axis=(1, 2))) for x in X) == 10    # every original once per epoch
    g2 = A.OnTheFly(X, y, ["red", "blue"], {"red": 0, "blue": 0}, factor=3, seed=7, batch_size=8, n_classes=2)
    np.testing.assert_array_equal(g2.batch(0)[0], seen[0])                   # same seed -> same batches
    g.next_epoch()
    assert not np.array_equal(g.batch(0)[0], seen[0])                       # new epoch -> new order / copies
