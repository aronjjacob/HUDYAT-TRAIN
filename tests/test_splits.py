"""Test 1 (random split) and Test 2 (leave-one-signer-out): stratified, seeded, no clip in two sets."""

import numpy as np
import pandas as pd
import pytest

from trainer import splits as S


def clips(signers=("S1", "S2", "S3"), signs=("a", "b", "c", "d"), takes=15):
    rows = []
    for g in signers:
        for s in signs:
            for t in range(takes):
                rows.append({"file": f"landmarks/cat/{s}/{g}/{s}_{g}_{t:02d}.npy", "label": s, "signer": g})
    return pd.DataFrame(rows)


# --- Test 1 ----------------------------------------------------------------------------------
def test_random_split_every_clip_in_exactly_one_set():
    df = clips()
    sp = S.random_split(df, seed=42)
    files = [it["file"] for part in S.PARTS for it in sp[part]]
    assert len(files) == len(set(files)) == len(df)


def test_random_split_is_stratified_70_15_15():
    df = clips(takes=20)                                   # 60 clips per sign
    sp = S.random_split(df, seed=1)
    for lab in "abcd":
        n = {p: sum(it["label"] == lab for it in sp[p]) for p in S.PARTS}
        assert n == {"train": 42, "val": 9, "test": 9}


def test_random_split_small_sign_still_gets_val_and_test():
    df = clips(signers=("S1",), signs=("a",), takes=4)
    sp = S.random_split(df, seed=3)
    assert all(len(sp[p]) >= 1 for p in S.PARTS)


def test_random_split_is_seeded():
    df = clips()
    a, b, c = S.random_split(df, seed=5), S.random_split(df, seed=5), S.random_split(df, seed=6)
    assert a == b
    assert a != c


def test_random_split_ignores_row_order():
    df = clips()
    a = S.random_split(df, seed=5)
    b = S.random_split(df.sample(frac=1, random_state=0), seed=5)
    assert {p: sorted(x["file"] for x in a[p]) for p in S.PARTS} == {p: sorted(x["file"] for x in b[p]) for p in S.PARTS}


# --- Test 2 (LOSO) ---------------------------------------------------------------------------
@pytest.mark.parametrize("signers", [("S1", "S2", "S3"), ("S1", "S2", "S3", "S4", "S5", "S6", "S7")])
def test_loso_one_round_per_signer_and_held_out_signer_never_trained(signers):
    df = clips(signers=signers)
    rounds = S.loso_rounds(df, seed=42)
    assert [r["held_out"] for r in rounds] == list(signers)
    for r in rounds:
        assert {it["signer"] for it in r["test"]} == {r["held_out"]}
        assert r["held_out"] not in {it["signer"] for it in r["train"] + r["val"]}
        files = [it["file"] for p in S.PARTS for it in r[p]]
        assert len(files) == len(set(files)) == len(df)


def test_loso_validation_is_15_percent_stratified_of_training_signers():
    df = clips(takes=20)                                   # 2 training signers x 20 takes = 40 per sign
    for r in S.loso_rounds(df, seed=0):
        for lab in "abcd":
            nv = sum(it["label"] == lab for it in r["val"])
            nt = sum(it["label"] == lab for it in r["train"])
            assert nv == 6 and nt == 34


def test_loso_is_seeded():
    df = clips()
    assert S.loso_rounds(df, seed=9) == S.loso_rounds(df, seed=9)
    assert S.loso_rounds(df, seed=9) != S.loso_rounds(df, seed=10)


def test_loso_needs_two_signers():
    with pytest.raises(S.SplitError):
        S.loso_rounds(clips(signers=("S1",)), seed=1)


# --- mean +- SD --------------------------------------------------------------------------------
def test_mean_sd_uses_the_sample_sd():
    m, sd = S.mean_sd([0.8, 0.9, 1.0])
    assert m == pytest.approx(0.9)
    assert sd == pytest.approx(np.std([0.8, 0.9, 1.0], ddof=1))
    assert S.mean_sd([0.7]) == (pytest.approx(0.7), None)
    assert S.mean_sd([]) == (None, None)
