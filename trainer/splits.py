"""Splits of the two tests, the label list, and mean +- SD.

Test 1 (random split): per sign, a seeded, stratified 70 / 15 / 15 % split of all kept clips of all signers.
Test 2 (leave-one-signer-out): one round per signer; round k tests on signer k only and trains on the others,
with a seeded, stratified 15 % of the training signers' clips held back for early stopping.

Clips are rows with at least file, label and signer. An optional `group` column keeps copies of one original
(e.g. augmented copies of one photo) in the same set. Every split is a pure function of the clips and the seed.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

PARTS = ("train", "val", "test")
FRACTIONS = (0.70, 0.15, 0.15)
LOSO_VAL = 0.15


class SplitError(ValueError):
    """The clips cannot be split this way (shown to the team in plain language)."""


def _items(df: pd.DataFrame) -> list[dict]:
    return [dict(r) for r in df.sort_values("file").to_dict("records")]


def _groups(df: pd.DataFrame) -> dict[str, list[dict]]:
    key = "group" if "group" in df.columns else "file"
    out: dict[str, list[dict]] = {}
    for it in _items(df):
        out.setdefault(str(it[key]), []).append(it)
    return out


def _assign(groups: dict[str, list[dict]], fractions: tuple[float, ...], names: tuple[str, ...],
            rng: np.random.Generator) -> dict[str, list[dict]]:
    """Shuffle the groups of one sign and fill the sets in order by clip count (cumulative fractions, rounded).
    With at least as many groups as sets, every set gets at least one group."""
    keys = sorted(groups)
    rng.shuffle(keys)
    total = sum(len(groups[k]) for k in keys)
    bounds = np.floor(np.cumsum(fractions)[:-1] * total + 0.5 + 1e-9)       # set sizes rounded half up
    chosen, done = [], 0
    for k in keys:
        chosen.append(names[int(np.searchsorted(bounds, done, side="right"))])
        done += len(groups[k])
    if len(keys) >= len(names):
        for i, name in enumerate(reversed(names[1:])):        # the last sets first: test, then val
            if name not in chosen:
                chosen[len(chosen) - 1 - i] = name
    out: dict[str, list[dict]] = {n: [] for n in names}
    for k, n in zip(keys, chosen):
        out[n] += groups[k]
    return out


def random_split(clips: pd.DataFrame, seed: int, fractions: tuple[float, float, float] = FRACTIONS) -> dict:
    """Test 1: {"train": [...], "val": [...], "test": [...]} (items sorted by file)."""
    if clips.empty:
        raise SplitError("There are no clips to split.")
    rng = np.random.default_rng(int(seed))
    out: dict[str, list[dict]] = {p: [] for p in PARTS}
    for label in sorted(clips["label"].unique()):
        parts = _assign(_groups(clips[clips["label"] == label]), fractions, PARTS, rng)
        for p in PARTS:
            out[p] += parts[p]
    return {p: sorted(out[p], key=lambda it: it["file"]) for p in PARTS}


def signer_order(signers) -> list[str]:
    """S1, S2, ..., S10 in number order."""
    def key(s: str):
        m = re.fullmatch(r"S(\d+)", str(s))
        return (0, int(m.group(1)), "") if m else (1, 0, str(s))
    return sorted(set(signers), key=key)


def loso_rounds(clips: pd.DataFrame, seed: int, val_fraction: float = LOSO_VAL) -> list[dict]:
    """Test 2: one round per signer: {"held_out", "train", "val", "test"}."""
    signers = signer_order(clips["signer"]) if not clips.empty else []
    if len(signers) < 2:
        raise SplitError("Leave-one-signer-out needs clips of at least two signers.")
    rounds = []
    for k, held in enumerate(signers):
        rng = np.random.default_rng([int(seed), k])
        test = clips[clips["signer"] == held]
        rest = clips[clips["signer"] != held]
        train: list[dict] = []
        val: list[dict] = []
        for label in sorted(rest["label"].unique()):
            parts = _assign(_groups(rest[rest["label"] == label]), (1 - val_fraction, val_fraction), ("train", "val"),
                            rng)
            train += parts["train"]
            val += parts["val"]
        rounds.append({"held_out": held, "train": sorted(train, key=lambda it: it["file"]),
                       "val": sorted(val, key=lambda it: it["file"]), "test": _items(test)})
    return rounds


def train_val_split(clips: pd.DataFrame, seed: int, val_fraction: float = LOSO_VAL) -> dict:
    """The final model: every clip, with a seeded, stratified share held back for early stopping."""
    rng = np.random.default_rng(int(seed))
    train: list[dict] = []
    val: list[dict] = []
    for label in sorted(clips["label"].unique()):
        parts = _assign(_groups(clips[clips["label"] == label]), (1 - val_fraction, val_fraction), ("train", "val"),
                        rng)
        train += parts["train"]
        val += parts["val"]
    return {"train": sorted(train, key=lambda it: it["file"]), "val": sorted(val, key=lambda it: it["file"])}


def mean_sd(values) -> tuple[float | None, float | None]:
    """Mean and sample SD (n - 1) across rounds; SD is None with fewer than two values."""
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None)
