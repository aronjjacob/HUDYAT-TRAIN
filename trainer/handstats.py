"""Preprocess results: which hands MediaPipe saw, per signer, and clips whose dominant hand looks wrong."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .landmarks import LEFT_FLAG, RIGHT_FLAG


def hand_stats(work: Path, clips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """clips: file, signer, hand, source_file (rich samples under `work`).

    Returns (per signer: clips, left_pct, right_pct, both_pct = share of sample frames with that hand;
    mismatches: clips whose Auto-detected dominant hand differs from the signer's signing_hand)."""
    per, bad = [], []
    for signer, g in clips.groupby("signer", sort=True):
        frames = left = right = both = 0
        for r in g.itertuples():
            try:
                x = np.load(Path(work) / r.file)
            except OSError:
                continue
            if x.shape[-1] != F.RICH_DIM:
                continue
            lp, rp = x[:, LEFT_FLAG] > 0.5, x[:, RIGHT_FLAG] > 0.5
            frames += len(x)
            left += int(lp.sum())
            right += int(rp.sum())
            both += int((lp & rp).sum())
            found = F.auto_hand(x)
            if (lp | rp).any() and found != r.hand:
                bad.append({"source_file": r.source_file, "signer": signer, "hand": r.hand, "detected": found})
        if frames:
            per.append({"signer": signer, "clips": len(g), "left_pct": left / frames, "right_pct": right / frames,
                        "both_pct": both / frames, "hand": ", ".join(sorted(set(g["hand"])))})
    return (pd.DataFrame(per, columns=["signer", "clips", "left_pct", "right_pct", "both_pct", "hand"]),
            pd.DataFrame(bad, columns=["source_file", "signer", "hand", "detected"]))
