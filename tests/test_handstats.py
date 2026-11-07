"""Preprocess results: per signer, how often the left / right hand was seen, and clips whose detected dominant
hand does not match the signer's signing_hand."""

import numpy as np
import pandas as pd

from trainer import features as F
from trainer.handstats import hand_stats


def sample(left: bool, right: bool, move_right=0.0, move_left=0.0):
    x = np.zeros((32, F.RICH_DIM), np.float32)
    t = np.arange(32)
    if left:
        x[:, 63] = 1
        x[:, F.L_WRIST] = np.stack([0.4 + move_left * t, np.full(32, 0.5), np.zeros(32)], 1)
        x[:, F.L_SCALE] = 0.1
    if right:
        x[:, 127] = 1
        x[:, F.R_WRIST] = np.stack([0.6 + move_right * t, np.full(32, 0.5), np.zeros(32)], 1)
        x[:, F.R_SCALE] = 0.1
    return x


def test_hand_stats_per_signer_and_mismatches(tmp_path):
    rows = []
    for i, (signer, hand, x) in enumerate([("S1", "right", sample(False, True)), ("S1", "right", sample(True, True, 0.02)),
                                           ("S2", "left", sample(True, False)), ("S2", "left", sample(False, True))]):
        f = f"landmarks/c/s/{signer}/x{i}.npy"
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / f, x)
        rows.append({"file": f, "signer": signer, "hand": hand, "source_file": f"{signer}_{i}.mp4"})
    per_signer, mismatches = hand_stats(tmp_path, pd.DataFrame(rows))
    s1 = per_signer.set_index("signer").loc["S1"]
    assert s1["right_pct"] == 1.0 and s1["left_pct"] == 0.5 and s1["clips"] == 2
    s2 = per_signer.set_index("signer").loc["S2"]
    assert s2["left_pct"] == 0.5 and s2["right_pct"] == 0.5
    assert list(mismatches["source_file"]) == ["S2_3.mp4"]                 # left-handed signer, right hand signing
    assert mismatches.iloc[0]["detected"] == "right"
