"""Split rules: fixed signer split, no signer in two splits, never PUB, none in every model."""

import pandas as pd

from trainer import dataset as ds
from trainer import settings as cfg
from trainer.split import SPLITS, build_split

SIGNS = [ds.Sign("colors", "red", "Red", 15, 1, 1), ds.Sign("colors", "blue", "Blue", 15, 1, 0),
         ds.Sign("shapes", "circle", "Circle", 15, 1, 0), ds.Sign("none", "none", "None", 10, 1, 0)]


def usable(rows):
    return pd.DataFrame([{"file": f"landmarks/{c}/{s}/{g}/{s}_{g}_{i}.npy", "category": c, "sign_id": s,
                          "signer": g, "excluded_reason": r} for i, (c, s, g, r) in enumerate(rows)])


def all_rows(extra=()):
    rows = []
    for g in cfg.SIGNERS:
        for c, s in (("colors", "red"), ("colors", "blue"), ("shapes", "circle"), ("none", "none")):
            rows.append((c, s, g, ""))
    return usable(rows + list(extra))


def test_fixed_signer_split():
    sp = build_split("colors", all_rows(), SIGNS)
    seen = {s: {it["signer"] for it in sp[s]} for s in SPLITS}
    assert seen["train"] == {"S1", "S2", "S3", "S4"}
    assert seen["val"] == {"S5"}
    assert seen["test"] == {"S6", "S7"}


def test_no_signer_in_two_splits():
    sp = build_split("colors", all_rows(), SIGNS)
    sets = [{it["signer"] for it in sp[s]} for s in SPLITS]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    files = [it["file"] for s in SPLITS for it in sp[s]]
    assert len(files) == len(set(files))


def test_never_pub_or_other_signers():
    sp = build_split("colors", all_rows([("colors", "red", "PUB", ""), ("colors", "red", "S8", "")]), SIGNS)
    signers = {it["signer"] for s in SPLITS for it in sp[s]}
    assert signers <= set(cfg.SIGNERS)


def test_none_class_in_every_model_and_only_own_category():
    for cat in ("colors", "shapes"):
        sp = build_split(cat, all_rows(), SIGNS)
        assert sp["labels"][-1] == "none"
        labels = {it["label"] for s in SPLITS for it in sp[s]}
        assert "none" in labels
        assert labels == set(sp["labels"])
    assert build_split("colors", all_rows(), SIGNS)["labels"] == ["red", "blue", "none"]


def test_excluded_rows_left_out_and_missing_sign_warned():
    rows = [r for r in all_rows().itertuples(index=False)]
    df = pd.DataFrame(rows)
    df.loc[(df["sign_id"] == "blue") & (df["signer"] == "S5"), "excluded_reason"] = "rejected in Validation"
    sp = build_split("colors", df, SIGNS)
    assert sp["counts"]["blue"]["val"] == 0
    assert any(w.startswith("blue:") and "validation" in w for w in sp["warnings"])
    assert sp["mirror_ok"]["red"] == 1 and sp["mirror_ok"]["blue"] == 0
