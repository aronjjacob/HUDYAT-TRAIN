"""Results for the paper: figures (300 dpi PNG + PDF), tables (CSV + Markdown), captions, summary, ZIP."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from trainer import experiments as E
from trainer import live, paper
from trainer import settings as cfg

from test_experiments import Ctx, FakeFit, env  # noqa: F401  (fixture)


@pytest.fixture
def runs(env):
    E.run(Ctx(), "random", "cat", {"seed": 1}, fit=FakeFit())
    r1 = E.run(Ctx(), "loso", "cat", {"seed": 1}, fit=FakeFit())
    r2 = E.run(Ctx(), "loso", "cat", {"seed": 2}, fit=FakeFit())
    return r1, r2


def test_generate_without_live_data_lists_what_is_missing(runs):
    out = paper.generate(Ctx(), "cat")
    d = paper.out_dir(cfg.Paths(), "cat")
    for name in ("dataset_clips", "detection_rate", "test1_confusion", "test1_recall", "test1_curves",
                 "test2_round_accuracy", "test2_recall", "test2_confusion", "test2_curves", "handedness",
                 "benchmark", "test2_confusion_S1", "test2_confusion_S2", "test2_confusion_S3"):
        png, pdf = d / "figures" / f"{name}.png", d / "figures" / f"{name}.pdf"
        assert png.exists() and pdf.exists(), name
        with Image.open(png) as im:
            dpi = im.info.get("dpi", (0, 0))
            assert round(dpi[0]) == 300 and im.width >= 2 * 300            # at least 2 inches at print size
    assert any("live" in m.lower() for m in out["missing"])
    assert not (d / "figures" / "live_accuracy.png").exists()


def test_tables_summary_captions_and_zip(runs):
    paths = cfg.Paths()
    live.save_attempt(paths, "cat", "T1", "right", "a", "a", 0.9, True, 300, "final-cat-1", 2)
    live.save_attempt(paths, "cat", "T2", "left", "b", "c", 0.8, False, 1300, "final-cat-1", 2)
    out = paper.generate(Ctx(), "cat")
    d = paper.out_dir(paths, "cat")
    for t in ("per_round", "summary_mean_sd", "per_sign", "handedness", "live_summary", "dataset_summary"):
        assert (d / "tables" / f"{t}.csv").exists(), t
    md = (d / "tables" / "tables.md").read_text(encoding="utf-8")
    assert "| Test signer |" in md and "± " in md
    summary = (d / "results_summary.md").read_text(encoding="utf-8")
    assert r_id(runs[1]) in summary and "SENYAS" in summary and "Live accuracy" in summary
    caps = (d / "captions.md").read_text(encoding="utf-8")
    assert "test2_recall" in caps and "Figure" in caps
    for name in ("live_accuracy", "response_time", "tester_hand"):
        assert (d / "figures" / f"{name}.png").exists()
    with zipfile.ZipFile(out["zip"]) as z:
        names = z.namelist()
    assert "results_summary.md" in names and "figures/test2_recall.pdf" in names
    meta = json.loads((d / "paper.json").read_text())
    assert meta["runs"]["test2"] == r_id(runs[1])                          # the newest finished Test 2


def test_live_results_are_reported_per_setup_with_the_settings_threshold(runs):
    paths = cfg.Paths()
    s = cfg.load_settings()
    s["confidence_threshold"] = 0.85
    cfg.save_settings(s)
    # trainer model: 0.9 counts, 0.8 does not (saved as correct under an older 70 % threshold)
    live.save_attempt(paths, "cat", "T1", "right", "a", "a", 0.9, True, 300, "final-cat-1", 1)
    live.save_attempt(paths, "cat", "T1", "right", "b", "b", 0.8, True, 300, "final-cat-1", 1)
    # app model, fixed capture: both pass the app's gates; app model, app capture today: margin too small
    for t in ("a", "b"):
        live.save_attempt(paths, "cat", "T1", "right", t, t, 0.82, True, 200, "final-cat-1", 1,
                          recognizer="app", capture="fixed", margin=0.5, time_from_hand_ms=900)
    live.save_attempt(paths, "cat", "T1", "right", "a", "a", 0.9, True, 2100, "final-cat-1", 1,
                      recognizer="app", capture="app", margin=0.1, time_from_hand_ms=2300)
    paper.generate(Ctx(), "cat")
    d = paper.out_dir(paths, "cat")
    tab = pd.read_csv(d / "tables" / "live_summary.csv")
    alls = tab[tab["Sign"] == "All"].set_index("Setup")["Accuracy"].to_dict()
    assert set(alls) == {"App model · fixed capture", "Trainer model", "App model · app capture today"}
    assert [alls["App model · fixed capture"], alls["Trainer model"], alls["App model · app capture today"]] ==         [paper._pct(1.0), paper._pct(0.5), paper._pct(0.0)]
    summary = (d / "results_summary.md").read_text(encoding="utf-8")
    assert "App model · fixed capture (reported)" in summary and "≥ 85% confidence" in summary
    caps = (d / "captions.md").read_text(encoding="utf-8")
    assert "70 % confidence" not in caps and "≥ 85% confidence" in caps and "Live test (App model · fixed capture)" in caps
    assert (d / "figures" / "live_setups.png").exists()


def r_id(r):
    return r["run_id"]


def test_senyas_values_are_editable_constants_with_a_source():
    assert paper.SENYAS["test_accuracy"] == 0.92 and paper.SENYAS["live_accuracy"] == 0.86
    assert "Oropesa" in paper.SENYAS["source"] and "2024" in paper.SENYAS["source"]


def test_nothing_to_show_without_runs(env):
    out = paper.generate(Ctx(), "cat")
    assert any("Test 2" in m for m in out["missing"])


def test_paper_tables_of_one_category(runs):
    from trainer import paper_tables as T
    paths = cfg.Paths()
    paper.generate(Ctx(), "cat")
    d = paper.out_dir(paths, "cat") / "tables"
    t6 = pd.read_csv(d / "table6_random.csv", dtype=str, keep_default_na=False)
    assert list(t6["Model"]) == ["Cat", "Benchmark", "SENYAS (Oropesa et al., 2024)"]
    assert t6.loc[0, "95% CI (%)"] and "(" in t6.loc[0, "Lowest recall (%)"]
    assert list(pd.read_csv(d / "table6_random_per_sign.csv")["Sign"])[-1] == "Mean (macro)"
    t7 = pd.read_csv(d / "table7_loso.csv", dtype=str, keep_default_na=False)
    assert t7.columns[3] == "Test S1 / S2 / S3 (%)" and t7.loc[0, "Every fold passes"] in ("yes", "no")
    folds = pd.read_csv(d / "table7_loso_per_fold.csv", dtype=str, keep_default_na=False)
    assert list(folds["Test signer"]) == ["S1", "S2", "S3", "Mean ± SD", "All folds pooled"]
    assert all(b.startswith(("PASS", "FAIL")) for b in folds["Benchmark"][:3])
    for f in ("errors_random", "errors_loso", "confusion_loso_counts", "confusion_loso_testS1_counts",
              "confusion_random_counts"):
        assert (d / f"{f}.csv").exists(), f
    # every error is listed with the test signer of its fold, and the counts add up to the wrong predictions
    run_dir = next(r["dir"] for r in E.list_runs(paths, "cat", "loso") if r["run_id"] == runs[1]["run_id"])
    res = E.load_results(Path(run_dir))
    wrong = sum(x["n"] - sum(x["confusion"][i][i] for i in range(len(x["confusion"]))) for x in res["rounds"])
    err = pd.read_csv(d / "errors_loso.csv", dtype=str, keep_default_na=False)
    assert sum(int(c) for c in err["Count"]) == wrong
    assert wrong == 0 or all(s.startswith("S") for s in err["Signer of the clips"])
    md = (d / "tables.md").read_text(encoding="utf-8")
    assert md.index("Table6 random") < md.index("Dataset summary")          # the paper's tables come first


def test_each_category_keeps_its_own_folder(runs):
    paths = cfg.Paths()
    paper.generate(Ctx(), "cat")
    (paper.out_dir(paths, "other") / "keep.txt").parent.mkdir(parents=True, exist_ok=True)
    (paper.out_dir(paths, "other") / "keep.txt").write_text("x")
    paper.generate(Ctx(), "cat")
    assert (paper.out_dir(paths, "other") / "keep.txt").exists()


def test_clopper_pearson_matches_the_audit():
    from trainer import paper_tables as T
    lo, hi = T.clopper_pearson(180, 180)
    assert round(lo * 100, 2) == 97.97 and hi == 1.0
    lo, hi = T.clopper_pearson(42, 42)
    assert round(lo * 100, 2) == 91.59
    assert T.clopper_pearson(0, 0) == (None, None)


def test_chosen_runs_are_used(runs):
    paths = cfg.Paths()
    t1 = next(r["run_id"] for r in E.list_runs(paths, "cat", "random"))
    paper.generate(Ctx(), "cat", test1=t1, test2=runs[0]["run_id"])                 # not the newest Test 2
    d = paper.out_dir(paths, "cat")
    meta = json.loads((d / "paper.json").read_text())
    assert meta["runs"]["test1"] == t1 and meta["runs"]["test2"] == runs[0]["run_id"]
    with pytest.raises(Exception, match="not a finished test run"):
        paper.generate(Ctx(), "cat", test2="nope")
