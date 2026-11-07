"""The final model: every signer, a chosen run's settings, and the preprocess.json the browser app copies."""

import json

import pytest

from trainer import experiments as E
from trainer import final
from trainer import settings as cfg

from test_experiments import SIGNS, Ctx, FakeFit, env  # noqa: F401  (fixture)


def test_final_model_uses_the_runs_settings_and_every_signer(env):
    res = E.run(Ctx(), "loso", "cat", {"seed": 7, "feature_version": 1, "canonical": True, "epochs": 3},
                fit=FakeFit())
    fit = FakeFit()
    out = final.run(Ctx(), "cat", from_run=res["run_id"], fit=fit)
    assert len(fit.calls) == 1
    c = fit.calls[0]
    trained = {f.split("/")[3] for f in c["train"] + c["val"]}
    assert trained == {"S1", "S2", "S3"}                               # all signers
    assert not set(c["train"]) & set(c["val"])
    assert len(c["val"]) == 6                                         # 15 % of 12 clips per sign, stratified
    paths = cfg.Paths()
    conf = json.loads((paths.model_dir("cat") / "config.json").read_text())
    assert conf["from_run"] == res["run_id"] and conf["opts"]["seed"] == 7 and conf["feature_version"] == 1
    assert conf["input_shape"] == [32, 128] and conf["model_run_id"] == out["model_run_id"]
    assert json.loads((paths.model_dir("cat") / "labels.json").read_text()) == list(SIGNS)


def test_final_model_has_exactly_one_output_per_sign_even_with_none_clips(env):
    """Real training (1 epoch): N signs -> N outputs, the labels are the signs, never a none class."""
    import keras
    from test_experiments import add_old_none_samples
    add_old_none_samples(env)
    final.run(Ctx(), "cat", opts={"epochs": 1, "seed": 1})
    d = cfg.Paths().model_dir("cat")
    labels = json.loads((d / "labels.json").read_text())
    assert labels == list(SIGNS) and "none" not in labels
    assert keras.models.load_model(d / "model.keras").output_shape == (None, len(SIGNS))
    assert json.loads((d / "config.json").read_text())["labels"] == list(SIGNS)


def test_preprocess_json_states_everything_the_browser_needs(env):
    E.run(Ctx(), "random", "cat", {"seed": 1, "feature_version": 1, "canonical": True}, fit=FakeFit())
    run_id = E.list_runs(cfg.Paths())[0]["run_id"]
    final.run(Ctx(), "cat", from_run=run_id, fit=FakeFit())
    p = json.loads((cfg.Paths().model_dir("cat") / "preprocess.json").read_text())
    assert p["frames"] == 32 and p["stored_frame_values"] == 149
    assert p["feature_version"] == 1 and p["features_per_frame"] == 128
    assert p["handedness"]["canonical"] is True and p["handedness"]["default_hand"] == "auto"
    assert p["labels"] == list(SIGNS)
    for k in ("trim_margin_s", "trim_min_frames", "min_hand_detection_confidence", "min_hand_presence_confidence",
              "min_tracking_confidence", "confidence_threshold", "live_window", "model_run_id", "contract"):
        assert k in p


def test_final_without_a_run_uses_the_defaults(env):
    fit = FakeFit()
    final.run(Ctx(), "cat", fit=fit)
    conf = json.loads((cfg.Paths().model_dir("cat") / "config.json").read_text())
    assert conf["from_run"] is None and conf["feature_version"] == 1 and conf["opts"]["canonical"] is True


def test_final_refuses_a_run_of_another_category(env):
    res = E.run(Ctx(), "random", "cat", {"seed": 1}, fit=FakeFit())
    with pytest.raises(final.ds.PlainError):
        final.run(Ctx(), "other", from_run=res["run_id"], fit=FakeFit())


def test_settings_without_a_test_run(env):
    final.run(Ctx(), "cat", opts={"feature_version": 1, "canonical": False, "epochs": 4, "seed": 3}, fit=FakeFit())
    conf = json.loads((cfg.Paths().model_dir("cat") / "config.json").read_text())
    assert conf["from_run"] is None and conf["feature_version"] == 1 and conf["input_shape"] == [32, 128]
    assert conf["opts"]["canonical"] is False and conf["opts"]["epochs"] == 4 and conf["opts"]["seed"] == 3


def test_several_categories_one_model_each_and_skip_the_ones_without_landmarks(env):
    fit = FakeFit()
    res = final.run_many(Ctx(), ["cat", "other"], opts={"epochs": 2}, fit=fit)
    assert len(fit.calls) == 1                                         # 'other' has no landmarks: skipped, not fatal
    assert res["trained"] == ["cat"] and res["skipped"] == ["other"]
    assert "other" in res["message"] and (cfg.Paths().model_dir("cat") / "model.keras").exists() is False \
        or (cfg.Paths().model_dir("cat") / "config.json").exists()


def test_several_categories_fail_plainly_when_none_has_landmarks(env):
    with pytest.raises(final.ds.PlainError):
        final.run_many(Ctx(), ["other"], fit=FakeFit())


def test_train_cli_accepts_all_and_settings(env, monkeypatch):
    from trainer import cli
    seen = {}

    def fake_run_many(ctx, cats, from_run=None, opts=None, fit=None):
        seen.update(cats=cats, from_run=from_run, opts=opts)
        return {"message": "ok"}
    monkeypatch.setattr(final, "run_many", fake_run_many)
    assert cli.main("train", ["--category", "all", "--no-canonical", "--epochs", "5"]) == 0
    assert seen["cats"] == ["cat", "other"] and seen["from_run"] is None             # 'all' = every model category
    assert "feature_version" not in seen["opts"] and seen["opts"]["canonical"] is False
    assert seen["opts"]["epochs"] == 5 and seen["opts"]["seed"] is None


def test_training_makes_the_paper_results_of_that_category(env):
    from trainer import paper
    E.run(Ctx(), "random", "cat", {"seed": 1, "feature_version": 1}, fit=FakeFit())
    E.run(Ctx(), "loso", "cat", {"seed": 1, "feature_version": 1}, fit=FakeFit())
    res = final.run_many(Ctx(), ["cat"], opts={"epochs": 2}, fit=FakeFit())
    d = paper.out_dir(cfg.Paths(), "cat") / "tables"
    assert (d / "table6_random.csv").exists() and (d / "table7_loso.csv").exists()
    assert "Paper results of cat updated" in res["message"]


def test_paper_problem_never_undoes_the_training(env, monkeypatch):
    from trainer import paper
    E.run(Ctx(), "loso", "cat", {"seed": 1, "feature_version": 1}, fit=FakeFit())

    def broken(ctx, category):
        raise RuntimeError("disk full")
    monkeypatch.setattr(paper, "generate", broken)
    ctx = Ctx()
    res = final.run_many(ctx, ["cat"], opts={"epochs": 2}, fit=FakeFit())
    assert res["trained"] == ["cat"] and "not made" in res["message"]
    assert any("disk full" in line for line in ctx.lines)


def test_preprocess_json_of_a_feature_v2_model(env):
    final.run(Ctx(), "cat", opts={"feature_version": 2}, fit=FakeFit())
    p = json.loads((cfg.Paths().model_dir("cat") / "preprocess.json").read_text())
    conf = json.loads((cfg.Paths().model_dir("cat") / "config.json").read_text())
    assert p["feature_version"] == conf["feature_version"] == 2
    assert p["normalization"] == conf["normalization"] == "hand-relative-sequence-v1"


def test_cli_passes_the_feature_version(env, monkeypatch):
    from trainer import cli
    from trainer import experiments as E2
    seen = {}

    def fake_run_many(ctx, cats, from_run=None, opts=None, fit=None):
        seen["train"] = opts
        return {"message": "ok"}

    def fake_experiment(ctx, test, category, opts):
        seen["experiment"] = opts
        return {"message": "ok", "run_id": "x"}
    monkeypatch.setattr(final, "run_many", fake_run_many)
    monkeypatch.setattr(E2, "run", fake_experiment)
    assert cli.main("train", ["--category", "cat", "--feature-version", "2"]) == 0
    assert cli.main("experiment", ["--test", "random", "--category", "cat", "--feature-version", "2"]) == 0
    assert seen["train"]["feature_version"] == seen["experiment"]["feature_version"] == 2
