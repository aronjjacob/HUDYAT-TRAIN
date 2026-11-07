"""Command-line entry points. The app starts exactly these (python -m trainer.runner <job> ...),
so every step can also run without the UI:

  python extract.py [--all] [--category C ...]
  python experiment.py --test random|loso --category C|all [--feature-version 1|2|3] [--no-canonical] [--mirror-aug]
                       [--epochs 150 --batch 32 --aug 5 --seed 42]
  python experiment.py --resume RUN_ID
  python train.py --category C [C ...]|all [--from-run RUN_ID] [--feature-version 1|2|3] [--no-canonical]
                  [--epochs 150 ...]                                     (one model per category)
  python export_model.py --category C [C ...]|all          (the HUDYAT app's model files)
  python paper.py --category C                                         (figures + tables for the paper)
  python live_test.py --category C [--mode test --tester T --tester-hand right|left --target SIGN --trials 25]
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _categories(arg: list[str] | None, trained_only: bool = False) -> list[str]:
    from . import dataset as ds
    from . import settings as cfg
    from .train import trained_categories
    if trained_only:
        avail = trained_categories(cfg.Paths())
    else:
        d = cfg.data_dir()
        if d is None:
            raise ds.PlainError("No dataset chosen yet.", "Choose it under Dataset in the sidebar, or add a folder or a .zip on the Data page.")
        avail = ds.model_categories(ds.load_signs(d))
    if not arg or arg == ["all"]:
        if not avail:
            raise ds.PlainError("Nothing to do yet.", "Run the earlier steps first.")
        return avail
    return arg


def main(job: str, argv: list[str] | None = None) -> int:
    from . import jobs
    from .dataset import PlainError
    p = argparse.ArgumentParser(prog=f"{job}")
    if job == "extract":
        p.add_argument("--all", action="store_true", help="re-extract every clip")
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--category", nargs="*", help="only these categories (alphabet_static_test only when named)")
    elif job == "experiment":
        p.add_argument("--test", choices=["random", "loso"])
        p.add_argument("--category", nargs="+", help="one or more categories, or all")
        p.add_argument("--feature-version", type=int, choices=[1, 2, 3], help="model input: 1 = hand-relative-v1 (wrist always 0), 2 = hand-relative-sequence-v1 (+ the wrist's movement), 3 = hand-face-location-v1 (v1 + where the hand is against the face; for Family; the app runs a face model for it)")
        p.add_argument("--no-canonical", action="store_true", help="do not mirror left-handed clips")
        p.add_argument("--mirror-aug", action="store_true", help="mirror a share of the training clips")
        p.add_argument("--mirror-share", type=float)
        p.add_argument("--epochs", type=int)
        p.add_argument("--batch", type=int)
        p.add_argument("--aug", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--patience", type=int)
        p.add_argument("--no-balance", action="store_true", help="do not weight rare signs more")
        p.add_argument("--resume", help="continue a stopped run (finished rounds are kept)")
    elif job == "train":
        p.add_argument("--category", nargs="+", required=True, help="one or more categories, or all")
        p.add_argument("--from-run", help="use the settings of this test run")
        p.add_argument("--feature-version", type=int, choices=[1, 2, 3], help="model input: 1 = hand-relative-v1 (wrist always 0), 2 = hand-relative-sequence-v1 (+ the wrist's movement), 3 = hand-face-location-v1 (v1 + where the hand is against the face; for Family; the app runs a face model for it)")
        p.add_argument("--no-canonical", action="store_true", help="do not mirror left-handed clips")
        p.add_argument("--mirror-aug", action="store_true", help="mirror a share of the training clips")
        p.add_argument("--mirror-share", type=float)
        p.add_argument("--epochs", type=int)
        p.add_argument("--batch", type=int)
        p.add_argument("--aug", type=int)
        p.add_argument("--seed", type=int)
    elif job == "export":
        p.add_argument("--category", nargs="+", required=True)
    elif job == "dataset":
        p.add_argument("--category", nargs="+", required=True, help="one or more categories, or all")
    elif job == "paper":
        p.add_argument("--category", required=True)
        p.add_argument("--test1", help="run ID of the Test 1 run to use (default: chosen automatically)")
        p.add_argument("--test2", help="run ID of the Test 2 run to use (default: chosen automatically)")
    elif job == "live":
        p.add_argument("--category", required=True)
        p.add_argument("--mode", choices=["free", "test"], default="free")
        p.add_argument("--tester", default="")
        p.add_argument("--tester-hand", choices=["right", "left"], default="right")
        p.add_argument("--recognizer", choices=["trainer", "app"], default="trainer",
                       help="the trainer's model, or the HUDYAT app's exported model files")
        p.add_argument("--capture", choices=["fixed", "app"], default="fixed",
                       help="app recognizer: the recommended sign window (fixed) or the app's 2-second clip (app)")
        p.add_argument("--export", help="app recognizer: the export folder to run (default: the current model's)")
        p.add_argument("--hand", choices=["right", "left", "auto"], help="hand setting (default: Settings)")
        p.add_argument("--screenshot", help=argparse.SUPPRESS)     # hidden: save the screens for the docs
        p.add_argument("--target", default="")
        p.add_argument("--trials", type=int, default=25)
        p.add_argument("--camera", type=int)
        p.add_argument("--backend", choices=["msmf", "dshow"])
        p.add_argument("--video", help=argparse.SUPPRESS)          # hidden: feed a recorded file
        p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    else:
        raise SystemExit(f"unknown job {job}")
    a = p.parse_args(argv)
    try:
        if job == "extract":
            from . import extract
            r = jobs.run_job("extract", extract.run, reextract_all=a.all, workers=a.workers,
                             categories=a.category or None)
        elif job == "experiment":
            from . import experiments
            if a.resume:
                r = jobs.run_job("experiment", experiments.run, resume=a.resume)
            else:
                if not a.test or not a.category:
                    raise PlainError("Choose the test and the category.", "--test random|loso --category C")
                opts = {"epochs": a.epochs, "batch_size": a.batch, "aug_factor": a.aug, "seed": a.seed,
                        "patience": a.patience, "canonical": not a.no_canonical, "mirror_aug": a.mirror_aug,
                        "mirror_share": a.mirror_share, "feature_version": a.feature_version}
                if a.no_balance:
                    opts["balance"] = False

                cats = _categories(a.category)

                def several(ctx):
                    out = []
                    for c in cats:
                        out.append(experiments.run(ctx, a.test, c, opts))
                    return {"message": " ".join(x["message"] for x in out), "runs": [x["run_id"] for x in out]}
                r = jobs.run_job("experiment", several)
        elif job == "train":
            from . import final
            opts = {"epochs": a.epochs, "batch_size": a.batch,
                    "aug_factor": a.aug, "seed": a.seed, "mirror_share": a.mirror_share}
            if a.feature_version:
                opts["feature_version"] = a.feature_version
            if a.no_canonical:
                opts["canonical"] = False
            if a.mirror_aug:
                opts["mirror_aug"] = True
            r = jobs.run_job("train", final.run_many, _categories(a.category), from_run=a.from_run, opts=opts)
        elif job == "paper":
            from . import paper
            r = jobs.run_job("paper", paper.run, a.category, test1=a.test1, test2=a.test2)
        elif job == "dataset":
            from . import dataset_export
            r = jobs.run_job("dataset", dataset_export.run, _categories(a.category))
        elif job == "export":
            from . import app_export
            r = jobs.run_job("export", app_export.run, _categories(a.category, trained_only=True))
        else:
            from . import live_window
            r = jobs.run_job("live", live_window.run, a.category, mode=a.mode, tester=a.tester,
                             tester_hand=a.tester_hand, target=a.target, trials=a.trials, hand=a.hand,
                             camera=a.camera, backend=a.backend, video=a.video, headless=a.headless,
                             screenshot=a.screenshot, recognizer=a.recognizer, capture=a.capture,
                             export=a.export)
    except PlainError as e:
        jobs.fail_before_start(job, e)
        print(f"\nERROR: {e.message}\n{e.hint}", file=sys.stderr)
        return 2
    print(r.get("message", "") if isinstance(r, dict) else "")
    return 0
