"""Command-line entry points. The app starts exactly these (python -m trainer.runner <job> ...),
so every step can also run without the UI:

  python extract.py [--all] [--category C ...]
  python split.py [--category C ...]
  python train.py --category C|all [--epochs 150 --batch 32 --aug 5 --seed 42]
  python evaluate.py --category C|all --confirm
  python export_tfjs.py --category C|all
  python live_test.py --category C [--mode test --tester T --target SIGN --trials 25]
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _categories(arg: list[str] | None, trained_only: bool = False, split_only: bool = False) -> list[str]:
    from . import dataset as ds
    from . import settings as cfg
    from .train import trained_categories
    if trained_only:
        avail = trained_categories(cfg.Paths())
    else:
        d = cfg.data_dir()
        if d is None:
            raise ds.PlainError("No dataset chosen yet.", "Choose it on the Dataset page (a folder or a .zip upload).")
        avail = ds.model_categories(ds.load_signs(d))
        if split_only:
            avail = [c for c in avail if cfg.Paths().split_json(c).exists()]
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
    elif job == "split":
        p.add_argument("--category", nargs="*")
    elif job == "train":
        p.add_argument("--category", nargs="+", required=True)
        p.add_argument("--epochs", type=int)
        p.add_argument("--batch", type=int)
        p.add_argument("--aug", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--patience", type=int)
        p.add_argument("--no-balance", action="store_true", help="do not weight rare signs more")
    elif job == "evaluate":
        p.add_argument("--category", nargs="+", required=True)
        p.add_argument("--confirm", action="store_true", help="tuning is finished; evaluate on S6/S7 once")
    elif job == "export":
        p.add_argument("--category", nargs="+", required=True)
    elif job == "live":
        p.add_argument("--category", required=True)
        p.add_argument("--mode", choices=["free", "test"], default="free")
        p.add_argument("--tester", default="")
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
        elif job == "split":
            from . import split
            r = jobs.run_job("split", split.run, categories=a.category or None)
        elif job == "train":
            from . import train
            r = jobs.run_job("train", train.run, _categories(a.category, split_only=True), epochs=a.epochs, batch_size=a.batch,
                             aug_factor=a.aug, seed=a.seed, patience=a.patience,
                             balance=not a.no_balance)
        elif job == "evaluate":
            if not a.confirm:
                raise PlainError("Evaluation on S6/S7 must be confirmed.",
                                 "Tick 'Tuning is finished' (app) or pass --confirm (command line).")
            from . import evaluate
            r = jobs.run_job("evaluate", evaluate.run, _categories(a.category, trained_only=True))
        elif job == "export":
            from . import tfjs_export
            r = jobs.run_job("export", tfjs_export.run, _categories(a.category, trained_only=True))
        else:
            from . import live_window
            r = jobs.run_job("live", live_window.run, a.category, mode=a.mode, tester=a.tester, target=a.target,
                             trials=a.trials, camera=a.camera, backend=a.backend, video=a.video,
                             headless=a.headless)
    except PlainError as e:
        jobs.fail_before_start(job, e)
        print(f"\nERROR: {e.message}\n{e.hint}", file=sys.stderr)
        return 2
    print(r.get("message", "") if isinstance(r, dict) else "")
    return 0
