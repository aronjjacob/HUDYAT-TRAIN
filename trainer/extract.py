"""Step 2: landmark extraction for kept S1-S7 clips (incremental; all categories or a chosen few).

The static A-Y test set (alphabet_static_test) is extracted only when it is chosen explicitly.
The preprocessing values (trim margin / minimum length, MediaPipe confidences) are the recorder's
unless adjusted on the Preprocess page; the values used are stored with every sample ('preprocess').
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import settings as cfg


_IMAGE_LM: dict = {}          # one IMAGE-mode landmarker per helper process (loading it per photo is slow)


def _image_landmarker(model: str, params: dict, pid: str):
    from .landmarks import create_image_landmarker
    if pid not in _IMAGE_LM:
        for lm in _IMAGE_LM.values():
            lm.close()
        _IMAGE_LM.clear()
        _IMAGE_LM[pid] = create_image_landmarker(Path(model), params)
    return _IMAGE_LM[pid]


def _worker(job: dict) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from .landmarks import IMAGE_EXT, extract_clip, extract_image
    row = dict(job["row"])
    try:
        if Path(job["video"]).suffix.lower() in IMAGE_EXT:
            r = extract_image(Path(job["video"]), Path(job["model"]), Path(job["data_dir"]), job["params"],
                              landmarker=_image_landmarker(job["model"], job["params"], job["params_id"]))
        else:
            r = extract_clip(Path(job["video"]), job["fps_hint"], Path(job["model"]), Path(job["data_dir"]),
                             job["params"])
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, r.sample.astype(np.float32))
        row.update(fps=f"{r.fps:.3f}", n_frames=r.n_frames, trim_start=r.trim_start, trim_end=r.trim_end,
                   frames_kept=r.frames_kept, hand_detect_rate=f"{r.hand_detect_rate:.3f}",
                   both_hands_rate=f"{r.both_hands_rate:.3f}", width=r.width, height=r.height,
                   face_detect_rate="" if r.face_detect_rate is None else f"{r.face_detect_rate:.3f}",
                   status=ds.STATUS_OK if r.hand_detect_rate >= job["thr"] else ds.STATUS_LOW)
        row["_dups"] = r.duplicate_frames
        row["preprocess"] = job["params_id"]
        row["format"] = ds.SAMPLE_FORMAT
    except Exception as e:  # noqa: BLE001
        row.update(status=ds.STATUS_FAILED, _error=f"{type(e).__name__}: {e}")
    row["extracted_at"] = datetime.now().isoformat(timespec="seconds")
    return row


HELPER_MB = 450              # memory one helper process needs (MediaPipe + OpenCV), with room to spare


def free_memory_gb() -> float:
    """Free physical memory (Windows; elsewhere a cautious guess)."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1024 ** 3
    except Exception:  # noqa: BLE001
        return 2.0


def auto_workers() -> int:
    """Helpers from the CPU count AND the free memory (an 8 GB laptop with a browser open gets 1-2)."""
    by_cpu = max(1, min(4, (os.cpu_count() or 2) // 2))
    by_mem = max(1, int((free_memory_gb() * 1024 - 300) // HELPER_MB))
    return min(by_cpu, by_mem)


def plan(paths: cfg.Paths, data_dir: Path, reextract_all: bool = False,
         categories: list[str] | None = None) -> tuple[list[dict], dict[str, dict]]:
    """(clips to extract, existing index rows by file). categories=None: every category except the static test."""
    static = bool(categories) and any(c in cfg.EXCLUDED_CATEGORIES for c in categories)
    kept = ds.kept_rows(data_dir, include_static=static)
    if categories:
        kept = kept[kept["category"].isin(categories)]
    existing = {r["file"]: r for r in ds.read_index(paths).to_dict("records")}
    todo = []
    for _, m in kept.iterrows():
        rel = ds.npy_rel_path(m)
        old = existing.get(rel)
        if (not reextract_all and old is not None and old.get("status") != ds.STATUS_FAILED
                and old.get("format") == ds.SAMPLE_FORMAT and (paths.work / rel).exists()):
            continue
        todo.append({"file": rel, "category": m["category"], "sign_id": m["sign_id"], "signer": m["signer"],
                     "take": m.get("take", ""), "source_file": m["file_path"], "split": m.get("split", ""),
                     "_fps": m.get("fps_actual", "")})
    return todo, existing


def run(ctx, reextract_all: bool = False, workers: int | None = None, categories: list[str] | None = None) -> dict:
    s = cfg.load_settings()
    data_dir = cfg.data_dir(s)
    problems = cfg.check_data_dir(data_dir)
    if problems:
        raise ds.PlainError(problems[0], "Choose the dataset under Dataset in the sidebar, or add a folder or a .zip on the Data page.")
    if cfg.landmarks_only(data_dir):        # no videos: re-extracting would replace good samples with failed ones
        raise ds.PlainError("This dataset has landmarks only, no videos.",
                            "Its samples are ready; go to Test the model.")
    if not cfg.MODEL_PATH.exists():
        raise ds.PlainError("The MediaPipe hand model models/hand_landmarker.task is missing.",
                            "Run 'SETUP (run once).bat' again; it copies the model from the recorder.")
    from .landmarks import face_model_path
    if not face_model_path(data_dir).exists():
        raise ds.PlainError("The MediaPipe face model models/face_landmarker.task is missing.",
                            "Download it as in Setup step 5 of the recorder's README; the trainer uses the "
                            "recorder's copy.")
    paths = cfg.Paths(s)
    signs = ds.load_signs(data_dir)
    known = ds.extract_categories(signs, include_static=True)
    if categories:
        categories = [c for c in categories if c != cfg.NONE_CATEGORY]          # none clips are never extracted
        if not categories:
            raise ds.PlainError("'none' is not a sign category: the models only learn signs.",
                                "Pick a sign category, or all.")
    unknown = [c for c in (categories or []) if c not in known]
    if unknown:
        raise ds.PlainError(f"Unknown category: {', '.join(unknown)}.", "Pick a category from the list.")
    todo, existing = plan(paths, data_dir, reextract_all, categories)
    thr = float(s.get("min_hand_detect_rate", 0.8))
    from . import recorder
    params = recorder.preprocess_params(s, data_dir)
    pid = recorder.params_id(params)
    ctx.log(f"Data folder: {data_dir}")
    ctx.log(f"Categories: {', '.join(categories) if categories else 'all (except the static test set)'}")
    ctx.log(f"Preprocessing: {pid}" + ("  (recorder values)" if recorder.is_default(params, data_dir)
                                       else "  (ADJUSTED: the browser app must use the same values)"))
    ctx.log(f"{len(todo)} clip(s) to extract ({'re-extract all' if reextract_all else 'new clips only'}); "
            f"{len(existing)} already in the index.")
    if not todo:
        ctx.progress(1.0, "Nothing new to extract")
        return {"message": "Nothing new to extract: every chosen clip is already done.", "extracted": 0}
    workers = workers or auto_workers()
    ctx.log(f"Using {workers} helper process(es) ({free_memory_gb():.1f} GB memory free).")
    jobs = []
    for t in todo:
        try:
            fps_hint = float(t["_fps"]) if t["_fps"] else None
        except ValueError:
            fps_hint = None
        row = {k: v for k, v in t.items() if not k.startswith("_")}
        jobs.append({"row": row, "video": str(data_dir / t["source_file"]), "fps_hint": fps_hint,
                     "out": str(paths.work / t["file"]), "model": str(cfg.MODEL_PATH),
                     "data_dir": str(data_dir), "thr": thr, "params": params, "params_id": pid})
    index = dict(existing)
    done = failed = low = 0
    t0 = time.time()

    def eta(n: int) -> str:
        if n < 3:
            return ""
        left = (time.time() - t0) / n * (len(jobs) - n)
        return f"  ·  about {round(left / 60)} min left" if left >= 90 else "  ·  less than 2 min left"
    dup_lines = []
    no_face: list[str] = []
    face_thr = 0.8

    def handle(r: dict) -> None:
        nonlocal done, failed, low
        done += 1
        if r["status"] == ds.STATUS_FAILED:
            failed += 1
            ctx.log(f"FAILED {r['source_file']}: {r.get('_error')}")
        elif r["status"] == ds.STATUS_LOW:
            low += 1
        if r.get("face_detect_rate") not in (None, "") and float(r["face_detect_rate"]) < face_thr:
            no_face.append(r["source_file"])
            ctx.log(f"Face found in only {float(r['face_detect_rate']):.0%} of the sign: {r['source_file']}")
        for fr in r.pop("_dups", []) or []:
            dup_lines.append(f"{r['source_file']}\tframe {fr}")
        r.pop("_error", None)
        index[r["file"]] = r
        if done % 20 == 0 or done == len(jobs):
            ds.write_index(cfg.Paths(s), sorted(index.values(), key=lambda x: x["file"]))
        ctx.progress(done / len(jobs), f"{done} / {len(jobs)} clips  ·  {r['sign_id']} {r['signer']}{eta(done)}",
                     done=done, total=len(jobs))

    pending = list(jobs)
    while pending:
        if workers == 1:
            for j in pending:
                handle(_worker(j))
            break
        finished: set[str] = set()
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_worker, j): j for j in pending}
                for fut in as_completed(futs):
                    r = fut.result()
                    finished.add(futs[fut]["out"])
                    handle(r)
            break
        except BrokenProcessPool:
            # a helper was killed (usually: the computer ran out of memory). Keep what is done, go on with fewer.
            pending = [j for j in pending if j["out"] not in finished]
            workers = max(1, workers // 2)
            ds.write_index(paths, sorted(index.values(), key=lambda x: x["file"]))
            ctx.log(f"A helper process stopped (low memory?). Continuing the remaining {len(pending)} with "
                    f"{workers} helper(s); finished clips are kept.")
    ds.write_index(paths, sorted(index.values(), key=lambda x: x["file"]))
    if dup_lines:
        paths.duplicate_log.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.duplicate_log, "a", encoding="utf-8") as f:
            f.write("\n".join(dup_lines) + "\n")
        ctx.log(f"{len(dup_lines)} frame(s) had two hands with the same label (logged).")
    msg = f"Extracted {ds.plural(done - failed, 'clip')}"
    if low:
        msg += f", {low} with low hand detection"
    if no_face:
        msg += f", {len(no_face)} with the face found in < {face_thr:.0%} of the sign; consider retaking them"
    if failed:
        msg += f", {failed} failed; see the log"
    ctx.log(msg)
    return {"message": msg + ".", "extracted": done - failed, "failed": failed, "low": low, "no_face": len(no_face)}
