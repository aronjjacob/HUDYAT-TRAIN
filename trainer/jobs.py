"""One long job at a time: lock file, background process, live log, status that survives a refresh.

The app (or a CLI script) starts a job; the job runs in its own process, writes
  work/job.lock            while it runs (holds job name, pid, token)
  work/jobs/<job>.log      the full log
  work/jobs/status.json    running / done / error (+ plain-language message and hint) of the last job
  work/jobs/status_<job>.json  the same, kept per job (so each page shows its own last run)
  work/jobs/progress.json  fraction + text for the progress bar
Nothing lives in the browser session, so a page refresh shows the same state.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from . import settings as cfg
from .dataset import PlainError

JOBS = ("extract", "split", "train", "evaluate", "export", "live")
JOB_TITLES = {"extract": "Extraction", "split": "Split", "train": "Training", "evaluate": "Evaluation",
              "export": "Export", "live": "Live test"}
TOKEN_ENV = "HUDYAT_JOB_TOKEN"
STALE_NO_PID_S = 60


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259   # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows: reader has it open; retry
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict | None:
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.05)
    return None


# ----------------------------------------------------------------------------
# Lock
# ----------------------------------------------------------------------------
def read_lock(paths: cfg.Paths) -> dict | None:
    """The current lock, or None. Stale locks (dead process) are removed."""
    info = _read_json(paths.lock)
    if info is None:
        if paths.lock.exists():          # unreadable/half-written: treat as held briefly
            age = time.time() - paths.lock.stat().st_mtime
            if age > STALE_NO_PID_S:
                paths.lock.unlink(missing_ok=True)
                return None
            return {"job": "?", "pid": None}
        return None
    pid = info.get("pid")
    if pid and not pid_alive(pid):
        _mark_crashed(paths, info)
        paths.lock.unlink(missing_ok=True)
        return None
    if not pid and time.time() - float(info.get("t", 0)) > STALE_NO_PID_S:
        paths.lock.unlink(missing_ok=True)
        return None
    return info


def _mark_crashed(paths: cfg.Paths, info: dict) -> None:
    st = read_status(paths)
    if st and st.get("state") == "running" and st.get("token") == info.get("token"):
        st.update(state="error", finished=_now(),
                  message=f"The {JOB_TITLES.get(info.get('job'), 'job')} stopped unexpectedly "
                          "(the window was closed or the computer went to sleep).",
                  hint="Start it again. Finished parts are kept.")
        save_status(paths, st)


def dataset_label(d) -> str:
    name = cfg.import_name(d)
    return name or "recordings"


def acquire_lock(paths: cfg.Paths, job: str, token: str, pid: int | None) -> bool:
    paths.work.mkdir(parents=True, exist_ok=True)
    read_lock(paths)                      # clears a stale lock
    try:
        fd = os.open(paths.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"job": job, "pid": pid, "token": token, "t": time.time(), "started": _now(),
                   "dataset": dataset_label(cfg.data_dir())}, f)
    return True


def _update_lock_pid(paths: cfg.Paths, pid: int) -> None:
    info = _read_json(paths.lock) or {}
    info["pid"] = pid
    _write_json(paths.lock, info)


def release_lock(paths: cfg.Paths, token: str) -> None:
    info = _read_json(paths.lock)
    if info and info.get("token") == token:
        paths.lock.unlink(missing_ok=True)


def busy(paths: cfg.Paths) -> str | None:
    """Name of the running job, or None."""
    info = read_lock(paths)
    return info.get("job") if info else None


# ----------------------------------------------------------------------------
# Status / progress
# ----------------------------------------------------------------------------
def read_status(paths: cfg.Paths, job: str | None = None) -> dict | None:
    """Status of the last job, or of the last run of `job`."""
    return _read_json(paths.jobs / (f"status_{job}.json" if job else "status.json"))


def save_status(paths: cfg.Paths, status: dict) -> None:
    _write_json(paths.jobs / "status.json", status)
    if status.get("job"):
        _write_json(paths.jobs / f"status_{status['job']}.json", status)


def read_progress(paths: cfg.Paths) -> dict:
    return _read_json(paths.jobs / "progress.json") or {}


def log_path(paths: cfg.Paths, job: str) -> Path:
    return paths.jobs / f"{job}.log"


def read_log(paths: cfg.Paths, job: str, tail_lines: int | None = None) -> str:
    p = log_path(paths, job)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if tail_lines:
        text = "\n".join(text.splitlines()[-tail_lines:])
    return text


class JobContext:
    """Passed to job functions: progress + log. Works the same from the CLI."""

    def __init__(self, paths: cfg.Paths, job: str) -> None:
        self.paths = paths
        self.job = job

    def progress(self, fraction: float, text: str = "", **extra) -> None:
        _write_json(self.paths.jobs / "progress.json",
                    {"job": self.job, "fraction": max(0.0, min(1.0, float(fraction))), "text": text,
                     "t": time.time(), **extra})

    def log(self, msg: str) -> None:
        print(msg, flush=True)


def run_job(job: str, func, *args, paths: cfg.Paths | None = None, **kwargs):
    """Run func(ctx, *args, **kwargs) under the lock. Used by the CLI scripts and the runner."""
    paths = paths or cfg.Paths()
    token = os.environ.get(TOKEN_ENV)
    lock = _read_json(paths.lock)
    owns = bool(token) and lock is not None and lock.get("token") == token
    if owns:
        _update_lock_pid(paths, os.getpid())
    else:
        token = uuid.uuid4().hex
        if not acquire_lock(paths, job, token, os.getpid()):
            other = busy(paths) or "another job"
            raise PlainError(f"{JOB_TITLES.get(other, other)} is still running.",
                             "Wait until it has finished, then try again.")
    status = {"job": job, "state": "running", "token": token, "pid": os.getpid(), "started": _now(),
              "finished": None, "message": "", "hint": "", "log": str(log_path(paths, job))}
    save_status(paths, status)
    ctx = JobContext(paths, job)
    ctx.progress(0.0, "Starting...")
    try:
        result = func(ctx, *args, **kwargs)
        status.update(state="done", finished=_now(),
                      message=(result.get("message") if isinstance(result, dict) else "") or "Finished.")
        ctx.progress(1.0, "Finished")
        return result
    except PlainError as e:
        print("ERROR:", e.message, "\n", traceback.format_exc(), flush=True)
        status.update(state="error", finished=_now(), message=e.message, hint=e.hint)
        raise
    except BaseException as e:  # noqa: BLE001
        print(traceback.format_exc(), flush=True)
        status.update(state="error", finished=_now(),
                      message=f"{JOB_TITLES.get(job, job)} failed: {type(e).__name__}: {e}",
                      hint="Open 'Full log' below for details and send it to the developers.")
        raise
    finally:
        save_status(paths, status)
        release_lock(paths, token)


def fail_before_start(job: str, error: PlainError, paths: cfg.Paths | None = None) -> None:
    """A job started by the app stopped with a plain error before run_job began: record it, free the lock."""
    paths = paths or cfg.Paths()
    token = os.environ.get(TOKEN_ENV)
    st = read_status(paths)
    if token and st and st.get("token") == token and st.get("state") == "running":
        st.update(state="error", finished=_now(), message=error.message, hint=error.hint)
        save_status(paths, st)
        release_lock(paths, token)


# ----------------------------------------------------------------------------
# Starting a job from the app
# ----------------------------------------------------------------------------
def python_exe() -> str:
    """The venv's python (pythonw would hide errors; the console is hidden by flags instead)."""
    return sys.executable


def start_background(job: str, args: list[str], paths: cfg.Paths | None = None,
                     show_window: bool = False) -> tuple[bool, str]:
    """Start `python -m trainer.runner <job> <args>` detached. (ok, message)."""
    paths = paths or cfg.Paths()
    token = uuid.uuid4().hex
    if not acquire_lock(paths, job, token, None):
        other = busy(paths) or "another job"
        return False, f"{JOB_TITLES.get(other, other)} is still running."
    paths.jobs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{TOKEN_ENV: token, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                              "TF_CPP_MIN_LOG_LEVEL": "2"})
    d = cfg.data_dir()
    if d is not None:
        env[cfg.DATA_DIR_ENV] = str(d)            # the job stays on this dataset even if the app switches
    save_status(paths,
                {"job": job, "state": "running", "token": token, "pid": None, "started": _now(),
                 "finished": None, "message": "", "hint": "", "log": str(log_path(paths, job))})
    _write_json(paths.jobs / "progress.json", {"job": job, "fraction": 0.0, "text": "Starting...", "t": time.time()})
    log = open(log_path(paths, job), "w", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if not show_window:
            flags |= subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.Popen([python_exe(), "-m", "trainer.runner", job, *args], cwd=str(cfg.PROJECT_DIR),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, creationflags=flags)
    except OSError as e:
        release_lock(paths, token)
        return False, f"Could not start: {e}"
    finally:
        log.close()
    _update_lock_pid(paths, proc.pid)
    return True, "Started."


def cancel(paths: cfg.Paths | None = None) -> bool:
    paths = paths or cfg.Paths()
    info = read_lock(paths)
    if not info or not info.get("pid"):
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(info["pid"]), "/T", "/F"], capture_output=True)
    else:
        os.kill(int(info["pid"]), 9)
    st = read_status(paths) or {}
    st.update(state="error", finished=_now(), message=f"{JOB_TITLES.get(info.get('job'), 'Job')} was stopped.",
              hint="Finished parts are kept. Start it again when you are ready.")
    save_status(paths, st)
    paths.lock.unlink(missing_ok=True)
    return True
