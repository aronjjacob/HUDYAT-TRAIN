"""Job lock: one long job at a time, stale locks cleared, lock released on error, status persisted."""

import json
import subprocess
import sys

import pytest

from trainer import jobs
from trainer import settings as cfg
from trainer.dataset import PlainError


@pytest.fixture
def paths(tmp_path, monkeypatch):
    s = tmp_path / "trainer_settings.json"
    s.write_text(json.dumps({"work_dir": str(tmp_path / "work")}))
    monkeypatch.setenv("HUDYAT_TRAINER_SETTINGS", str(s))
    monkeypatch.delenv(jobs.TOKEN_ENV, raising=False)
    return cfg.Paths()


def test_second_job_refused_while_first_holds_lock(paths):
    assert jobs.acquire_lock(paths, "train", "tok1", None)
    assert jobs.busy(paths) == "train"
    assert not jobs.acquire_lock(paths, "extract", "tok2", None)
    with pytest.raises(PlainError, match="Model training is still running"):
        jobs.run_job("extract", lambda ctx: None, paths=paths)
    jobs.release_lock(paths, "wrong-token")
    assert jobs.busy(paths) == "train"
    jobs.release_lock(paths, "tok1")
    assert jobs.busy(paths) is None


def test_stale_lock_of_dead_process_is_cleared(paths):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert jobs.acquire_lock(paths, "train", "tok", p.pid)
    jobs._write_json(paths.jobs / "status.json", {"job": "train", "state": "running", "token": "tok"})
    assert jobs.busy(paths) is None                     # dead pid -> lock removed
    st = jobs.read_status(paths)
    assert st["state"] == "error" and "stopped unexpectedly" in st["message"]
    assert jobs.acquire_lock(paths, "extract", "tok2", None)


def test_run_job_success_releases_lock_and_records_status(paths):
    def work(ctx, n):
        assert jobs.busy(paths) == "split"
        ctx.progress(0.5, "half")
        return {"message": f"did {n}"}
    assert jobs.run_job("split", work, 3, paths=paths) == {"message": "did 3"}
    assert jobs.busy(paths) is None
    st = jobs.read_status(paths)
    assert st["state"] == "done" and st["message"] == "did 3"
    assert jobs.read_progress(paths)["fraction"] == 1.0


def test_run_job_error_releases_lock_with_plain_message(paths):
    def bad(ctx):
        raise PlainError("No split yet.", "Run Split first.")
    with pytest.raises(PlainError):
        jobs.run_job("train", bad, paths=paths)
    assert jobs.busy(paths) is None
    st = jobs.read_status(paths)
    assert st["state"] == "error" and st["message"] == "No split yet." and st["hint"] == "Run Split first."

    def crash(ctx):
        raise ZeroDivisionError("x")
    with pytest.raises(ZeroDivisionError):
        jobs.run_job("train", crash, paths=paths)
    assert jobs.busy(paths) is None
    assert "ZeroDivisionError" in jobs.read_status(paths)["message"]


def test_background_job_runs_and_survives_as_files(paths, monkeypatch):
    """start_background -> separate process; the app only reads files (so a refresh shows the same)."""
    import time
    ok, msg = jobs.start_background("experiment", ["--test", "random", "--category", "colors"], paths=paths)
    assert ok, msg
    assert jobs.busy(paths) == "experiment"
    ok2, msg2 = jobs.start_background("extract", [], paths=paths)
    assert not ok2 and "still running" in msg2
    for _ in range(600):
        if jobs.busy(paths) is None:
            break
        time.sleep(0.1)
    assert jobs.busy(paths) is None
    st = jobs.read_status(paths)
    assert st["state"] == "error"                     # no data folder configured -> plain error
    assert "No dataset chosen" in st["message"]
    assert jobs.log_path(paths, "experiment").exists()
