"""UI test: the real app, headless, driven by Playwright (Chromium) against a TEMPORARY dataset.

Opens every page, clicks each main button in pipeline order, waits for the background jobs,
and fails on any Streamlit exception or error banner. Screenshots of every page (1366x768,
plus a full-length version and a 1920x1080 version) go to docs/screenshots/.

Slow (several minutes). Run alone with:  python -m pytest tests/test_ui.py
"""

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

import fake_dataset as fd
from trainer.settings import PROJECT_DIR

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not fd.real_available(), reason="recorder folder with FSL-105 clips not found")]
playwright = pytest.importorskip("playwright.sync_api")

SHOTS = PROJECT_DIR / "docs" / "screenshots"
PAGES = [("", "1_dataset"), ("preprocess", "2_preprocess"), ("validation", "3_validation"), ("split", "4_split"),
         ("train", "5_train"), ("evaluate", "6_evaluate"), ("export", "7_export"), ("live", "8_live_test"),
         ("settings", "9_settings")]
JOB_TIMEOUT_S = 900


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui")
    data = fd.build(root, per_signer=1)
    settings = fd.make_settings(root, data)
    s = json.loads(settings.read_text())
    s["camera"] = 9                      # no such camera: the live test must fail with a plain message
    settings.write_text(json.dumps(s, indent=2))
    port = free_port()
    env = dict(os.environ, HUDYAT_TRAINER_SETTINGS=str(settings), TF_CPP_MIN_LOG_LEVEL="2")
    log = open(root / "streamlit.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "trainer_app.py", "--server.headless", "true",
                             "--server.port", str(port)], cwd=PROJECT_DIR, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    url = f"http://localhost:{port}"
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"{url}/_stcore/health", timeout=1).read() == b"ok":
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("the app did not start")
    yield {"url": url, "root": root, "log": root / "streamlit.log"}
    proc.kill()
    log.close()


@pytest.fixture(scope="module")
def page(app):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1366, "height": 768})
        yield pg
        browser.close()


def open_page(pg, app, path):
    pg.goto(f"{app['url']}/{path}")
    pg.wait_for_selector('[data-testid="stSidebar"] .sb-title', timeout=60000)
    settle(pg)


def settle(pg, ms=1500):
    """Wait until the script run triggered by the last action has finished (twice, to skip a gap)."""
    for _ in range(2):
        pg.wait_for_timeout(400)
        try:
            pg.wait_for_selector('[data-test-script-state="notRunning"]', timeout=60000)
        except Exception:  # noqa: BLE001
            pass
    pg.wait_for_timeout(ms)


def assert_clean(pg, app, where):
    exc = pg.locator('[data-testid="stException"]').count()
    oops = pg.get_by_text("Something went wrong on this page").count()
    if exc or oops:
        pytest.fail(f"{where}: Streamlit exception on the page\n" + app["log"].read_text(errors="replace")[-4000:])


def click(pg, name):
    pg.get_by_role("button", name=re.compile(name)).first.click()
    settle(pg, 800)


def wait_job(pg, app, where, expect="success"):
    """Wait until no job is running (sidebar) and the page shows the job's result banner."""
    t0 = time.time()
    while time.time() - t0 < JOB_TIMEOUT_S:
        pg.wait_for_timeout(1500)
        if pg.locator(".sb-job").count() == 0 and pg.locator(f".hb-{expect}").count() > 0:
            settle(pg)
            assert_clean(pg, app, where)
            return pg.locator(f".hb-{expect}").first.inner_text()
        if expect == "success" and pg.locator(".sb-job").count() == 0 and pg.locator(".hb-error").count():
            pytest.fail(f"{where}: job failed: {pg.locator('.hb-error').first.inner_text()}")
    pytest.fail(f"{where}: job did not finish in {JOB_TIMEOUT_S} s")


def shoot(pg, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    pg.screenshot(path=str(SHOTS / f"{name}.png"))
    h = pg.evaluate("document.querySelector('[data-testid=stMain]').scrollHeight")
    pg.set_viewport_size({"width": 1366, "height": max(768, int(h))})
    pg.wait_for_timeout(700)
    pg.screenshot(path=str(SHOTS / f"{name}_full.png"))
    pg.set_viewport_size({"width": 1366, "height": 768})
    pg.wait_for_timeout(300)


def test_pages_open_before_anything_ran(page, app):
    for path, _ in PAGES:
        open_page(page, app, path)
        assert_clean(page, app, path or "dataset")
    # the step list is always visible: no collapse button, and the reopen button is shown if collapsed
    page.hover('[data-testid="stSidebar"]')
    assert page.locator('[data-testid="stSidebar"]').get_attribute("aria-expanded") == "true"
    assert not page.locator('[data-testid="stSidebarCollapseButton"]').is_visible()
    page.evaluate("document.querySelector('[data-testid=stSidebarCollapseButton] button').click()")
    page.wait_for_timeout(1000)
    page.locator('[data-testid="stExpandSidebarButton"]').click()
    page.wait_for_timeout(1000)
    assert page.locator('[data-testid="stSidebar"]').get_attribute("aria-expanded") == "true"
    open_page(page, app, "train")
    assert page.get_by_role("button", name="Start training").is_disabled()        # locked before Split
    assert page.get_by_text("Run step 4 (Split) first.").count() == 1


def test_preprocess(page, app):
    open_page(page, app, "preprocess")
    click(page, r"Extract new clips \(\d+\)")
    msg = wait_job(page, app, "preprocess")
    assert "Extracted" in msg
    assert page.get_by_role("button", name=re.compile(r"Extract new clips \(0\)")).is_disabled()
    shoot(page, "2_preprocess")


def test_validation(page, app):
    open_page(page, app, "validation")
    page.get_by_label("Reviewer name").fill("UI test")
    page.get_by_label("Reviewer name").press("Enter")
    settle(page)
    click(page, "^Approve$")
    page.get_by_label("Reason (needed to reject)").fill("test rejection")
    page.get_by_label("Reason (needed to reject)").press("Enter")
    settle(page)
    click(page, "^Reject$")
    assert_clean(page, app, "validation")
    shoot(page, "3_validation")
    rows = (app["root"] / "work" / "validation.csv").read_text().splitlines()
    assert len(rows) == 3 and "approved" in rows[1] and "rejected" in rows[2]


def test_split(page, app):
    open_page(page, app, "split")
    click(page, "^Run split$")
    wait_job(page, app, "split")
    assert page.get_by_text("Samples per sign and split").count() == 1
    shoot(page, "4_split")


def test_train(page, app):
    open_page(page, app, "train")
    for label, value in (("Epochs", "4"), ("Aug. factor", "1"), ("Batch", "16")):
        box = page.get_by_role("spinbutton", name=label)
        box.fill(value)
        box.press("Tab")
        settle(page, 600)
    click(page, "^Start training$")
    page.wait_for_selector(".sb-job", timeout=20000)
    page.wait_for_timeout(6000)
    shoot(page, "5_train_running")
    wait_job(page, app, "train")
    assert page.get_by_text("Saved models").count() == 1
    shoot(page, "5_train")


def test_evaluate(page, app):
    open_page(page, app, "evaluate")
    assert page.get_by_role("button", name="Evaluate on the test set").is_disabled()     # needs the tick box
    page.get_by_text("Tuning is finished; evaluate on the test set once").click()
    settle(page)
    page.get_by_role("button", name="Evaluate on the test set", exact=True).click()
    settle(page, 800)
    wait_job(page, app, "evaluate")
    assert page.locator(".verdict").count() == 1
    shoot(page, "6_evaluate")


def test_export(page, app):
    open_page(page, app, "export")
    click(page, "^Export to TF.js$")
    wait_job(page, app, "export")
    assert page.get_by_text("Check passed").count() >= 1
    shoot(page, "7_export")


def test_live_test_page(page, app):
    open_page(page, app, "live")
    click(page, "^Start live test$")
    msg = wait_job(page, app, "live", expect="error")
    assert "Could not open camera 9" in msg                                     # plain message, no crash
    shoot(page, "8_live_test")


def test_all_pages_after_pipeline_and_screenshots(page, app):
    for path, name in PAGES:
        open_page(page, app, path)
        assert_clean(page, app, name)
        page.screenshot(path=str(SHOTS / f"{name}.png"))
        if name in ("1_dataset", "9_settings"):
            shoot(page, name)
    page.set_viewport_size({"width": 1920, "height": 1080})
    (SHOTS / "1920x1080").mkdir(parents=True, exist_ok=True)
    for path, name in PAGES:
        open_page(page, app, path)
        assert_clean(page, app, name)
        page.screenshot(path=str(SHOTS / "1920x1080" / f"{name}.png"))
    page.set_viewport_size({"width": 1366, "height": 768})
