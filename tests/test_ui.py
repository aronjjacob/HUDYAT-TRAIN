"""UI test: the real app, headless, driven by Playwright (Chromium) against a TEMPORARY dataset.

Opens every page, clicks each main button in pipeline order (Preprocess, Data viewer, Test 1 / Test 2, Final
model, Export, Live, Paper), waits for the background jobs, and fails on any Streamlit exception or error banner.
Screenshots of every page (1366x768 with a full-length version, and 1920x1080) go to docs/screenshots/v2/.

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
              pytest.mark.skipif(not fd.real_available(), reason="recorder folder with S1-S3 clips not found")]
playwright = pytest.importorskip("playwright.sync_api")

SHOTS = PROJECT_DIR / "docs" / "screenshots" / "v2"
PAGES = [("", "1_data"), ("preprocess", "2_preprocess"), ("test", "3_test"), ("final", "4_final"),
         ("live", "5_live"), ("paper", "6_paper"), ("settings", "7_settings")]
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


def tab(pg, name):
    pg.get_by_role("tab", name=name, exact=True).click()
    settle(pg, 600)


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


def set_number(pg, label, value):
    box = pg.get_by_role("spinbutton", name=label)
    box.fill(str(value))
    box.press("Tab")
    settle(pg, 600)


def test_pages_open_before_anything_ran(page, app):
    for path, _ in PAGES:
        open_page(page, app, path)
        assert_clean(page, app, path or "data")
    # the step list is always visible: no collapse button, and the reopen button is shown if collapsed
    page.hover('[data-testid="stSidebar"]')
    assert page.locator('[data-testid="stSidebar"]').get_attribute("aria-expanded") == "true"
    assert not page.locator('[data-testid="stSidebarCollapseButton"]').is_visible()
    page.evaluate("document.querySelector('[data-testid=stSidebarCollapseButton] button').click()")
    page.wait_for_timeout(1000)
    page.locator('[data-testid="stExpandSidebarButton"]').click()
    page.wait_for_timeout(1000)
    assert page.locator('[data-testid="stSidebar"]').get_attribute("aria-expanded") == "true"
    side = page.locator('[data-testid="stSidebar"]').inner_text()
    # the usual order, numbered, with the next step named; nothing is locked
    assert "01" in side and "Data" in side and "06" in side and "Results for the paper" in side
    assert "Next:" in side and "locked" not in side
    assert "Train model" in side
    open_page(page, app, "test")
    assert page.get_by_text("No landmarks yet.").count() == 1                      # says what it needs; still opens
    open_page(page, app, "")
    assert page.get_by_text("1 left-handed", exact=False).count() >= 1


def test_preprocess(page, app):
    open_page(page, app, "preprocess")
    assert page.get_by_role("tab", name="Categories", exact=True).get_attribute("aria-selected") == "true"
    click(page, r"^Extract \d+$")                                              # the category card's own button
    msg = wait_job(page, app, "preprocess")
    assert "Extracted" in msg
    open_page(page, app, "preprocess")
    assert page.get_by_role("button", name=re.compile(r"^Extract \d+$")).count() == 0     # nothing left to extract
    assert page.get_by_role("button", name="Re-extract", exact=True).count() >= 1
    click(page, "^Export$")
    assert "Exported" in wait_job(page, app, "dataset export")
    assert list((app["root"] / "work" / "datasets").glob("*.zip"))
    open_page(page, app, "preprocess")
    tab(page, "Check the results")
    assert page.get_by_text("Hands per signer").count() == 1
    shoot(page, "2_preprocess")


def test_data_viewer_reject_and_restore(page, app):
    open_page(page, app, "")
    page.get_by_role("tab", name="Review clips").click()
    settle(page)
    page.get_by_label("Reason").fill("UI test rejection")
    page.get_by_label("Reason").press("Enter")
    settle(page)
    click(page, "^Reject$")
    assert page.get_by_text("Rejected: left out of every test and model.").count() == 1
    click(page, "^Restore$")
    assert_clean(page, app, "data viewer")
    rows = (app["root"] / "work" / "validation.csv").read_text().splitlines()
    assert len(rows) == 3 and "rejected" in rows[1] and "restored" in rows[2]
    shoot(page, "1_data")


def test_test_page_runs_both_tests(page, app):
    open_page(page, app, "test")
    page.get_by_text("Settings for this run").click()
    settle(page, 600)
    set_number(page, "Max epochs", 6)
    set_number(page, "Extra copies", 1)
    set_number(page, "Batch", 16)
    click(page, "^Run Test 2$")
    page.wait_for_selector(".sb-job", timeout=20000)
    page.wait_for_timeout(12000)
    shoot(page, "3_test_running")
    wait_job(page, app, "test 2")
    click(page, "^Run Test 1$")
    wait_job(page, app, "test 1")
    open_page(page, app, "test")
    assert page.locator(".verdict").count() == 1
    assert page.get_by_text("Compare runs").count() == 1
    assert page.get_by_text("Left-hand check").count() == 1
    shoot(page, "3_test")


def train_quickly(pg):
    pg.get_by_text("Training settings").click()
    settle(pg, 600)
    set_number(pg, "Max epochs", 6)
    set_number(pg, "Extra copies", 1)
    set_number(pg, "Batch", 16)


def test_final_model_and_export(page, app):
    open_page(page, app, "final")
    train_quickly(page)
    assert page.get_by_text("Use the settings of a test run").count() == 1        # inside Training settings
    click(page, r"^Train model · v1$")
    wait_job(page, app, "train model")
    open_page(page, app, "final")
    tab(page, "Export for the app")
    click(page, "^Export for the app$")
    wait_job(page, app, "export")
    open_page(page, app, "final")
    assert page.get_by_text("Exported and checked", exact=False).count() == 1
    shoot(page, "4_final")


def test_live_page_fails_plainly_without_a_camera(page, app):
    open_page(page, app, "live")
    click(page, "^Start$")
    msg = wait_job(page, app, "live", expect="error")
    assert "Could not open camera 9" in msg                                     # plain message, no crash
    shoot(page, "5_live")


def test_paper(page, app):
    open_page(page, app, "paper")
    click(page, "^Generate figures$")
    wait_job(page, app, "paper")
    open_page(page, app, "paper")
    assert page.get_by_role("button", name="Download everything as .zip").count() == 1
    assert page.get_by_text("Figure 1", exact=False).count() >= 1
    shoot(page, "6_paper")


def test_all_pages_after_pipeline_and_screenshots(page, app):
    for path, name in PAGES:
        open_page(page, app, path)
        assert_clean(page, app, name)
        page.screenshot(path=str(SHOTS / f"{name}.png"))
        if name == "7_settings":
            shoot(page, name)
    page.set_viewport_size({"width": 1920, "height": 1080})
    (SHOTS / "1920x1080").mkdir(parents=True, exist_ok=True)
    for path, name in PAGES:
        open_page(page, app, path)
        assert_clean(page, app, name)
        page.screenshot(path=str(SHOTS / "1920x1080" / f"{name}.png"))
    page.set_viewport_size({"width": 1366, "height": 768})
