"""UI test of the upload path: Data page -> upload a .zip of photos -> Scan -> Import -> Preprocess ->
Test 1 (Test 2 is greyed out for photos) -> Train model -> Export, all by clicking in the real app (headless Chromium).

Uses 3 letters x 30 photos of the alphabet-static dataset from Downloads (skipped when it is not there).
Screenshots: docs/screenshots/v2/upload_*.png
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pytest

from test_ui import assert_clean, click, free_port, open_page, set_number, settle, shoot, tab, train_quickly, wait_job
from trainer.settings import PROJECT_DIR

PHOTOS = Path.home() / "Downloads" / "alphabet-static" / "Collated"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not PHOTOS.exists(), reason="alphabet-static photos not found")]
playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def upapp(tmp_path_factory):
    root = tmp_path_factory.mktemp("uiup")
    zp = root / "alphabet_mini.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for letter in "ABC":
            for f in sorted((PHOTOS / letter).glob("*.jpg"))[:30]:
                z.write(f, f"Collated/{letter}/{f.name}")
    settings = root / "trainer_settings.json"
    settings.write_text(json.dumps({"work_dir": str(root / "work"), "camera": 9, "backend": "msmf"}))
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
    yield {"url": url, "root": root, "zip": zp, "log": root / "streamlit.log", "settings": settings}
    proc.kill()
    log.close()


@pytest.fixture(scope="module")
def pg(upapp):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        yield page
        browser.close()


def test_scan_then_import_zip(pg, upapp):
    open_page(pg, upapp, "")
    add = pg.get_by_role("tab", name="Add or combine datasets")
    if add.count():                                   # a dataset is already in use (recordings found)
        add.click()
        settle(pg, 600)
    pg.locator('[data-testid="stFileUploaderDropzoneInput"]').first.set_input_files(str(upapp["zip"]))
    settle(pg, 3000)
    assert_clean(pg, upapp, "scan")
    assert pg.get_by_text("none, photos").count() == 1          # the scan shows what was found first
    assert not (upapp["root"] / "work" / "imports" / "alphabet_mini").exists()   # nothing copied yet
    shoot(pg, "upload_1_scan")
    click(pg, "^Import$")
    settle(pg, 3000)
    assert_clean(pg, upapp, "data after import")
    assert pg.get_by_text("Imported 'alphabet_mini'", exact=False).count() == 1
    assert "alphabet_mini" in pg.locator("body").inner_text()
    s = json.loads(upapp["settings"].read_text())
    assert "imports" in s["data_dir"] and "alphabet_mini" in s["data_dir"]
    shoot(pg, "upload_2_data")


def test_preprocess_and_test_1_only(pg, upapp):
    open_page(pg, upapp, "preprocess")
    click(pg, r"^Extract 90$")
    assert "Extracted" in wait_job(pg, upapp, "preprocess")
    open_page(pg, upapp, "test")
    assert pg.get_by_role("button", name="Run Test 2", exact=True).is_disabled()
    assert pg.get_by_text("Photos have no signers", exact=False).count() >= 1
    pg.get_by_text("Settings for this run").click()
    settle(pg, 600)
    set_number(pg, "Max epochs", 6)
    set_number(pg, "Extra copies", 1)
    set_number(pg, "Batch", 16)
    click(pg, "^Run Test 1$")
    wait_job(pg, upapp, "test 1")
    open_page(pg, upapp, "test")
    assert pg.locator(".verdict").count() == 1
    shoot(pg, "upload_3_test")


def test_final_model_and_export(pg, upapp):
    open_page(pg, upapp, "final")
    train_quickly(pg)
    click(pg, r"^Train model · v1$")
    wait_job(pg, upapp, "train model")
    open_page(pg, upapp, "final")
    tab(pg, "Export for the app")
    click(pg, "^Export for the app$")
    wait_job(pg, upapp, "export")
    open_page(pg, upapp, "final")
    assert pg.get_by_text("Exported and checked", exact=False).count() == 1
    out = next((upapp["root"] / "work" / "imports" / "alphabet_mini" / "work" / "exports").iterdir())
    assert out.name.endswith("-dabest-alphabet-mini-v1")
    assert (out / "model.json").exists() and (out / "weights.bin").exists() and (out / "parity.json").exists()
    shoot(pg, "upload_4_final")
