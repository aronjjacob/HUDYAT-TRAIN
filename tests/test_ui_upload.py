"""UI test of the upload path: Dataset page -> upload a .zip of photos -> Preprocess -> Split -> Train ->
Evaluate -> Export -> Live Test, all by clicking in the real app (headless Chromium).

Uses 3 letters x 30 photos of the alphabet-static dataset from Downloads (skipped when it is not there).
Screenshots: docs/screenshots/upload_*.png
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

from test_ui import SHOTS, assert_clean, click, free_port, open_page, settle, shoot, wait_job
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


def test_upload_zip_on_dataset_page(pg, upapp):
    open_page(pg, upapp, "")
    pg.get_by_text("Add a dataset (upload a .zip or a folder)").click()
    settle(pg, 600)
    pg.locator('[data-testid="stFileUploaderDropzoneInput"]').first.set_input_files(str(upapp["zip"]))
    settle(pg, 2000)
    click(pg, "^Upload and use$")
    settle(pg, 3000)
    assert_clean(pg, upapp, "dataset after upload")
    assert pg.get_by_text("Added 'alphabet_mini'").count() == 1
    assert "alphabet_mini" in pg.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first.inner_text()
    assert pg.get_by_text("Files per sign").count() == 1
    s = json.loads(upapp["settings"].read_text())
    assert "imports" in s["data_dir"] and "alphabet_mini" in s["data_dir"]
    shoot(pg, "upload_1_dataset")


def test_preprocess_split_train(pg, upapp):
    open_page(pg, upapp, "preprocess")
    click(pg, r"Extract new clips \(90\)")
    assert "Extracted" in wait_job(pg, upapp, "preprocess")
    shoot(pg, "upload_2_preprocess")
    open_page(pg, upapp, "split")
    click(pg, "^Run split")
    wait_job(pg, upapp, "split")
    assert pg.get_by_text("Train (70 %)").count() == 1
    shoot(pg, "upload_4_split")
    open_page(pg, upapp, "train")
    for label, value in (("Epochs", "8"), ("Aug. factor", "1"), ("Batch", "16")):
        box = pg.get_by_role("spinbutton", name=label)
        box.fill(value)
        box.press("Tab")
        settle(pg, 600)
    click(pg, "^Start training")
    wait_job(pg, upapp, "train")
    assert pg.get_by_text("Saved models").count() == 1
    shoot(pg, "upload_5_train")


def test_evaluate_export_live(pg, upapp):
    open_page(pg, upapp, "evaluate")
    pg.get_by_text("Tuning is finished; evaluate on the test set once").click()
    settle(pg)
    pg.get_by_role("button", name="Evaluate on the test set", exact=True).click()
    settle(pg, 800)
    wait_job(pg, upapp, "evaluate")
    assert pg.locator(".verdict").count() == 1
    assert pg.get_by_text("test split").count() >= 1
    shoot(pg, "upload_6_evaluate")
    open_page(pg, upapp, "export")
    click(pg, "^Export to TF.js$")
    wait_job(pg, upapp, "export")
    assert pg.get_by_text("Check passed").count() >= 1
    tfjs = upapp["root"] / "work" / "imports" / "alphabet_mini" / "work" / "models"
    assert (tfjs / "alphabet_mini" / "tfjs" / "model.json").exists()
    assert (tfjs / "alphabet_mini" / "tfjs" / "preprocess.json").exists()
    open_page(pg, upapp, "live")
    click(pg, "^Start live test$")
    assert "Could not open camera 9" in wait_job(pg, upapp, "live", expect="error")


def test_switch_back_to_recordings_keeps_outputs_apart(pg, upapp):
    open_page(pg, upapp, "")
    use = pg.get_by_role("button", name="Use", exact=True)
    if use.count() == 0:
        pytest.skip("no HAND TRACKER recordings on this computer")
    use.first.click()
    settle(pg, 2000)
    assert_clean(pg, upapp, "switched")
    assert "recordings" in pg.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first.inner_text()
    open_page(pg, upapp, "train")
    assert pg.get_by_text("Saved models").count() == 0          # the upload's model is not shown here
    assert (SHOTS / "upload_6_evaluate.png").exists()
