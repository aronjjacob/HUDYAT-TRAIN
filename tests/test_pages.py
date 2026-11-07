"""Every page renders without an error, on a small synthetic dataset with finished test runs (Streamlit AppTest,
no browser). The Playwright click-through with real clips is in test_ui.py."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from trainer import experiments as E
from trainer import final
from trainer import settings as cfg

from test_experiments import Ctx, FakeFit, env  # noqa: F401  (fixture)

PAGES = ["data", "preprocess", "test", "final", "live", "paper", "settings"]


@pytest.fixture
def app(env, tmp_path):
    E.run(Ctx(), "random", "cat", {"seed": 1}, fit=FakeFit())
    E.run(Ctx(), "loso", "cat", {"seed": 1}, fit=FakeFit())
    final.run(Ctx(), "cat", from_run=E.list_runs(cfg.Paths())[0]["run_id"], fit=FakeFit())
    from trainer import paper
    paper.generate(Ctx(), "cat")
    return env


def errors(at: AppTest) -> list[str]:
    out = [e.value for e in at.exception]
    out += [c.value for c in at.code if "Traceback" in str(c.value)]          # ui guarded(): error + traceback
    return out


def _render(page: str) -> None:
    from ui import common, p_data, p_final, p_live, p_paper, p_preprocess, p_settings, p_test, state
    common.inject_css()
    if page == "settings":
        p_settings.render()
        return
    {"data": p_data, "preprocess": p_preprocess, "test": p_test, "final": p_final, "live": p_live,
     "paper": p_paper}[page].render(state.snapshot())


def test_whole_app_starts(app):
    at = AppTest.from_file(str(cfg.PROJECT_DIR / "trainer_app.py"), default_timeout=60)
    at.run()
    assert not errors(at), errors(at)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    at = AppTest.from_function(_render, args=(page,), default_timeout=60)
    at.run()
    assert not errors(at), errors(at)
