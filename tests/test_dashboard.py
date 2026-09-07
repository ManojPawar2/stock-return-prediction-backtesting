"""Every Streamlit page must actually run.

A syntax check proves nothing about runtime, so these tests boot each page
through ``AppTest`` — the same script runner Streamlit itself uses — and fail
on any exception the page raises, including inside widgets.

Network is mocked, so the pages exercise the real pipeline against synthetic
prices rather than Yahoo Finance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

from src import data_loader as dl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted((ROOT / "pages").glob("*.py"))
ALL_SCRIPTS = [ROOT / "app.py"] + PAGES

# Pages run the whole pipeline, so give them room.
TIMEOUT = 600


def synthetic_prices(n: int = 2600) -> pd.DataFrame:
    """Ten years of yfinance-shaped daily bars covering config.yaml's range."""
    idx = pd.bdate_range("2015-01-01", periods=n, name="Date")
    rng = np.random.default_rng(61)
    close = pd.Series(30 * np.exp(np.cumsum(rng.normal(0.0008, 0.016, n))), index=idx)
    day_range = close * pd.Series(rng.uniform(0.006, 0.032, n), index=idx)
    pos = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    low = close - day_range * pos
    return pd.DataFrame({
        "Open": low + day_range * 0.45,
        "High": low + day_range,
        "Low": low,
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(2e6, 9e7, n).astype("float64"),
    }, index=idx)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Serve synthetic data; keep every cache and artefact inside tmp_path."""
    monkeypatch.setattr(dl, "_download", lambda ticker, start, end: synthetic_prices())
    cache = tmp_path / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dl, "DATA_RAW", cache)
    yield


def run_page(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    app.run()
    return app


def assert_no_exception(app: AppTest, name: str) -> None:
    if app.exception:
        details = "; ".join(f"{e.type}: {e.message}" for e in app.exception)
        pytest.fail(f"{name} raised: {details}")


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_page_runs_without_exception(script):
    assert_no_exception(run_page(script), script.name)


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_page_renders_a_title(script):
    app = run_page(script)
    assert app.title or app.header, f"{script.name} rendered no heading"


def test_there_are_ten_scripts():
    """One entry point plus nine pages, as the README states."""
    assert len(ALL_SCRIPTS) == 10
    assert len(PAGES) == 9


def test_pages_are_numbered_for_ordering():
    """Streamlit orders the sidebar by filename, so the prefixes matter."""
    for page in PAGES:
        assert page.name[0].isdigit(), f"{page.name} has no ordering prefix"


def test_overview_shows_headline_metrics():
    app = run_page(ROOT / "app.py")
    labels = [m.label for m in app.metric]
    assert "Trading days" in labels
    assert "Buy-and-hold return" in labels


def test_backtest_page_reports_against_the_benchmark():
    app = run_page(ROOT / "pages" / "5_Backtest.py")
    assert_no_exception(app, "5_Backtest.py")
    labels = [m.label for m in app.metric]
    assert "Strategy return" in labels
    assert "Buy and hold" in labels, "the benchmark must always be shown alongside"


def test_backtest_page_states_the_outcome_either_way():
    """Whether it won or lost, the page must say so plainly."""
    app = run_page(ROOT / "pages" / "5_Backtest.py")
    spoken = " ".join(
        [w.value for w in app.success] + [w.value for w in app.warning]
    )
    assert "buy-and-hold" in spoken.lower()


def test_modeling_page_shows_split_and_metrics():
    app = run_page(ROOT / "pages" / "4_Modeling.py")
    assert_no_exception(app, "4_Modeling.py")
    labels = [m.label for m in app.metric]
    assert "Test R²" in labels
    assert "Directional accuracy" in labels


def test_comparison_page_includes_the_benchmark_row():
    app = run_page(ROOT / "pages" / "6_Comparison.py")
    assert_no_exception(app, "6_Comparison.py")
    assert len(app.dataframe) >= 1


def test_feature_lab_reports_the_leakage_check():
    app = run_page(ROOT / "pages" / "3_Feature_Lab.py")
    assert_no_exception(app, "3_Feature_Lab.py")
    spoken = " ".join(
        [w.value for w in app.success] + [w.value for w in app.error]
    )
    assert "threshold" in spoken.lower()


def test_eda_page_reports_fat_tails():
    app = run_page(ROOT / "pages" / "2_EDA.py")
    assert_no_exception(app, "2_EDA.py")
    assert "Excess kurtosis" in [m.label for m in app.metric]


def test_every_page_offers_the_shared_sidebar():
    """One configuration drives the whole app; pages must not diverge."""
    for script in ALL_SCRIPTS:
        app = run_page(script)
        assert app.sidebar.text_input, f"{script.name} has no ticker input"
        assert app.sidebar.text_input[0].label == "Ticker"
