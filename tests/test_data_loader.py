"""Tests for the data loader.

Network is never touched: ``_download`` is monkeypatched with synthetic frames
shaped exactly like the awkward things yfinance actually returns.
"""

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src.data_loader import COLUMNS, DataLoadError


def _raw_single(n: int = 20) -> pd.DataFrame:
    """A flat-column frame with yfinance's capitalisation."""
    idx = pd.bdate_range("2024-01-01", periods=n, name="Date")
    close = np.linspace(100, 120, n)
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Adj Close": close * 0.99,
            "Volume": np.full(n, 1_000_000),
        },
        index=idx,
    )


def _raw_multiindex(ticker: str = "AAPL", n: int = 20) -> pd.DataFrame:
    """The MultiIndex (field, ticker) frame yfinance returns for downloads."""
    flat = _raw_single(n)
    flat.columns = pd.MultiIndex.from_product(
        [flat.columns, [ticker]], names=["Price", "Ticker"]
    )
    return flat


# ---------------------------------------------------------------- normalise


def test_normalise_flat_columns():
    out = dl.normalise(_raw_single(), ticker="AAPL")
    assert list(out.columns) == list(COLUMNS)
    assert out.index.name == "date"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing


def test_normalise_flattens_multiindex():
    out = dl.normalise(_raw_multiindex(), ticker="AAPL")
    assert list(out.columns) == list(COLUMNS)
    assert not isinstance(out.columns, pd.MultiIndex)
    assert len(out) == 20


def test_normalise_selects_the_requested_ticker():
    a = _raw_single(10)
    b = _raw_single(10) * 2
    frame = pd.concat(
        {"AAPL": a, "MSFT": b}, axis=1, names=["Ticker", "Price"]
    ).swaplevel(axis=1)
    out = dl.normalise(frame, ticker="MSFT")
    assert list(out.columns) == list(COLUMNS)
    # MSFT's close is double AAPL's, so we must have picked the right slice.
    assert out["close"].iloc[0] == pytest.approx(a["Close"].iloc[0] * 2)


def test_normalise_synthesises_adj_close_when_absent():
    raw = _raw_single().drop(columns=["Adj Close"])
    out = dl.normalise(raw, ticker="AAPL")
    pd.testing.assert_series_equal(
        out["adj_close"], out["close"], check_names=False
    )


def test_normalise_strips_timezone_and_dedupes():
    raw = _raw_single(5)
    raw.index = raw.index.tz_localize("America/New_York")
    raw = pd.concat([raw, raw.iloc[[-1]]])  # duplicate final date
    out = dl.normalise(raw, ticker="AAPL")
    assert out.index.tz is None
    assert not out.index.duplicated().any()
    assert len(out) == 5


def test_normalise_sorts_unsorted_input():
    raw = _raw_single(10).iloc[::-1]
    out = dl.normalise(raw, ticker="AAPL")
    assert out.index.is_monotonic_increasing


def test_normalise_rejects_empty_frame():
    with pytest.raises(DataLoadError, match="No rows"):
        dl.normalise(pd.DataFrame(), ticker="NOPE")


def test_normalise_rejects_missing_columns():
    raw = _raw_single().drop(columns=["High", "Low"])
    with pytest.raises(DataLoadError, match="Missing required columns"):
        dl.normalise(raw, ticker="AAPL")


# ------------------------------------------------------------------- cache


def test_cache_path_is_deterministic(tmp_path):
    a = dl.cache_path("aapl", "2020-01-01", "2021-01-01", tmp_path)
    b = dl.cache_path("AAPL", "2020-01-01", "2021-01-01", tmp_path)
    assert a == b
    assert a.name == "AAPL_2020-01-01_2021-01-01.parquet"


def test_download_is_cached_and_reused(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_download(ticker, start, end):
        calls["n"] += 1
        return _raw_multiindex(ticker)

    monkeypatch.setattr(dl, "_download", fake_download)

    first = dl.load_prices("AAPL", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    assert calls["n"] == 1
    assert dl.is_cached("AAPL", "2024-01-01", "2024-02-01", tmp_path)

    second = dl.load_prices("AAPL", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    assert calls["n"] == 1, "second call must be served from the Parquet cache"
    pd.testing.assert_frame_equal(first, second)


def test_use_cache_false_forces_redownload(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_download(ticker, start, end):
        calls["n"] += 1
        return _raw_single()

    monkeypatch.setattr(dl, "_download", fake_download)
    dl.load_prices("AAPL", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    dl.load_prices(
        "AAPL", "2024-01-01", "2024-02-01", use_cache=False, cache_dir=tmp_path
    )
    assert calls["n"] == 2


def test_cache_roundtrip_preserves_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_download", lambda t, s, e: _raw_multiindex(t))
    dl.load_prices("AAPL", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    reread = dl.load_prices("AAPL", "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    assert list(reread.columns) == list(COLUMNS)
    assert reread.index.name == "date"
    assert reread.index.is_monotonic_increasing


# ------------------------------------------------------------------- misc


def test_load_many_skips_failures(tmp_path, monkeypatch):
    def fake_download(ticker, start, end):
        if ticker == "BAD":
            raise RuntimeError("delisted")
        return _raw_single()

    monkeypatch.setattr(dl, "_download", fake_download)
    out = dl.load_many(["AAPL", "BAD", "MSFT"], "2024-01-01", "2024-02-01",
                       cache_dir=tmp_path)
    assert set(out) == {"AAPL", "MSFT"}


def test_load_many_raises_when_all_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dl, "_download", lambda t, s, e: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with pytest.raises(DataLoadError):
        dl.load_many(["BAD1", "BAD2"], "2024-01-01", "2024-02-01", cache_dir=tmp_path)


def test_describe_reports_expected_keys():
    df = dl.normalise(_raw_single(30), ticker="AAPL")
    stats = dl.describe(df)
    assert stats["rows"] == 30
    assert stats["missing_values"] == 0
    # adj_close rises linearly, so total return must be positive.
    assert stats["total_return_pct"] > 0
    assert stats["start"] < stats["end"]
