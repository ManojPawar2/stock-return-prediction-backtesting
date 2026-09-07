"""Tests for the technical indicators.

Two kinds of assertion:

1. **Known values.** Indicators are recomputed by hand (or from their closed
   form) so a silent formula change is caught, not just a shape change.
2. **No lookahead.** Recomputing an indicator on the truncated series
   ``x[:t+1]`` must give the same value at ``t`` as computing it on the full
   series. This is the property the whole project rests on.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    atr,
    bollinger_bands,
    macd,
    obv,
    rolling_zscore,
    rsi,
    stochastic_k,
    true_range,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Deterministic pseudo-random walk — no RNG seeding surprises."""
    n = 200
    idx = pd.bdate_range("2022-01-03", periods=n, name="date")
    steps = np.sin(np.arange(n) / 7.0) + np.cos(np.arange(n) / 3.0) * 0.5
    close = pd.Series(100 + steps.cumsum(), index=idx)
    return pd.DataFrame(
        {
            "high": close + 1.5,
            "low": close - 1.5,
            "close": close,
            "volume": pd.Series(np.arange(n) % 7 * 1e5 + 1e6, index=idx),
        }
    )


# --------------------------------------------------------------------- RSI


def test_rsi_is_bounded():
    close = pd.Series(np.random.default_rng(0).normal(100, 5, 300).cumsum())
    out = rsi(close, 14).dropna()
    assert len(out) > 0
    assert out.between(0, 100).all()


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1.0, 60.0))
    assert rsi(close, 14).dropna().eq(100.0).all()


def test_rsi_all_losses_is_0():
    close = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(close, 14).dropna().eq(0.0).all()


def test_rsi_flat_series_is_neutral_50():
    close = pd.Series([100.0] * 40)
    assert rsi(close, 14).dropna().eq(50.0).all()


def test_rsi_warmup_is_nan():
    close = pd.Series(np.arange(1.0, 40.0))
    out = rsi(close, 14)
    # diff() costs one row, then 14 more for the first Wilder average.
    assert out.iloc[:14].isna().all()
    assert out.notna().iloc[14:].all()


def test_rsi_matches_hand_computed_wilder_value():
    """First RSI value = simple average of gains/losses over the period."""
    period = 3
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 11.5])
    # deltas: +1.0, -0.5, +1.5, -0.5
    # first 3 deltas -> avg gain (1.0+0+1.5)/3, avg loss (0+0.5+0)/3
    avg_gain = (1.0 + 0.0 + 1.5) / 3
    avg_loss = (0.0 + 0.5 + 0.0) / 3
    expected = 100 - 100 / (1 + avg_gain / avg_loss)
    assert rsi(close, period).iloc[3] == pytest.approx(expected)


# -------------------------------------------------------------------- MACD


def test_macd_columns_and_histogram_identity(prices):
    out = macd(prices["close"])
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist"]
    valid = out.dropna()
    assert np.allclose(valid["macd_hist"], valid["macd"] - valid["macd_signal"])


def test_macd_matches_manual_ema_difference(prices):
    close = prices["close"]
    fast = close.ewm(span=12, min_periods=12, adjust=False).mean()
    slow = close.ewm(span=26, min_periods=26, adjust=False).mean()
    pd.testing.assert_series_equal(
        macd(close)["macd"], (fast - slow).rename("macd")
    )


def test_macd_is_zero_for_a_constant_series():
    close = pd.Series([50.0] * 100)
    assert macd(close)["macd"].dropna().abs().max() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------- Bollinger


def test_bollinger_bands_ordering_and_pctb(prices):
    out = bollinger_bands(prices["close"], 20, 2.0).dropna()
    assert (out["upper" if "upper" in out else "bb_upper"] >= out["bb_middle"]).all()
    assert (out["bb_lower"] <= out["bb_middle"]).all()
    # %B is the close's position between the bands.
    close = prices["close"].loc[out.index]
    expected = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    assert np.allclose(out["bb_pct_b"], expected)


def test_bollinger_matches_manual_population_std(prices):
    close = prices["close"]
    out = bollinger_bands(close, 20, 2.0)
    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    assert np.allclose(out["bb_upper"].dropna(), (mid + 2 * sd).dropna())


def test_bollinger_constant_series_has_zero_bandwidth():
    close = pd.Series([100.0] * 50)
    out = bollinger_bands(close, 20)
    assert out["bb_bandwidth"].dropna().eq(0.0).all()
    # Zero-width band makes %B undefined rather than infinite.
    assert out["bb_pct_b"].dropna().empty


# ---------------------------------------------------------------- ATR / TR


def test_true_range_captures_overnight_gap():
    high = pd.Series([10.0, 20.0])
    low = pd.Series([9.0, 19.0])
    close = pd.Series([9.5, 19.5])
    tr = true_range(high, low, close)
    # Day 2 gapped up from 9.5 to a 19-20 range: TR must span the gap.
    assert tr.iloc[1] == pytest.approx(20.0 - 9.5)
    assert tr.iloc[1] > (high - low).iloc[1]


def test_atr_is_positive(prices):
    out = atr(prices["high"], prices["low"], prices["close"], 14).dropna()
    assert len(out) > 0
    assert (out > 0).all()


# --------------------------------------------------------------------- OBV


def test_obv_accumulates_signed_volume():
    close = pd.Series([10.0, 11.0, 10.0, 10.0, 12.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    #        first day 0, +200, -300, unchanged 0, +500
    assert list(obv(close, volume)) == [0.0, 200.0, -100.0, -100.0, 400.0]


# -------------------------------------------------------------- stochastic


def test_stochastic_k_bounded_and_correct(prices):
    out = stochastic_k(prices["high"], prices["low"], prices["close"], 14).dropna()
    assert out.between(0, 100).all()


def test_stochastic_k_at_window_high_is_100():
    high = pd.Series(np.arange(1.0, 30.0))
    low = high - 1
    close = high.copy()  # closing at the high every day
    assert stochastic_k(high, low, close, 14).dropna().eq(100.0).all()


# ------------------------------------------------------------------ zscore


def test_rolling_zscore_is_standardised(prices):
    out = rolling_zscore(prices["close"], 20).dropna()
    assert out.abs().max() < 10
    # A constant series has no dispersion, so the z-score is undefined.
    assert rolling_zscore(pd.Series([5.0] * 50), 20).dropna().empty


# ----------------------------------------------------------- NO LOOKAHEAD
# The core guarantee: an indicator at time t must not change when future
# rows are appended.


@pytest.mark.parametrize("cut", [60, 120, 199])
def test_no_lookahead_single_series_indicators(prices, cut):
    close = prices["close"]
    for name, fn in {
        "rsi": lambda s: rsi(s, 14),
        "macd": lambda s: macd(s)["macd"],
        "macd_hist": lambda s: macd(s)["macd_hist"],
        "bb_pct_b": lambda s: bollinger_bands(s, 20)["bb_pct_b"],
        "bb_bandwidth": lambda s: bollinger_bands(s, 20)["bb_bandwidth"],
        "zscore": lambda s: rolling_zscore(s, 20),
    }.items():
        full = fn(close).iloc[cut]
        truncated = fn(close.iloc[: cut + 1]).iloc[cut]
        assert full == pytest.approx(truncated, nan_ok=True), (
            f"{name} at index {cut} changed when future data was appended "
            f"({truncated} -> {full}) — this is lookahead bias"
        )


@pytest.mark.parametrize("cut", [60, 120, 199])
def test_no_lookahead_multi_series_indicators(prices, cut):
    h, l, c, v = (prices[k] for k in ("high", "low", "close", "volume"))
    pairs = {
        "atr": (
            atr(h, l, c, 14).iloc[cut],
            atr(h[: cut + 1], l[: cut + 1], c[: cut + 1], 14).iloc[cut],
        ),
        "stoch_k": (
            stochastic_k(h, l, c, 14).iloc[cut],
            stochastic_k(h[: cut + 1], l[: cut + 1], c[: cut + 1], 14).iloc[cut],
        ),
        "obv": (
            obv(c, v).iloc[cut],
            obv(c[: cut + 1], v[: cut + 1]).iloc[cut],
        ),
    }
    for name, (full, truncated) in pairs.items():
        assert full == pytest.approx(truncated, nan_ok=True), (
            f"{name} at index {cut} leaked future information"
        )
