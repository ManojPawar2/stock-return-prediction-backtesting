"""Classic technical indicators, as pure functions.

Every function here is **strictly backward-looking**: the value at index ``t``
depends only on observations up to and including ``t``.  This is the property
that keeps the feature matrix leakage-free (README section 4), and it is
asserted directly in ``tests/test_indicators.py``.

Warm-up periods are left as ``NaN`` rather than back-filled.  Inventing an
early RSI value would be inventing information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} expects a Series, got a DataFrame")
    return x.astype("float64")


def _wilder_smooth(x: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, seeded the way Wilder defined it.

    ``y[n-1] = mean(x[0..n-1])``  then  ``y[t] = y[t-1] + (x[t] - y[t-1]) / n``

    A plain ``ewm(alpha=1/n, adjust=False)`` is *not* equivalent: it seeds the
    recursion with the single first observation instead of the simple average
    of the first ``n``.  The two converge eventually but disagree for the first
    several dozen bars, which is exactly the region a short backtest uses.

    Implemented by overwriting the seed position and running the (recursive,
    therefore causal) EWM from there, so the result stays lookahead-free.
    """
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    positions = np.flatnonzero(x.notna().to_numpy())
    if len(positions) < period:
        return out

    seed_pos = int(positions[period - 1])
    sub = x.iloc[seed_pos:].copy()
    sub.iloc[0] = float(x.iloc[: seed_pos + 1].dropna().iloc[-period:].mean())
    out.iloc[seed_pos:] = (
        sub.ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    )
    return out


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    ``RSI = 100 - 100 / (1 + average_gain / average_loss)``

    Averages use :func:`_wilder_smooth`, i.e. Wilder's original SMA-seeded
    recursion rather than a bare EWM.

    Bounded in [0, 100].  An all-gains window gives exactly 100.
    """
    close = _as_series(close, "rsi")
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 means an unbroken run of gains -> RSI is 100 by definition.
    out = out.where(avg_loss != 0.0, 100.0)
    # Both zero means a flat series; 50 is the neutral convention.
    out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return out.where(avg_gain.notna()).rename(f"rsi_{period}")


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns ``macd`` (fast EMA minus slow EMA), ``macd_signal`` (EMA of the
    MACD line) and ``macd_hist`` (their difference).
    """
    close = _as_series(close, "macd")
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": sig, "macd_hist": line - sig}
    )


# --------------------------------------------------------------------------
# Volatility / bands
# --------------------------------------------------------------------------


def bollinger_bands(
    close: pd.Series, window: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands plus the two scale-free derivatives.

    ``bb_pct_b`` places the price within the band (0 = lower, 1 = upper) and
    ``bb_bandwidth`` measures band width relative to the middle band.  Only
    these two are used as model features — the raw bands are price levels and
    are non-stationary.
    """
    close = _as_series(close, "bollinger_bands")
    middle = close.rolling(window, min_periods=window).mean()
    # ddof=0: the population standard deviation, as Bollinger defined it.
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper = middle + n_std * sd
    lower = middle - n_std * sd

    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": (close - lower) / width,
            "bb_bandwidth": (upper - lower) / middle.replace(0.0, np.nan),
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: the largest of the three candidate ranges.

    Using the previous close is what makes this capture overnight gaps, which
    a plain high-low range would miss entirely.
    """
    high, low, close = (
        _as_series(high, "true_range"),
        _as_series(low, "true_range"),
        _as_series(close, "true_range"),
    )
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1).rename("true_range")


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range, using Wilder's SMA-seeded smoothing."""
    tr = true_range(high, low, close)
    return _wilder_smooth(tr, period).rename(f"atr_{period}")


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by the daily price change."""
    close = _as_series(close, "obv")
    volume = _as_series(volume, "obv")
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum().rename("obv")


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------


def stochastic_k(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Stochastic %K: where the close sits inside the recent high-low range."""
    highest = high.rolling(window, min_periods=window).max()
    lowest = low.rolling(window, min_periods=window).min()
    span = (highest - lowest).replace(0.0, np.nan)
    return (100.0 * (close - lowest) / span).rename(f"stoch_k_{window}")


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Standardise a series against its own trailing window.

    Turns a non-stationary level into something a model can use across regimes.
    """
    series = _as_series(series, "rolling_zscore")
    mean = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / sd
