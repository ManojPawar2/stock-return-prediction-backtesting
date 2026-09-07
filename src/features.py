"""Feature engineering.

Every column produced here obeys one rule (README section 4, Rule 1):

    The value at row ``t`` depends only on observations at or before ``t``.

Concretely that means no ``.shift(-n)``, no ``center=True``, no ``.bfill()``,
and no full-sample statistic (a global mean, a global z-score) ever entering a
feature.  ``tests/test_features.py`` enforces this by recomputing every single
feature on a truncated frame and demanding an identical value.

A second, subtler rule governs *what* is built: features must be
**stationary**.  A raw price or a raw moving average is not — a model trained
on AAPL at $30 would have to extrapolate wildly at $250, and a tree simply
cannot.  So levels are always converted to ratios, returns, or bounded
oscillators before they reach a model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config, FeatureConfig
from .indicators import (
    atr,
    bollinger_bands,
    macd,
    obv,
    rolling_zscore,
    rsi,
    stochastic_k,
)
from .preprocessing import adjust_ohlc

logger = logging.getLogger(__name__)

#: Columns that are inputs or bookkeeping, never model features.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "adj_close", "volume",
    "target", "return_1d_fwd", "prediction", "signal", "position",
    # Raw levels kept for plotting but excluded from the model matrix.
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "bb_middle", "bb_upper", "bb_lower", "volume_sma", "obv", "atr",
)


def build_features(
    df: pd.DataFrame,
    cfg: Config | FeatureConfig | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix from a cleaned OHLCV frame.

    Parameters
    ----------
    df
        Cleaned OHLCV, as produced by :func:`src.preprocessing.clean`.
    cfg
        A :class:`~src.config.Config` or :class:`~src.config.FeatureConfig`.
    dropna
        Drop the warm-up rows whose long rolling windows are still NaN.
        Done **before** the chronological split so that every split is a
        contiguous block of complete rows (README section 4).

    Returns
    -------
    DataFrame
        The original price columns plus every engineered feature.
    """
    fc = _feature_config(cfg)

    if df.empty:
        raise ValueError("build_features received an empty frame")

    # Put open/high/low on the same adjusted basis as the close before any
    # indicator touches two different price series. See preprocessing.adjust_ohlc.
    out = adjust_ohlc(df)

    close = out["close"]
    high, low, volume = out["high"], out["low"], out["volume"]
    ret1 = close.pct_change()

    # ---------------------------------------------------------- returns
    out["ret_1d"] = ret1
    for w in fc.return_windows:
        if w == 1:
            continue
        out[f"ret_{w}d"] = close.pct_change(w)

    # Lagged one-day returns: yesterday's move, the day before, and so on.
    for lag in fc.return_lags:
        out[f"ret_lag_{lag}"] = ret1.shift(lag)

    # Log return is better behaved in the tails than the simple return.
    out["log_ret_1d"] = np.log(close / close.shift(1))

    # ---------------------------------------------- moving averages (ratios)
    for w in fc.sma_windows:
        sma = close.rolling(w, min_periods=w).mean()
        out[f"sma_{w}"] = sma                       # level: plotting only
        out[f"close_sma_{w}_ratio"] = close / sma   # feature: stationary

    if len(fc.sma_windows) >= 2:
        fast, slow = fc.sma_windows[0], fc.sma_windows[2 if len(fc.sma_windows) > 2 else 1]
        out[f"sma_{fast}_{slow}_ratio"] = (
            out[f"sma_{fast}"] / out[f"sma_{slow}"]
        )

    # Exponential moving averages react faster to a change in trend.
    out["ema_12_26_ratio"] = (
        close.ewm(span=12, min_periods=12, adjust=False).mean()
        / close.ewm(span=26, min_periods=26, adjust=False).mean()
    )

    # ------------------------------------------------------- volatility
    for w in fc.volatility_windows:
        out[f"vol_{w}"] = ret1.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(252)

    if len(fc.volatility_windows) >= 2:
        short, long = fc.volatility_windows[0], fc.volatility_windows[1]
        # >1 means volatility is rising relative to its own baseline.
        out[f"vol_ratio_{short}_{long}"] = (
            out[f"vol_{short}"] / out[f"vol_{long}"].replace(0.0, np.nan)
        )

    # Downside deviation: only losing days contribute.
    out["downside_vol_20"] = (
        ret1.where(ret1 < 0, 0.0).rolling(20, min_periods=20).std(ddof=1)
        * np.sqrt(252)
    )

    # ----------------------------------------------------------- volume
    vw = fc.volume_window
    out["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    out["volume_sma"] = volume.rolling(vw, min_periods=vw).mean()
    out["volume_ratio"] = volume / out["volume_sma"].replace(0.0, np.nan)
    out["volume_zscore"] = rolling_zscore(volume, vw)
    # Dollar volume as a ratio, not a level — the level grows with the price.
    dollar = close * volume
    out["dollar_volume_ratio"] = dollar / dollar.rolling(
        vw, min_periods=vw
    ).mean().replace(0.0, np.nan)
    out["obv"] = obv(close, volume)
    out["obv_zscore"] = rolling_zscore(out["obv"], vw)

    # ------------------------------------------------------ intraday range
    out["high_low_range"] = (high - low) / close.replace(0.0, np.nan)
    span = (high - low).replace(0.0, np.nan)
    # Where in the day's range did it close? 1 = at the high, 0 = at the low.
    out["close_position"] = (close - low) / span
    out["gap_open"] = (out["open"] - close.shift(1)) / close.shift(1)

    # ------------------------------------------------ technical indicators
    out[f"rsi_{fc.rsi_period}"] = rsi(close, fc.rsi_period)

    macd_df = macd(close, fc.macd_fast, fc.macd_slow, fc.macd_signal)
    # Scale MACD by price so it is comparable across price levels and tickers.
    out["macd_norm"] = macd_df["macd"] / close
    out["macd_signal_norm"] = macd_df["macd_signal"] / close
    out["macd_hist_norm"] = macd_df["macd_hist"] / close

    bb = bollinger_bands(close, fc.bollinger_window, fc.bollinger_std)
    out["bb_middle"], out["bb_upper"], out["bb_lower"] = (
        bb["bb_middle"], bb["bb_upper"], bb["bb_lower"]
    )
    out["bb_pct_b"] = bb["bb_pct_b"]
    out["bb_bandwidth"] = bb["bb_bandwidth"]

    out["atr"] = atr(high, low, close, fc.atr_period)
    out["atr_pct"] = out["atr"] / close.replace(0.0, np.nan)

    out["stoch_k"] = stochastic_k(high, low, close, 14)

    # Distance from the 52-week extremes: a bounded, scale-free trend measure.
    roll_max = close.rolling(252, min_periods=60).max()
    roll_min = close.rolling(252, min_periods=60).min()
    out["dist_from_52w_high"] = close / roll_max - 1.0
    out["dist_from_52w_low"] = close / roll_min.replace(0.0, np.nan) - 1.0

    # --------------------------------------------------------- calendar
    if fc.include_calendar:
        out["day_of_week"] = out.index.dayofweek.astype("float64")
        out["month"] = out.index.month.astype("float64")
        # Cyclical encoding so December and January are adjacent, not 11 apart.
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # ----------------------------------------------------------- cleanup
    out = out.replace([np.inf, -np.inf], np.nan)

    if dropna:
        before = len(out)
        out = out.dropna(subset=feature_columns(out))
        logger.info(
            "build_features: dropped %d warm-up rows, %d remain",
            before - len(out), len(out),
        )

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input columns: everything except prices and raw levels."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Just the model inputs, in a stable column order."""
    return df[feature_columns(df)]


def _feature_config(cfg: Config | FeatureConfig | None) -> FeatureConfig:
    if cfg is None:
        return FeatureConfig()
    if isinstance(cfg, FeatureConfig):
        return cfg
    return cfg.features


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature summary for the Feature Lab page."""
    cols = feature_columns(df)
    stats = df[cols].describe().T
    stats["missing"] = df[cols].isna().sum()
    stats["missing_pct"] = (stats["missing"] / len(df) * 100).round(2)
    stats["skew"] = df[cols].skew()
    stats["kurtosis"] = df[cols].kurtosis()
    return stats


def future_return_correlation(
    df: pd.DataFrame, horizon: int = 1
) -> pd.Series:
    """Correlation of each feature with the *future* return.

    This is the honest headline of the whole project.  For daily equity data
    these values are almost always tiny (|rho| well under 0.1).  A feature
    showing |rho| above ~0.3 is far more likely to be a leak than an edge, so
    this doubles as a leakage smoke test.
    """
    future = df["close"].pct_change(horizon).shift(-horizon)
    cols = feature_columns(df)
    corr = df[cols].corrwith(future).sort_values(key=abs, ascending=False)
    return corr.rename(f"corr_with_{horizon}d_forward_return")


def leakage_report(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.3) -> pd.DataFrame:
    """Flag features suspiciously correlated with the future.

    Used by the Feature Lab's self-check panel.
    """
    corr = future_return_correlation(df, horizon)
    return pd.DataFrame(
        {
            "feature": corr.index,
            "corr_with_future": corr.to_numpy().round(4),
            "suspicious": np.abs(corr.to_numpy()) > threshold,
        }
    )
