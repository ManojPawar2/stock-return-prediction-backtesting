"""Tests for feature engineering.

The headline test is :func:`test_no_feature_uses_future_information`, which
rebuilds the entire feature matrix on a truncated frame and demands that every
feature at time ``t`` is bit-for-bit unchanged.  If a future-peeking feature is
ever added, that test fails immediately.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config, FeatureConfig
from src.features import (
    NON_FEATURE_COLUMNS,
    build_features,
    describe_features,
    feature_columns,
    feature_matrix,
    future_return_correlation,
    leakage_report,
)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """400 rows of a deterministic, realistic-looking price series."""
    n = 400
    idx = pd.bdate_range("2021-01-04", periods=n, name="date")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.015, n)
    close = pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)

    # The intraday range must vary day to day. With a fixed proportional
    # spread, high_low_range and close_position become mathematically
    # constant, which silently turns two features into dead columns and
    # makes any correlation against them undefined.
    day_range = close * pd.Series(rng.uniform(0.005, 0.03, n), index=idx)
    close_pos = pd.Series(rng.uniform(0.15, 0.85, n), index=idx)
    low = close - day_range * close_pos
    high = low + day_range
    open_ = low + day_range * pd.Series(rng.uniform(0.1, 0.9, n), index=idx)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": pd.Series(rng.integers(5e5, 5e6, n).astype("float64"), index=idx),
        }
    )


@pytest.fixture
def built(ohlcv) -> pd.DataFrame:
    return build_features(ohlcv, Config())


# ------------------------------------------------------------------- shape


def test_produces_a_useful_number_of_features(built):
    cols = feature_columns(built)
    assert len(cols) >= 35, f"expected 35+ features, got {len(cols)}"


def test_no_missing_values_after_warmup(built):
    assert not built[feature_columns(built)].isna().any().any()


def test_no_infinities(built):
    assert np.isfinite(built[feature_columns(built)].to_numpy()).all()


def test_index_stays_sorted_and_unique(built):
    assert built.index.is_monotonic_increasing
    assert not built.index.duplicated().any()


def test_price_columns_are_excluded_from_the_model_matrix(built):
    cols = set(feature_columns(built))
    for banned in ("open", "high", "low", "close", "adj_close", "volume"):
        assert banned not in cols


def test_raw_levels_are_excluded_but_ratios_are_kept(built):
    cols = set(feature_columns(built))
    # Non-stationary levels must not reach the model...
    for level in ("sma_20", "bb_upper", "bb_middle", "volume_sma", "obv", "atr"):
        assert level not in cols, f"{level} is a raw price level"
    # ...but their scale-free counterparts must.
    for ratio in ("close_sma_20_ratio", "bb_pct_b", "volume_ratio",
                  "obv_zscore", "atr_pct"):
        assert ratio in cols


def test_expected_feature_families_are_present(built):
    cols = set(feature_columns(built))
    for expected in (
        "ret_1d", "ret_5d", "ret_20d", "ret_lag_1", "ret_lag_5", "log_ret_1d",
        "close_sma_5_ratio", "ema_12_26_ratio", "vol_5", "vol_20", "vol_60",
        "vol_ratio_5_20", "downside_vol_20", "volume_change", "volume_ratio",
        "volume_zscore", "dollar_volume_ratio", "high_low_range",
        "close_position", "gap_open", "rsi_14", "macd_norm", "macd_hist_norm",
        "bb_pct_b", "bb_bandwidth", "atr_pct", "stoch_k",
        "dist_from_52w_high", "dist_from_52w_low",
    ):
        assert expected in cols, f"missing feature {expected}"


def test_feature_matrix_has_stable_column_order(built):
    a = feature_matrix(built)
    b = feature_matrix(built)
    assert list(a.columns) == list(b.columns)
    assert list(a.columns) == feature_columns(built)


def test_calendar_features_can_be_disabled(ohlcv):
    fc = FeatureConfig(include_calendar=False)
    cols = set(feature_columns(build_features(ohlcv, fc)))
    assert "month_sin" not in cols and "day_of_week" not in cols


def test_dropna_false_keeps_every_row(ohlcv):
    out = build_features(ohlcv, Config(), dropna=False)
    assert len(out) == len(ohlcv)


def test_empty_frame_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        build_features(pd.DataFrame(), Config())


# -------------------------------------------------------- value correctness


def test_ret_1d_matches_manual_pct_change(ohlcv, built):
    expected = ohlcv["close"].pct_change().loc[built.index]
    assert np.allclose(built["ret_1d"], expected)


def test_ret_lag_1_is_yesterdays_return(built):
    # Today's lag-1 feature must equal yesterday's one-day return.
    assert np.allclose(
        built["ret_lag_1"].iloc[1:], built["ret_1d"].iloc[:-1], atol=1e-12
    )


def test_close_sma_ratio_is_around_one(built):
    assert built["close_sma_20_ratio"].between(0.5, 2.0).all()
    assert built["close_sma_20_ratio"].mean() == pytest.approx(1.0, abs=0.1)


def test_close_position_is_bounded(built):
    assert built["close_position"].between(-0.01, 1.01).all()


def test_bounded_indicators_stay_in_range(built):
    assert built["rsi_14"].between(0, 100).all()
    assert built["stoch_k"].between(0, 100).all()


def test_volatility_is_annualised_and_positive(built):
    assert (built["vol_20"] > 0).all()
    # 1.5% daily vol annualises to roughly 24%.
    assert 0.05 < built["vol_20"].mean() < 1.0


def test_dist_from_52w_high_is_non_positive(built):
    assert (built["dist_from_52w_high"] <= 1e-9).all()


def test_dist_from_52w_low_is_non_negative(built):
    assert (built["dist_from_52w_low"] >= -1e-9).all()


def test_month_cyclical_encoding_wraps():
    """December and January must be neighbours in the encoded space."""
    dec = np.array([np.sin(2 * np.pi * 12 / 12), np.cos(2 * np.pi * 12 / 12)])
    jan = np.array([np.sin(2 * np.pi * 1 / 12), np.cos(2 * np.pi * 1 / 12)])
    jun = np.array([np.sin(2 * np.pi * 6 / 12), np.cos(2 * np.pi * 6 / 12)])
    assert np.linalg.norm(dec - jan) < np.linalg.norm(dec - jun)


def test_features_use_adjusted_prices(ohlcv):
    """A raw/adjusted mix would corrupt every multi-series indicator."""
    df = ohlcv.copy()
    df["adj_close"] = df["close"] * 0.25          # a 4:1 split
    out = build_features(df, Config())
    # %K would blow far outside 0-100 if high/low were left unadjusted.
    assert out["stoch_k"].between(0, 100).all()
    assert out["atr_pct"].between(0, 0.5).all()


# ================================================================= LEAKAGE
# The property the entire project depends on.


@pytest.mark.parametrize("cut", [300, 350, 399])
def test_no_feature_uses_future_information(ohlcv, cut):
    """Rebuild on df[:cut+1]; every feature at `cut` must be unchanged."""
    full = build_features(ohlcv, Config(), dropna=False)
    truncated = build_features(ohlcv.iloc[: cut + 1], Config(), dropna=False)

    date = ohlcv.index[cut]
    assert date in truncated.index

    offenders = []
    for col in feature_columns(full):
        a = full.loc[date, col]
        b = truncated.loc[date, col]
        if pd.isna(a) and pd.isna(b):
            continue
        if not np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            offenders.append(f"{col}: full={a!r} truncated={b!r}")

    assert not offenders, (
        f"{len(offenders)} feature(s) changed at {date.date()} once future "
        "rows were appended — this is lookahead bias:\n  "
        + "\n  ".join(offenders)
    )


def test_appending_a_future_row_does_not_change_the_past(ohlcv):
    """A stronger phrasing: adding tomorrow must not rewrite today."""
    base = build_features(ohlcv.iloc[:-1], Config(), dropna=False)
    extended = build_features(ohlcv, Config(), dropna=False)
    common = base.index[-50:]
    cols = feature_columns(base)
    pd.testing.assert_frame_equal(
        base.loc[common, cols], extended.loc[common, cols],
        check_exact=False, rtol=1e-9,
    )


def test_no_feature_is_implausibly_correlated_with_the_future(built):
    """A real feature has |rho| well under 0.3 against tomorrow's return."""
    corr = future_return_correlation(built, horizon=1).abs()
    worst = corr.idxmax()
    assert corr.max() < 0.3, (
        f"{worst} correlates {corr.max():.3f} with the future return — "
        "that is a leak, not an edge"
    )


def test_leakage_report_flags_a_planted_leak(built):
    """Sanity-check the detector itself by planting an obvious cheat."""
    cheating = built.copy()
    cheating["tomorrow_peek"] = cheating["close"].pct_change().shift(-1)
    report = leakage_report(cheating, threshold=0.3)
    flagged = report.loc[report["suspicious"], "feature"].tolist()
    assert "tomorrow_peek" in flagged


def test_future_return_correlations_are_small(built):
    """The honest headline: daily returns are close to unpredictable."""
    corr = future_return_correlation(built).abs()
    assert corr.median() < 0.1


# --------------------------------------------------------------- reporting


def test_describe_features_covers_every_feature(built):
    stats = describe_features(built)
    assert len(stats) == len(feature_columns(built))
    for col in ("mean", "std", "missing", "missing_pct", "skew", "kurtosis"):
        assert col in stats.columns


def test_non_feature_columns_constant_is_respected(built):
    for col in NON_FEATURE_COLUMNS:
        if col in built.columns:
            assert col not in feature_columns(built)
