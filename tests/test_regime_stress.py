"""Tests for regime analysis and stress testing.

These two modules answer the same underlying question from different angles:
is the result robust, or did it come from one lucky slice of history?
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.dataset import build_dataset
from src.regime import (
    TREND_LABELS,
    VOL_LABELS,
    consistency_score,
    label_regimes,
    label_trend_regime,
    label_volatility_regime,
    performance_by_regime,
    regime_report,
    regime_summary_text,
)
from src.stress import (
    break_even_cost,
    robustness_summary,
    stress_verdict,
    sweep_costs,
    sweep_periods,
    sweep_seeds,
    sweep_thresholds,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """800 days built with three genuinely distinct volatility regimes.

    Three levels, not two: with only a calm and a crash level, the tercile
    boundaries fall *inside* the calm cluster and split it arbitrarily, so the
    labels would carry no information about the planted structure.

    days   0-250 : quiet      (sigma 0.005)  -> bottom tercile
    days 250-400 : normal     (sigma 0.015)
    days 400-550 : crash      (sigma 0.045)  -> top tercile
    days 550-800 : normal     (sigma 0.015)
    """
    n = 800
    idx = pd.bdate_range("2018-01-01", periods=n, name="date")
    rng = np.random.default_rng(31)
    day = np.arange(n)
    crash = (day >= 400) & (day < 550)
    quiet = day < 250

    vol = np.where(crash, 0.045, np.where(quiet, 0.005, 0.015))
    drift = np.where(crash, -0.003, 0.0008)
    r = rng.normal(drift, vol)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    day_range = close * pd.Series(rng.uniform(0.006, 0.03, n), index=idx)
    pos = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    low = close - day_range * pos
    return pd.DataFrame({
        "open": low + day_range * 0.4, "high": low + day_range, "low": low,
        "close": close, "adj_close": close,
        "volume": pd.Series(rng.integers(1e6, 9e6, n).astype("float64"), index=idx),
    })


@pytest.fixture
def series(prices):
    rng = np.random.default_rng(32)
    returns = prices["close"].pct_change().fillna(0.0)
    predictions = pd.Series(
        returns.shift(-1).fillna(0.0) * 0.15 + rng.normal(0, 0.006, len(returns)),
        index=returns.index,
    )
    return returns, predictions


# ============================================================= REGIME LABELS


def test_volatility_labels_are_the_expected_three(prices):
    labels = label_volatility_regime(prices["close"].pct_change()).dropna()
    assert set(labels.unique()) <= set(VOL_LABELS)


def test_volatility_terciles_are_roughly_balanced(prices):
    labels = label_volatility_regime(prices["close"].pct_change()).dropna()
    counts = labels.value_counts(normalize=True)
    for label in VOL_LABELS:
        assert 0.15 < counts.get(label, 0) < 0.55


def test_the_planted_crash_is_labelled_high_volatility(prices):
    labels = label_volatility_regime(prices["close"].pct_change())
    crash = labels.iloc[430:540]
    assert (crash == "high_vol").mean() > 0.8


def test_the_quiet_stretch_is_labelled_low_volatility(prices):
    labels = label_volatility_regime(prices["close"].pct_change())
    # Start at 40 so the 20-day window is clear of the warm-up, and stop at
    # 245 so it is clear of the transition into the normal-volatility block.
    quiet = labels.iloc[40:245]
    assert (quiet == "low_vol").mean() > 0.8
    assert (quiet == "high_vol").sum() == 0


def test_trend_labels_are_bull_or_bear(prices):
    labels = label_trend_regime(prices["close"], 200).dropna()
    assert set(labels.unique()) <= set(TREND_LABELS)


def test_trend_regime_is_causal(prices):
    """A trailing moving average cannot see the future."""
    full = label_trend_regime(prices["close"], 200)
    truncated = label_trend_regime(prices["close"].iloc[:600], 200)
    assert (full.iloc[:600].dropna() == truncated.dropna()).all()


def test_causal_volatility_labels_do_not_change_with_future_data(prices):
    returns = prices["close"].pct_change()
    full = label_volatility_regime(returns, causal=True)
    truncated = label_volatility_regime(returns.iloc[:600], causal=True)
    common = truncated.dropna().index
    assert (full.loc[common] == truncated.loc[common]).all()


def test_non_causal_labels_may_change_with_future_data(prices):
    """Documents exactly why the causal variant exists."""
    returns = prices["close"].pct_change()
    full = label_volatility_regime(returns, causal=False)
    truncated = label_volatility_regime(returns.iloc[:600], causal=False)
    common = truncated.dropna().index
    # Adding the crash shifts the whole-sample terciles.
    assert not (full.loc[common] == truncated.loc[common]).all()


def test_label_regimes_combines_both(prices):
    labels = label_regimes(prices)
    assert set(labels.columns) == {"vol_regime", "trend_regime", "regime"}
    combined = labels["regime"].dropna()
    assert all("_" in value for value in combined)


# ========================================================== REGIME SCORING


def test_performance_by_regime_scores_each_bucket(prices, series):
    returns, _ = series
    labels = label_regimes(prices)["vol_regime"]
    breakdown = performance_by_regime(returns * 0.5, returns, labels)
    assert len(breakdown) >= 2
    assert "excess_return" in breakdown.columns
    assert breakdown["days"].sum() <= len(returns)


def test_regime_days_sum_to_the_labelled_sample(prices, series):
    returns, _ = series
    labels = label_regimes(prices)["vol_regime"]
    breakdown = performance_by_regime(returns, returns, labels, min_days=1)
    assert breakdown["days"].sum() == int(labels.notna().sum())


def test_identical_series_show_zero_excess_in_every_regime(prices, series):
    returns, _ = series
    labels = label_regimes(prices)["vol_regime"]
    breakdown = performance_by_regime(returns, returns.copy(), labels)
    assert breakdown["excess_return"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_tiny_regimes_are_dropped(prices, series):
    returns, _ = series
    labels = pd.Series("only_here", index=returns.index, dtype="object")
    labels.iloc[5:8] = "rare"
    breakdown = performance_by_regime(returns, returns, labels, min_days=20)
    assert "rare" not in breakdown.index


def test_regime_report_returns_three_breakdowns(prices, series):
    returns, _ = series
    report = regime_report(prices, returns * 0.6, returns)
    assert set(report) == {"volatility", "trend", "combined"}


def test_consistency_score_counts_wins():
    breakdown = pd.DataFrame({
        "outperformed": [True, True, False, False],
        "excess_return": [0.1, 0.05, -0.02, -0.08],
    }, index=["a", "b", "c", "d"])
    score = consistency_score(breakdown)
    assert score["n_regimes"] == 4
    assert score["n_outperformed"] == 2
    assert score["pct_outperformed"] == pytest.approx(50.0)
    assert score["worst_excess"] == pytest.approx(-0.08)


def test_consistency_score_of_an_empty_breakdown():
    assert consistency_score(pd.DataFrame())["n_regimes"] == 0


def test_summary_text_calls_out_a_single_regime_edge():
    breakdown = pd.DataFrame({
        "outperformed": [True, False, False, False],
        "excess_return": [0.5, -0.1, -0.2, -0.1],
    }, index=["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"])
    text = regime_summary_text(breakdown)
    assert "one environment" in text


def test_summary_text_recognises_consistency():
    breakdown = pd.DataFrame({
        "outperformed": [True, True, True],
        "excess_return": [0.1, 0.2, 0.05],
    }, index=["a", "b", "c"])
    assert "consistent" in regime_summary_text(breakdown)


# ================================================================== STRESS


def test_cost_sweep_returns_decrease_with_cost(series):
    returns, preds = series
    sweep = sweep_costs(returns, preds, Config(), [0, 5, 20, 50])
    assert len(sweep) == 4
    assert sweep["total_return"].is_monotonic_decreasing
    assert sweep["total_cost"].is_monotonic_increasing


def test_zero_cost_is_the_best_case(series):
    returns, preds = series
    sweep = sweep_costs(returns, preds, Config(), [0, 10])
    assert sweep.loc[0, "total_return"] > sweep.loc[1, "total_return"]


def test_threshold_sweep_reduces_exposure_and_trades(series):
    returns, preds = series
    sweep = sweep_thresholds(returns, preds, Config(), [0.0, 0.005, 0.02, 0.10])
    assert sweep["exposure"].is_monotonic_decreasing
    # A threshold far above any prediction stops trading entirely.
    assert sweep["exposure"].iloc[-1] == pytest.approx(0.0)
    assert sweep["n_trades"].iloc[-1] == 0


def test_a_strategy_that_never_trades_has_flat_equity(series):
    returns, preds = series
    sweep = sweep_thresholds(returns, preds, Config(), [10.0])
    assert sweep["final_equity"].iloc[0] == pytest.approx(1.0)


def test_period_sweep_splits_the_window(series):
    returns, preds = series
    sweep = sweep_periods(returns, preds, Config(), n_periods=4)
    assert len(sweep) == 4
    assert sweep["days"].sum() == pytest.approx(len(returns), abs=4)
    for a, b in zip(sweep.itertuples(), list(sweep.itertuples())[1:]):
        assert a.end <= b.start


def test_period_sweep_rejects_zero_periods(series):
    returns, preds = series
    with pytest.raises(ValueError):
        sweep_periods(returns, preds, Config(), n_periods=0)


def test_seed_sweep_measures_model_stability(prices):
    cfg = Config(start="2018-01-01", end="2021-02-01",
                 train_end="2019-12-31", valid_end="2020-06-30",
                 model="random_forest")
    data, _, features = build_dataset(prices, cfg)
    sweep = sweep_seeds(data, cfg, features, seeds=[0, 1, 2])
    assert len(sweep) == 3
    assert sweep["seed"].tolist() == [0, 1, 2]
    assert np.isfinite(sweep["total_return"]).all()


def test_break_even_cost_finds_the_crossover():
    sweep = pd.DataFrame({
        "cost_bps": [0, 5, 10, 20],
        "total_return": [0.30, 0.20, 0.05, -0.10],
    })
    assert break_even_cost(sweep, benchmark_return=0.10) == pytest.approx(5)


def test_break_even_cost_is_nan_when_never_winning():
    sweep = pd.DataFrame({"cost_bps": [0, 5], "total_return": [0.01, 0.0]})
    assert np.isnan(break_even_cost(sweep, benchmark_return=0.50))


def test_robustness_summary_aggregates_sweeps(series):
    returns, preds = series
    cfg = Config()
    sweeps = {
        "cost": sweep_costs(returns, preds, cfg, [0, 5, 20]),
        "threshold": sweep_thresholds(returns, preds, cfg, [0.0, 0.005]),
    }
    summary = robustness_summary(sweeps)
    assert set(summary["sweep"]) == {"cost", "threshold"}
    assert "pct_beating_benchmark" in summary.columns


def test_stress_verdict_is_honest_when_the_strategy_fails():
    sweeps = {"cost": pd.DataFrame({
        "total_return": [-0.1, -0.2, -0.3],
        "sharpe_ratio": [-0.5, -0.6, -0.7],
        "beats_buy_and_hold": [False, False, False],
    })}
    verdict = stress_verdict(sweeps)
    assert "no robust edge" in verdict


def test_stress_verdict_recognises_robustness():
    sweeps = {"cost": pd.DataFrame({
        "total_return": [0.3, 0.28, 0.25],
        "sharpe_ratio": [1.1, 1.0, 0.9],
        "beats_buy_and_hold": [True, True, True],
    })}
    assert "not an artefact" in stress_verdict(sweeps)


def test_stress_verdict_on_empty_input():
    assert "No stress results" in stress_verdict({})
