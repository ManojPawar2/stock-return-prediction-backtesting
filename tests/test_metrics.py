"""Tests for performance and risk metrics.

Every metric is checked against a series whose answer is known in closed form.
A quietly wrong metric is worse than a missing one: it yields a confident,
plausible, false conclusion.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    TRADING_DAYS,
    annualised_volatility,
    average_daily_return,
    cagr,
    calmar_ratio,
    compare_to_benchmark,
    conditional_value_at_risk,
    downside_deviation,
    drawdown_series,
    equity_curve,
    exposure,
    format_summary,
    max_drawdown,
    max_drawdown_duration,
    performance_summary,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summary_table,
    total_return,
    trade_statistics,
    value_at_risk,
    win_rate,
)


def const(value: float, n: int = TRADING_DAYS) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.Series([value] * n, index=idx)


# ------------------------------------------------------------------ returns


def test_equity_curve_compounds():
    r = pd.Series([0.10, 0.10, -0.10])
    eq = equity_curve(r)
    assert eq.iloc[0] == pytest.approx(1.10)
    assert eq.iloc[1] == pytest.approx(1.21)
    assert eq.iloc[2] == pytest.approx(1.089)


def test_total_return_is_compounded_not_summed():
    r = pd.Series([0.10, 0.10])
    assert total_return(r) == pytest.approx(0.21), "0.21, not 0.20"


def test_total_return_of_a_round_trip_is_zero_ish():
    r = pd.Series([0.5, -1 / 3])   # up 50%, then back down
    assert total_return(r) == pytest.approx(0.0)


def test_cagr_of_exactly_one_year_equals_total_return():
    """252 days at a constant rate: CAGR must equal the period return."""
    r = const(0.001, TRADING_DAYS)
    assert cagr(r) == pytest.approx(total_return(r), rel=1e-9)


def test_cagr_of_two_years_is_the_annualised_rate():
    """Doubling over exactly two years is sqrt(2) - 1 per year."""
    n = TRADING_DAYS * 2
    daily = 2 ** (1 / n) - 1
    assert cagr(const(daily, n)) == pytest.approx(np.sqrt(2) - 1, rel=1e-6)


def test_cagr_of_a_wipeout():
    r = pd.Series([-1.0, 0.0, 0.0])
    assert cagr(r) == pytest.approx(-1.0)


def test_average_daily_return():
    assert average_daily_return(pd.Series([0.01, -0.01, 0.02])) == pytest.approx(
        0.02 / 3
    )


# --------------------------------------------------------------------- risk


def test_annualised_volatility_scales_with_sqrt_252():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    assert annualised_volatility(r) == pytest.approx(
        r.std(ddof=1) * np.sqrt(252), rel=1e-9
    )
    # 1% daily is about 15.9% annualised.
    assert annualised_volatility(r) == pytest.approx(0.159, abs=0.01)


def test_volatility_of_a_constant_series_is_zero():
    assert annualised_volatility(const(0.001)) == pytest.approx(0.0)


def test_downside_deviation_ignores_gains():
    """Two series with identical losses but different gains must match."""
    a = pd.Series([-0.01, -0.02, 0.01, 0.02])
    b = pd.Series([-0.01, -0.02, 0.50, 0.90])
    assert downside_deviation(a) == pytest.approx(downside_deviation(b))


def test_downside_deviation_is_below_total_volatility():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.02, 1000))
    assert downside_deviation(r) < annualised_volatility(r)


def test_max_drawdown_is_hand_computable():
    # 1.0 -> 1.5 -> 0.75: peak 1.5, trough 0.75, so -50%.
    r = pd.Series([0.5, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_of_a_monotonic_rise_is_zero():
    assert max_drawdown(const(0.001)) == pytest.approx(0.0)


def test_max_drawdown_is_never_positive():
    rng = np.random.default_rng(2)
    assert max_drawdown(pd.Series(rng.normal(0.001, 0.02, 500))) <= 0


def test_drawdown_series_is_non_positive_and_zero_at_new_highs():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.015, 400))
    dd = drawdown_series(r)
    assert (dd <= 1e-12).all()
    # Wherever equity sets a new high above the starting capital, drawdown is 0.
    eq = equity_curve(r)
    at_high = eq >= eq.cummax() - 1e-12
    assert dd[at_high & (eq >= 1.0)].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_a_first_day_loss_counts_as_a_drawdown():
    """Falling 10% on day one is a real 10% drawdown from starting capital."""
    dd = drawdown_series(pd.Series([-0.10, 0.0]))
    assert dd.iloc[0] == pytest.approx(-0.10)
    assert max_drawdown(pd.Series([-0.10, 0.0])) == pytest.approx(-0.10)


def test_max_drawdown_duration_counts_underwater_days():
    # Down, down, then a full recovery on the third day.
    r = pd.Series([-0.1, -0.1, 0.2345679, 0.0])
    assert max_drawdown_duration(r) >= 2


def test_max_drawdown_duration_of_a_rising_series_is_zero():
    assert max_drawdown_duration(const(0.001, 50)) == 0


def test_value_at_risk_is_the_quantile():
    r = pd.Series(np.linspace(-0.10, 0.10, 101))
    assert value_at_risk(r, 0.05) == pytest.approx(np.quantile(r, 0.05))
    assert value_at_risk(r, 0.05) < 0


def test_cvar_is_worse_than_var():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.02, 2000))
    assert conditional_value_at_risk(r) <= value_at_risk(r)


# ------------------------------------------------------------ risk-adjusted


def test_sharpe_is_hand_computable():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-9)


def test_sharpe_subtracts_the_risk_free_rate():
    r = const(0.001, 500) + pd.Series(
        np.random.default_rng(6).normal(0, 0.005, 500),
        index=pd.bdate_range("2020-01-01", periods=500),
    )
    assert sharpe_ratio(r, risk_free_rate=0.05) < sharpe_ratio(r, risk_free_rate=0.0)


def test_sharpe_of_a_constant_series_is_nan():
    """Zero volatility makes the ratio undefined, not infinite."""
    assert np.isnan(sharpe_ratio(const(0.001)))


def test_sharpe_sign_follows_the_mean():
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, 1000)
    assert sharpe_ratio(pd.Series(noise + 0.002)) > 0
    assert sharpe_ratio(pd.Series(noise - 0.002)) < 0


def test_sortino_exceeds_sharpe_when_upside_dominates():
    """Big gains inflate Sharpe's denominator but not Sortino's.

    Losses must actually vary: a series of identical losses has zero downside
    deviation, which makes Sortino undefined rather than large.
    """
    rng = np.random.default_rng(20)
    losses = list(rng.uniform(-0.012, -0.008, 10))
    gains = list(rng.uniform(0.03, 0.07, 10))
    r = pd.Series(losses + gains)
    assert sortino_ratio(r) > sharpe_ratio(r)


def test_sortino_is_nan_when_every_loss_is_identical():
    """Zero downside dispersion is undefined, not infinitely good."""
    assert np.isnan(sortino_ratio(pd.Series([-0.01] * 10 + [0.05] * 10)))


def test_sortino_is_nan_without_losses():
    assert np.isnan(sortino_ratio(const(0.001, 100)))


def test_calmar_is_cagr_over_max_drawdown():
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.0008, 0.015, 1000))
    assert calmar_ratio(r) == pytest.approx(cagr(r) / abs(max_drawdown(r)), rel=1e-9)


def test_calmar_is_nan_without_a_drawdown():
    assert np.isnan(calmar_ratio(const(0.001, 100)))


# ----------------------------------------------------------------- trading


def test_win_rate_excludes_flat_days():
    """A day with no position is neither a win nor a loss."""
    r = pd.Series([0.01, -0.01, 0.0, 0.0, 0.01])
    assert win_rate(r) == pytest.approx(2 / 3)


def test_win_rate_is_nan_when_never_active():
    assert np.isnan(win_rate(pd.Series([0.0, 0.0, 0.0])))


def test_profit_factor_is_gains_over_losses():
    r = pd.Series([0.03, 0.01, -0.02])
    assert profit_factor(r) == pytest.approx(0.04 / 0.02)


def test_profit_factor_above_one_means_profitable():
    assert profit_factor(pd.Series([0.05, -0.01])) > 1
    assert profit_factor(pd.Series([0.01, -0.05])) < 1


def test_profit_factor_without_losses_is_infinite():
    assert profit_factor(pd.Series([0.01, 0.02])) == float("inf")


def test_exposure_counts_active_days():
    assert exposure(pd.Series([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.5)
    assert exposure(pd.Series([-1.0, -1.0])) == pytest.approx(1.0)


def test_trade_statistics_on_a_known_log():
    trades = pd.DataFrame({
        "net_return": [0.05, -0.02, 0.03, -0.01],
        "holding_days": [5, 3, 7, 1],
    })
    stats = trade_statistics(trades)
    assert stats["n_trades"] == 4
    assert stats["n_wins"] == 2
    assert stats["n_losses"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["avg_win"] == pytest.approx(0.04)
    assert stats["avg_loss"] == pytest.approx(-0.015)
    assert stats["best_trade"] == pytest.approx(0.05)
    assert stats["worst_trade"] == pytest.approx(-0.02)
    assert stats["avg_holding_days"] == pytest.approx(4.0)
    assert stats["payoff_ratio"] == pytest.approx(0.04 / 0.015)


def test_trade_statistics_on_an_empty_log():
    assert trade_statistics(pd.DataFrame())["n_trades"] == 0


# --------------------------------------------------------------- assembly


def test_performance_summary_has_every_metric():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.0005, 0.015, 500),
                  index=pd.bdate_range("2020-01-01", periods=500))
    summary = performance_summary(r, positions=pd.Series([1.0] * 500))
    for key in ("total_return", "cagr", "annualised_volatility", "sharpe_ratio",
                "sortino_ratio", "max_drawdown", "max_drawdown_days",
                "calmar_ratio", "win_rate", "profit_factor", "var_95",
                "cvar_95", "n_days", "exposure"):
        assert key in summary


def test_compare_to_benchmark_computes_the_difference():
    rng = np.random.default_rng(10)
    strat = pd.Series(rng.normal(0.001, 0.01, 300))
    bench = pd.Series(rng.normal(0.0005, 0.012, 300))
    frame = compare_to_benchmark(strat, bench)
    assert list(frame.columns) == ["strategy", "buy_and_hold", "difference"]
    assert frame.loc["total_return", "difference"] == pytest.approx(
        frame.loc["total_return", "strategy"] - frame.loc["total_return", "buy_and_hold"]
    )


def test_identical_series_compare_to_zero_difference():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.001, 0.01, 300))
    frame = compare_to_benchmark(r, r.copy())
    assert frame["difference"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_summary_table_builds_a_model_comparison():
    table = summary_table({
        "linear": {"sharpe_ratio": 0.5, "total_return": 0.1},
        "xgboost": {"sharpe_ratio": 0.7, "total_return": 0.2},
    })
    assert list(table.index) == ["linear", "xgboost"]


# ---------------------------------------------------------------- edge cases


def test_metrics_handle_an_empty_series():
    empty = pd.Series([], dtype=float)
    assert np.isnan(total_return(empty))
    assert np.isnan(cagr(empty))
    assert np.isnan(max_drawdown(empty))


def test_metrics_drop_nan_and_inf():
    r = pd.Series([0.01, np.nan, np.inf, 0.02])
    assert total_return(r) == pytest.approx(1.01 * 1.02 - 1)


def test_single_observation_gives_nan_for_dispersion():
    one = pd.Series([0.01])
    assert np.isnan(annualised_volatility(one))
    assert np.isnan(sharpe_ratio(one))


def test_format_summary_renders_percentages_and_na():
    rendered = format_summary({
        "total_return": 0.1234,
        "sharpe_ratio": 1.5,
        "calmar_ratio": float("nan"),
        "n_days": 500,
    })
    values = dict(zip(rendered["metric"], rendered["value"]))
    assert values["Total Return"] == "12.34%"
    assert values["Sharpe Ratio"] == "1.500"
    assert values["Calmar Ratio"] == "n/a"
    assert values["N Days"] == "500"
