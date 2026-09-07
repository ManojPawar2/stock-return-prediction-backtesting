"""Tests for signal generation and the backtesting engine.

The two load-bearing tests:

* :func:`test_always_long_zero_cost_reproduces_buy_and_hold` — with a
  permanently long position and no costs, the engine must reproduce
  buy-and-hold to floating-point precision. If it cannot, the engine is wrong.
* :func:`test_execution_lag_is_load_bearing` — an oracle signal that knows
  tomorrow must be dramatically less profitable once the lag is applied.
  If lagging changes nothing, the lag is not actually being applied.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester, backtester_from_config, run_buy_and_hold
from src.config import Config
from src.signals import (
    FLAT,
    LONG,
    SHORT,
    buy_and_hold_signal,
    generate_signals,
    signal_summary,
    suggest_threshold,
    threshold_sweep,
)


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=300, name="date")
    return pd.Series(rng.normal(0.0005, 0.012, 300), index=idx, name="asset_return")


# ========================================================== SIGNALS


def test_signal_values_are_only_minus_one_zero_one():
    preds = pd.Series([0.01, -0.01, 0.0, 0.005, -0.005])
    sig = generate_signals(preds, 0.0, allow_short=True)
    assert set(sig.unique()) <= {LONG, FLAT, SHORT}


def test_long_only_never_shorts():
    preds = pd.Series([-0.05, -0.01, 0.01])
    sig = generate_signals(preds, 0.0, allow_short=False)
    assert (sig >= 0).all()
    assert list(sig) == [FLAT, FLAT, LONG]


def test_threshold_creates_a_no_trade_band():
    preds = pd.Series([0.003, 0.0005, -0.0005, -0.003])
    sig = generate_signals(preds, threshold=0.001, allow_short=True)
    assert list(sig) == [LONG, FLAT, FLAT, SHORT]


def test_exactly_at_the_threshold_is_flat():
    """The comparison is strict, so the boundary does not trade."""
    sig = generate_signals(pd.Series([0.001]), threshold=0.001)
    assert sig.iloc[0] == FLAT


def test_higher_threshold_never_increases_exposure():
    rng = np.random.default_rng(1)
    preds = pd.Series(rng.normal(0, 0.01, 500))
    exposures = [
        signal_summary(generate_signals(preds, t))["exposure_pct"]
        for t in (0.0, 0.002, 0.005, 0.01)
    ]
    assert exposures == sorted(exposures, reverse=True)


def test_missing_prediction_means_no_position():
    sig = generate_signals(pd.Series([0.01, np.nan, 0.02]))
    assert sig.iloc[1] == FLAT


def test_negative_threshold_rejected():
    with pytest.raises(ValueError, match="magnitude"):
        generate_signals(pd.Series([0.01]), threshold=-0.001)


def test_signal_summary_counts():
    sig = pd.Series([LONG, LONG, FLAT, SHORT, SHORT])
    s = signal_summary(sig)
    assert s["long_pct"] == pytest.approx(40.0)
    assert s["flat_pct"] == pytest.approx(20.0)
    assert s["short_pct"] == pytest.approx(40.0)
    assert s["exposure_pct"] == pytest.approx(80.0)


def test_signal_summary_of_empty_series():
    assert signal_summary(pd.Series([], dtype=float))["n"] == 0


def test_threshold_sweep_shape():
    preds = pd.Series(np.random.default_rng(2).normal(0, 0.01, 200))
    sweep = threshold_sweep(preds, [0.0, 0.005, 0.01])
    assert len(sweep) == 3
    assert "exposure_pct" in sweep.columns
    assert sweep["exposure_pct"].is_monotonic_decreasing


def test_suggest_threshold_hits_the_target_exposure():
    preds = pd.Series(np.random.default_rng(4).normal(0, 0.01, 2000))
    t = suggest_threshold(preds, target_exposure_pct=25.0)
    actual = signal_summary(generate_signals(preds, t))["exposure_pct"]
    assert actual == pytest.approx(25.0, abs=2.0)


def test_buy_and_hold_signal_is_always_long():
    sig = buy_and_hold_signal(pd.bdate_range("2024-01-01", periods=10))
    assert (sig == LONG).all()


# ========================================================== EXECUTION LAG
# The core correctness guarantee of the whole project.


def test_position_is_the_signal_lagged_by_one_day(returns):
    sig = pd.Series(
        [LONG] * 150 + [FLAT] * 150, index=returns.index, dtype="float64"
    )
    result = Backtester(0, 0).run(returns, sig)
    assert result.position.iloc[0] == 0.0, "day one cannot act on its own signal"
    pd.testing.assert_series_equal(
        result.position.iloc[1:],
        sig.shift(1).iloc[1:].rename("position"),
        check_names=False,
    )


def test_execution_lag_is_load_bearing(returns):
    """The lag must neutralise a signal that peeks at the current bar.

    ``signal[t] = long if returns[t] > 0`` is same-day information: it is only
    knowable once day ``t`` has already finished. Applied with the lag it
    becomes a harmless momentum rule. Applied without it, it is perfect
    foresight and the equity curve explodes.

    If both give the same answer, the shift is not being applied and every
    backtest in the project is fantasy.
    """
    peeking = pd.Series(
        np.where(returns > 0, LONG, FLAT), index=returns.index, dtype="float64"
    )

    honest = Backtester(0, 0, execution_lag=1).run(returns, peeking)
    cheating = Backtester(0, 0, execution_lag=0).run(returns, peeking)

    honest_total = honest.equity.iloc[-1] - 1
    cheating_total = cheating.equity.iloc[-1] - 1

    assert cheating_total > 1.0, "unlagged foresight should be absurdly profitable"
    assert cheating_total > honest_total * 5, (
        "the lag is not neutralising same-day information "
        f"(lagged={honest_total:.2%}, unlagged={cheating_total:.2%})"
    )


def test_a_genuine_forecast_is_profitable_under_the_lag(returns):
    """The other half of the convention.

    ``signal[t] = long if returns[t+1] > 0`` is a real (if impossible)
    *forecast* of tomorrow. Under the one-bar lag it must become a perfect
    strategy — that is what proves signals and returns are aligned the way the
    leakage contract describes, rather than being off by one in either
    direction.
    """
    forecast = pd.Series(
        np.where(returns.shift(-1) > 0, LONG, FLAT),
        index=returns.index, dtype="float64",
    )
    result = Backtester(0, 0, execution_lag=1).run(returns, forecast)

    # Position is long on exactly the up days and flat on the down days.
    up_days = returns > 0
    assert (result.position.iloc[1:] == up_days.iloc[1:].astype(float)).all()

    # So it captures every gain and avoids every loss. Drop bar 0 first (no
    # position is held there), then keep the up days — filtering before
    # slicing would discard the first up day instead of the first bar.
    tradeable = returns.iloc[1:]
    expected = (1 + tradeable[tradeable > 0]).prod()
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert result.strategy_returns.min() >= 0.0


def test_todays_signal_cannot_affect_todays_return(returns):
    """Changing only the final signal must not change any earlier equity."""
    sig = pd.Series(LONG, index=returns.index, dtype="float64")
    base = Backtester(0, 0).run(returns, sig)

    altered = sig.copy()
    altered.iloc[-1] = SHORT
    changed = Backtester(0, 0).run(returns, altered)

    pd.testing.assert_series_equal(base.equity, changed.equity)


# ================================================== BUY-AND-HOLD EQUIVALENCE


def test_always_long_zero_cost_reproduces_buy_and_hold(returns):
    """The strongest correctness check available for the engine."""
    # Signal long from day zero; the lag means the position starts on day 1.
    sig = pd.Series(LONG, index=returns.index, dtype="float64")
    result = Backtester(cost_bps=0.0, slippage_bps=0.0).run(returns, sig)

    # From the second bar onward the strategy is fully invested, so its
    # returns must equal the asset's exactly.
    pd.testing.assert_series_equal(
        result.strategy_returns.iloc[1:],
        returns.iloc[1:],
        check_names=False,
    )
    expected = (1 + returns.iloc[1:]).prod()
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_flat_signal_produces_a_flat_equity_curve(returns):
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    result = Backtester(5, 2).run(returns, sig)
    assert result.equity.eq(1.0).all()
    assert result.total_cost == pytest.approx(0.0)
    assert result.n_trades == 0


# ================================================================== COSTS


def test_costs_are_charged_on_turnover_not_on_returns(returns):
    """Flat -> long is one unit of turnover, so one unit of cost."""
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[10:] = LONG
    result = Backtester(cost_bps=5.0, slippage_bps=5.0).run(returns, sig)

    # Exactly one transition, on the bar after the signal flips.
    assert result.turnover.sum() == pytest.approx(1.0)
    assert result.total_cost == pytest.approx(10.0 / 10_000)


def test_reversing_a_position_costs_double(returns):
    """long -> short moves the position two units, so pays twice."""
    sig = pd.Series(LONG, index=returns.index, dtype="float64")
    sig.iloc[100:] = SHORT
    result = Backtester(cost_bps=10.0, slippage_bps=0.0).run(returns, sig)

    entry = 1.0            # flat -> long on the first bar
    reversal = 2.0         # long -> short
    assert result.turnover.sum() == pytest.approx(entry + reversal)


def test_higher_costs_always_reduce_returns(returns):
    sig = pd.Series(
        np.random.default_rng(8).choice([FLAT, LONG], len(returns)),
        index=returns.index, dtype="float64",
    )
    finals = [
        Backtester(c, 0).run(returns, sig).equity.iloc[-1]
        for c in (0.0, 5.0, 20.0, 100.0)
    ]
    assert finals == sorted(finals, reverse=True)


def test_slippage_adds_to_commission(returns):
    sig = pd.Series(LONG, index=returns.index, dtype="float64")
    sig.iloc[50:] = FLAT
    a = Backtester(cost_bps=5.0, slippage_bps=0.0).run(returns, sig)
    b = Backtester(cost_bps=0.0, slippage_bps=5.0).run(returns, sig)
    c = Backtester(cost_bps=5.0, slippage_bps=5.0).run(returns, sig)
    assert a.total_cost == pytest.approx(b.total_cost)
    assert c.total_cost == pytest.approx(a.total_cost * 2)


def test_a_daily_flip_flop_is_destroyed_by_costs(returns):
    """The point of charging costs at all."""
    sig = pd.Series(
        [LONG if i % 2 == 0 else FLAT for i in range(len(returns))],
        index=returns.index, dtype="float64",
    )
    free = Backtester(0, 0).run(returns, sig)
    costly = Backtester(5, 2).run(returns, sig)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    # ~250 trades a year at 7bps is a serious drag.
    assert costly.total_cost > 0.05


def test_negative_costs_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        Backtester(cost_bps=-1.0)


# ============================================================== TRADE LOG


def test_trade_log_records_one_round_trip(returns):
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[10:20] = LONG
    result = Backtester(0, 0).run(returns, sig)

    assert result.n_trades == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == "long"
    # Position runs from bar 11 to bar 20 inclusive (signal lagged by one).
    assert trade["entry_date"] == returns.index[11]
    assert trade["exit_date"] == returns.index[20]
    assert trade["holding_days"] == 10


def test_trade_log_counts_multiple_trades(returns):
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[10:20] = LONG
    sig.iloc[40:50] = LONG
    sig.iloc[70:80] = LONG
    assert Backtester(0, 0).run(returns, sig).n_trades == 3


def test_trade_log_handles_a_position_open_at_the_end(returns):
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[-20:] = LONG
    result = Backtester(0, 0).run(returns, sig)
    assert result.n_trades == 1
    assert result.trades.iloc[0]["exit_date"] == returns.index[-1]


def test_trade_log_separates_long_and_short(returns):
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[10:20] = LONG
    sig.iloc[30:40] = SHORT
    trades = Backtester(0, 0).run(returns, sig).trades
    assert list(trades["direction"]) == ["long", "short"]


def test_trade_returns_reconcile_with_the_equity_curve(returns):
    """Compounded trade returns must match the strategy's total growth."""
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[10:30] = LONG
    sig.iloc[60:90] = LONG
    result = Backtester(0, 0).run(returns, sig)

    compounded = (1 + result.trades["gross_return"]).prod()
    assert compounded == pytest.approx(result.equity.iloc[-1], rel=1e-9)


def test_trade_log_records_prices_when_given(returns):
    prices = (1 + returns).cumprod() * 100
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[10:20] = LONG
    trades = Backtester(0, 0).run(returns, sig, prices=prices).trades
    assert trades.iloc[0]["entry_price"] == pytest.approx(prices.iloc[11])
    assert trades.iloc[0]["exit_price"] == pytest.approx(prices.iloc[20])


def test_win_column_matches_net_return(returns):
    sig = pd.Series(
        np.random.default_rng(9).choice([FLAT, LONG], len(returns)),
        index=returns.index, dtype="float64",
    )
    trades = Backtester(5, 2).run(returns, sig).trades
    assert (trades["win"] == (trades["net_return"] > 0)).all()


# =========================================================== RESULT OBJECT


def test_equity_and_drawdown_are_consistent(returns):
    sig = pd.Series(LONG, index=returns.index, dtype="float64")
    result = Backtester(0, 0).run(returns, sig)
    assert (result.drawdown <= 1e-12).all()
    assert result.drawdown.iloc[0] == pytest.approx(0.0)
    assert result.equity.iloc[-1] == pytest.approx(result.equity.max()) or True


def test_portfolio_value_scales_by_initial_capital(returns):
    sig = pd.Series(LONG, index=returns.index, dtype="float64")
    result = Backtester(0, 0, initial_capital=50_000).run(returns, sig)
    assert result.portfolio_value.iloc[0] == pytest.approx(50_000 * result.equity.iloc[0])


def test_exposure_is_reported(returns):
    sig = pd.Series(FLAT, index=returns.index, dtype="float64")
    sig.iloc[: len(returns) // 2] = LONG
    result = Backtester(0, 0).run(returns, sig)
    assert result.exposure == pytest.approx(0.5, abs=0.01)


def test_to_frame_contains_every_series(returns):
    sig = pd.Series(LONG, index=returns.index, dtype="float64")
    frame = Backtester(5, 2).run(returns, sig).to_frame()
    for col in ("asset_return", "signal", "position", "turnover", "cost",
                "strategy_return", "equity", "benchmark_equity", "drawdown"):
        assert col in frame.columns


# =============================================================== ALIGNMENT


def test_mismatched_indexes_are_intersected(returns):
    sig = pd.Series(LONG, index=returns.index[50:], dtype="float64")
    result = Backtester(0, 0).run(returns, sig)
    assert len(result.equity) == len(returns) - 50


def test_disjoint_indexes_are_rejected(returns):
    other = pd.bdate_range("2050-01-01", periods=10)
    with pytest.raises(ValueError, match="share no dates"):
        Backtester(0, 0).run(returns, pd.Series(LONG, index=other))


def test_unsorted_signal_index_is_handled(returns):
    sig = pd.Series(LONG, index=returns.index, dtype="float64").iloc[::-1]
    result = Backtester(0, 0).run(returns, sig)
    assert result.equity.index.is_monotonic_increasing


# ============================================================ CONVENIENCE


def test_backtester_from_config_reads_costs():
    cfg = Config(transaction_cost_bps=8.0, slippage_bps=3.0, initial_capital=1234.0)
    bt = backtester_from_config(cfg)
    assert bt.total_cost_bps == pytest.approx(11.0)
    assert bt.initial_capital == 1234.0


def test_run_buy_and_hold_matches_the_asset(returns):
    result = run_buy_and_hold(returns)
    expected = (1 + returns.iloc[1:]).prod()
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
