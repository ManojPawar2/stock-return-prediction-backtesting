"""Vectorised backtesting engine.

This module owns Rule 3 of the leakage contract, and it is the single most
important line in the project::

    position = signal.shift(1)

A prediction made at the close of day ``t`` cannot influence the position held
*during* day ``t``.  It can only set the position for ``t+1``.  Dropping that
shift is the most common way a backtest becomes fantasy, and it is a silent
failure: the equity curve simply becomes beautiful.

``tests/test_backtester.py`` proves the shift is load-bearing by feeding the
engine a perfect oracle signal and asserting that the lagged version is far
less profitable than the unlagged one.

Costs are charged on **turnover**, not on returns::

    turnover = |position_t - position_{t-1}|
    cost     = turnover * (commission + slippage) / 10_000

Turnover is 1.0 for flat->long, and 2.0 for long->short (the position moves two
units), which is exactly right: reversing a position costs twice as much as
opening one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .signals import buy_and_hold_signal

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Everything one backtest produced."""

    returns: pd.Series          # realised asset return per day
    signal: pd.Series           # decision made at the close of each day
    position: pd.Series         # position actually held that day (lagged)
    turnover: pd.Series         # |change in position|
    cost: pd.Series             # friction paid that day
    gross_returns: pd.Series    # position * asset return
    strategy_returns: pd.Series # gross minus cost
    equity: pd.Series           # cumulative growth of 1 unit
    benchmark_equity: pd.Series # buy-and-hold on the same dates
    trades: pd.DataFrame        # one row per completed round trip
    initial_capital: float = 10_000.0
    cost_bps: float = 0.0

    # ------------------------------------------------------------- helpers

    @property
    def portfolio_value(self) -> pd.Series:
        return self.equity * self.initial_capital

    @property
    def benchmark_value(self) -> pd.Series:
        return self.benchmark_equity * self.initial_capital

    @property
    def drawdown(self) -> pd.Series:
        return self.equity / self.equity.cummax() - 1.0

    @property
    def benchmark_drawdown(self) -> pd.Series:
        return self.benchmark_equity / self.benchmark_equity.cummax() - 1.0

    @property
    def benchmark_returns(self) -> pd.Series:
        """Buy-and-hold daily returns, on exactly the strategy's dates."""
        return self.returns.rename("benchmark_returns")

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_cost(self) -> float:
        return float(self.cost.sum())

    @property
    def exposure(self) -> float:
        """Fraction of days holding a non-zero position."""
        if not len(self.position):
            return 0.0
        return float((self.position != 0).mean())

    def to_frame(self) -> pd.DataFrame:
        """Everything per-day, for inspection and CSV export."""
        return pd.DataFrame({
            "asset_return": self.returns,
            "signal": self.signal,
            "position": self.position,
            "turnover": self.turnover,
            "cost": self.cost,
            "gross_return": self.gross_returns,
            "strategy_return": self.strategy_returns,
            "equity": self.equity,
            "benchmark_equity": self.benchmark_equity,
            "drawdown": self.drawdown,
        })


class Backtester:
    """Applies signals to returns with an execution lag, costs and slippage."""

    def __init__(
        self,
        cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        initial_capital: float = 10_000.0,
        execution_lag: int = 1,
    ) -> None:
        if cost_bps < 0 or slippage_bps < 0:
            raise ValueError("costs must be non-negative")
        if execution_lag < 0:
            raise ValueError("execution_lag must be >= 0")
        if execution_lag == 0:
            # Permitted only so the test suite can demonstrate what breaks.
            logger.warning(
                "execution_lag=0 lets a prediction trade the bar it was made "
                "on. This is lookahead bias and is only valid in tests."
            )
        self.cost_bps = float(cost_bps)
        self.slippage_bps = float(slippage_bps)
        self.initial_capital = float(initial_capital)
        self.execution_lag = int(execution_lag)

    @property
    def total_cost_bps(self) -> float:
        return self.cost_bps + self.slippage_bps

    # ---------------------------------------------------------------- run

    def run(
        self,
        returns: pd.Series,
        signals: pd.Series,
        prices: pd.Series | None = None,
    ) -> BacktestResult:
        """Backtest ``signals`` against realised ``returns``.

        Parameters
        ----------
        returns
            Realised asset return **on** each date (``close_t / close_{t-1} - 1``).
        signals
            Target position decided at the close of each date.
        prices
            Optional price series, used only to record entry/exit prices in
            the trade log.
        """
        returns = pd.Series(returns, dtype="float64").rename("asset_return")
        signals = pd.Series(signals, dtype="float64").rename("signal")

        # Work on the shared dates only, so a mismatched index cannot silently
        # misalign a signal against the wrong day's return.
        index = returns.index.intersection(signals.index).sort_values()
        if len(index) == 0:
            raise ValueError("returns and signals share no dates")
        returns = returns.loc[index].fillna(0.0)
        signals = signals.loc[index].fillna(0.0)

        # ---- THE EXECUTION LAG (Rule 3) --------------------------------
        # Decided at the close of t, therefore held during t+1.
        position = signals.shift(self.execution_lag).fillna(0.0).rename("position")

        # ---- costs on turnover -----------------------------------------
        # The first day's turnover is the cost of establishing the position
        # from a flat book, hence fillna with |position| rather than 0.
        turnover = position.diff().abs()
        turnover.iloc[0] = abs(position.iloc[0])
        turnover = turnover.fillna(0.0).rename("turnover")

        cost = (turnover * self.total_cost_bps / 10_000.0).rename("cost")

        gross = (position * returns).rename("gross_return")
        net = (gross - cost).rename("strategy_return")

        equity = (1.0 + net).cumprod().rename("equity")
        benchmark = (1.0 + returns).cumprod().rename("benchmark_equity")

        trades = self._build_trade_log(position, returns, cost, prices)

        logger.info(
            "Backtest: %d days, %d trades, exposure %.1f%%, cost %.4f",
            len(index), len(trades), (position != 0).mean() * 100, cost.sum(),
        )

        return BacktestResult(
            returns=returns,
            signal=signals,
            position=position,
            turnover=turnover,
            cost=cost,
            gross_returns=gross,
            strategy_returns=net,
            equity=equity,
            benchmark_equity=benchmark,
            trades=trades,
            initial_capital=self.initial_capital,
            cost_bps=self.total_cost_bps,
        )

    # --------------------------------------------------------- trade log

    @staticmethod
    def _build_trade_log(
        position: pd.Series,
        returns: pd.Series,
        cost: pd.Series,
        prices: pd.Series | None,
    ) -> pd.DataFrame:
        """One row per completed round trip.

        A trade runs from the first day of a non-zero position until the
        position changes to something else (including flat).
        """
        columns = [
            "entry_date", "exit_date", "direction", "holding_days",
            "entry_price", "exit_price", "gross_return", "cost", "net_return",
        ]
        if position.empty:
            return pd.DataFrame(columns=columns)

        rows: list[dict] = []
        values = position.to_numpy()
        dates = position.index

        start = None
        for i in range(len(values)):
            current = values[i]
            previous = values[i - 1] if i > 0 else 0.0

            if current != previous:
                if previous != 0.0 and start is not None:
                    rows.append(_close_trade(
                        start, i - 1, previous, position, returns, cost, prices
                    ))
                start = i if current != 0.0 else None

        if values[-1] != 0.0 and start is not None:
            rows.append(_close_trade(
                start, len(values) - 1, values[-1], position, returns, cost, prices
            ))

        if not rows:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame(rows)[columns]
        frame["win"] = frame["net_return"] > 0
        return frame


def _close_trade(
    start: int,
    end: int,
    direction: float,
    position: pd.Series,
    returns: pd.Series,
    cost: pd.Series,
    prices: pd.Series | None,
) -> dict:
    """Summarise one completed round trip from bar ``start`` to bar ``end``."""
    dates = position.index
    window = slice(start, end + 1)

    gross = float((1.0 + direction * returns.iloc[window]).prod() - 1.0)
    # Include the exit cost, charged on the bar after the position closes.
    cost_window = slice(start, min(end + 2, len(cost)))
    trade_cost = float(cost.iloc[cost_window].sum())

    entry_price = exit_price = float("nan")
    if prices is not None:
        aligned = prices.reindex(dates)
        entry_price = float(aligned.iloc[start])
        exit_price = float(aligned.iloc[end])

    return {
        "entry_date": dates[start],
        "exit_date": dates[end],
        "direction": "long" if direction > 0 else "short",
        "holding_days": end - start + 1,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross,
        "cost": trade_cost,
        "net_return": gross - trade_cost,
    }


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def backtester_from_config(cfg: Config) -> Backtester:
    return Backtester(
        cost_bps=cfg.transaction_cost_bps,
        slippage_bps=cfg.slippage_bps,
        initial_capital=cfg.initial_capital,
    )


def run_buy_and_hold(returns: pd.Series, cfg: Config | None = None) -> BacktestResult:
    """The baseline: long from the first day, never trading again."""
    bt = backtester_from_config(cfg) if cfg else Backtester(0.0, 0.0)
    # Signal one day early so the position is live on the very first bar.
    signal = buy_and_hold_signal(returns.index)
    result = bt.run(returns, signal)
    return result
