"""Strategy performance and risk metrics.

Formulas match README section 16 exactly.  Each one is verified in
``tests/test_metrics.py`` against a hand-constructed series whose answer is
known in closed form, because a metric that is quietly wrong is worse than no
metric at all — it produces a confident, plausible, false conclusion.

Two conventions used throughout:

* Returns are **simple** (not log) daily returns, so equity compounds as
  ``(1 + r).cumprod()``.
* Annualisation uses ``trading_days = 252``.  Volatility scales with the square
  root of time; mean return scales linearly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()


# --------------------------------------------------------------------------
# Return metrics
# --------------------------------------------------------------------------


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compounded growth of ``initial`` through the return series."""
    return initial * (1.0 + _clean(returns)).cumprod()


def total_return(returns: pd.Series) -> float:
    """Cumulative return over the whole period."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Compound annual growth rate.

    Uses the number of observations rather than the calendar span, so a series
    with gaps is not silently annualised over the wrong horizon.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        # A total wipeout has no meaningful growth rate.
        return -1.0
    years = len(r) / trading_days
    return float(growth ** (1.0 / years) - 1.0)


def average_daily_return(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(r.mean()) if len(r) else float("nan")


# --------------------------------------------------------------------------
# Risk metrics
# --------------------------------------------------------------------------


def annualised_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Standard deviation of daily returns, scaled by sqrt(252)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, trading_days: int = TRADING_DAYS
) -> float:
    """Volatility of returns below a minimum acceptable return.

    Only losses count, which is why Sortino is often preferred to Sharpe: an
    upside surprise is not a risk.
    """
    r = _clean(returns)
    downside = r[r < mar]
    if len(downside) < 2:
        return float("nan")
    return float(downside.std(ddof=1) * np.sqrt(trading_days))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak, at every point in time.

    The starting capital counts as the first peak.  Taking the peak from the
    equity curve alone would mean a strategy that falls 10% on its very first
    day reports a drawdown of zero, because that day is its own running
    maximum — yet the investor is unambiguously 10% down on the money they
    started with.
    """
    equity = equity_curve(returns)
    if equity.empty:
        return equity
    peak = np.maximum(equity.cummax(), 1.0)
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """The worst peak-to-trough fall. Returned as a negative number."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest run of consecutive days spent below a previous peak.

    Often more informative than the depth: a 20% drawdown recovered in a month
    is a different experience from a 20% drawdown lasting three years.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return 0
    underwater = dd < -1e-12
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical VaR: the loss exceeded only ``level`` of the time."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(np.quantile(r, level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Average loss on the days worse than the VaR threshold (expected shortfall)."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    threshold = np.quantile(r, level)
    tail = r[r <= threshold]
    return float(tail.mean()) if len(tail) else float("nan")


# --------------------------------------------------------------------------
# Risk-adjusted metrics
# --------------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised excess return per unit of volatility.

    ``risk_free_rate`` is an annual rate and is converted to a daily one before
    subtraction.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    # A tolerance, not `== 0`: the float std of a repeated constant is on the
    # order of 1e-19, which sails past an equality check and produces a
    # nonsensical Sharpe of ~1e16 instead of "undefined".
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    excess = r - risk_free_rate / trading_days
    return float(excess.mean() / sd * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Sharpe, but penalising only downside volatility."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    dd = downside_deviation(r, 0.0, trading_days)
    if not np.isfinite(dd) or dd < 1e-12:
        return float("nan")
    excess = r.mean() - risk_free_rate / trading_days
    return float(excess * trading_days / dd)


def calmar_ratio(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """CAGR divided by the magnitude of the worst drawdown."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or abs(mdd) < 1e-12:
        return float("nan")
    return float(cagr(returns, trading_days) / abs(mdd))


# --------------------------------------------------------------------------
# Trading statistics
# --------------------------------------------------------------------------


def win_rate(returns: pd.Series) -> float:
    """Share of active days that were profitable.

    Flat days are excluded — a day with no position is neither a win nor a
    loss, and counting them would make an inactive strategy look consistent.
    """
    r = _clean(returns)
    active = r[r != 0]
    if active.empty:
        return float("nan")
    return float((active > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross gains divided by gross losses. Above 1.0 is profitable."""
    r = _clean(returns)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def exposure(positions: pd.Series) -> float:
    """Fraction of days holding a non-zero position."""
    p = pd.Series(positions, dtype="float64").dropna()
    return float((p != 0).mean()) if len(p) else float("nan")


def trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Summary of a trade log produced by the backtester."""
    empty = {
        "n_trades": 0, "n_wins": 0, "n_losses": 0, "win_rate": float("nan"),
        "avg_win": float("nan"), "avg_loss": float("nan"),
        "avg_trade": float("nan"), "best_trade": float("nan"),
        "worst_trade": float("nan"), "avg_holding_days": float("nan"),
        "payoff_ratio": float("nan"),
    }
    if trades is None or trades.empty:
        return empty

    net = trades["net_return"]
    wins, losses = net[net > 0], net[net <= 0]
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")

    return {
        "n_trades": int(len(trades)),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate": float(len(wins) / len(trades)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": float(net.mean()),
        "best_trade": float(net.max()),
        "worst_trade": float(net.min()),
        "avg_holding_days": float(trades["holding_days"].mean()),
        "payoff_ratio": (
            float(avg_win / abs(avg_loss))
            if len(wins) and len(losses) and avg_loss != 0 else float("nan")
        ),
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
    positions: pd.Series | None = None,
) -> dict[str, float]:
    """Every headline metric for one return series."""
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, trading_days),
        "avg_daily_return": average_daily_return(returns),
        "annualised_volatility": annualised_volatility(returns, trading_days),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, trading_days),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, trading_days),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_days": max_drawdown_duration(returns),
        "calmar_ratio": calmar_ratio(returns, trading_days),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "var_95": value_at_risk(returns, 0.05),
        "cvar_95": conditional_value_at_risk(returns, 0.05),
        "n_days": int(len(_clean(returns))),
    }
    if positions is not None:
        out["exposure"] = exposure(positions)
    return out


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
    positions: pd.Series | None = None,
) -> pd.DataFrame:
    """Side-by-side strategy vs buy-and-hold, plus the difference.

    Reporting the strategy alone is meaningless: a Sharpe of 0.8 is only good
    news if the benchmark did worse.
    """
    strategy = performance_summary(
        strategy_returns, risk_free_rate, trading_days, positions
    )
    benchmark = performance_summary(benchmark_returns, risk_free_rate, trading_days)

    frame = pd.DataFrame({"strategy": strategy, "buy_and_hold": benchmark})
    frame["difference"] = frame["strategy"] - frame["buy_and_hold"]
    return frame


def summary_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn ``{model_name: performance_summary}`` into a comparison table."""
    return pd.DataFrame(results).T


#: Metrics whose display should be a percentage rather than a raw ratio.
PERCENT_METRICS = frozenset({
    "total_return", "cagr", "avg_daily_return", "annualised_volatility",
    "max_drawdown", "win_rate", "var_95", "cvar_95", "exposure",
    "avg_win", "avg_loss", "avg_trade", "best_trade", "worst_trade",
})


def format_summary(summary: dict[str, float]) -> pd.DataFrame:
    """Human-readable rendering for the dashboard and the report."""
    rows = []
    for key, value in summary.items():
        if isinstance(value, (int, np.integer)) and key not in PERCENT_METRICS:
            display = f"{value:,}"
        elif value is None or (isinstance(value, float) and not np.isfinite(value)):
            display = "n/a" if value is None or np.isnan(value) else "inf"
        elif key in PERCENT_METRICS:
            display = f"{value * 100:.2f}%"
        else:
            display = f"{value:.3f}"
        rows.append({"metric": key.replace("_", " ").title(), "value": display})
    return pd.DataFrame(rows)
