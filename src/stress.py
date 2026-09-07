"""Stress testing.

One backtest is one number from one configuration.  The question that matters
is whether the result survives when the assumptions move.  Each sweep here
varies a single dimension and holds the rest fixed, so the sensitivity is
attributable.

The sweeps deliberately include settings that should *break* the strategy —
50 bps costs, a threshold high enough to stop trading.  A result that survives
every configuration is usually a bug; a result that degrades smoothly and
predictably is a result you can reason about.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .backtester import Backtester
from .config import Config
from .dataset import build_dataset, realised_returns
from .metrics import max_drawdown, performance_summary, sharpe_ratio, total_return
from .models import ModelPipeline
from .signals import generate_signals

logger = logging.getLogger(__name__)


def _run_one(
    returns: pd.Series,
    predictions: pd.Series,
    threshold: float,
    cost_bps: float,
    slippage_bps: float,
    allow_short: bool,
    initial_capital: float = 10_000.0,
) -> dict[str, float]:
    """Backtest one parameter combination and summarise it."""
    signals = generate_signals(predictions, threshold, allow_short)
    result = Backtester(cost_bps, slippage_bps, initial_capital).run(returns, signals)
    summary = performance_summary(result.strategy_returns, positions=result.position)
    return {
        "total_return": summary["total_return"],
        "cagr": summary["cagr"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "max_drawdown": summary["max_drawdown"],
        "win_rate": summary["win_rate"],
        "exposure": summary["exposure"],
        "n_trades": int(result.n_trades),
        "total_cost": result.total_cost,
        "final_equity": float(result.equity.iloc[-1]),
    }


# --------------------------------------------------------------------------
# Single-dimension sweeps
# --------------------------------------------------------------------------


def sweep_costs(
    returns: pd.Series,
    predictions: pd.Series,
    cfg: Config,
    cost_grid_bps: list[float] | None = None,
) -> pd.DataFrame:
    """Vary transaction cost. Answers: at what cost does the edge vanish?"""
    grid = cost_grid_bps if cost_grid_bps is not None else cfg.stress.cost_grid_bps
    benchmark = total_return(returns)
    rows = []
    for cost in grid:
        stats = _run_one(returns, predictions, cfg.signal_threshold, float(cost),
                         0.0, cfg.allow_short, cfg.initial_capital)
        rows.append({
            "cost_bps": float(cost),
            **stats,
            "beats_buy_and_hold": stats["total_return"] > benchmark,
        })
    return pd.DataFrame(rows)


def sweep_thresholds(
    returns: pd.Series,
    predictions: pd.Series,
    cfg: Config,
    threshold_grid: list[float] | None = None,
) -> pd.DataFrame:
    """Vary the signal threshold. Answers: is one number doing all the work?"""
    grid = threshold_grid if threshold_grid is not None else cfg.stress.threshold_grid
    benchmark = total_return(returns)
    rows = []
    for threshold in grid:
        stats = _run_one(returns, predictions, float(threshold),
                         cfg.transaction_cost_bps, cfg.slippage_bps,
                         cfg.allow_short, cfg.initial_capital)
        rows.append({
            "threshold": float(threshold),
            **stats,
            "beats_buy_and_hold": stats["total_return"] > benchmark,
        })
    return pd.DataFrame(rows)


def sweep_seeds(
    data: pd.DataFrame,
    cfg: Config,
    features: list[str],
    model_name: str | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Refit a tree model under several seeds.

    Answers: is the result a property of the data, or of one lucky
    initialisation? A large spread means the model is fitting noise.
    """
    from .dataset import TARGET, split_from_config

    name = model_name or cfg.model
    grid = seeds if seeds is not None else cfg.stress.seed_grid
    split = split_from_config(data, cfg)
    test = split.test
    returns = realised_returns(test)

    rows = []
    for seed in grid:
        pipe = ModelPipeline(name, cfg.params_for(name), int(seed),
                             cfg.target_kind).fit(
            split.train[features], split.train[TARGET]
        )
        preds = pipe.predict(test[features])
        stats = _run_one(returns, preds, cfg.signal_threshold,
                         cfg.transaction_cost_bps, cfg.slippage_bps,
                         cfg.allow_short, cfg.initial_capital)
        rows.append({"seed": int(seed), **stats})
    return pd.DataFrame(rows)


def sweep_tickers(
    cfg: Config,
    tickers: list[str] | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Run the identical pipeline across a basket. Answers: does it generalise?

    A strategy that only works on the ticker it was developed against has been
    fitted to that ticker's history.
    """
    from .data_loader import load_prices
    from .dataset import TARGET, split_from_config
    from .preprocessing import clean

    name = model_name or cfg.model
    symbols = tickers if tickers is not None else cfg.universe

    rows = []
    for ticker in symbols:
        try:
            raw = load_prices(ticker, cfg.start, cfg.end, cfg.use_cache)
            cleaned, _ = clean(raw)
            data, split, features = build_dataset(cleaned, cfg)
            pipe = ModelPipeline(name, cfg.params_for(name), cfg.seed,
                                 cfg.target_kind).fit(
                split.train[features], split.train[TARGET]
            )
            test = split.test
            returns = realised_returns(test)
            preds = pipe.predict(test[features])
            stats = _run_one(returns, preds, cfg.signal_threshold,
                             cfg.transaction_cost_bps, cfg.slippage_bps,
                             cfg.allow_short, cfg.initial_capital)
            bh = total_return(returns)
            rows.append({
                "ticker": ticker, **stats,
                "buy_and_hold_return": bh,
                "excess_return": stats["total_return"] - bh,
                "beats_buy_and_hold": stats["total_return"] > bh,
            })
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the sweep
            logger.warning("Stress sweep skipped %s: %s", ticker, exc)
    return pd.DataFrame(rows)


def sweep_periods(
    returns: pd.Series,
    predictions: pd.Series,
    cfg: Config,
    n_periods: int = 4,
) -> pd.DataFrame:
    """Split the test window into consecutive blocks and score each.

    Answers: was the whole result produced by one good stretch?
    """
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")

    index = returns.index.intersection(predictions.index)
    edges = np.linspace(0, len(index), n_periods + 1).astype(int)

    rows = []
    for i in range(n_periods):
        block = index[edges[i]:edges[i + 1]]
        if len(block) < 20:
            continue
        r, p = returns.loc[block], predictions.loc[block]
        stats = _run_one(r, p, cfg.signal_threshold, cfg.transaction_cost_bps,
                         cfg.slippage_bps, cfg.allow_short, cfg.initial_capital)
        bh = total_return(r)
        rows.append({
            "period": i + 1,
            "start": block.min().date(),
            "end": block.max().date(),
            "days": len(block),
            **stats,
            "buy_and_hold_return": bh,
            "excess_return": stats["total_return"] - bh,
            "beats_buy_and_hold": stats["total_return"] > bh,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def break_even_cost(sweep: pd.DataFrame, benchmark_return: float) -> float:
    """Highest cost at which the strategy still beats buy-and-hold.

    ``nan`` when it never does — which is itself a clear answer.
    """
    winners = sweep[sweep["total_return"] > benchmark_return]
    return float(winners["cost_bps"].max()) if len(winners) else float("nan")


def robustness_summary(sweeps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How often the strategy survived, per sweep dimension."""
    rows = []
    for name, frame in sweeps.items():
        if frame.empty:
            continue
        row = {
            "sweep": name,
            "n_configurations": len(frame),
            "return_min": frame["total_return"].min(),
            "return_max": frame["total_return"].max(),
            "return_mean": frame["total_return"].mean(),
            "return_std": frame["total_return"].std(),
            "sharpe_min": frame["sharpe_ratio"].min(),
            "sharpe_max": frame["sharpe_ratio"].max(),
        }
        if "beats_buy_and_hold" in frame.columns:
            row["pct_beating_benchmark"] = float(
                frame["beats_buy_and_hold"].mean() * 100
            )
        rows.append(row)
    return pd.DataFrame(rows)


def stress_verdict(sweeps: dict[str, pd.DataFrame]) -> str:
    """Plain-English reading of the sweeps."""
    summary = robustness_summary(sweeps)
    if summary.empty:
        return "No stress results to assess."

    if "pct_beating_benchmark" not in summary.columns:
        return "Sweeps completed, but no benchmark comparison was available."

    rates = summary["pct_beating_benchmark"].dropna()
    if rates.empty:
        return "Sweeps completed, but no benchmark comparison was available."

    overall = rates.mean()
    if overall >= 75:
        verdict = (
            "The strategy beats buy-and-hold across most configurations, which "
            "suggests the result is not an artefact of one specific setting."
        )
    elif overall >= 40:
        verdict = (
            "The strategy beats buy-and-hold in some configurations but not "
            "others, so the headline result depends materially on the settings "
            "chosen."
        )
    else:
        verdict = (
            "The strategy fails to beat buy-and-hold across most "
            "configurations. The honest conclusion is that no robust edge was "
            "demonstrated."
        )

    spread = summary["return_std"].max()
    if np.isfinite(spread) and spread > 0.5:
        verdict += (
            " Returns also vary widely between configurations, which is itself "
            "evidence of fragility."
        )
    return verdict
