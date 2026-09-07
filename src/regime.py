"""Market regime analysis.

A strategy that works only during one market environment has not been
validated; it has been lucky once.  This module labels every day by the
environment it belonged to, then scores the strategy separately inside each.

Two independent labellings:

**Volatility** — terciles of 20-day realised volatility (low / medium / high).
**Trend** — price above or below its 200-day moving average (bull / bear).

Both labels are computed from **trailing** windows, so a day's regime is
knowable on that day.  Using the full sample to define terciles would be a
mild form of lookahead; that trade-off is called out explicitly in
:func:`label_regimes` and a causal variant is provided.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metrics import performance_summary

logger = logging.getLogger(__name__)

VOL_LABELS = ("low_vol", "medium_vol", "high_vol")
TREND_LABELS = ("bear", "bull")


def label_volatility_regime(
    returns: pd.Series,
    window: int = 20,
    causal: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    """Tag each day low / medium / high volatility.

    Parameters
    ----------
    causal
        When True, the tercile boundaries are computed from an expanding
        window, so a day's label uses only information available then.  When
        False (the default) boundaries come from the whole sample.

        The non-causal version is fine for *describing* history, which is what
        this analysis does — it asks "how did the strategy behave in calm
        periods?", a question asked after the fact.  It must never be used to
        build a feature or drive a signal, and :func:`regime_report` never does.
    """
    vol = returns.rolling(window, min_periods=min_periods or window).std()

    if not causal:
        lower, upper = vol.quantile(1 / 3), vol.quantile(2 / 3)
        labels = pd.Series(VOL_LABELS[1], index=vol.index, dtype="object")
        labels[vol <= lower] = VOL_LABELS[0]
        labels[vol > upper] = VOL_LABELS[2]
        return labels.where(vol.notna()).rename("vol_regime")

    # Expanding quantiles: each day compared only to its own past.
    lower = vol.expanding(min_periods=window * 3).quantile(1 / 3)
    upper = vol.expanding(min_periods=window * 3).quantile(2 / 3)
    labels = pd.Series(VOL_LABELS[1], index=vol.index, dtype="object")
    labels[vol <= lower] = VOL_LABELS[0]
    labels[vol > upper] = VOL_LABELS[2]
    return labels.where(vol.notna() & lower.notna()).rename("vol_regime")


def label_trend_regime(prices: pd.Series, window: int = 200) -> pd.Series:
    """Bull when price is above its trailing moving average, bear below.

    Always causal: the moving average is backward-looking by construction.
    """
    ma = prices.rolling(window, min_periods=window).mean()
    labels = pd.Series(np.where(prices > ma, TREND_LABELS[1], TREND_LABELS[0]),
                       index=prices.index, dtype="object")
    return labels.where(ma.notna()).rename("trend_regime")


def label_regimes(
    df: pd.DataFrame,
    price_col: str = "close",
    vol_window: int = 20,
    trend_window: int = 200,
    causal: bool = False,
) -> pd.DataFrame:
    """Attach both regime labels to a price frame."""
    returns = df[price_col].pct_change()
    out = pd.DataFrame(index=df.index)
    out["vol_regime"] = label_volatility_regime(returns, vol_window, causal)
    out["trend_regime"] = label_trend_regime(df[price_col], trend_window)
    out["regime"] = (
        out["trend_regime"].astype(str) + "_" + out["vol_regime"].astype(str)
    ).where(out["vol_regime"].notna() & out["trend_regime"].notna())
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def performance_by_regime(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    labels: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    min_days: int = 20,
) -> pd.DataFrame:
    """Score strategy and benchmark separately inside each regime.

    Regimes with fewer than ``min_days`` observations are dropped: a Sharpe
    computed from eight days is not a measurement.
    """
    index = strategy_returns.index.intersection(labels.dropna().index)
    strategy = strategy_returns.loc[index]
    benchmark = benchmark_returns.reindex(index)
    tags = labels.loc[index]

    rows = []
    for regime in sorted(tags.dropna().unique()):
        mask = tags == regime
        n = int(mask.sum())
        if n < min_days:
            logger.info("Skipping regime %s: only %d days", regime, n)
            continue

        s = performance_summary(strategy[mask], risk_free_rate, trading_days)
        b = performance_summary(benchmark[mask], risk_free_rate, trading_days)
        rows.append({
            "regime": regime,
            "days": n,
            "pct_of_sample": n / len(tags) * 100,
            "strategy_total_return": s["total_return"],
            "benchmark_total_return": b["total_return"],
            "excess_return": s["total_return"] - b["total_return"],
            "strategy_sharpe": s["sharpe_ratio"],
            "benchmark_sharpe": b["sharpe_ratio"],
            "strategy_max_drawdown": s["max_drawdown"],
            "benchmark_max_drawdown": b["max_drawdown"],
            "strategy_win_rate": s["win_rate"],
            "outperformed": s["total_return"] > b["total_return"],
        })

    if not rows:
        return pd.DataFrame(columns=["regime", "days", "pct_of_sample"])
    return pd.DataFrame(rows).set_index("regime")


def regime_report(
    df: pd.DataFrame,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    price_col: str = "close",
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Volatility, trend and combined regime breakdowns."""
    labels = label_regimes(df, price_col=price_col)
    return {
        "volatility": performance_by_regime(
            strategy_returns, benchmark_returns, labels["vol_regime"], **kwargs
        ),
        "trend": performance_by_regime(
            strategy_returns, benchmark_returns, labels["trend_regime"], **kwargs
        ),
        "combined": performance_by_regime(
            strategy_returns, benchmark_returns, labels["regime"], **kwargs
        ),
    }


def consistency_score(breakdown: pd.DataFrame) -> dict[str, float]:
    """How reliably the strategy beat its benchmark across regimes.

    A strategy winning in one regime out of four is a strategy that got lucky
    once, however good the headline number looks.
    """
    if breakdown.empty or "outperformed" not in breakdown.columns:
        return {"n_regimes": 0, "n_outperformed": 0, "pct_outperformed": float("nan"),
                "worst_excess": float("nan"), "best_excess": float("nan"),
                "mean_excess": float("nan")}

    wins = int(breakdown["outperformed"].sum())
    n = int(len(breakdown))
    return {
        "n_regimes": n,
        "n_outperformed": wins,
        "pct_outperformed": wins / n * 100,
        "worst_excess": float(breakdown["excess_return"].min()),
        "best_excess": float(breakdown["excess_return"].max()),
        "mean_excess": float(breakdown["excess_return"].mean()),
    }


def regime_summary_text(breakdown: pd.DataFrame) -> str:
    """One-paragraph plain-English reading of a regime breakdown."""
    if breakdown.empty:
        return "Not enough data in any regime to draw a conclusion."

    score = consistency_score(breakdown)
    winners = breakdown.index[breakdown["outperformed"]].tolist()
    losers = breakdown.index[~breakdown["outperformed"]].tolist()

    parts = [
        f"The strategy beat buy-and-hold in {score['n_outperformed']} of "
        f"{score['n_regimes']} regimes ({score['pct_outperformed']:.0f}%)."
    ]
    if winners:
        parts.append(f"It outperformed in: {', '.join(winners)}.")
    if losers:
        parts.append(f"It underperformed in: {', '.join(losers)}.")
    if score["n_outperformed"] == score["n_regimes"]:
        parts.append("Performance is consistent across every environment tested.")
    elif score["n_outperformed"] <= 1 and score["n_regimes"] > 2:
        parts.append(
            "Because the edge appears in only one environment, it is more "
            "likely a property of that period than a durable effect."
        )
    return " ".join(parts)
