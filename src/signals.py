"""Turning predictions into positions.

Deliberately simple.  A complicated signal rule is hard to explain and easy to
overfit; a threshold on the predicted return is one number the stress tester
can sweep, so its influence on the result is measurable rather than assumed.

    prediction >  threshold                -> +1  (long)
    prediction < -threshold and shorting   -> -1  (short)
    otherwise                              ->  0  (flat)

The threshold is a friction control.  At ``threshold = 0`` the strategy takes
a position on nearly every day and pays the maximum in transaction costs; a
higher threshold trades only on stronger convictions.

Note what this module does *not* do: it does not lag anything.  The signal at
row ``t`` is the decision made using information available at ``t``.  Turning
that into a position held on ``t+1`` is the backtester's job, and keeping the
two steps separate is what makes the execution lag visible and testable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

LONG, FLAT, SHORT = 1.0, 0.0, -1.0


def generate_signals(
    predictions: pd.Series,
    threshold: float = 0.0,
    allow_short: bool = False,
) -> pd.Series:
    """Map predicted returns to target positions in {-1, 0, +1}.

    Parameters
    ----------
    threshold
        A magnitude, always non-negative. The band ``[-threshold, threshold]``
        is the no-trade zone.
    allow_short
        When False (the default) a negative prediction means flat, not short.
    """
    if threshold < 0:
        raise ValueError(f"threshold is a magnitude and must be >= 0, got {threshold}")

    preds = pd.Series(predictions, dtype="float64")
    signal = pd.Series(FLAT, index=preds.index, name="signal", dtype="float64")

    signal[preds > threshold] = LONG
    if allow_short:
        signal[preds < -threshold] = SHORT

    # A missing prediction means no view, which means no position.
    signal[preds.isna()] = FLAT
    return signal


def signals_from_config(predictions: pd.Series, cfg: Config) -> pd.Series:
    return generate_signals(predictions, cfg.signal_threshold, cfg.allow_short)


def buy_and_hold_signal(index: pd.Index) -> pd.Series:
    """Always long: the baseline every strategy is measured against."""
    return pd.Series(LONG, index=index, name="signal", dtype="float64")


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def signal_summary(signal: pd.Series) -> dict[str, float]:
    """How often the strategy is in the market, and how often it switches."""
    n = len(signal)
    if n == 0:
        return {"n": 0, "long_pct": 0.0, "flat_pct": 0.0, "short_pct": 0.0,
                "exposure_pct": 0.0, "n_changes": 0, "avg_holding_days": 0.0}

    changes = int((signal.diff().fillna(signal) != 0).sum())
    active = int((signal != FLAT).sum())
    return {
        "n": n,
        "long_pct": float((signal == LONG).mean() * 100),
        "flat_pct": float((signal == FLAT).mean() * 100),
        "short_pct": float((signal == SHORT).mean() * 100),
        "exposure_pct": float(active / n * 100),
        "n_changes": changes,
        # How long an average position survives before the signal flips.
        "avg_holding_days": float(active / changes) if changes else float(n),
    }


def threshold_sweep(
    predictions: pd.Series,
    thresholds: list[float],
    allow_short: bool = False,
) -> pd.DataFrame:
    """Exposure and turnover across candidate thresholds.

    Shows the trade-off directly: raising the threshold cuts costly churn but
    also cuts time in the market.
    """
    rows = []
    for t in thresholds:
        summary = signal_summary(generate_signals(predictions, t, allow_short))
        rows.append({"threshold": t, **summary})
    return pd.DataFrame(rows)


def suggest_threshold(predictions: pd.Series, target_exposure_pct: float = 50.0) -> float:
    """The threshold that puts the strategy in the market a target share of days.

    Useful as a starting point, but it must be computed on **training or
    validation** predictions only — deriving it from test predictions would be
    tuning on the test set.
    """
    preds = pd.Series(predictions).dropna()
    if preds.empty:
        return 0.0
    quantile = 1.0 - min(max(target_exposure_pct, 0.0), 100.0) / 100.0
    return float(max(np.quantile(preds, quantile), 0.0))
