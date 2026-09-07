"""Target construction and chronological splitting.

This module owns Rule 2 of the leakage contract: **the target is strictly in
the future**, and the alignment is explicit rather than implied.

    y[t] = close[t + h] / close[t] - 1

So the row at date ``t`` pairs features known at ``t``'s close with the return
earned *after* it.  The last ``h`` rows have no target and are removed.

It also owns the splitting policy.  ``sklearn.model_selection.train_test_split``
is never imported here, deliberately: shuffling time series lets the model
train on 2024 and test on 2016, which inflates every metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .features import feature_columns

logger = logging.getLogger(__name__)

TARGET = "target"


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------


def make_target(
    df: pd.DataFrame,
    horizon: int = 1,
    kind: str = "regression",
    price_col: str = "close",
) -> pd.Series:
    """Future return (or its sign) aligned to the decision date.

    Parameters
    ----------
    horizon
        How many days ahead. ``1`` = tomorrow's return.
    kind
        ``"regression"`` returns the forward return itself;
        ``"classification"`` returns 1 when it is strictly positive, else 0.

    Notes
    -----
    ``shift(-horizon)`` is the only negative shift in the entire project, and
    it is correct precisely because this is the label, not a feature.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if kind not in ("regression", "classification"):
        raise ValueError(f"kind must be 'regression' or 'classification', got {kind!r}")
    if price_col not in df.columns:
        raise KeyError(f"price column {price_col!r} not in frame")

    forward = df[price_col].pct_change(horizon).shift(-horizon)

    if kind == "classification":
        # NaN must stay NaN: an unknown future is not a "down" day.
        return forward.gt(0).astype("float64").where(forward.notna()).rename(TARGET)

    return forward.rename(TARGET)


def attach_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the target column and drop rows whose future is unknown."""
    out = df.copy()
    out[TARGET] = make_target(out, cfg.target_horizon, cfg.target_kind)
    # Always keep the realised next-day return: the backtester needs it even
    # when the target itself is a classification label.
    out["return_1d_fwd"] = out["close"].pct_change().shift(-1)

    before = len(out)
    out = out.dropna(subset=[TARGET, "return_1d_fwd"])
    logger.info("attach_target: dropped %d row(s) with no future", before - len(out))
    return out


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


@dataclass
class Split:
    """One chronological train/validation/test partition."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def __post_init__(self) -> None:
        self._check()

    def _check(self) -> None:
        """Assert the partition is genuinely ordered and disjoint."""
        parts = [("train", self.train), ("valid", self.valid), ("test", self.test)]
        non_empty = [(n, p) for n, p in parts if len(p)]
        for (n1, p1), (n2, p2) in zip(non_empty, non_empty[1:]):
            if p1.index.max() >= p2.index.min():
                raise ValueError(
                    f"{n1} ends {p1.index.max().date()} but {n2} starts "
                    f"{p2.index.min().date()} — splits overlap in time"
                )
        combined = np.concatenate([p.index.to_numpy() for _, p in non_empty]) \
            if non_empty else np.array([])
        if len(combined) != len(np.unique(combined)):
            raise ValueError("splits share rows")

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
        }

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("valid", self.valid),
                           ("test", self.test)):
            rows.append({
                "split": name,
                "rows": len(part),
                "start": part.index.min().date().isoformat() if len(part) else "-",
                "end": part.index.max().date().isoformat() if len(part) else "-",
                "years": round(len(part) / 252, 2),
            })
        return pd.DataFrame(rows)

    def xy(self, part: str, features: list[str] | None = None
           ) -> tuple[pd.DataFrame, pd.Series]:
        """Feature matrix and target for one split."""
        frame = getattr(self, part)
        cols = features if features is not None else feature_columns(frame)
        return frame[cols], frame[TARGET]


def chronological_split(
    df: pd.DataFrame,
    train_end: str,
    valid_end: str,
) -> Split:
    """Partition by date. Never shuffles, never samples.

    Boundaries are inclusive of the named day: ``train_end="2021-12-31"``
    puts every 2021 row in train.
    """
    if df.empty:
        raise ValueError("cannot split an empty frame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("chronological_split requires a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("frame must be sorted by date before splitting")

    t_end, v_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    if t_end >= v_end:
        raise ValueError(f"train_end ({train_end}) must precede valid_end ({valid_end})")

    split = Split(
        train=df.loc[df.index <= t_end],
        valid=df.loc[(df.index > t_end) & (df.index <= v_end)],
        test=df.loc[df.index > v_end],
    )

    for name, part in (("train", split.train), ("valid", split.valid),
                       ("test", split.test)):
        if part.empty:
            raise ValueError(
                f"The {name} split is empty. Data spans "
                f"{df.index.min().date()}..{df.index.max().date()} but "
                f"train_end={train_end}, valid_end={valid_end}."
            )
    logger.info("chronological_split: %s", split.sizes)
    return split


def split_from_config(df: pd.DataFrame, cfg: Config) -> Split:
    return chronological_split(df, cfg.train_end, cfg.valid_end)


# --------------------------------------------------------------------------
# Full assembly
# --------------------------------------------------------------------------


def build_dataset(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, Split, list[str]]:
    """Features -> target -> chronological split, in the leakage-safe order.

    The ordering is load-bearing.  Warm-up NaNs are dropped in
    :func:`~src.features.build_features` and target NaNs here, *before*
    splitting, so every split is a contiguous block of complete rows.  Dropping
    afterwards would punch holes in the middle of each period.
    """
    from .features import build_features

    featured = build_features(df, cfg, dropna=True)
    with_target = attach_target(featured, cfg)
    split = split_from_config(with_target, cfg)
    return with_target, split, feature_columns(with_target)


def realised_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """The return realised **on** each day: ``close[t] / close[t-1] - 1``.

    This is what the backtester consumes, and it is deliberately *not* the
    same series as the target.  The target at ``t`` looks forward to ``t+1``;
    this looks back from ``t``.  Getting the two confused shifts every
    strategy return by one day, which is the exact failure the execution lag
    exists to prevent, so the conversion lives here once rather than being
    re-derived at each call site.
    """
    return df[price_col].pct_change().fillna(0.0).rename("asset_return")


def class_balance(y: pd.Series) -> dict[str, float]:
    """Up/down day proportions, for the classification variant."""
    up = float((y > 0).mean())
    return {"up_pct": up * 100, "down_pct": (1 - up) * 100, "n": int(len(y))}


def target_summary(df: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Headline statistics of the target, for the Modeling page."""
    y = df[TARGET]
    return {
        "horizon_days": cfg.target_horizon,
        "kind": cfg.target_kind,
        "n": int(len(y)),
        "mean_pct": float(y.mean() * 100),
        "std_pct": float(y.std() * 100),
        "min_pct": float(y.min() * 100),
        "max_pct": float(y.max() * 100),
        "pct_positive": float((y > 0).mean() * 100),
        "skew": float(y.skew()),
        "kurtosis": float(y.kurtosis()),
    }
