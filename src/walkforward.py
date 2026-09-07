"""Walk-forward validation.

A single train/test split gives one estimate from one period, and that period
may simply have been kind.  Walk-forward retrains repeatedly, always on the
past, always testing on the next unseen block::

    fold 1: TRAIN[2015-2019] -> TEST[2020]
    fold 2: TRAIN[2015-2020] -> TEST[2021]     (expanding)
    fold 3: TRAIN[2015-2021] -> TEST[2022]

Every fold's predictions are out-of-sample by construction, and stitching them
together yields one continuous out-of-sample series covering most of the
history — far more evidence than a single hold-out provides.

Two window modes:

``expanding``
    Train on everything available so far.  More data each fold; assumes old
    relationships still hold.

``rolling``
    Train on a fixed-length recent window.  Adapts to regime change, at the
    cost of forgetting.

The retraining is what makes this honest.  Fitting once and predicting across
all folds would leak later data into earlier predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .dataset import TARGET
from .evaluation import regression_metrics
from .models import ModelPipeline

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    """One train/test block of a walk-forward run."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    predictions: pd.Series
    actuals: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)
    importance: pd.Series | None = None

    def to_row(self) -> dict:
        return {
            "fold": self.index,
            "train_start": self.train_start.date(),
            "train_end": self.train_end.date(),
            "test_start": self.test_start.date(),
            "test_end": self.test_end.date(),
            "n_train": self.n_train,
            "n_test": self.n_test,
            **{k: v for k, v in self.metrics.items() if k != "n"},
        }


@dataclass
class WalkForwardResult:
    """All folds plus the stitched out-of-sample series."""

    folds: list[Fold]
    predictions: pd.Series      # concatenated, chronological, out-of-sample
    actuals: pd.Series
    model_name: str
    mode: str

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def fold_table(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_row() for f in self.folds])

    def overall_metrics(self) -> dict[str, float]:
        """Metrics over the whole stitched out-of-sample series."""
        return regression_metrics(self.actuals, self.predictions)

    def stability(self) -> pd.DataFrame:
        """Mean and spread of each metric across folds.

        A strategy whose directional accuracy swings from 45% to 60% between
        folds has not shown an edge; it has shown noise.
        """
        table = self.fold_table()
        numeric = table.select_dtypes(include=[np.number]).drop(
            columns=["fold"], errors="ignore"
        )
        return pd.DataFrame({
            "mean": numeric.mean(),
            "std": numeric.std(),
            "min": numeric.min(),
            "max": numeric.max(),
        })

    def average_importance(self) -> pd.Series:
        """Feature importance averaged over folds.

        More trustworthy than any single fit: a feature that only matters in
        one fold was probably fitting that fold's noise.
        """
        frames = [f.importance for f in self.folds if f.importance is not None]
        if not frames:
            return pd.Series(dtype="float64")
        return (
            pd.concat(frames, axis=1).mean(axis=1)
            .sort_values(ascending=False)
            .rename("importance")
        )


# --------------------------------------------------------------------------
# Fold construction
# --------------------------------------------------------------------------


def make_folds(
    n_rows: int,
    n_folds: int,
    mode: str = "expanding",
    min_train: int = 250,
) -> list[tuple[slice, slice]]:
    """Positional (train, test) slices for a walk-forward run.

    The data after ``min_train`` is divided into ``n_folds`` equal test blocks.
    Fold ``i`` trains on everything before its test block (expanding) or on the
    most recent ``min_train`` rows before it (rolling).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if mode not in ("expanding", "rolling"):
        raise ValueError("mode must be 'expanding' or 'rolling'")
    if n_rows <= min_train:
        raise ValueError(
            f"Need more than min_train={min_train} rows to walk forward, "
            f"got {n_rows}. Lower min_train_days or widen the date range."
        )

    testable = n_rows - min_train
    if testable < n_folds:
        raise ValueError(
            f"Only {testable} rows are available for testing but {n_folds} "
            "folds were requested; each fold needs at least one row."
        )

    edges = np.linspace(min_train, n_rows, n_folds + 1).astype(int)
    folds: list[tuple[slice, slice]] = []
    for i in range(n_folds):
        test_start, test_end = int(edges[i]), int(edges[i + 1])
        if test_end <= test_start:
            continue
        train_start = 0 if mode == "expanding" else max(0, test_start - min_train)
        folds.append((slice(train_start, test_start), slice(test_start, test_end)))
    return folds


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def walk_forward(
    data: pd.DataFrame,
    cfg: Config,
    features: list[str],
    model_name: str | None = None,
    n_folds: int | None = None,
    mode: str | None = None,
    min_train: int | None = None,
) -> WalkForwardResult:
    """Retrain on each fold and stitch the out-of-sample predictions together.

    ``data`` must already carry features and a target (see
    :func:`src.dataset.build_dataset`).
    """
    name = model_name or cfg.model
    n_folds = n_folds if n_folds is not None else cfg.walk_forward.n_folds
    mode = mode or cfg.walk_forward.mode
    min_train = min_train if min_train is not None else cfg.walk_forward.min_train_days

    if TARGET not in data.columns:
        raise ValueError("data must contain a target column; call build_dataset first")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data must be sorted by date")

    slices = make_folds(len(data), n_folds, mode, min_train)
    folds: list[Fold] = []

    for i, (train_slice, test_slice) in enumerate(slices, start=1):
        train, test = data.iloc[train_slice], data.iloc[test_slice]

        # Retraining per fold is what keeps every prediction out-of-sample.
        pipe = ModelPipeline(
            name, cfg.params_for(name), cfg.seed, cfg.target_kind
        ).fit(train[features], train[TARGET])

        preds = pipe.predict(test[features])
        actuals = test[TARGET]

        try:
            importance = pipe.feature_importance()
        except AttributeError:  # pragma: no cover - all current models support it
            importance = None

        folds.append(Fold(
            index=i,
            train_start=train.index.min(), train_end=train.index.max(),
            test_start=test.index.min(), test_end=test.index.max(),
            n_train=len(train), n_test=len(test),
            predictions=preds, actuals=actuals,
            metrics=regression_metrics(actuals, preds),
            importance=importance,
        ))
        logger.info(
            "Fold %d/%d: train %s..%s (%d) -> test %s..%s (%d)",
            i, len(slices), train.index.min().date(), train.index.max().date(),
            len(train), test.index.min().date(), test.index.max().date(), len(test),
        )

    stitched_pred = pd.concat([f.predictions for f in folds]).sort_index()
    stitched_actual = pd.concat([f.actuals for f in folds]).sort_index()

    return WalkForwardResult(
        folds=folds,
        predictions=stitched_pred.rename("prediction"),
        actuals=stitched_actual.rename(TARGET),
        model_name=name,
        mode=mode,
    )


def walk_forward_backtest(
    data: pd.DataFrame,
    cfg: Config,
    features: list[str],
    model_name: str | None = None,
    **kwargs,
):
    """Walk-forward, then backtest the stitched out-of-sample predictions.

    This is the most realistic figure the project produces: every prediction
    behind this equity curve was made by a model that had never seen the day
    it was predicting.
    """
    from .backtester import backtester_from_config
    from .dataset import realised_returns
    from .signals import generate_signals

    result = walk_forward(data, cfg, features, model_name, **kwargs)

    covered = data.loc[result.predictions.index]
    returns = realised_returns(covered)
    signals = generate_signals(
        result.predictions, cfg.signal_threshold, cfg.allow_short
    )
    backtest = backtester_from_config(cfg).run(
        returns, signals, prices=covered["close"]
    )
    return result, backtest
