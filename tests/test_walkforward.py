"""Tests for walk-forward validation.

The guarantee that matters: every fold trains strictly before it tests, and no
fold's test rows appear in its own training set. If that breaks, walk-forward
becomes an elaborate way of scoring the training data.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.dataset import TARGET, build_dataset
from src.walkforward import (
    make_folds,
    walk_forward,
    walk_forward_backtest,
)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    n = 1400
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    rng = np.random.default_rng(21)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, n))), index=idx)
    day_range = close * pd.Series(rng.uniform(0.006, 0.028, n), index=idx)
    pos = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    low = close - day_range * pos
    return pd.DataFrame({
        "open": low + day_range * 0.45,
        "high": low + day_range,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": pd.Series(rng.integers(1e6, 9e6, n).astype("float64"), index=idx),
    })


@pytest.fixture
def cfg() -> Config:
    return Config(
        start="2016-01-01", end="2021-06-30",
        train_end="2019-12-31", valid_end="2020-06-30",
        model="ridge",
    )


@pytest.fixture
def prepared(ohlcv, cfg):
    data, _, features = build_dataset(ohlcv, cfg)
    return data, features


# --------------------------------------------------------------- fold maths


def test_folds_are_ordered_and_non_overlapping():
    folds = make_folds(1000, 5, "expanding", min_train=300)
    assert len(folds) == 5
    for (_, test_a), (_, test_b) in zip(folds, folds[1:]):
        assert test_a.stop <= test_b.start


def test_test_blocks_tile_the_remaining_data():
    folds = make_folds(1000, 4, "expanding", min_train=200)
    assert folds[0][1].start == 200
    assert folds[-1][1].stop == 1000
    covered = sum(t.stop - t.start for _, t in folds)
    assert covered == 800


def test_training_always_precedes_testing():
    """The core guarantee."""
    for mode in ("expanding", "rolling"):
        for train, test in make_folds(1200, 6, mode, min_train=300):
            assert train.stop <= test.start, f"{mode}: train overlaps test"
            assert train.start < train.stop, "empty training window"


def test_expanding_windows_grow():
    sizes = [t.stop - t.start for t, _ in make_folds(1000, 5, "expanding", 200)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_rolling_windows_stay_fixed():
    sizes = [t.stop - t.start for t, _ in make_folds(1000, 5, "rolling", 200)]
    assert len(set(sizes)) == 1
    assert sizes[0] == 200


def test_rolling_forgets_old_data():
    folds = make_folds(1000, 5, "rolling", 200)
    assert folds[-1][0].start > folds[0][0].start


@pytest.mark.parametrize("bad", [
    dict(n_folds=0),
    dict(mode="random"),
    dict(n_rows=100, min_train=250),
])
def test_invalid_fold_requests_rejected(bad):
    kwargs = dict(n_rows=1000, n_folds=5, mode="expanding", min_train=250)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        make_folds(**kwargs)


def test_too_many_folds_for_the_data_is_rejected():
    with pytest.raises(ValueError, match="folds"):
        make_folds(255, 20, "expanding", min_train=250)


# ------------------------------------------------------------------ runner


def test_walk_forward_produces_the_requested_folds(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=4, min_train=400)
    assert result.n_folds == 4
    assert result.model_name == "ridge"


def test_every_fold_trains_before_it_tests(prepared, cfg):
    """No fold may train on data at or after its own test period."""
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=5, min_train=400)
    for fold in result.folds:
        assert fold.train_end < fold.test_start, (
            f"fold {fold.index} trains to {fold.train_end.date()} but tests "
            f"from {fold.test_start.date()}"
        )


def test_fold_test_periods_do_not_overlap(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=5, min_train=400)
    for a, b in zip(result.folds, result.folds[1:]):
        assert a.test_end < b.test_start


def test_stitched_predictions_are_unique_and_chronological(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=5, min_train=400)
    preds = result.predictions
    assert preds.index.is_monotonic_increasing
    assert not preds.index.duplicated().any()
    assert len(preds) == sum(f.n_test for f in result.folds)


def test_stitched_series_align(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=4, min_train=400)
    assert result.predictions.index.equals(result.actuals.index)
    # And the actuals really are the dataset's target on those dates.
    pd.testing.assert_series_equal(
        result.actuals,
        data.loc[result.predictions.index, TARGET],
        check_names=False,
    )


def test_walk_forward_covers_more_data_than_a_single_split(prepared, cfg):
    """The practical reason to prefer it."""
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=5, min_train=400)
    single_holdout = len(data) - 400
    assert len(result.predictions) == single_holdout


def test_expanding_and_rolling_give_different_answers(prepared, cfg):
    data, features = prepared
    a = walk_forward(data, cfg, features, model_name="random_forest",
                     n_folds=3, mode="expanding", min_train=400)
    b = walk_forward(data, cfg, features, model_name="random_forest",
                     n_folds=3, mode="rolling", min_train=400)
    assert a.mode == "expanding" and b.mode == "rolling"
    assert not np.allclose(a.predictions, b.predictions)


def test_retraining_actually_happens(prepared, cfg):
    """Different folds must produce genuinely different models.

    If the model were fit once and reused, later folds would score
    suspiciously well and every fold's importance vector would be identical.
    """
    data, features = prepared
    result = walk_forward(data, cfg, features, model_name="random_forest",
                          n_folds=4, min_train=400)
    importances = [f.importance for f in result.folds]
    assert not np.allclose(importances[0], importances[-1]), (
        "all folds share one importance vector — the model was not retrained"
    )


def test_missing_target_is_rejected(prepared, cfg):
    data, features = prepared
    with pytest.raises(ValueError, match="target"):
        walk_forward(data.drop(columns=[TARGET]), cfg, features, n_folds=3)


def test_unsorted_data_is_rejected(prepared, cfg):
    data, features = prepared
    with pytest.raises(ValueError, match="sorted"):
        walk_forward(data.iloc[::-1], cfg, features, n_folds=3)


# --------------------------------------------------------------- reporting


def test_fold_table_has_a_row_per_fold(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=4, min_train=400)
    table = result.fold_table()
    assert len(table) == 4
    for col in ("fold", "train_start", "test_end", "n_train", "rmse",
                "directional_accuracy"):
        assert col in table.columns


def test_overall_metrics_cover_the_stitched_series(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=4, min_train=400)
    overall = result.overall_metrics()
    assert overall["n"] == len(result.predictions)


def test_stability_reports_spread_across_folds(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, n_folds=5, min_train=400)
    stability = result.stability()
    assert set(stability.columns) == {"mean", "std", "min", "max"}
    assert "directional_accuracy" in stability.index
    assert (stability["max"] >= stability["min"]).all()


def test_average_importance_is_normalised(prepared, cfg):
    data, features = prepared
    result = walk_forward(data, cfg, features, model_name="random_forest",
                          n_folds=3, min_train=400)
    imp = result.average_importance()
    assert len(imp) == len(features)
    assert imp.sum() == pytest.approx(1.0, abs=1e-6)
    assert imp.is_monotonic_decreasing


# ------------------------------------------------------- walk-forward + bt


def test_walk_forward_backtest_runs_end_to_end(prepared, cfg):
    data, features = prepared
    result, backtest = walk_forward_backtest(
        data, cfg, features, n_folds=4, min_train=400
    )
    assert len(backtest.equity) == len(result.predictions)
    assert backtest.equity.index.equals(result.predictions.index)
    assert np.isfinite(backtest.equity).all()


def test_walk_forward_backtest_respects_the_execution_lag(prepared, cfg):
    data, features = prepared
    _, backtest = walk_forward_backtest(data, cfg, features, n_folds=3, min_train=400)
    assert backtest.position.iloc[0] == 0.0
    pd.testing.assert_series_equal(
        backtest.position.iloc[1:],
        backtest.signal.shift(1).iloc[1:].rename("position"),
        check_names=False,
    )


def test_walk_forward_backtest_charges_costs(prepared, cfg):
    data, features = prepared
    free = cfg.replace(transaction_cost_bps=0.0, slippage_bps=0.0)
    _, cheap = walk_forward_backtest(data, free, features, n_folds=3, min_train=400)
    _, dear = walk_forward_backtest(
        data, cfg.replace(transaction_cost_bps=50.0), features,
        n_folds=3, min_train=400,
    )
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]
    assert dear.total_cost > cheap.total_cost
