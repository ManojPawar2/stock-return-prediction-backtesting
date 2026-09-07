"""Tests for target construction and chronological splitting.

The two things that must be exactly right:

* ``y[t]`` really is the return earned *after* ``t`` (off-by-one here silently
  destroys every result downstream), and
* splits are ordered, disjoint and contiguous in time.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.dataset import (
    TARGET,
    Split,
    attach_target,
    build_dataset,
    chronological_split,
    class_balance,
    make_target,
    target_summary,
)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    n = 900
    idx = pd.bdate_range("2019-01-01", periods=n, name="date")
    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.014, n))), index=idx)
    rng2 = np.random.default_rng(12)
    day_range = close * pd.Series(rng2.uniform(0.005, 0.03, n), index=idx)
    pos = pd.Series(rng2.uniform(0.2, 0.8, n), index=idx)
    low = close - day_range * pos
    return pd.DataFrame(
        {
            "open": low + day_range * 0.4,
            "high": low + day_range,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": pd.Series(rng2.integers(1e6, 9e6, n).astype("float64"), index=idx),
        }
    )


@pytest.fixture
def cfg() -> Config:
    return Config(
        start="2019-01-01", end="2022-06-30",
        train_end="2020-12-31", valid_end="2021-12-31",
    )


# ------------------------------------------------------------------ target


def test_target_is_tomorrows_return_exactly():
    close = pd.Series(
        [100.0, 110.0, 99.0, 99.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    y = make_target(pd.DataFrame({"close": close}), horizon=1)
    assert y.iloc[0] == pytest.approx(0.10)          # 100 -> 110
    assert y.iloc[1] == pytest.approx(-0.10)         # 110 -> 99
    assert y.iloc[2] == pytest.approx(0.0)           # 99 -> 99
    assert pd.isna(y.iloc[3]), "the final row has no known future"


def test_target_respects_horizon():
    close = pd.Series(
        [100.0, 101.0, 102.0, 120.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    y = make_target(pd.DataFrame({"close": close}), horizon=3)
    assert y.iloc[0] == pytest.approx(0.20)          # 100 -> 120, three days on
    assert y.iloc[1:].isna().all()


def test_classification_target_is_the_sign():
    close = pd.Series(
        [100.0, 110.0, 99.0, 99.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    y = make_target(pd.DataFrame({"close": close}), kind="classification")
    assert y.iloc[0] == 1.0
    assert y.iloc[1] == 0.0
    assert y.iloc[2] == 0.0, "an unchanged close is not an up day"
    assert pd.isna(y.iloc[3]), "unknown future must stay NaN, not become 0"


def test_target_never_uses_the_present():
    """y[t] must be independent of everything at or before t."""
    close = pd.Series(
        np.linspace(100, 200, 50), index=pd.bdate_range("2024-01-01", periods=50)
    )
    df = pd.DataFrame({"close": close})
    full = make_target(df, 1)
    truncated = make_target(df.iloc[:31], 1)
    # Index 29 has its future (index 30) inside the truncated frame.
    assert full.iloc[29] == pytest.approx(truncated.iloc[29])
    # Index 30 is the last truncated row, so its future is unknown there.
    assert pd.isna(truncated.iloc[30])
    assert pd.notna(full.iloc[30])


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_horizon_rejected(bad):
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        make_target(df, horizon=bad)


def test_invalid_kind_rejected():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="kind"):
        make_target(df, kind="ranking")


def test_missing_price_column_rejected():
    with pytest.raises(KeyError):
        make_target(pd.DataFrame({"px": [1.0, 2.0]}))


def test_attach_target_drops_the_unknowable_tail(ohlcv, cfg):
    out = attach_target(ohlcv, cfg)
    assert len(out) == len(ohlcv) - 1
    assert not out[TARGET].isna().any()
    assert not out["return_1d_fwd"].isna().any()


def test_forward_return_is_kept_even_for_classification(ohlcv):
    cfg = Config(target_kind="classification")
    out = attach_target(ohlcv, cfg)
    assert set(out[TARGET].unique()) <= {0.0, 1.0}
    # The backtester still needs the real return magnitude.
    assert out["return_1d_fwd"].std() > 0


# ------------------------------------------------------------------- split


def test_split_boundaries_are_inclusive(ohlcv, cfg):
    df = attach_target(ohlcv, cfg)
    s = chronological_split(df, "2020-12-31", "2021-12-31")
    assert s.train.index.max() <= pd.Timestamp("2020-12-31")
    assert s.valid.index.min() > pd.Timestamp("2020-12-31")
    assert s.valid.index.max() <= pd.Timestamp("2021-12-31")
    assert s.test.index.min() > pd.Timestamp("2021-12-31")


def test_splits_are_ordered_and_disjoint(ohlcv, cfg):
    df = attach_target(ohlcv, cfg)
    s = chronological_split(df, cfg.train_end, cfg.valid_end)
    assert s.train.index.max() < s.valid.index.min()
    assert s.valid.index.max() < s.test.index.min()
    assert len(s.train) + len(s.valid) + len(s.test) == len(df)
    assert not s.train.index.intersection(s.test.index).size


def test_each_split_is_contiguous(ohlcv, cfg):
    """No holes: a split must be one unbroken block of the original frame."""
    df = attach_target(ohlcv, cfg)
    s = chronological_split(df, cfg.train_end, cfg.valid_end)
    for part in (s.train, s.valid, s.test):
        block = df.loc[part.index.min():part.index.max()]
        assert len(block) == len(part)


def test_overlapping_splits_are_rejected(ohlcv, cfg):
    df = attach_target(ohlcv, cfg)
    a = df.iloc[:400]
    with pytest.raises(ValueError, match="overlap"):
        Split(train=a, valid=df.iloc[300:500], test=df.iloc[500:])


def test_empty_split_is_rejected(ohlcv, cfg):
    df = attach_target(ohlcv, cfg)
    with pytest.raises(ValueError, match="empty"):
        chronological_split(df, "2035-01-01", "2036-01-01")


def test_reversed_boundaries_are_rejected(ohlcv, cfg):
    df = attach_target(ohlcv, cfg)
    with pytest.raises(ValueError, match="precede"):
        chronological_split(df, "2021-12-31", "2020-12-31")


def test_unsorted_frame_is_rejected(ohlcv, cfg):
    df = attach_target(ohlcv, cfg).iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(df, cfg.train_end, cfg.valid_end)


def test_split_summary_and_sizes(ohlcv, cfg):
    df = attach_target(ohlcv, cfg)
    s = chronological_split(df, cfg.train_end, cfg.valid_end)
    assert sum(s.sizes.values()) == len(df)
    summary = s.summary()
    assert list(summary["split"]) == ["train", "valid", "test"]
    assert summary["rows"].sum() == len(df)


def test_xy_returns_aligned_matrix_and_target(ohlcv, cfg):
    _, split, features = build_dataset(ohlcv, cfg)
    X, y = split.xy("train", features)
    assert len(X) == len(y)
    assert X.index.equals(y.index)
    assert list(X.columns) == features
    assert TARGET not in X.columns


# --------------------------------------------------------------- assembly


def test_build_dataset_end_to_end(ohlcv, cfg):
    df, split, features = build_dataset(ohlcv, cfg)
    assert len(features) >= 35
    assert TARGET in df.columns
    assert not df[features + [TARGET]].isna().any().any()
    assert sum(split.sizes.values()) == len(df)


def test_train_precedes_test_in_time(ohlcv, cfg):
    """The whole point: the model may only learn from the past."""
    _, split, _ = build_dataset(ohlcv, cfg)
    assert split.train.index.max() < split.test.index.min()


def test_target_column_is_never_a_feature(ohlcv, cfg):
    _, _, features = build_dataset(ohlcv, cfg)
    assert TARGET not in features
    assert "return_1d_fwd" not in features, "the forward return would be a total leak"


def test_no_split_shares_a_single_date(ohlcv, cfg):
    _, split, _ = build_dataset(ohlcv, cfg)
    all_dates = np.concatenate(
        [split.train.index.values, split.valid.index.values, split.test.index.values]
    )
    assert len(all_dates) == len(np.unique(all_dates))


# -------------------------------------------------------------- reporting


def test_class_balance(ohlcv, cfg):
    df = attach_target(ohlcv, cfg)
    bal = class_balance(df[TARGET])
    assert bal["up_pct"] + bal["down_pct"] == pytest.approx(100.0)
    assert 30 < bal["up_pct"] < 70, "a sane series is not 90% up days"


def test_target_summary_keys(ohlcv, cfg):
    df, _, _ = build_dataset(ohlcv, cfg)
    s = target_summary(df, cfg)
    assert s["horizon_days"] == 1
    assert s["n"] > 0
    assert 20 < s["pct_positive"] < 80
