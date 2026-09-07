"""Tests for validation and cleaning.

Each check gets its own synthetic corruption, so a regression points straight
at the check that broke.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CleaningLog,
    ValidationReport,
    clean,
    validate,
    validate_and_clean,
)


def make_clean(n: int = 60) -> pd.DataFrame:
    """A well-formed price frame with no problems at all."""
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    close = np.linspace(100, 130, n)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def checks(report: ValidationReport) -> set[str]:
    return {i.check for i in report.issues}


# ---------------------------------------------------------------- validate


def test_clean_data_passes_every_check():
    report = validate(make_clean())
    assert report.is_valid
    assert not report.errors
    # Only holiday-consistent calendar notes are acceptable here.
    assert checks(report) <= {"calendar_gaps"}


def test_empty_frame_is_an_error():
    report = validate(pd.DataFrame())
    assert not report.is_valid
    assert "empty" in checks(report)


def test_missing_column_is_an_error():
    report = validate(make_clean().drop(columns=["volume"]))
    assert "schema" in checks(report)


def test_detects_missing_values():
    df = make_clean()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    report = validate(df)
    assert "missing_values" in checks(report)
    assert not report.is_valid


def test_detects_duplicate_dates():
    df = make_clean()
    df = pd.concat([df, df.iloc[[10]]]).sort_index()
    assert "duplicate_dates" in checks(validate(df))


def test_detects_unsorted_index():
    df = make_clean().iloc[::-1]
    assert "date_ordering" in checks(validate(df))


def test_detects_infinite_values():
    df = make_clean()
    df.iloc[3, df.columns.get_loc("high")] = np.inf
    assert "infinite_values" in checks(validate(df))


def test_detects_nonpositive_prices():
    df = make_clean()
    df.iloc[7, df.columns.get_loc("low")] = 0.0
    assert "nonpositive_prices" in checks(validate(df))


def test_detects_negative_volume():
    df = make_clean()
    df.iloc[4, df.columns.get_loc("volume")] = -5.0
    assert "negative_volume" in checks(validate(df))


def test_detects_high_below_low():
    df = make_clean()
    df.iloc[9, df.columns.get_loc("high")] = df["low"].iloc[9] - 5
    assert "high_below_low" in checks(validate(df))


def test_detects_close_outside_range():
    df = make_clean()
    df.iloc[11, df.columns.get_loc("close")] = df["high"].iloc[11] + 10
    assert "ohlc_inconsistent" in checks(validate(df))


def test_zero_volume_is_a_warning_not_an_error():
    df = make_clean()
    df.iloc[6, df.columns.get_loc("volume")] = 0.0
    report = validate(df)
    assert "zero_volume" in checks(report)
    assert report.is_valid, "a trading halt must not block the pipeline"


def test_extreme_return_is_flagged_but_not_blocking():
    df = make_clean()
    df.iloc[20, df.columns.get_loc("adj_close")] *= 2.0  # +100% day
    report = validate(df)
    assert "extreme_returns" in checks(report)
    assert report.is_valid


def test_large_calendar_gap_is_a_warning():
    df = make_clean(120)
    df = pd.concat([df.iloc[:20], df.iloc[80:]])  # remove ~3 months
    report = validate(df)
    assert "calendar_gaps" in checks(report)
    gap = next(i for i in report.issues if i.check == "calendar_gaps")
    assert gap.severity == "warning"


def test_report_frame_and_summary():
    df = make_clean()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    report = validate(df)
    frame = report.to_frame()
    assert "check" in frame.columns and len(frame) == len(report.issues)
    assert "error" in report.summary()


# ------------------------------------------------------------------- clean


def test_clean_leaves_good_data_untouched():
    df = make_clean()
    out, log = clean(df)
    pd.testing.assert_frame_equal(out, df)
    assert log.rows_removed == 0
    assert log.actions == []


def test_clean_sorts_and_dedupes():
    df = make_clean(30)
    corrupted = pd.concat([df.iloc[10:], df.iloc[:10], df.iloc[[5]]])
    out, log = clean(corrupted)
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    assert len(out) == 30
    assert any("duplicate" in a for a in log.actions)


def test_clean_forward_fills_short_price_gaps():
    df = make_clean()
    df.iloc[10, df.columns.get_loc("close")] = np.nan
    out, log = clean(df)
    assert len(out) == len(df), "a one-day gap should be filled, not dropped"
    assert out["close"].iloc[10] == pytest.approx(df["close"].iloc[9])
    assert any("Forward-filled" in a for a in log.actions)


def test_clean_drops_long_price_gaps_instead_of_inventing_data():
    df = make_clean()
    df.iloc[10:20, df.columns.get_loc("close")] = np.nan
    out, log = clean(df, max_ffill=3)
    assert len(out) < len(df)
    assert any("Dropped" in a for a in log.actions)


def test_clean_never_forward_fills_volume():
    df = make_clean()
    df.iloc[10, df.columns.get_loc("volume")] = np.nan
    out, _ = clean(df)
    # The row is dropped rather than given a fabricated volume.
    assert df.index[10] not in out.index
    assert len(out) == len(df) - 1


def test_clean_removes_impossible_bars():
    df = make_clean()
    df.iloc[15, df.columns.get_loc("high")] = df["low"].iloc[15] - 3
    out, log = clean(df)
    assert df.index[15] not in out.index
    assert any("impossible" in a for a in log.actions)


def test_clean_converts_infinities():
    df = make_clean()
    df.iloc[8, df.columns.get_loc("open")] = np.inf
    out, log = clean(df)
    assert np.isfinite(out.to_numpy()).all()
    assert any("infinite" in a for a in log.actions)


def test_clean_keeps_zero_volume_by_default():
    df = make_clean()
    df.iloc[12, df.columns.get_loc("volume")] = 0.0
    kept, _ = clean(df)
    assert df.index[12] in kept.index
    dropped, _ = clean(df, drop_zero_volume=True)
    assert df.index[12] not in dropped.index


def test_clean_keeps_real_crashes():
    """A -30%% day is a market event, not a data error."""
    df = make_clean()
    for col in ("open", "high", "low", "close", "adj_close"):
        df.iloc[25, df.columns.get_loc(col)] *= 0.7
    out, _ = clean(df)
    assert len(out) == len(df)
    assert df.index[25] in out.index


def test_clean_handles_empty_frame():
    out, log = clean(pd.DataFrame())
    assert out.empty
    assert log.rows_after == 0


# ------------------------------------------------------- validate_and_clean


def test_cleaning_resolves_the_errors_it_should():
    df = make_clean()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    df.iloc[8, df.columns.get_loc("open")] = np.inf
    df = pd.concat([df, df.iloc[[3]]]).sort_index()

    cleaned, before, after, log = validate_and_clean(df)
    assert not before.is_valid
    assert after.is_valid, f"cleaning left errors: {after.errors}"
    assert log.rows_before >= log.rows_after
    assert "->" in log.summary()
