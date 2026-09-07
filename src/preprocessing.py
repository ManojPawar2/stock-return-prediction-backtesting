"""Data validation and cleaning.

Two deliberately separate stages:

``validate``
    Never mutates.  Returns a :class:`ValidationReport` describing everything
    suspicious about the frame, so the dashboard can show the user what is
    wrong before anything is silently "fixed".

``clean``
    Mutates, and records every action in :class:`CleaningLog`.

Keeping them apart matters because in finance a "problem" is often real: a
-20% day is not a data error, it is a crash.  Those are flagged, never dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import COLUMNS

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")

#: Daily moves beyond this magnitude are flagged for review, not removed.
EXTREME_RETURN = 0.50


@dataclass
class Issue:
    """One validation finding."""

    check: str
    severity: str  # "error" | "warning" | "info"
    count: int
    message: str
    dates: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.severity.upper():7s}] {self.check}: {self.message}"


@dataclass
class ValidationReport:
    """The result of :func:`validate`."""

    n_rows: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        """True when nothing blocking was found. Warnings do not block."""
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        """Tabular form for the Streamlit validation panel."""
        if not self.issues:
            return pd.DataFrame(
                columns=["check", "severity", "count", "message", "example_dates"]
            )
        return pd.DataFrame(
            [
                {
                    "check": i.check,
                    "severity": i.severity,
                    "count": i.count,
                    "message": i.message,
                    "example_dates": ", ".join(i.dates[:5]),
                }
                for i in self.issues
            ]
        )

    def summary(self) -> str:
        if not self.issues:
            return f"{self.n_rows} rows: all checks passed."
        return (
            f"{self.n_rows} rows: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), "
            f"{len(self.issues) - len(self.errors) - len(self.warnings)} note(s)."
        )


@dataclass
class CleaningLog:
    """What :func:`clean` actually changed."""

    actions: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    def add(self, message: str) -> None:
        logger.info("clean: %s", message)
        self.actions.append(message)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    def summary(self) -> str:
        if not self.actions:
            return f"No changes needed ({self.rows_before} rows)."
        return (
            f"{self.rows_before} -> {self.rows_after} rows. "
            + " ".join(self.actions)
        )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _dates(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in index]


def validate(df: pd.DataFrame, extreme_return: float = EXTREME_RETURN) -> ValidationReport:
    """Inspect a price frame without modifying it."""
    report = ValidationReport(n_rows=len(df))

    def add(check: str, severity: str, mask_or_count: Any, message: str,
            index: pd.Index | None = None) -> None:
        if isinstance(mask_or_count, (int, np.integer)):
            count = int(mask_or_count)
            dates: list[str] = [] if index is None else _dates(index)
        else:
            count = int(mask_or_count.sum())
            dates = _dates(df.index[mask_or_count]) if count else []
        if count:
            report.issues.append(Issue(check, severity, count, message, dates))

    if df.empty:
        report.issues.append(Issue("empty", "error", 1, "Frame contains no rows."))
        return report

    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        report.issues.append(
            Issue("schema", "error", len(missing_cols),
                  f"Missing columns: {missing_cols}")
        )
        return report

    # --- structural ------------------------------------------------------
    dup = df.index.duplicated(keep=False)
    add("duplicate_dates", "error", dup,
        f"{int(dup.sum())} rows share a date with another row.")

    if not df.index.is_monotonic_increasing:
        report.issues.append(
            Issue("date_ordering", "error", 1,
                  "Index is not sorted ascending by date.")
        )

    # --- values ----------------------------------------------------------
    na = df[list(COLUMNS)].isna().any(axis=1)
    add("missing_values", "error", na,
        f"{int(na.sum())} rows contain a missing OHLCV value.")

    numeric = df[list(COLUMNS)].apply(pd.to_numeric, errors="coerce")
    inf = numeric.replace([np.inf, -np.inf], np.nan).isna() & numeric.notna()
    inf_rows = inf.any(axis=1)
    add("infinite_values", "error", inf_rows,
        f"{int(inf_rows.sum())} rows contain an infinite value.")

    nonpositive = (df[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    add("nonpositive_prices", "error", nonpositive,
        f"{int(nonpositive.sum())} rows have a price <= 0.")

    negative_volume = df["volume"] < 0
    add("negative_volume", "error", negative_volume,
        f"{int(negative_volume.sum())} rows have negative volume.")

    # --- OHLC internal consistency ---------------------------------------
    hl = df["high"] < df["low"]
    add("high_below_low", "error", hl,
        f"{int(hl.sum())} rows have high < low.")

    outside = (
        (df["close"] > df["high"] + 1e-6) | (df["close"] < df["low"] - 1e-6)
        | (df["open"] > df["high"] + 1e-6) | (df["open"] < df["low"] - 1e-6)
    )
    add("ohlc_inconsistent", "error", outside,
        f"{int(outside.sum())} rows have open/close outside the high-low range.")

    # --- soft signals -----------------------------------------------------
    zero_volume = df["volume"] == 0
    add("zero_volume", "warning", zero_volume,
        f"{int(zero_volume.sum())} rows have zero volume (halt or stale quote).")

    returns = df["adj_close"].pct_change()
    extreme = returns.abs() > extreme_return
    add("extreme_returns", "warning", extreme.fillna(False),
        f"{int(extreme.sum())} days moved more than "
        f"{extreme_return:.0%} — verify these are real events, not bad ticks.")

    if len(df) > 1:
        expected = pd.bdate_range(df.index.min(), df.index.max())
        gaps = expected.difference(df.index)
        # US markets close ~9-10 days a year for holidays.
        tolerance = int(np.ceil(len(expected) / 252 * 12))
        if len(gaps) > tolerance:
            report.issues.append(
                Issue("calendar_gaps", "warning", len(gaps),
                      f"{len(gaps)} business days absent (about "
                      f"{tolerance} expected from market holidays).",
                      _dates(gaps)))
        elif len(gaps):
            report.issues.append(
                Issue("calendar_gaps", "info", len(gaps),
                      f"{len(gaps)} business days absent, consistent with "
                      "market holidays.", _dates(gaps)))

    return report


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def clean(
    df: pd.DataFrame,
    max_ffill: int = 3,
    drop_zero_volume: bool = False,
) -> tuple[pd.DataFrame, CleaningLog]:
    """Return a cleaned copy of ``df`` plus a log of what changed.

    Parameters
    ----------
    max_ffill
        Longest run of consecutive missing prices to forward-fill.  Longer
        gaps are dropped rather than invented.  Volume is never forward-filled
        — a fabricated volume would corrupt every volume feature.
    drop_zero_volume
        Whether to remove zero-volume rows.  Off by default; they are real
        (trading halts) and dropping them creates artificial return jumps.
    """
    log = CleaningLog(rows_before=len(df))
    out = df.copy()

    if out.empty:
        log.rows_after = 0
        return out, log

    # 1. Ordering.
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
        log.add("Sorted rows by date.")

    # 2. Duplicate dates — keep the last, which is the latest revision.
    dupes = int(out.index.duplicated(keep="last").sum())
    if dupes:
        out = out[~out.index.duplicated(keep="last")]
        log.add(f"Dropped {dupes} duplicate date row(s), keeping the latest.")

    # 3. Infinities become NaN so the fill/drop logic handles them uniformly.
    n_inf = int(np.isinf(out[list(COLUMNS)].to_numpy(dtype="float64")).sum())
    if n_inf:
        out[list(COLUMNS)] = out[list(COLUMNS)].replace([np.inf, -np.inf], np.nan)
        log.add(f"Converted {n_inf} infinite value(s) to missing.")

    # 4. Non-positive prices are impossible; mark them missing.
    bad_price = (out[list(PRICE_COLUMNS)] <= 0)
    n_bad = int(bad_price.to_numpy().sum())
    if n_bad:
        out[list(PRICE_COLUMNS)] = out[list(PRICE_COLUMNS)].mask(bad_price)
        log.add(f"Marked {n_bad} non-positive price value(s) as missing.")

    neg_vol = out["volume"] < 0
    if int(neg_vol.sum()):
        out.loc[neg_vol, "volume"] = np.nan
        log.add(f"Marked {int(neg_vol.sum())} negative volume value(s) as missing.")

    # 5. Forward-fill short price gaps only.
    n_missing_price = int(out[list(PRICE_COLUMNS)].isna().to_numpy().sum())
    if n_missing_price:
        out[list(PRICE_COLUMNS)] = out[list(PRICE_COLUMNS)].ffill(limit=max_ffill)
        filled = n_missing_price - int(
            out[list(PRICE_COLUMNS)].isna().to_numpy().sum()
        )
        if filled:
            log.add(
                f"Forward-filled {filled} missing price value(s) "
                f"(runs of at most {max_ffill} days)."
            )

    # 6. Anything still missing cannot be trusted.
    still_na = out[list(COLUMNS)].isna().any(axis=1)
    if int(still_na.sum()):
        out = out[~still_na]
        log.add(f"Dropped {int(still_na.sum())} row(s) still missing values.")

    # 7. Internally impossible bars.
    inconsistent = (
        (out["high"] < out["low"])
        | (out["close"] > out["high"] + 1e-6) | (out["close"] < out["low"] - 1e-6)
        | (out["open"] > out["high"] + 1e-6) | (out["open"] < out["low"] - 1e-6)
    )
    if int(inconsistent.sum()):
        out = out[~inconsistent]
        log.add(f"Dropped {int(inconsistent.sum())} row(s) with impossible OHLC bars.")

    if drop_zero_volume:
        zero = out["volume"] == 0
        if int(zero.sum()):
            out = out[~zero]
            log.add(f"Dropped {int(zero.sum())} zero-volume row(s).")

    log.rows_after = len(out)
    return out, log


# --------------------------------------------------------------------------
# Split / dividend adjustment
# --------------------------------------------------------------------------


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale open/high/low onto the same basis as ``adj_close``.

    Yahoo adjusts only the close for splits and dividends and leaves
    open/high/low raw.  Mixing the two is a silent, serious bug: any indicator
    touching both series (ATR, stochastic %K, anything using the previous
    close) sees a fictitious gap on every split date.

    On AAPL, whose 4:1 split lands mid-sample, the unadjusted mix pushes
    stochastic %K to -296 (its range is 0-100) and inflates ATR from roughly
    2% of price to 5.9%.

    The fix is one shared factor per bar::

        factor = adj_close / close
        adj_open, adj_high, adj_low = open * factor, high * factor, low * factor

    Because every field in a bar is scaled identically, OHLC ordering
    (``low <= open, close <= high``) is preserved exactly.  ``volume`` is left
    alone: it is a share count, not a price.
    """
    out = df.copy()
    factor = out["adj_close"] / out["close"].replace(0.0, np.nan)
    for col in ("open", "high", "low"):
        out[col] = out[col] * factor
    out["close"] = out["adj_close"]
    return out


def validate_and_clean(
    df: pd.DataFrame, **kwargs: Any
) -> tuple[pd.DataFrame, ValidationReport, ValidationReport, CleaningLog]:
    """Validate, clean, then re-validate.

    Returning both reports lets the dashboard show a genuine before/after,
    and lets a test assert that cleaning actually resolved the errors.
    """
    before = validate(df)
    cleaned, log = clean(df, **kwargs)
    after = validate(cleaned)
    return cleaned, before, after, log
