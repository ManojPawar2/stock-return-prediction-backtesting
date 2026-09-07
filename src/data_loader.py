"""Market data ingestion.

Responsibilities:

* download daily OHLCV from Yahoo Finance,
* normalise whatever shape yfinance returns into one fixed schema,
* cache to Parquet so a study is reproducible and runs offline.

The normalisation step matters more than it looks.  yfinance returns a
MultiIndex column frame when given a list of tickers (and, depending on the
version, even for a single ticker), sometimes drops ``Adj Close`` when
``auto_adjust=True``, and capitalises differently across versions.  Every
downstream module depends on exactly one schema, so it is pinned here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW, Config

logger = logging.getLogger(__name__)

#: The one schema every downstream module may assume.
COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")

_ALIASES: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}


class DataLoadError(RuntimeError):
    """Raised when data cannot be obtained or fails the schema contract."""


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def cache_path(ticker: str, start: str, end: str, cache_dir: Path | None = None) -> Path:
    """Deterministic on-disk location for one (ticker, range) request."""
    directory = Path(cache_dir) if cache_dir is not None else DATA_RAW
    stem = f"{ticker.upper().replace('/', '-')}_{start}_{end}"
    return directory / f"{stem}.parquet"


def is_cached(ticker: str, start: str, end: str, cache_dir: Path | None = None) -> bool:
    return cache_path(ticker, start, end, cache_dir).exists()


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def normalise(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Coerce a raw yfinance frame into the project schema.

    Returns a frame indexed by a tz-naive ``DatetimeIndex`` named ``date`` with
    exactly the columns in :data:`COLUMNS`, sorted ascending.
    """
    if df is None or len(df) == 0:
        raise DataLoadError(
            f"No rows returned for {ticker or 'request'} — check the ticker "
            "symbol and that the date range covers trading days."
        )

    out = df.copy()

    # 1. Flatten a MultiIndex column frame down to the price-field level.
    if isinstance(out.columns, pd.MultiIndex):
        levels = [
            lvl for lvl in range(out.columns.nlevels)
            if any(str(v).strip().lower() in _ALIASES for v in out.columns.get_level_values(lvl))
        ]
        if not levels:
            raise DataLoadError(f"Cannot find OHLCV fields in columns: {list(out.columns)}")
        field_level = levels[0]
        if ticker and out.columns.nlevels > 1:
            # Keep only this ticker's slice when several are present.
            for lvl in range(out.columns.nlevels):
                if lvl == field_level:
                    continue
                values = {str(v).upper() for v in out.columns.get_level_values(lvl)}
                if ticker.upper() in values and len(values) > 1:
                    out = out.xs(ticker.upper(), axis=1, level=lvl, drop_level=True)
                    break
        out.columns = [str(c).strip() for c in out.columns.get_level_values(
            field_level if out.columns.nlevels > 1 else 0
        )] if isinstance(out.columns, pd.MultiIndex) else [str(c).strip() for c in out.columns]

    # 2. Rename to canonical lowercase names.
    renamed: dict[str, str] = {}
    for col in out.columns:
        key = str(col).strip().lower().replace(" ", "_")
        canonical = _ALIASES.get(key) or _ALIASES.get(key.replace("_", " "))
        if canonical:
            renamed[col] = canonical
    out = out.rename(columns=renamed)

    # Duplicate canonical names can appear after flattening; keep the first.
    out = out.loc[:, ~out.columns.duplicated()]

    # 3. adj_close is optional upstream (auto_adjust=True drops it); if it is
    #    missing, close is already adjusted, so they are the same series.
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]

    missing = [c for c in COLUMNS if c not in out.columns]
    if missing:
        raise DataLoadError(
            f"Missing required columns {missing} for {ticker or 'request'}; "
            f"got {sorted(out.columns)}"
        )

    out = out[list(COLUMNS)]

    # 4. Index: tz-naive datetimes named 'date', ascending, unique.
    #    Building from .values deliberately drops any inferred `freq`, which
    #    does not survive a Parquet round-trip — without this, a fresh frame
    #    and its cached copy would compare unequal on index metadata alone.
    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out.index = pd.DatetimeIndex(index.normalize().values, name="date")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # 5. Numeric dtypes; volume stays float so NaNs survive to validation.
    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch from Yahoo Finance. Isolated so tests can monkeypatch it."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment problem
        raise DataLoadError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    logger.info("Downloading %s from %s to %s", ticker, start, end)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,   # keep a separate adj_close column
        progress=False,
        actions=False,
        threads=False,
    )
    return raw


def load_prices(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return normalised daily OHLCV for one ticker.

    Reads from the Parquet cache when available; otherwise downloads, writes
    the cache, and returns the frame.
    """
    path = cache_path(ticker, start, end, cache_dir)

    if use_cache and path.exists():
        logger.info("Loading %s from cache %s", ticker, path.name)
        cached = pd.read_parquet(path)
        cached.index = pd.DatetimeIndex(pd.to_datetime(cached.index), name="date")
        return cached[list(COLUMNS)]

    frame = normalise(_download(ticker, start, end), ticker=ticker)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    logger.info("Cached %d rows to %s", len(frame), path.name)
    return frame


def load_many(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load several tickers, skipping (with a warning) any that fail.

    Used by the multi-ticker stress sweep, where one delisted or mistyped
    symbol should not abort the whole study.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            out[ticker] = load_prices(ticker, start, end, use_cache, cache_dir)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the sweep
            logger.warning("Skipping %s: %s", ticker, exc)
    if not out:
        raise DataLoadError(f"None of the requested tickers loaded: {tickers}")
    return out


def load_from_config(cfg: Config) -> pd.DataFrame:
    """Convenience wrapper used by the CLI and the dashboard."""
    return load_prices(cfg.ticker, cfg.start, cfg.end, cfg.use_cache)


# --------------------------------------------------------------------------
# Rolling cache for live use
# --------------------------------------------------------------------------

#: How many trailing days to re-request on an incremental update.
OVERLAP_DAYS = 7


def live_cache_path(ticker: str, cache_dir: Path | None = None) -> Path:
    """One rolling file per ticker, with no date range in the name.

    The range-keyed :func:`cache_path` is wrong for live use: changing the end
    date every day would mint a new file and re-download ten years of history
    each time.
    """
    directory = Path(cache_dir) if cache_dir is not None else DATA_RAW
    return directory / f"{ticker.upper().replace('/', '-')}_live.parquet"


def _adjustment_basis(frame: pd.DataFrame) -> float:
    """``adj_close / close`` on the earliest row — the split/dividend factor."""
    row = frame.iloc[0]
    close = float(row["close"])
    if close == 0:
        return float("nan")
    return float(row["adj_close"]) / close


def update_cache(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: Path | None = None,
    overlap_days: int = OVERLAP_DAYS,
) -> pd.DataFrame:
    """Bring a ticker's rolling cache up to date, downloading only what is new.

    Two details here are easy to get wrong and both corrupt the series
    silently.

    **Overlap, not append-from-last.** The refetch starts ``overlap_days``
    *before* the last cached bar rather than one day after it. Yahoo revises
    recent bars, and de-duplicating with ``keep="last"`` lets those
    corrections land instead of freezing the first version in forever.

    **Dividend basis.** ``adj_close`` is retroactive: when a dividend is paid,
    every historical adjusted close shifts onto a new basis. Appending only
    new rows would leave old rows on the old basis and new rows on the new
    one, quietly corrupting every return that spans the boundary. So the
    factor is compared before and after, and a full re-download is triggered
    when it moves.
    """
    path = live_cache_path(ticker, cache_dir)
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()

    if not path.exists():
        logger.info("No live cache for %s; downloading full history", ticker)
        frame = normalise(_download(ticker, start, end), ticker=ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    existing = pd.read_parquet(path)
    existing.index = pd.DatetimeIndex(pd.to_datetime(existing.index), name="date")

    since = (existing.index.max() - pd.Timedelta(days=overlap_days)).date().isoformat()
    if pd.Timestamp(since) >= pd.Timestamp(end):
        return existing[list(COLUMNS)]

    try:
        fresh = normalise(_download(ticker, since, end), ticker=ticker)
    except DataLoadError:
        # No new bars is normal outside trading hours, not an error.
        logger.info("No new data for %s since %s", ticker, since)
        return existing[list(COLUMNS)]

    old_basis, new_basis = _adjustment_basis(existing), _adjustment_basis(fresh)
    overlap = existing.index.intersection(fresh.index)
    if len(overlap):
        # Compare on a shared date so the check is not confused by the
        # factor legitimately differing across distant points in time.
        shared = overlap.max()
        old_basis = float(
            existing.loc[shared, "adj_close"] / existing.loc[shared, "close"]
        )
        new_basis = float(fresh.loc[shared, "adj_close"] / fresh.loc[shared, "close"])

    if not np.isclose(old_basis, new_basis, rtol=1e-6, equal_nan=True):
        logger.warning(
            "Adjustment basis for %s moved (%.6f -> %.6f); re-downloading the "
            "full history so every row shares one basis.",
            ticker, old_basis, new_basis,
        )
        frame = normalise(_download(ticker, start, end), ticker=ticker)
    else:
        frame = pd.concat([existing, fresh])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    frame = frame[list(COLUMNS)]
    frame.to_parquet(path)
    logger.info(
        "Live cache for %s now runs to %s (%d rows)",
        ticker, frame.index.max().date(), len(frame),
    )
    return frame


def latest_bar_date(ticker: str, cache_dir: Path | None = None) -> pd.Timestamp | None:
    """Date of the most recent cached bar, or None when nothing is cached."""
    path = live_cache_path(ticker, cache_dir)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    return pd.Timestamp(pd.to_datetime(frame.index).max())


def describe(df: pd.DataFrame) -> dict[str, object]:
    """Headline dataset statistics for the Overview page."""
    returns = df["adj_close"].pct_change()
    return {
        "rows": int(len(df)),
        "start": df.index.min().date().isoformat(),
        "end": df.index.max().date().isoformat(),
        "trading_days": int(len(df)),
        "years": round(len(df) / 252, 2),
        "first_close": float(df["adj_close"].iloc[0]),
        "last_close": float(df["adj_close"].iloc[-1]),
        "total_return_pct": float(
            (df["adj_close"].iloc[-1] / df["adj_close"].iloc[0] - 1) * 100
        ),
        "mean_daily_return_pct": float(returns.mean() * 100),
        "daily_volatility_pct": float(returns.std() * 100),
        "annualised_volatility_pct": float(returns.std() * (252 ** 0.5) * 100),
        "mean_volume": float(df["volume"].mean()),
        "missing_values": int(df.isna().sum().sum()),
    }
