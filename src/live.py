"""Live daily predictions.

Runs once per trading day, after the close, from a scheduled job.  It fetches
the bar that just completed, predicts the next session, and appends the result
to a log that later gets its realised outcome backfilled.

Three properties matter more than anything else here.

**Completed bars only.** The model was trained on finished daily bars, so a
prediction is made only from a bar that has actually closed.  Feeding it a
partial intraday bar would hand it a distribution it never saw in training —
the same class of error as lookahead bias, just in the opposite direction.

**Idempotent.** A scheduled job can fire twice, be retried, or be run by hand.
Predicting for a bar that is already logged is a no-op, so the log can never
gain duplicate rows.

**No calendar arithmetic.** Nothing here computes "the next trading day".
Holidays and half-days make that fragile.  Instead a prediction records the
bar it was made from, and the outcome is backfilled later by looking up
whichever bar actually came next in the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODELS, OUTPUTS, Config
from .data_loader import latest_bar_date, update_cache
from .dataset import realised_returns
from .features import build_features, feature_columns
from .models import ModelPipeline, model_path
from .preprocessing import clean
from .signals import generate_signals

logger = logging.getLogger(__name__)

#: Where the running record of live predictions lives.
PREDICTION_LOG = OUTPUTS / "live_predictions.csv"

#: Trailing bars pulled for feature construction. The longest warm-up is 60,
#: so this leaves generous margin without rebuilding a decade of features.
TAIL_BARS = 400

#: A daily return prediction beyond this magnitude means something is wrong.
SANITY_BOUND = 0.10

#: Column dtypes are declared explicitly. Without this, a fresh log's text
#: columns are inferred as float64 from their all-NaN state, and writing a
#: date or "True" into one raises a pandas dtype warning that will become an
#: error in a future release.
LOG_DTYPES: dict[str, str] = {
    "bar_date": "object",
    "target_date": "object",
    "ticker": "object",
    "model": "object",
    "fingerprint": "object",
    "prediction": "float64",
    "signal": "float64",
    "realised_return": "float64",
    "correct": "object",
    "logged_at": "object",
}

LOG_COLUMNS = list(LOG_DTYPES)


def empty_log() -> pd.DataFrame:
    """A zero-row log with the correct dtypes already set."""
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in LOG_DTYPES.items()}
    )


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    """Force a log frame onto the declared dtypes."""
    out = frame.copy()
    for name, dtype in LOG_DTYPES.items():
        if name not in out.columns:
            out[name] = np.nan
        if dtype == "object":
            out[name] = out[name].fillna("").astype(str).replace("nan", "")
        else:
            out[name] = pd.to_numeric(out[name], errors="coerce")
    return out[LOG_COLUMNS]


@dataclass
class LivePrediction:
    """One prediction, made from one completed bar."""

    bar_date: str
    ticker: str
    model: str
    fingerprint: str
    prediction: float
    signal: float
    logged_at: str
    close: float

    def to_row(self) -> dict:
        return {
            "bar_date": self.bar_date,
            "target_date": "",          # filled in at backfill time
            "ticker": self.ticker,
            "model": self.model,
            "fingerprint": self.fingerprint,
            "prediction": self.prediction,
            "signal": self.signal,
            "realised_return": np.nan,
            "correct": "",
            "logged_at": self.logged_at,
        }

    @property
    def direction(self) -> str:
        if self.signal > 0:
            return "LONG"
        if self.signal < 0:
            return "SHORT"
        return "FLAT"


class LiveError(RuntimeError):
    """Raised when a live prediction cannot be made safely."""


# --------------------------------------------------------------------------
# Bar completeness
# --------------------------------------------------------------------------

MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16  # 16:00 ET


def bar_is_final(bar_date, now: datetime | None = None) -> bool:
    """Has the session for ``bar_date`` actually finished?

    A bar dated earlier than today is always final.  A bar dated *today* is
    only final once the market has closed, because until then the provider is
    serving a partial bar whose open/high/low/close are all still moving.

    This guard matters because the job is idempotent: if a partial bar were
    ever logged, ``already_logged`` would refuse to revisit it and the
    incomplete version would be frozen into the track record permanently. The
    schedule already avoids this, but a manual run at lunchtime would not.
    """
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        eastern = now.astimezone(ZoneInfo(MARKET_TZ))
    except Exception:  # noqa: BLE001 - missing tz database
        # Without a timezone database, fall back to UTC and the later of the
        # two possible close times (21:00 UTC during EST). Conservative:
        # it may decline a valid bar, never accept a partial one.
        eastern = now
        return pd.Timestamp(bar_date).date() < now.date() or now.hour >= 21

    bar = pd.Timestamp(bar_date).date()
    if bar < eastern.date():
        return True
    if bar > eastern.date():
        return False  # dated in the future; something is wrong upstream
    return eastern.hour >= MARKET_CLOSE_HOUR


# --------------------------------------------------------------------------
# Prediction log
# --------------------------------------------------------------------------


def load_log(path: Path | None = None) -> pd.DataFrame:
    """Read the prediction log, returning an empty frame when absent."""
    path = Path(path) if path is not None else PREDICTION_LOG
    if not path.exists():
        return empty_log()
    return _coerce(pd.read_csv(path))


def save_log(frame: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else PREDICTION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce(frame).to_csv(path, index=False)
    return path


def already_logged(log: pd.DataFrame, ticker: str, bar_date: str, model: str) -> bool:
    """Has this (ticker, bar, model) combination already been predicted?"""
    if log.empty:
        return False
    match = (
        (log["ticker"].astype(str) == ticker)
        & (log["bar_date"].astype(str) == str(bar_date))
        & (log["model"].astype(str) == model)
    )
    return bool(match.any())


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def resolve_model(cfg: Config, model_file: Path | None = None) -> ModelPipeline:
    """Load the persisted model for this config, or fail loudly.

    Deliberately never falls back to training on the fly: a scheduled job that
    silently retrains would change the model behind the track record without
    anyone noticing.
    """
    path = Path(model_file) if model_file is not None else model_path(cfg)
    if not path.exists():
        raise LiveError(
            f"No trained model at {path}. Run: python cli.py train --save "
            f"--ticker {cfg.ticker} --model {cfg.model}"
        )
    return ModelPipeline.load(path)


def predict_latest(
    cfg: Config,
    model_file: Path | None = None,
    cache_dir: Path | None = None,
) -> LivePrediction:
    """Fetch the newest completed bar and predict the next session."""
    raw = update_cache(cfg.ticker, cfg.start, cache_dir=cache_dir)
    cleaned, _ = clean(raw)

    if len(cleaned) < 100:
        raise LiveError(
            f"Only {len(cleaned)} bars available for {cfg.ticker}; at least "
            "100 are needed for the feature warm-up."
        )

    # Only the tail is needed: the longest lookback is 60 bars, so rebuilding
    # a decade of features on every scheduled run would be pure waste.
    tail = cleaned.tail(TAIL_BARS)
    featured = build_features(tail, cfg, dropna=True)
    if featured.empty:
        raise LiveError("Feature construction produced no complete rows.")

    latest = featured.iloc[[-1]]
    bar_date = pd.Timestamp(latest.index[-1])

    pipeline = resolve_model(cfg, model_file)
    missing = [c for c in pipeline.feature_names_ if c not in featured.columns]
    if missing:
        raise LiveError(f"Model expects features not present live: {missing}")

    prediction = float(pipeline.predict(latest[pipeline.feature_names_]).iloc[0])

    if not np.isfinite(prediction) or abs(prediction) > SANITY_BOUND:
        raise LiveError(
            f"Prediction {prediction:.4f} is outside the plausible range for a "
            f"daily return (+/-{SANITY_BOUND:.0%}). Refusing to emit a signal."
        )

    signal = float(
        generate_signals(
            pd.Series([prediction], index=[bar_date]),
            cfg.signal_threshold, cfg.allow_short,
        ).iloc[0]
    )

    return LivePrediction(
        bar_date=bar_date.date().isoformat(),
        ticker=cfg.ticker,
        model=cfg.model,
        fingerprint=cfg.fingerprint(),
        prediction=prediction,
        signal=signal,
        logged_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        close=float(cleaned["close"].loc[bar_date]),
    )


# --------------------------------------------------------------------------
# Backfill
# --------------------------------------------------------------------------


def backfill_outcomes(
    log: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, int]:
    """Fill in what actually happened, for rows whose next bar now exists.

    Rather than computing "the next trading day", this looks up whichever bar
    genuinely follows the prediction's bar in the data — so holidays, weekends
    and unscheduled closures need no special handling.
    """
    if log.empty:
        return log, 0

    returns = realised_returns(prices)
    dates = prices.index
    updated = log.copy()
    filled = 0

    for i, row in updated.iterrows():
        if str(row["ticker"]) != ticker:
            continue
        if pd.notna(row["realised_return"]) and str(row["realised_return"]) != "":
            continue

        bar = pd.Timestamp(str(row["bar_date"]))
        later = dates[dates > bar]
        if len(later) == 0:
            continue  # the next session has not happened yet

        target = later[0]
        realised = float(returns.loc[target])
        signal = float(row["signal"])

        updated.at[i, "target_date"] = target.date().isoformat()
        updated.at[i, "realised_return"] = realised
        # "Correct" means the position taken was the right one. A flat day is
        # neither right nor wrong, so it is recorded as such rather than
        # counted as a win.
        if signal == 0:
            updated.at[i, "correct"] = "flat"
        else:
            updated.at[i, "correct"] = str(bool(np.sign(signal) == np.sign(realised)))
        filled += 1

    return updated, filled


# --------------------------------------------------------------------------
# The scheduled entry point
# --------------------------------------------------------------------------


def run_daily(
    cfg: Config,
    model_file: Path | None = None,
    log_path: Path | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """One scheduled run: update data, backfill outcomes, predict if new.

    Returns a summary describing what happened, so the job's output is
    readable in a CI log.
    """
    log_path = Path(log_path) if log_path is not None else PREDICTION_LOG

    raw = update_cache(cfg.ticker, cfg.start, cache_dir=cache_dir)
    prices, _ = clean(raw)
    log = load_log(log_path)

    # Backfill first: yesterday's prediction can be scored before today's is made.
    log, filled = backfill_outcomes(log, prices, cfg.ticker)

    bar_date = pd.Timestamp(prices.index.max()).date().isoformat()
    status = "predicted"
    prediction: LivePrediction | None = None

    if already_logged(log, cfg.ticker, bar_date, cfg.model):
        # No new bar since the last run. Normal on weekends, holidays, and
        # any retry — not an error.
        status = "no_new_bar"
        logger.info("Bar %s already logged for %s; nothing to do",
                    bar_date, cfg.ticker)
    elif not bar_is_final(bar_date):
        # Mid-session: the latest bar is still forming. Logging it now would
        # freeze a partial bar into the record, since the idempotency check
        # would then refuse to revisit it.
        status = "bar_not_final"
        logger.info(
            "Bar %s for %s has not closed yet; declining to predict",
            bar_date, cfg.ticker,
        )
    else:
        prediction = predict_latest(cfg, model_file, cache_dir)
        new_row = _coerce(pd.DataFrame([prediction.to_row()]))
        # Concatenating onto a zero-row frame triggers a pandas dtype-inference
        # warning, and there is nothing to preserve in that case anyway.
        log = new_row if log.empty else pd.concat([log, new_row], ignore_index=True)
        logger.info("Logged %s prediction for %s from bar %s",
                    prediction.direction, cfg.ticker, prediction.bar_date)

    log = log.sort_values(["ticker", "bar_date"]).reset_index(drop=True)
    save_log(log, log_path)

    return {
        "status": status,
        "ticker": cfg.ticker,
        "model": cfg.model,
        "bar_date": bar_date,
        "outcomes_backfilled": filled,
        "log_rows": int(len(log)),
        "prediction": prediction.prediction if prediction else None,
        "signal": prediction.signal if prediction else None,
        "direction": prediction.direction if prediction else None,
    }


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------


def scorecard(log: pd.DataFrame, ticker: str | None = None) -> dict:
    """Live track record so far.

    This is the number that cannot be curve-fitted: every row was written
    before its outcome was known.
    """
    frame = log.copy()
    if ticker:
        frame = frame[frame["ticker"].astype(str) == ticker]

    scored = frame[frame["realised_return"].notna()]
    directional = scored[scored["correct"].astype(str).isin(["True", "False"])]

    empty = {
        "n_predictions": int(len(frame)),
        "n_scored": int(len(scored)),
        "n_directional": int(len(directional)),
        "hit_rate": float("nan"),
        "strategy_return": float("nan"),
        "buy_and_hold_return": float("nan"),
        "mean_prediction": float("nan"),
        "first_date": "",
        "last_date": "",
    }
    if scored.empty:
        return empty

    signals = scored["signal"].astype(float)
    realised = scored["realised_return"].astype(float)

    return {
        **empty,
        "hit_rate": (
            float((directional["correct"].astype(str) == "True").mean())
            if len(directional) else float("nan")
        ),
        # Gross of costs: this is a signal scorecard, not a backtest.
        "strategy_return": float((1 + signals * realised).prod() - 1),
        "buy_and_hold_return": float((1 + realised).prod() - 1),
        "mean_prediction": float(scored["prediction"].astype(float).mean()),
        "first_date": str(scored["bar_date"].min()),
        "last_date": str(scored["bar_date"].max()),
    }
