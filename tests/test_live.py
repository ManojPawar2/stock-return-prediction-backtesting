"""Tests for the incremental cache and live daily predictions.

The three things that would break a scheduled job silently:

* an append-only update that misses Yahoo's revisions or a dividend
  re-basing every historical adjusted close,
* a job that is not idempotent, so a retry duplicates rows in the track
  record,
* a prediction made from a bar that has not actually closed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import live as lv
from src.config import Config
from src.live import (
    LiveError,
    LivePrediction,
    already_logged,
    backfill_outcomes,
    load_log,
    predict_latest,
    resolve_model,
    run_daily,
    save_log,
    scorecard,
)
from src.models import ModelPipeline
from src.pipeline import load, prepare, train


def raw_frame(n: int, start: str = "2015-01-02", factor: float = 1.0,
              seed: int = 71) -> pd.DataFrame:
    """A yfinance-shaped frame; ``factor`` sets the adj_close basis."""
    idx = pd.bdate_range(start, periods=n, name="Date")
    rng = np.random.default_rng(seed)
    close = pd.Series(80 * np.exp(np.cumsum(rng.normal(0.0005, 0.014, n))), index=idx)
    span = close * pd.Series(rng.uniform(0.006, 0.03, n), index=idx)
    pos = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    low = close - span * pos
    return pd.DataFrame({
        "Open": low + span * 0.45,
        "High": low + span,
        "Low": low,
        "Close": close,
        "Adj Close": close * factor,
        "Volume": rng.integers(1e6, 9e6, n).astype("float64"),
    }, index=idx)


# ==================================================== INCREMENTAL CACHE


def test_first_update_downloads_full_history(tmp_path, monkeypatch):
    calls = []

    def fake(ticker, start, end):
        calls.append((start, end))
        return raw_frame(300)

    monkeypatch.setattr(dl, "_download", fake)
    frame = dl.update_cache("TEST", cache_dir=tmp_path)

    assert len(frame) == 300
    assert len(calls) == 1
    assert dl.live_cache_path("TEST", tmp_path).exists()


def test_second_update_refetches_only_the_recent_window(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(300))
    dl.update_cache("TEST", cache_dir=tmp_path)

    requested = []

    def fake(ticker, start, end):
        requested.append(start)
        return raw_frame(10, start="2016-02-24")

    monkeypatch.setattr(dl, "_download", fake)
    dl.update_cache("TEST", cache_dir=tmp_path)

    # The second request must start near the cache end, not at 2015.
    assert requested and pd.Timestamp(requested[0]) > pd.Timestamp("2016-01-01")


def test_update_overlaps_rather_than_appending_from_the_next_day(tmp_path, monkeypatch):
    """The overlap is what lets Yahoo's revisions land."""
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(200))
    first = dl.update_cache("TEST", cache_dir=tmp_path)
    last_date = first.index.max()

    requested = {}

    def fake(ticker, start, end):
        requested["start"] = pd.Timestamp(start)
        return raw_frame(5, start=last_date.date().isoformat())

    monkeypatch.setattr(dl, "_download", fake)
    dl.update_cache("TEST", cache_dir=tmp_path, overlap_days=7)

    assert requested["start"] < last_date, "refetch must overlap the cached tail"


def test_revised_bars_replace_the_cached_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(200))
    first = dl.update_cache("TEST", cache_dir=tmp_path)
    overlap_date = first.index[-3]
    original_close = float(first.loc[overlap_date, "close"])

    revised = raw_frame(200)
    revised.loc[overlap_date, "Close"] = original_close * 1.5
    revised.loc[overlap_date, "High"] = original_close * 1.6
    revised.loc[overlap_date, "Adj Close"] = original_close * 1.5

    monkeypatch.setattr(dl, "_download", lambda t, s, e: revised.tail(20))
    updated = dl.update_cache("TEST", cache_dir=tmp_path)

    assert float(updated.loc[overlap_date, "close"]) == pytest.approx(
        original_close * 1.5
    ), "a revised bar must overwrite the cached one"


def test_new_bars_are_appended(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(200))
    first = dl.update_cache("TEST", cache_dir=tmp_path)

    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(220))
    second = dl.update_cache("TEST", cache_dir=tmp_path)

    assert len(second) > len(first)
    assert second.index.max() > first.index.max()
    assert second.index.is_monotonic_increasing
    assert not second.index.duplicated().any()


def test_a_dividend_rebasing_triggers_a_full_redownload(tmp_path, monkeypatch):
    """The subtle one: adj_close shifts retroactively across all history."""
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(200, factor=1.0))
    dl.update_cache("TEST", cache_dir=tmp_path)

    ranges = []

    def fake(ticker, start, end):
        ranges.append(start)
        # Same bars, but every adjusted close now on a new basis.
        return raw_frame(200, factor=0.95)

    monkeypatch.setattr(dl, "_download", fake)
    updated = dl.update_cache("TEST", start="2015-01-01", cache_dir=tmp_path)

    assert pd.Timestamp(ranges[-1]) == pd.Timestamp("2015-01-01"), (
        "a basis change must trigger a full re-download, not an append"
    )
    basis = updated["adj_close"] / updated["close"]
    assert basis.std() == pytest.approx(0.0, abs=1e-9), (
        "every row must end up on one shared adjustment basis"
    )


def test_unchanged_basis_does_not_redownload_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(200, factor=0.9))
    dl.update_cache("TEST", cache_dir=tmp_path)

    ranges = []

    def fake(ticker, start, end):
        ranges.append(pd.Timestamp(start))
        return raw_frame(210, factor=0.9).tail(15)

    monkeypatch.setattr(dl, "_download", fake)
    dl.update_cache("TEST", start="2015-01-01", cache_dir=tmp_path)

    assert ranges[-1] > pd.Timestamp("2015-06-01"), "should be an incremental fetch"


def test_no_new_data_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(200))
    first = dl.update_cache("TEST", cache_dir=tmp_path)

    def empty(ticker, start, end):
        raise dl.DataLoadError("No rows returned")

    monkeypatch.setattr(dl, "_download", empty)
    second = dl.update_cache("TEST", cache_dir=tmp_path)

    pd.testing.assert_frame_equal(first, second)


def test_latest_bar_date(tmp_path, monkeypatch):
    assert dl.latest_bar_date("TEST", tmp_path) is None
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(50))
    frame = dl.update_cache("TEST", cache_dir=tmp_path)
    assert dl.latest_bar_date("TEST", tmp_path) == frame.index.max()


def test_live_cache_path_has_no_date_range(tmp_path):
    path = dl.live_cache_path("aapl", tmp_path)
    assert path.name == "AAPL_live.parquet"
    assert "2015" not in path.name


# ========================================================== LIVE PREDICT


@pytest.fixture
def live_env(tmp_path, monkeypatch):
    """A trained model plus a cache, all inside tmp_path."""
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(1400))
    monkeypatch.setattr(dl, "DATA_RAW", tmp_path / "raw")
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    cfg = Config(
        ticker="TEST", start="2015-01-01", end="2020-12-31",
        train_end="2018-12-31", valid_end="2019-12-31", model="ridge",
    )
    prepared = prepare(cfg, load(cfg).cleaned)
    model = train(cfg, prepared, "ridge")
    model_file = tmp_path / "model.joblib"
    model.pipeline.save(model_file)

    return cfg, model_file, tmp_path


def test_predict_latest_uses_the_most_recent_bar(live_env):
    cfg, model_file, tmp = live_env
    result = predict_latest(cfg, model_file, cache_dir=tmp / "raw")

    cached = dl.update_cache(cfg.ticker, cache_dir=tmp / "raw")
    assert result.bar_date == cached.index.max().date().isoformat()
    assert result.ticker == "TEST"
    assert np.isfinite(result.prediction)
    assert result.signal in (-1.0, 0.0, 1.0)
    assert result.direction in ("LONG", "FLAT", "SHORT")


def test_prediction_matches_a_full_pipeline_prediction(live_env):
    """The live path must agree with the batch path on the same bar."""
    cfg, model_file, tmp = live_env
    result = predict_latest(cfg, model_file, cache_dir=tmp / "raw")

    from src.preprocessing import clean

    cached = dl.update_cache(cfg.ticker, cache_dir=tmp / "raw")
    cleaned, _ = clean(cached)
    featured = build = __import__(
        "src.features", fromlist=["build_features"]
    ).build_features(cleaned, cfg, dropna=True)
    pipeline = ModelPipeline.load(model_file)
    expected = float(
        pipeline.predict(featured[pipeline.feature_names_].iloc[[-1]]).iloc[0]
    )
    assert result.prediction == pytest.approx(expected, rel=1e-9)


def test_missing_model_fails_loudly(live_env):
    cfg, _, tmp = live_env
    with pytest.raises(LiveError, match="No trained model"):
        resolve_model(cfg, tmp / "nope.joblib")


def test_absurd_prediction_is_refused(live_env, monkeypatch):
    """A daily return of 500% means something is broken; emit no signal."""
    cfg, model_file, tmp = live_env

    monkeypatch.setattr(
        ModelPipeline, "predict",
        lambda self, X: pd.Series([5.0], index=X.index),
    )
    with pytest.raises(LiveError, match="plausible range"):
        predict_latest(cfg, model_file, cache_dir=tmp / "raw")


def test_too_little_history_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(40))
    cfg = Config(ticker="TINY")
    with pytest.raises(LiveError, match="at least"):
        predict_latest(cfg, tmp_path / "m.joblib", cache_dir=tmp_path)


# ================================================================ LOG


def test_log_roundtrip(tmp_path):
    path = tmp_path / "log.csv"
    assert load_log(path).empty

    row = LivePrediction(
        bar_date="2026-09-04", ticker="AAPL", model="ridge",
        fingerprint="abc", prediction=0.001, signal=1.0,
        logged_at="2026-09-04T20:30:00+00:00", close=320.0,
    ).to_row()
    save_log(pd.DataFrame([row]), path)

    reread = load_log(path)
    assert len(reread) == 1
    assert list(reread.columns) == lv.LOG_COLUMNS
    assert reread.iloc[0]["ticker"] == "AAPL"


def test_already_logged_matches_on_ticker_bar_and_model():
    log = pd.DataFrame([{
        "ticker": "AAPL", "bar_date": "2026-09-04", "model": "ridge",
    }])
    assert already_logged(log, "AAPL", "2026-09-04", "ridge")
    assert not already_logged(log, "MSFT", "2026-09-04", "ridge")
    assert not already_logged(log, "AAPL", "2026-09-05", "ridge")
    assert not already_logged(log, "AAPL", "2026-09-04", "xgboost")


def test_already_logged_on_empty_log():
    assert not already_logged(pd.DataFrame(), "AAPL", "2026-09-04", "ridge")


# =========================================================== BACKFILL


def make_prices(n: int = 10) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", periods=n, name="date")
    close = pd.Series(np.linspace(100, 110, n), index=idx)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "adj_close": close,
        "volume": pd.Series(1e6, index=idx),
    })


def test_backfill_uses_the_next_actual_bar_not_a_calendar():
    """Holidays and weekends need no special handling."""
    prices = make_prices(10)
    # Drop a bar to simulate a holiday.
    prices = prices.drop(prices.index[3])

    log = pd.DataFrame([{
        "bar_date": prices.index[2].date().isoformat(), "target_date": "",
        "ticker": "T", "model": "ridge", "fingerprint": "f",
        "prediction": 0.01, "signal": 1.0, "realised_return": np.nan,
        "correct": "", "logged_at": "x",
    }])

    updated, filled = backfill_outcomes(log, prices, "T")
    assert filled == 1
    # The next *available* bar, which is the one after the missing day.
    assert updated.iloc[0]["target_date"] == prices.index[3].date().isoformat()


def test_backfill_computes_the_right_return():
    prices = make_prices(5)
    log = pd.DataFrame([{
        "bar_date": prices.index[1].date().isoformat(), "target_date": "",
        "ticker": "T", "model": "m", "fingerprint": "f",
        "prediction": 0.01, "signal": 1.0, "realised_return": np.nan,
        "correct": "", "logged_at": "x",
    }])
    updated, _ = backfill_outcomes(log, prices, "T")
    expected = prices["close"].iloc[2] / prices["close"].iloc[1] - 1
    assert float(updated.iloc[0]["realised_return"]) == pytest.approx(expected)
    assert updated.iloc[0]["correct"] == "True"   # rising series, long signal


def test_backfill_marks_a_wrong_call_false():
    prices = make_prices(5)
    log = pd.DataFrame([{
        "bar_date": prices.index[1].date().isoformat(), "target_date": "",
        "ticker": "T", "model": "m", "fingerprint": "f",
        "prediction": -0.01, "signal": -1.0, "realised_return": np.nan,
        "correct": "", "logged_at": "x",
    }])
    updated, _ = backfill_outcomes(log, prices, "T")
    assert updated.iloc[0]["correct"] == "False"


def test_flat_signals_are_neither_right_nor_wrong():
    prices = make_prices(5)
    log = pd.DataFrame([{
        "bar_date": prices.index[1].date().isoformat(), "target_date": "",
        "ticker": "T", "model": "m", "fingerprint": "f",
        "prediction": 0.0, "signal": 0.0, "realised_return": np.nan,
        "correct": "", "logged_at": "x",
    }])
    updated, _ = backfill_outcomes(log, prices, "T")
    assert updated.iloc[0]["correct"] == "flat"


def test_backfill_skips_predictions_whose_future_has_not_happened():
    prices = make_prices(5)
    log = pd.DataFrame([{
        "bar_date": prices.index[-1].date().isoformat(), "target_date": "",
        "ticker": "T", "model": "m", "fingerprint": "f",
        "prediction": 0.01, "signal": 1.0, "realised_return": np.nan,
        "correct": "", "logged_at": "x",
    }])
    updated, filled = backfill_outcomes(log, prices, "T")
    assert filled == 0
    assert pd.isna(updated.iloc[0]["realised_return"])


def test_backfill_does_not_rewrite_scored_rows():
    prices = make_prices(5)
    log = pd.DataFrame([{
        "bar_date": prices.index[1].date().isoformat(), "target_date": "old",
        "ticker": "T", "model": "m", "fingerprint": "f",
        "prediction": 0.01, "signal": 1.0, "realised_return": 0.999,
        "correct": "True", "logged_at": "x",
    }])
    updated, filled = backfill_outcomes(log, prices, "T")
    assert filled == 0
    assert float(updated.iloc[0]["realised_return"]) == 0.999


# ========================================================== RUN DAILY


def test_run_daily_logs_a_prediction(live_env):
    cfg, model_file, tmp = live_env
    log_path = tmp / "log.csv"

    result = run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")
    assert result["status"] == "predicted"
    assert result["direction"] in ("LONG", "FLAT", "SHORT")
    assert load_log(log_path).shape[0] == 1


def test_run_daily_is_idempotent(live_env):
    """A retry, or a manual re-run, must not duplicate the track record."""
    cfg, model_file, tmp = live_env
    log_path = tmp / "log.csv"

    first = run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")
    second = run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")
    third = run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")

    assert first["status"] == "predicted"
    assert second["status"] == "no_new_bar"
    assert third["status"] == "no_new_bar"
    assert len(load_log(log_path)) == 1, "the log gained a duplicate row"


def test_run_daily_backfills_on_the_next_run(live_env, monkeypatch):
    cfg, model_file, tmp = live_env
    log_path = tmp / "log.csv"

    run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")
    assert pd.isna(load_log(log_path).iloc[0]["realised_return"])

    # A new bar arrives, so the previous prediction becomes scoreable.
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(1401))
    result = run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")

    log = load_log(log_path)
    assert result["outcomes_backfilled"] >= 1
    assert pd.notna(log.iloc[0]["realised_return"])
    assert log.iloc[0]["target_date"] != ""
    assert len(log) == 2


def test_run_daily_keeps_the_log_sorted(live_env, monkeypatch):
    cfg, model_file, tmp = live_env
    log_path = tmp / "log.csv"
    for extra in (0, 1, 2):
        monkeypatch.setattr(
            dl, "_download", lambda t, s, e, k=extra: raw_frame(1400 + k)
        )
        run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")

    log = load_log(log_path)
    assert list(log["bar_date"]) == sorted(log["bar_date"])


# =========================================================== SCORECARD


def test_scorecard_on_an_empty_log():
    card = scorecard(load_log(Path("does_not_exist.csv")))
    assert card["n_predictions"] == 0
    assert np.isnan(card["hit_rate"])


def test_scorecard_counts_hits():
    log = pd.DataFrame([
        {"ticker": "T", "bar_date": "2026-01-01", "signal": 1.0,
         "prediction": 0.01, "realised_return": 0.02, "correct": "True"},
        {"ticker": "T", "bar_date": "2026-01-02", "signal": 1.0,
         "prediction": 0.01, "realised_return": -0.01, "correct": "False"},
        {"ticker": "T", "bar_date": "2026-01-05", "signal": 0.0,
         "prediction": 0.0, "realised_return": 0.03, "correct": "flat"},
        {"ticker": "T", "bar_date": "2026-01-06", "signal": 1.0,
         "prediction": 0.01, "realised_return": np.nan, "correct": ""},
    ])
    card = scorecard(log, "T")
    assert card["n_predictions"] == 4
    assert card["n_scored"] == 3
    assert card["n_directional"] == 2, "flat days are excluded from the hit rate"
    assert card["hit_rate"] == pytest.approx(0.5)


def test_scorecard_compares_against_buy_and_hold():
    log = pd.DataFrame([
        {"ticker": "T", "bar_date": "2026-01-01", "signal": 0.0,
         "prediction": 0.0, "realised_return": 0.10, "correct": "flat"},
        {"ticker": "T", "bar_date": "2026-01-02", "signal": 1.0,
         "prediction": 0.01, "realised_return": 0.10, "correct": "True"},
    ])
    card = scorecard(log, "T")
    # Flat on the first day, long on the second.
    assert card["strategy_return"] == pytest.approx(0.10)
    assert card["buy_and_hold_return"] == pytest.approx(0.21)


# ================================================== PARTIAL-BAR GUARD


def _utc(y, m, d, h):
    from datetime import datetime, timezone
    return datetime(y, m, d, h, 0, tzinfo=timezone.utc)


def test_a_past_bar_is_always_final():
    from src.live import bar_is_final
    # 2026-09-08 09:00 UTC = 05:00 ET, market not open; but the 09-04 bar
    # closed days ago.
    assert bar_is_final("2026-09-04", _utc(2026, 9, 8, 9))


def test_todays_bar_is_not_final_mid_session():
    """13:00 UTC = 09:00 ET (EDT) — the session has not even opened."""
    from src.live import bar_is_final
    assert not bar_is_final("2026-09-08", _utc(2026, 9, 8, 13))


def test_todays_bar_is_not_final_just_before_the_close():
    """19:30 UTC = 15:30 ET (EDT) — half an hour of trading left."""
    from src.live import bar_is_final
    assert not bar_is_final("2026-09-08", _utc(2026, 9, 8, 19, ))


def test_todays_bar_is_final_after_the_close():
    """21:30 UTC = 17:30 ET (EDT) — the scheduled slot."""
    from src.live import bar_is_final
    assert bar_is_final("2026-09-08", _utc(2026, 9, 8, 21))


def test_guard_holds_in_winter_too():
    """In January, 21:30 UTC = 16:30 ET (EST) — still after the close."""
    from src.live import bar_is_final
    assert bar_is_final("2026-01-14", _utc(2026, 1, 14, 21))
    # ...but 20:00 UTC = 15:00 ET (EST) is mid-session.
    assert not bar_is_final("2026-01-14", _utc(2026, 1, 14, 20))


def test_a_future_dated_bar_is_rejected():
    from src.live import bar_is_final
    assert not bar_is_final("2026-09-20", _utc(2026, 9, 8, 21))


def test_run_daily_declines_a_partial_bar(live_env, monkeypatch):
    """A manual mid-session run must not freeze an incomplete bar into the log."""
    import src.live as live_mod

    cfg, model_file, tmp = live_env
    log_path = tmp / "log.csv"

    monkeypatch.setattr(live_mod, "bar_is_final", lambda *a, **k: False)
    result = run_daily(cfg, model_file, log_path, cache_dir=tmp / "raw")

    assert result["status"] == "bar_not_final"
    assert result["prediction"] is None
    assert load_log(log_path).empty, "no row may be written for a partial bar"


# =============================================== ROLLING-CACHE PREFERENCE


def test_load_range_prefers_the_rolling_cache(tmp_path, monkeypatch):
    """A deployed app must not hit Yahoo on every cold start."""
    monkeypatch.setattr(dl, "_download", lambda t, s, e: raw_frame(600))
    dl.update_cache("TEST", cache_dir=tmp_path)

    calls = []

    def should_not_run(ticker, start, end):
        calls.append(ticker)
        return raw_frame(600)

    monkeypatch.setattr(dl, "_download", should_not_run)
    frame = dl.load_range("TEST", "2015-01-02", "2016-06-01", cache_dir=tmp_path)

    assert not calls, "the rolling cache should have served this without a download"
    assert len(frame) > 0
    assert list(frame.columns) == list(dl.COLUMNS)
    assert frame.index.min() >= pd.Timestamp("2015-01-02")
    assert frame.index.max() <= pd.Timestamp("2016-06-01")


def test_load_range_falls_back_when_the_cache_starts_too_late(tmp_path, monkeypatch):
    """A cache of only recent bars must not pose as a decade of history."""
    monkeypatch.setattr(
        dl, "_download", lambda t, s, e: raw_frame(60, start="2024-01-02")
    )
    dl.update_cache("TEST", cache_dir=tmp_path)

    calls = []

    def fallback(ticker, start, end):
        calls.append(start)
        return raw_frame(600, start="2015-01-02")

    monkeypatch.setattr(dl, "_download", fallback)
    dl.load_range("TEST", "2015-01-02", "2017-01-01", cache_dir=tmp_path)

    assert calls, "should have fallen back to a ranged download"


def test_load_range_falls_back_with_no_rolling_cache(tmp_path, monkeypatch):
    calls = []

    def fetch(ticker, start, end):
        calls.append(ticker)
        return raw_frame(300)

    monkeypatch.setattr(dl, "_download", fetch)
    dl.load_range("TEST", "2015-01-02", "2016-01-01", cache_dir=tmp_path)
    assert calls
