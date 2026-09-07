"""Tests for the assembled pipeline and the command-line interface.

Network is never touched: ``data_loader._download`` is monkeypatched with a
synthetic but realistic price series, so these tests are fast and offline.
"""

import numpy as np
import pandas as pd
import pytest

import cli
from src import data_loader as dl
from src.config import Config
from src.pipeline import (
    backtest,
    compare_models,
    load,
    prepare,
    run_full_study,
    train,
)


def synthetic(n: int = 1500) -> pd.DataFrame:
    """A yfinance-shaped frame with a realistic price path."""
    idx = pd.bdate_range("2016-01-04", periods=n, name="Date")
    rng = np.random.default_rng(51)
    close = pd.Series(60 * np.exp(np.cumsum(rng.normal(0.0005, 0.014, n))), index=idx)
    day_range = close * pd.Series(rng.uniform(0.006, 0.03, n), index=idx)
    pos = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    low = close - day_range * pos
    return pd.DataFrame({
        "Open": low + day_range * 0.45,
        "High": low + day_range,
        "Low": low,
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(1e6, 9e6, n).astype("float64"),
    }, index=idx)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Serve synthetic data and keep every artefact inside tmp_path."""
    monkeypatch.setattr(dl, "_download", lambda ticker, start, end: synthetic())
    monkeypatch.setattr(dl, "DATA_RAW", tmp_path / "raw")
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def cfg() -> Config:
    return Config(
        ticker="TEST", start="2016-01-01", end="2021-12-31",
        train_end="2019-12-31", valid_end="2020-12-31",
        model="ridge",
    )


# ================================================================ PIPELINE


def test_load_returns_data_and_reports(cfg):
    loaded = load(cfg)
    assert len(loaded.cleaned) > 1000
    assert loaded.report.is_valid
    assert loaded.summary["rows"] == len(loaded.cleaned)


def test_prepare_builds_features_and_splits(cfg):
    prepared = prepare(cfg, load(cfg).cleaned)
    assert len(prepared.features) >= 35
    assert prepared.split.train.index.max() < prepared.split.test.index.min()


def test_train_scores_every_split(cfg):
    prepared = prepare(cfg, load(cfg).cleaned)
    model = train(cfg, prepared)
    assert set(model.metrics) == {"train", "valid", "test"}
    assert set(model.predictions) == {"train", "valid", "test"}
    for part, preds in model.predictions.items():
        assert preds.index.equals(getattr(prepared.split, part).index)


def test_train_is_reproducible(cfg):
    prepared = prepare(cfg, load(cfg).cleaned)
    a = train(cfg, prepared, "random_forest").predictions["test"]
    b = train(cfg, prepared, "random_forest").predictions["test"]
    pd.testing.assert_series_equal(a, b)


def test_backtest_covers_the_test_split(cfg):
    prepared = prepare(cfg, load(cfg).cleaned)
    result = backtest(cfg, prepared, train(cfg, prepared))
    assert result.equity.index.equals(prepared.split.test.index)
    assert np.isfinite(result.equity).all()


def test_backtest_applies_the_execution_lag(cfg):
    prepared = prepare(cfg, load(cfg).cleaned)
    result = backtest(cfg, prepared, train(cfg, prepared))
    assert result.position.iloc[0] == 0.0


def test_compare_models_uses_identical_data(cfg):
    prepared = prepare(cfg, load(cfg).cleaned)
    table, models, backtests = compare_models(cfg, prepared, ["linear", "ridge"])

    assert set(models) == {"linear", "ridge"}
    assert "buy_and_hold" in table.index
    # Every model must be scored on exactly the same days.
    lengths = {len(bt.equity) for bt in backtests.values()}
    assert len(lengths) == 1


def test_comparison_benchmark_row_is_the_same_for_all(cfg):
    prepared = prepare(cfg, load(cfg).cleaned)
    table, _, backtests = compare_models(cfg, prepared, ["linear", "ridge"])
    benchmarks = [bt.benchmark_equity.iloc[-1] for bt in backtests.values()]
    assert len(set(np.round(benchmarks, 10))) == 1
    assert table.loc["buy_and_hold", "total_return"] == pytest.approx(
        benchmarks[0] - 1, rel=1e-6
    )


def test_run_full_study_produces_everything(cfg):
    study = run_full_study(cfg, "ridge")
    for key in ("loaded", "prepared", "model", "backtest", "report"):
        assert key in study
    assert "# Research Report" in study["report"]
    assert "## Verdict" in study["report"]


def test_run_full_study_can_skip_optional_stages(cfg):
    study = run_full_study(
        cfg, "ridge", include_walkforward=False,
        include_regimes=False, include_stress=False,
    )
    assert study["walkforward"] is None
    assert study["stress"] == {}
    assert "## 6. Stress Tests" not in study["report"]


# ===================================================================== CLI


def run_cli(*args) -> int:
    return cli.main(list(args))


BASE = ["--ticker", "TEST", "--start", "2016-01-01", "--end", "2021-12-31",
        "--train-end", "2019-12-31", "--valid-end", "2020-12-31"]


def test_cli_fetch(capsys):
    assert run_cli("fetch", *BASE) == 0
    out = capsys.readouterr().out
    assert "DATA" in out and "Validation:" in out


def test_cli_features(capsys):
    assert run_cli("features", *BASE) == 0
    out = capsys.readouterr().out
    assert "FEATURES" in out
    assert "would indicate a leak" in out


def test_cli_train(capsys):
    assert run_cli("train", *BASE, "--model", "ridge") == 0
    out = capsys.readouterr().out
    assert "MODEL" in out and "Top features:" in out


def test_cli_train_reports_no_sanity_warnings_on_realistic_data(capsys):
    run_cli("train", *BASE, "--model", "ridge")
    assert "No sanity warnings" in capsys.readouterr().out


def test_cli_backtest(capsys):
    assert run_cli("backtest", *BASE, "--model", "ridge") == 0
    out = capsys.readouterr().out
    assert "BACKTEST" in out and "buy_and_hold" in out
    assert "Total cost paid" in out


def test_cli_compare(capsys):
    assert run_cli("compare", *BASE) == 0
    out = capsys.readouterr().out
    assert "MODEL COMPARISON" in out
    assert "buy_and_hold" in out


def test_cli_walkforward(capsys):
    assert run_cli("walkforward", *BASE, "--model", "ridge", "--folds", "3") == 0
    out = capsys.readouterr().out
    assert "WALK-FORWARD" in out
    assert "Stability across folds" in out


def test_cli_stress(capsys):
    assert run_cli("stress", *BASE, "--model", "ridge", "--sweep", "cost") == 0
    out = capsys.readouterr().out
    assert "STRESS TESTS" in out and "VERDICT:" in out


def test_cli_report_writes_a_file(capsys, tmp_path, monkeypatch):
    from src import config as config_module
    from src import report as report_module

    monkeypatch.setattr(report_module, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config_module, "EXPERIMENT_LOG", tmp_path / "log.csv")

    assert run_cli("report", *BASE, "--model", "ridge") == 0
    out = capsys.readouterr().out
    assert "VERDICT:" in out
    assert list((tmp_path / "reports").glob("*.md"))


def test_cli_overrides_reach_the_config(capsys):
    """A CLI flag must actually change the run."""
    run_cli("backtest", *BASE, "--model", "ridge", "--cost-bps", "0")
    cheap = capsys.readouterr().out
    run_cli("backtest", *BASE, "--model", "ridge", "--cost-bps", "80")
    dear = capsys.readouterr().out
    assert cheap != dear


def test_cli_rejects_an_invalid_configuration(capsys):
    code = run_cli("train", *BASE, "--train-end", "2035-01-01")
    assert code == 2
    assert "Configuration error" in capsys.readouterr().err


def test_cli_rejects_an_unknown_command():
    with pytest.raises(SystemExit):
        run_cli("nonsense")


def test_cli_rejects_an_unknown_model():
    with pytest.raises(SystemExit):
        run_cli("train", "--model", "transformer")


def test_unset_flags_do_not_override_the_config_file():
    """Config.replace ignores None, so unset flags leave config.yaml alone."""
    parser = cli.build_parser()
    args = parser.parse_args(["fetch"])
    overrides = {f: getattr(args, f, None) for f in cli.CONFIG_FIELDS}
    assert all(value is None for value in overrides.values())


def test_every_command_is_registered():
    parser = cli.build_parser()
    choices = parser._actions[1].choices
    assert set(choices) == set(cli.COMMANDS)
