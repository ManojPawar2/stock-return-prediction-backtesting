"""Tests for the configuration layer."""

import pytest

from src.config import Config, FeatureConfig, load_config


def test_defaults_are_valid():
    Config().validate()


def test_loads_project_yaml():
    cfg = load_config()
    assert cfg.ticker
    assert cfg.model in ("linear", "ridge", "random_forest", "xgboost")
    # config.yaml must actually supply hyperparameters, not silently fall back.
    assert cfg.params_for("xgboost")["max_depth"] == 3


def test_overrides_apply_and_none_is_ignored():
    cfg = load_config(ticker="MSFT", seed=None)
    assert cfg.ticker == "MSFT"
    assert cfg.seed == 42


def test_replace_rejects_unknown_field():
    with pytest.raises(KeyError):
        Config().replace(not_a_field=1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_end": "2010-01-01"},          # before start
        {"valid_end": "2016-01-01"},          # before train_end
        {"model": "nonexistent"},
        {"target_kind": "clustering"},
        {"target_horizon": 0},
        {"signal_threshold": -0.1},
        {"transaction_cost_bps": -1.0},
    ],
)
def test_invalid_configs_are_rejected(overrides):
    cfg = Config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_total_cost_adds_slippage():
    cfg = Config(transaction_cost_bps=5.0, slippage_bps=2.0)
    assert cfg.total_cost_bps == 7.0


def test_fingerprint_is_deterministic_and_sensitive():
    a, b = Config(), Config()
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != Config(ticker="MSFT").fingerprint()


def test_max_window_covers_longest_lookback():
    fc = FeatureConfig()
    # MACD needs slow EMA plus the signal EMA on top of it.
    assert fc.max_window >= fc.macd_slow + fc.macd_signal
    assert fc.max_window >= max(fc.volatility_windows)
