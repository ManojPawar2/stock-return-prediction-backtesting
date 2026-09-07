"""Typed configuration for the whole project.

Every module takes a :class:`Config` rather than reading globals, so a run is
fully described by one object.  That object is hashable into a short digest
(:meth:`Config.fingerprint`) which is what makes experiments reproducible and
comparable in ``outputs/experiment_log.csv``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml

# Project root = the directory containing this file's parent (src/ -> repo root).
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"
EXPERIMENT_LOG = OUTPUTS / "experiment_log.csv"

DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for d in (DATA_RAW, DATA_PROCESSED, FIGURES, MODELS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Nested config sections
# --------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    """Window lengths for every engineered feature.

    All windows are backward-looking; see README section 4 (Leakage Contract).
    """

    return_windows: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    return_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    sma_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    volatility_windows: list[int] = field(default_factory=lambda: [5, 20, 60])
    volume_window: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    include_calendar: bool = True

    @property
    def max_window(self) -> int:
        """Longest lookback used, i.e. how many warm-up rows will be NaN."""
        return max(
            [*self.return_windows, *self.sma_windows, *self.volatility_windows,
             self.volume_window, self.rsi_period, self.macd_slow + self.macd_signal,
             self.bollinger_window, self.atr_period, max(self.return_lags) + 1]
        )


@dataclass
class WalkForwardConfig:
    n_folds: int = 5
    mode: str = "expanding"  # expanding | rolling
    min_train_days: int = 500


@dataclass
class StressConfig:
    cost_grid_bps: list[float] = field(
        default_factory=lambda: [0, 1, 5, 10, 20, 50]
    )
    threshold_grid: list[float] = field(
        default_factory=lambda: [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]
    )
    seed_grid: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


# --------------------------------------------------------------------------
# Top-level config
# --------------------------------------------------------------------------

MODEL_NAMES = ("linear", "ridge", "random_forest", "xgboost")


@dataclass
class Config:
    # data
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    use_cache: bool = True
    universe: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "JPM", "XOM", "KO"]
    )

    # chronological split
    train_end: str = "2021-12-31"
    valid_end: str = "2022-12-31"

    # target
    target_horizon: int = 1
    target_kind: str = "regression"

    # features
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # models
    model: str = "xgboost"
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    # signals
    signal_threshold: float = 0.0
    allow_short: bool = False

    # backtest
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_capital: float = 10_000.0
    risk_free_rate: float = 0.0
    trading_days: int = 252

    # walk-forward / stress
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    stress: StressConfig = field(default_factory=StressConfig)

    # reproducibility
    seed: int = 42

    # ---------------------------------------------------------------- API

    def params_for(self, model_name: str | None = None) -> dict[str, Any]:
        """Hyperparameters for one model, defaulting to an empty dict."""
        name = model_name or self.model
        return dict(self.model_params.get(name, {}))

    @property
    def total_cost_bps(self) -> float:
        """One-way friction charged on turnover: commission plus slippage."""
        return float(self.transaction_cost_bps) + float(self.slippage_bps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self, length: int = 10) -> str:
        """Short deterministic digest of the whole configuration."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]

    def replace(self, **overrides: Any) -> "Config":
        """Return a copy with top-level fields overridden.

        ``None`` values are ignored, which lets CLI parsers pass every flag
        through without checking which ones the user actually supplied.
        """
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in data:
                raise KeyError(f"Unknown config field: {key!r}")
            data[key] = value
        return Config(**data)

    def validate(self) -> None:
        """Fail fast on a configuration that cannot produce a valid study."""
        import pandas as pd

        start, train_end = pd.Timestamp(self.start), pd.Timestamp(self.train_end)
        valid_end, end = pd.Timestamp(self.valid_end), pd.Timestamp(self.end)
        if not start < train_end < valid_end < end:
            raise ValueError(
                "Dates must be strictly increasing: "
                f"start({self.start}) < train_end({self.train_end}) < "
                f"valid_end({self.valid_end}) < end({self.end})"
            )
        if self.model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}, got {self.model!r}")
        if self.target_kind not in ("regression", "classification"):
            raise ValueError("target_kind must be 'regression' or 'classification'")
        if self.target_horizon < 1:
            raise ValueError("target_horizon must be >= 1")
        if self.signal_threshold < 0:
            raise ValueError("signal_threshold must be >= 0 (it is a magnitude)")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be non-negative")
        if self.walk_forward.mode not in ("expanding", "rolling"):
            raise ValueError("walk_forward.mode must be 'expanding' or 'rolling'")
        if self.walk_forward.n_folds < 1:
            raise ValueError("walk_forward.n_folds must be >= 1")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load ``config.yaml`` (or a given path) and apply keyword overrides.

    Missing file falls back to dataclass defaults, so the project still runs
    from a bare checkout.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    raw["features"] = FeatureConfig(**(raw.get("features") or {}))
    raw["walk_forward"] = WalkForwardConfig(**(raw.get("walk_forward") or {}))
    raw["stress"] = StressConfig(**(raw.get("stress") or {}))

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path.name}: {sorted(unknown)}")

    cfg = Config(**raw)
    if overrides:
        cfg = cfg.replace(**overrides)
    cfg.validate()
    return cfg


def set_seed(seed: int) -> None:
    """Seed every source of randomness the project touches."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
