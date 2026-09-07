"""The end-to-end pipeline, assembled once.

Both the CLI and the Streamlit dashboard call these functions, so the two
front-ends cannot drift apart and produce different numbers from the same
configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .backtester import BacktestResult, backtester_from_config
from .config import Config, ensure_dirs, set_seed
from .data_loader import describe, load_prices
from .dataset import TARGET, Split, build_dataset, realised_returns
from .evaluation import evaluate
from .features import future_return_correlation
from .models import ModelPipeline, available_models
from .preprocessing import CleaningLog, ValidationReport, clean, validate
from .signals import generate_signals

logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    """Raw prices plus the validation and cleaning record."""

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    report: ValidationReport
    log: CleaningLog
    summary: dict


def load(cfg: Config) -> LoadedData:
    """Download (or read from cache), validate and clean."""
    ensure_dirs()
    raw = load_prices(cfg.ticker, cfg.start, cfg.end, cfg.use_cache)
    report = validate(raw)
    cleaned, log = clean(raw)
    return LoadedData(raw, cleaned, report, log, describe(cleaned))


@dataclass
class PreparedData:
    """Feature matrix, target and chronological splits."""

    data: pd.DataFrame
    split: Split
    features: list[str]


def prepare(cfg: Config, cleaned: pd.DataFrame) -> PreparedData:
    data, split, features = build_dataset(cleaned, cfg)
    return PreparedData(data, split, features)


@dataclass
class TrainedModel:
    """A fitted pipeline with its per-split predictions and metrics."""

    pipeline: ModelPipeline
    predictions: dict[str, pd.Series]
    metrics: dict[str, dict[str, float]]

    @property
    def name(self) -> str:
        return self.pipeline.name

    def importance(self) -> pd.Series:
        return self.pipeline.feature_importance()


def train(cfg: Config, prepared: PreparedData, model_name: str | None = None) -> TrainedModel:
    """Fit on train, then score every split.

    The scaler inside :class:`ModelPipeline` is fit on the training rows only,
    so validation and test statistics never reach it.
    """
    set_seed(cfg.seed)
    name = model_name or cfg.model
    X_train, y_train = prepared.split.xy("train", prepared.features)
    pipeline = ModelPipeline(
        name, cfg.params_for(name), cfg.seed, cfg.target_kind
    ).fit(X_train, y_train)

    predictions: dict[str, pd.Series] = {}
    metrics: dict[str, dict[str, float]] = {}
    for part in ("train", "valid", "test"):
        frame = getattr(prepared.split, part)
        if frame.empty:
            continue
        preds = pipeline.predict(frame[prepared.features])
        predictions[part] = preds
        metrics[part] = evaluate(frame[TARGET], preds, cfg.target_kind)

    return TrainedModel(pipeline, predictions, metrics)


def backtest(
    cfg: Config,
    prepared: PreparedData,
    model: TrainedModel,
    part: str = "test",
) -> BacktestResult:
    """Turn one split's predictions into an equity curve."""
    frame = getattr(prepared.split, part)
    returns = realised_returns(frame)
    signals = generate_signals(
        model.predictions[part], cfg.signal_threshold, cfg.allow_short
    )
    return backtester_from_config(cfg).run(returns, signals, prices=frame["close"])


def compare_models(
    cfg: Config,
    prepared: PreparedData,
    model_names: list[str] | None = None,
    part: str = "test",
) -> tuple[pd.DataFrame, dict[str, TrainedModel], dict[str, BacktestResult]]:
    """Train every model on identical data and score them side by side.

    Identical splits, identical features, identical seed — so any difference
    is the model, not the setup.
    """
    from .metrics import performance_summary

    names = model_names or available_models()
    rows, models, backtests = [], {}, {}

    for name in names:
        model = train(cfg, prepared, name)
        result = backtest(cfg, prepared, model, part)
        models[name], backtests[name] = model, result

        stats = performance_summary(
            result.strategy_returns, cfg.risk_free_rate, cfg.trading_days,
            result.position,
        )
        metrics = model.metrics.get(part, {})
        rows.append({
            "model": name,
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "information_coefficient": metrics.get("information_coefficient"),
            "total_return": stats["total_return"],
            "cagr": stats["cagr"],
            "sharpe_ratio": stats["sharpe_ratio"],
            "max_drawdown": stats["max_drawdown"],
            "n_trades": result.n_trades,
            "exposure": stats.get("exposure"),
        })

    table = pd.DataFrame(rows).set_index("model")

    # The benchmark is identical for every model, so add it once as a row.
    if backtests:
        any_result = next(iter(backtests.values()))
        bench = performance_summary(
            any_result.benchmark_returns, cfg.risk_free_rate, cfg.trading_days
        )
        # np.nan rather than None: assigning None into float columns makes
        # pandas re-infer dtypes and emits a FutureWarning. The model-accuracy
        # columns are genuinely undefined for a benchmark that makes no
        # prediction, so NaN is also the honest value.
        import numpy as np

        table.loc["buy_and_hold"] = {
            "rmse": np.nan, "mae": np.nan, "r2": np.nan,
            "directional_accuracy": np.nan, "information_coefficient": np.nan,
            "total_return": bench["total_return"], "cagr": bench["cagr"],
            "sharpe_ratio": bench["sharpe_ratio"],
            "max_drawdown": bench["max_drawdown"],
            "n_trades": 1, "exposure": 1.0,
        }
    return table, models, backtests


def run_full_study(
    cfg: Config,
    model_name: str | None = None,
    include_walkforward: bool = True,
    include_regimes: bool = True,
    include_stress: bool = True,
) -> dict:
    """Everything: load, prepare, train, backtest, validate, stress, report."""
    from .regime import regime_report
    from .report import assemble_report
    from .stress import sweep_costs, sweep_periods, sweep_thresholds
    from .walkforward import walk_forward

    name = model_name or cfg.model
    loaded = load(cfg)
    prepared = prepare(cfg, loaded.cleaned)
    model = train(cfg, prepared, name)
    result = backtest(cfg, prepared, model)

    test = prepared.split.test
    returns = realised_returns(test)
    preds = model.predictions["test"]

    extras: dict = {
        "feature_importance": model.importance(),
        "future_correlations": future_return_correlation(prepared.data),
    }

    walkforward = None
    if include_walkforward:
        try:
            walkforward = walk_forward(prepared.data, cfg, prepared.features, name)
            extras["walkforward_table"] = walkforward.fold_table()
            extras["walkforward_stability"] = walkforward.stability()
        except ValueError as exc:
            logger.warning("Skipping walk-forward: %s", exc)

    if include_regimes:
        extras["regime_breakdown"] = regime_report(
            test, result.strategy_returns, result.benchmark_returns
        )

    sweeps: dict[str, pd.DataFrame] = {}
    if include_stress:
        sweeps = {
            "cost": sweep_costs(returns, preds, cfg),
            "threshold": sweep_thresholds(returns, preds, cfg),
            "period": sweep_periods(returns, preds, cfg, 4),
        }
        extras["stress_sweeps"] = sweeps

    report_text = assemble_report(
        cfg, loaded.summary, prepared.split.summary(), name,
        model.metrics, result, **extras,
    )

    return {
        "loaded": loaded,
        "prepared": prepared,
        "model": model,
        "backtest": result,
        "walkforward": walkforward,
        "stress": sweeps,
        "report": report_text,
    }
