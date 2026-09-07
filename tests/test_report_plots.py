"""Tests for the research report and the plotting layer.

The report's job is to state what the numbers show — including when they show
the strategy lost. The verdict tests pin that behaviour, because a generator
that only ever congratulates the author is not a research tool.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.config import Config
from src.plots import (
    correlation_heatmap,
    cumulative_returns_chart,
    drawdown_chart,
    equity_curve_chart,
    feature_importance_chart,
    future_correlation_chart,
    model_comparison_chart,
    prediction_scatter,
    price_chart,
    regime_chart,
    returns_distribution,
    rolling_volatility_chart,
    sweep_chart,
    trade_markers_chart,
    volume_chart,
    walkforward_chart,
)
from src.report import (
    ReportInputs,
    build_report,
    log_experiment,
    save_report,
    verdict,
)


def comparison(total_diff: float, sharpe_diff: float, dd_diff: float) -> pd.DataFrame:
    """A strategy-vs-benchmark table with controlled differences."""
    bench = {"total_return": 0.50, "sharpe_ratio": 1.00, "max_drawdown": -0.30}
    strat = {
        "total_return": bench["total_return"] + total_diff,
        "sharpe_ratio": bench["sharpe_ratio"] + sharpe_diff,
        "max_drawdown": bench["max_drawdown"] + dd_diff,
    }
    frame = pd.DataFrame({"strategy": strat, "buy_and_hold": bench})
    frame["difference"] = frame["strategy"] - frame["buy_and_hold"]
    return frame


# ================================================================= VERDICT


def test_verdict_reports_a_genuine_win():
    headline, text = verdict(comparison(0.20, 0.30, 0.05))
    assert "outperformed" in headline.lower()
    assert "not evidence of future profitability" in text


def test_verdict_reports_a_loss_honestly():
    """The behaviour that matters most."""
    headline, text = verdict(comparison(-0.30, -0.20, -0.05))
    assert "did not beat" in headline.lower()
    assert "legitimate research finding" in text


def test_verdict_distinguishes_return_from_risk_adjusted_return():
    """More return bought with more risk is not outperformance."""
    headline, text = verdict(comparison(0.20, -0.30, -0.05))
    assert "more risk" in headline.lower()
    assert "extra risk rather than skill" in text


def test_verdict_credits_a_drawdown_improvement():
    headline, _ = verdict(comparison(-0.20, -0.10, 0.12))
    assert "reduced drawdown" in headline.lower()


def test_verdict_handles_a_broken_comparison():
    headline, _ = verdict(pd.DataFrame())
    assert headline == "Inconclusive"


# ================================================================== REPORT


@pytest.fixture
def inputs() -> ReportInputs:
    rng = np.random.default_rng(41)
    idx = pd.bdate_range("2023-01-02", periods=250, name="date")
    return ReportInputs(
        cfg=Config(),
        data_summary={"rows": 2515, "years": 9.98, "total_return_pct": 935.8},
        split_summary=pd.DataFrame({
            "split": ["train", "valid", "test"],
            "rows": [1703, 251, 500],
            "start": ["2015-03-31", "2022-01-03", "2023-01-03"],
            "end": ["2021-12-31", "2022-12-30", "2024-12-27"],
        }),
        model_name="xgboost",
        model_metrics={
            "train": {"rmse": 0.0148, "r2": 0.34, "directional_accuracy": 0.65},
            "test": {"rmse": 0.0136, "r2": -0.02, "directional_accuracy": 0.54},
        },
        comparison=comparison(-0.24, -0.07, -0.03),
        trades=pd.DataFrame({
            "net_return": rng.normal(0.005, 0.03, 40),
            "holding_days": rng.integers(1, 15, 40),
        }),
        feature_importance=pd.Series(
            rng.random(10), index=[f"feat_{i}" for i in range(10)]
        ).sort_values(ascending=False),
        future_correlations=pd.Series(
            rng.normal(0, 0.03, 10), index=[f"feat_{i}" for i in range(10)]
        ),
        generated_at="2026-01-01 12:00",
    )


def test_report_contains_every_section(inputs):
    text = build_report(inputs)
    for heading in ("# Research Report", "## Verdict", "## 1. Study Design",
                    "## 2. Model Performance", "## 3. Backtest vs Buy-and-Hold",
                    "## 7. Limitations"):
        assert heading in text


def test_report_states_the_loss_plainly(inputs):
    text = build_report(inputs)
    assert "did not beat buy-and-hold" in text.lower()
    assert "legitimate research finding" in text


def test_report_records_the_config_fingerprint(inputs):
    assert inputs.cfg.fingerprint() in build_report(inputs)


def test_report_notes_the_absence_of_sanity_warnings(inputs):
    """Realistic metrics should be explained, not silently accepted."""
    text = build_report(inputs)
    assert "No sanity warnings" in text
    assert "lookahead bias rather than skill" in text


def test_report_raises_sanity_warnings_when_results_are_implausible(inputs):
    inputs.model_metrics["test"] = {
        "rmse": 0.001, "r2": 0.92, "directional_accuracy": 0.95,
    }
    text = build_report(inputs)
    assert "Sanity warnings" in text
    assert "lookahead" in text


def test_report_includes_limitations(inputs):
    text = build_report(inputs)
    assert "Survivorship bias" in text
    assert "not a prediction about the future" in text


def test_report_includes_optional_sections_when_supplied(inputs):
    inputs.walkforward_table = pd.DataFrame({
        "fold": [1, 2], "n_train": [800, 900], "n_test": [100, 100],
        "rmse": [0.02, 0.019],
    })
    inputs.regime_breakdown = {"volatility": pd.DataFrame({
        "days": [100, 120], "strategy_total_return": [0.1, -0.05],
        "benchmark_total_return": [0.05, 0.02], "excess_return": [0.05, -0.07],
        "outperformed": [True, False],
    }, index=pd.Index(["low_vol", "high_vol"], name="regime"))}
    inputs.stress_sweeps = {"cost": pd.DataFrame({
        "cost_bps": [0, 5, 20], "total_return": [0.3, 0.2, 0.05],
        "sharpe_ratio": [1.0, 0.8, 0.3],
        "beats_buy_and_hold": [False, False, False],
    })}
    text = build_report(inputs)
    assert "## 4. Walk-Forward Validation" in text
    assert "## 5. Regime Analysis" in text
    assert "## 6. Stress Tests" in text
    assert "Robustness assessment" in text


def test_report_omits_optional_sections_when_absent(inputs):
    text = build_report(inputs)
    assert "## 4. Walk-Forward Validation" not in text
    assert "## 6. Stress Tests" not in text


def test_report_survives_nan_values(inputs):
    inputs.model_metrics["test"]["r2"] = float("nan")
    assert "n/a" in build_report(inputs)


def test_save_report_writes_a_file(inputs, tmp_path):
    path = save_report(build_report(inputs), inputs.cfg, "xgboost",
                       directory=tmp_path, timestamp="20260101_120000")
    assert path.exists()
    assert path.name == "AAPL_xgboost_20260101_120000.md"
    assert "Research Report" in path.read_text(encoding="utf-8")


# --------------------------------------------------------- experiment log


def test_experiment_log_creates_and_appends(tmp_path):
    cfg = Config()
    path = tmp_path / "log.csv"
    log_experiment(cfg, "ridge", {"rmse": 0.02}, {"total_return": 0.1},
                   path=path, timestamp="2026-01-01T00:00:00")
    log_experiment(cfg, "xgboost", {"rmse": 0.019}, {"total_return": 0.2},
                   path=path, timestamp="2026-01-02T00:00:00")

    frame = pd.read_csv(path)
    assert len(frame) == 2
    assert list(frame["model"]) == ["ridge", "xgboost"]
    assert "model_rmse" in frame.columns
    assert "strategy_total_return" in frame.columns
    # Same config, so the fingerprints must match and be comparable.
    assert frame["fingerprint"].nunique() == 1


def test_experiment_log_distinguishes_configs(tmp_path):
    path = tmp_path / "log.csv"
    log_experiment(Config(ticker="AAPL"), "ridge", {}, {}, path=path)
    log_experiment(Config(ticker="MSFT"), "ridge", {}, {}, path=path)
    assert pd.read_csv(path)["fingerprint"].nunique() == 2


# =================================================================== PLOTS


@pytest.fixture
def market() -> pd.DataFrame:
    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n, name="date")
    rng = np.random.default_rng(42)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.014, n))), index=idx)
    span = close * 0.02
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + span, "low": close - span,
        "close": close, "adj_close": close,
        "volume": pd.Series(rng.integers(1e6, 9e6, n).astype("float64"), index=idx),
    })


@pytest.fixture
def series(market):
    returns = market["close"].pct_change().fillna(0.0)
    return returns, (1 + returns).cumprod()


def test_price_and_volume_charts(market):
    for fig in (price_chart(market), volume_chart(market)):
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


def test_candlestick_chart_has_price_and_volume(market):
    from src.plots import candlestick_chart

    fig = candlestick_chart(market)
    assert any(trace.type == "candlestick" for trace in fig.data)
    assert any(trace.type == "bar" for trace in fig.data)


def test_returns_distribution_overlays_a_normal(series):
    returns, _ = series
    fig = returns_distribution(returns)
    names = [trace.name for trace in fig.data]
    assert "Observed" in names and "Normal fit" in names


def test_returns_distribution_skips_the_fit_for_a_constant_series():
    fig = returns_distribution(pd.Series([0.0] * 50))
    assert "Normal fit" not in [trace.name for trace in fig.data]


def test_equity_chart_draws_both_curves(series):
    _, equity = series
    fig = equity_curve_chart(equity, equity * 0.9)
    names = [trace.name for trace in fig.data]
    assert "Strategy" in names and "Buy and hold" in names


def test_equity_chart_works_without_a_benchmark(series):
    _, equity = series
    assert len(equity_curve_chart(equity).data) == 1


def test_drawdown_chart(series):
    _, equity = series
    dd = equity / equity.cummax() - 1
    assert isinstance(drawdown_chart(dd, dd * 1.2), go.Figure)


def test_trade_markers_chart_plots_entries_and_exits(market):
    trades = pd.DataFrame({
        "entry_date": [market.index[10], market.index[50]],
        "exit_date": [market.index[20], market.index[70]],
    })
    fig = trade_markers_chart(market["close"], trades)
    names = [trace.name for trace in fig.data]
    assert "Entry" in names and "Exit" in names


def test_trade_markers_chart_handles_no_trades(market):
    fig = trade_markers_chart(market["close"], pd.DataFrame())
    assert len(fig.data) == 1


def test_feature_importance_chart_limits_to_top_n():
    importance = pd.Series(
        np.linspace(0.1, 0.01, 50), index=[f"f{i}" for i in range(50)]
    )
    fig = feature_importance_chart(importance, top_n=10)
    assert len(fig.data[0].y) == 10


def test_prediction_scatter_includes_a_reference_line(series):
    returns, _ = series
    fig = prediction_scatter(returns, returns * 0.1)
    assert "Perfect prediction" in [trace.name for trace in fig.data]


def test_correlation_heatmap(market):
    frame = market[["open", "high", "low", "close", "volume"]]
    fig = correlation_heatmap(frame)
    assert fig.data[0].type == "heatmap"


def test_future_correlation_chart_sorts_by_magnitude():
    corr = pd.Series({"a": 0.01, "b": -0.25, "c": 0.05})
    fig = future_correlation_chart(corr, top_n=3)
    # Bars are drawn smallest-first, so the largest magnitude is last.
    assert fig.data[0].y[-1] == "b"


def test_model_comparison_chart():
    table = pd.DataFrame(
        {"total_return": [0.1, -0.05]}, index=["linear", "xgboost"]
    )
    assert isinstance(model_comparison_chart(table), go.Figure)


def test_model_comparison_chart_rejects_a_missing_metric():
    table = pd.DataFrame({"total_return": [0.1]}, index=["linear"])
    with pytest.raises(KeyError):
        model_comparison_chart(table, metric="sharpe_ratio")


def test_sweep_chart_draws_the_benchmark_line():
    sweep = pd.DataFrame({"cost_bps": [0, 5, 20], "total_return": [0.3, 0.2, 0.0]})
    fig = sweep_chart(sweep, "cost_bps", benchmark=0.25)
    assert len(fig.layout.shapes) >= 1


def test_regime_chart_groups_strategy_and_benchmark():
    breakdown = pd.DataFrame({
        "strategy_total_return": [0.1, -0.05],
        "benchmark_total_return": [0.05, 0.02],
    }, index=pd.Index(["low_vol", "high_vol"], name="regime"))
    fig = regime_chart(breakdown)
    assert fig.layout.barmode == "group"
    assert len(fig.data) == 2


def test_walkforward_chart_marks_the_coin_flip_line():
    table = pd.DataFrame({"fold": [1, 2, 3], "directional_accuracy": [0.52, 0.48, 0.55]})
    fig = walkforward_chart(table)
    # A mean line and the 0.5 reference line.
    assert len(fig.layout.shapes) >= 2


def test_charts_have_titles(market, series):
    returns, equity = series
    for fig in (price_chart(market), volume_chart(market),
                returns_distribution(returns), cumulative_returns_chart(returns),
                rolling_volatility_chart(returns), equity_curve_chart(equity)):
        assert fig.layout.title.text
