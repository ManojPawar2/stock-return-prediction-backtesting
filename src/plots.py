"""Shared Plotly figures.

Kept out of the Streamlit pages so charts are testable and reusable: a page
should assemble figures, not define them.

Every chart that compares a strategy to its benchmark draws both on the same
axes.  A lone equity curve rising to the right always looks like success; the
comparison is what carries the information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# One palette used everywhere, so a colour means the same thing on every page.
STRATEGY_COLOUR = "#2563eb"
BENCHMARK_COLOUR = "#94a3b8"
POSITIVE_COLOUR = "#16a34a"
NEGATIVE_COLOUR = "#dc2626"
ACCENT_COLOUR = "#f59e0b"

LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _finish(fig: go.Figure, title: str, yaxis: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(title=title, yaxis_title=yaxis, height=height, **LAYOUT)
    return fig


# --------------------------------------------------------------------------
# Price and volume
# --------------------------------------------------------------------------


def price_chart(df: pd.DataFrame, moving_averages: tuple[int, ...] = (20, 50, 200)) -> go.Figure:
    """Adjusted close with trailing moving averages."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Close",
        line=dict(color=STRATEGY_COLOUR, width=1.6),
    ))
    for window in moving_averages:
        if len(df) >= window:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["close"].rolling(window).mean(),
                name=f"SMA {window}", line=dict(width=1, dash="dot"),
            ))
    return _finish(fig, "Price", "Price")


def candlestick_chart(df: pd.DataFrame) -> go.Figure:
    """OHLC candles above a volume histogram."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=("Price", "Volume"),
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
    ), row=1, col=1)

    rising = df["close"] >= df["open"]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=np.where(rising, POSITIVE_COLOUR, NEGATIVE_COLOUR),
        opacity=0.55,
    ), row=2, col=1)

    fig.update_layout(height=640, xaxis_rangeslider_visible=False, **LAYOUT)
    return fig


def volume_chart(df: pd.DataFrame, window: int = 20) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", opacity=0.5))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["volume"].rolling(window).mean(),
        name=f"{window}-day average", line=dict(color=ACCENT_COLOUR, width=2),
    ))
    return _finish(fig, "Trading volume", "Shares")


# --------------------------------------------------------------------------
# Returns and risk
# --------------------------------------------------------------------------


def returns_distribution(returns: pd.Series, bins: int = 80) -> go.Figure:
    """Return histogram against a fitted normal.

    The gap between the two is the point: market returns have fat tails, so
    the extremes occur far more often than a normal curve predicts.
    """
    clean = returns.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clean, nbinsx=bins, name="Observed",
        histnorm="probability density",
        marker_color=STRATEGY_COLOUR, opacity=0.7,
    ))
    if len(clean) > 2 and clean.std() > 0:
        grid = np.linspace(clean.min(), clean.max(), 300)
        density = (
            1 / (clean.std() * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((grid - clean.mean()) / clean.std()) ** 2)
        )
        fig.add_trace(go.Scatter(
            x=grid, y=density, name="Normal fit",
            line=dict(color=NEGATIVE_COLOUR, width=2),
        ))
    return _finish(fig, "Daily return distribution", "Density")


def rolling_volatility_chart(returns: pd.Series, windows: tuple[int, ...] = (20, 60)) -> go.Figure:
    fig = go.Figure()
    for window in windows:
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns.rolling(window).std() * np.sqrt(252),
            name=f"{window}-day",
        ))
    return _finish(fig, "Rolling annualised volatility", "Volatility")


def cumulative_returns_chart(returns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns.index, y=(1 + returns.fillna(0)).cumprod() - 1,
        name="Cumulative return", line=dict(color=STRATEGY_COLOUR, width=2),
    ))
    return _finish(fig, "Cumulative return", "Return")


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------


def equity_curve_chart(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    initial_capital: float = 1.0,
) -> go.Figure:
    """Strategy against buy-and-hold on shared axes."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_equity.index, y=strategy_equity * initial_capital,
        name="Strategy", line=dict(color=STRATEGY_COLOUR, width=2.2),
    ))
    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index, y=benchmark_equity * initial_capital,
            name="Buy and hold",
            line=dict(color=BENCHMARK_COLOUR, width=2, dash="dash"),
        ))
    label = "Portfolio value" if initial_capital != 1.0 else "Growth of 1"
    return _finish(fig, "Equity curve", label, height=460)


def drawdown_chart(
    strategy_drawdown: pd.Series,
    benchmark_drawdown: pd.Series | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_drawdown.index, y=strategy_drawdown, name="Strategy",
        fill="tozeroy", line=dict(color=NEGATIVE_COLOUR, width=1.4),
    ))
    if benchmark_drawdown is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_drawdown.index, y=benchmark_drawdown, name="Buy and hold",
            line=dict(color=BENCHMARK_COLOUR, width=1.4, dash="dash"),
        ))
    return _finish(fig, "Drawdown", "Drawdown", height=320)


def positions_chart(position: pd.Series, prices: pd.Series) -> go.Figure:
    """Price with the held position shaded underneath."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=("Price", "Position held"),
    )
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices, name="Price",
        line=dict(color=STRATEGY_COLOUR, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=position.index, y=position, name="Position",
        line=dict(color=ACCENT_COLOUR, width=1), fill="tozeroy",
        line_shape="hv",
    ), row=2, col=1)
    fig.update_layout(height=560, **LAYOUT)
    return fig


def trade_markers_chart(prices: pd.Series, trades: pd.DataFrame) -> go.Figure:
    """Price with entry and exit markers for every completed trade."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices, name="Price",
        line=dict(color=BENCHMARK_COLOUR, width=1.4),
    ))
    if trades is not None and not trades.empty:
        for label, column, symbol, colour in (
            ("Entry", "entry_date", "triangle-up", POSITIVE_COLOUR),
            ("Exit", "exit_date", "triangle-down", NEGATIVE_COLOUR),
        ):
            dates = pd.DatetimeIndex(trades[column]).intersection(prices.index)
            fig.add_trace(go.Scatter(
                x=dates, y=prices.reindex(dates), mode="markers", name=label,
                marker=dict(symbol=symbol, size=9, color=colour),
            ))
    return _finish(fig, "Trades", "Price", height=460)


# --------------------------------------------------------------------------
# Model diagnostics
# --------------------------------------------------------------------------


def feature_importance_chart(importance: pd.Series, top_n: int = 20) -> go.Figure:
    top = importance.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=STRATEGY_COLOUR,
    ))
    return _finish(fig, f"Top {min(top_n, len(importance))} features",
                   "", height=max(360, 22 * len(top)))


def prediction_scatter(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    """Predicted against realised returns.

    On daily data this is expected to look like a shapeless cloud. A tight
    diagonal would mean lookahead bias, not a good model.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode="markers", name="Days",
        marker=dict(size=4, opacity=0.45, color=STRATEGY_COLOUR),
    ))
    if len(actual):
        span = [float(np.nanmin(actual)), float(np.nanmax(actual))]
        fig.add_trace(go.Scatter(
            x=span, y=span, mode="lines", name="Perfect prediction",
            line=dict(color=NEGATIVE_COLOUR, dash="dash", width=1.4),
        ))
    fig.update_layout(xaxis_title="Actual return")
    return _finish(fig, "Predicted vs actual return", "Predicted return")


def correlation_heatmap(frame: pd.DataFrame, max_features: int = 25) -> go.Figure:
    corr = frame.iloc[:, :max_features].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="rho"),
    ))
    return _finish(fig, "Feature correlation", "", height=max(460, 20 * len(corr)))


def future_correlation_chart(correlations: pd.Series, top_n: int = 20) -> go.Figure:
    """Each feature's correlation with the next day's return."""
    top = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    top = top.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=np.where(top.values > 0, POSITIVE_COLOUR, NEGATIVE_COLOUR),
    ))
    fig.add_vline(x=0, line_width=1, line_color="#334155")
    return _finish(
        fig, "Correlation with next-day return (expect all to be small)",
        "", height=max(360, 22 * len(top)),
    )


# --------------------------------------------------------------------------
# Comparison and stress
# --------------------------------------------------------------------------


def model_comparison_chart(table: pd.DataFrame, metric: str = "total_return") -> go.Figure:
    if metric not in table.columns:
        raise KeyError(f"{metric!r} not in the comparison table")
    values = table[metric]
    fig = go.Figure(go.Bar(
        x=values.index.astype(str), y=values.values,
        marker_color=np.where(values.values >= 0, POSITIVE_COLOUR, NEGATIVE_COLOUR),
    ))
    return _finish(fig, f"{metric.replace('_', ' ').title()} by model",
                   metric.replace("_", " ").title())


def sweep_chart(
    sweep: pd.DataFrame,
    x: str,
    y: str = "total_return",
    benchmark: float | None = None,
) -> go.Figure:
    """Sensitivity curve for one stress sweep, with the benchmark drawn in."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep[x], y=sweep[y], mode="lines+markers", name="Strategy",
        line=dict(color=STRATEGY_COLOUR, width=2),
    ))
    if benchmark is not None:
        fig.add_hline(
            y=benchmark, line_dash="dash", line_color=BENCHMARK_COLOUR,
            annotation_text="Buy and hold", annotation_position="top left",
        )
    fig.update_layout(xaxis_title=x.replace("_", " ").title())
    return _finish(fig, f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ')}",
                   y.replace("_", " ").title())


def regime_chart(breakdown: pd.DataFrame) -> go.Figure:
    """Strategy vs benchmark return within each market regime."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=breakdown.index.astype(str), y=breakdown["strategy_total_return"],
        name="Strategy", marker_color=STRATEGY_COLOUR,
    ))
    fig.add_trace(go.Bar(
        x=breakdown.index.astype(str), y=breakdown["benchmark_total_return"],
        name="Buy and hold", marker_color=BENCHMARK_COLOUR,
    ))
    fig.update_layout(barmode="group")
    return _finish(fig, "Performance by market regime", "Total return")


def walkforward_chart(fold_table: pd.DataFrame, metric: str = "directional_accuracy") -> go.Figure:
    """One metric across folds, with the all-fold mean marked."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fold_table["fold"].astype(str), y=fold_table[metric],
        name=metric.replace("_", " ").title(), marker_color=STRATEGY_COLOUR,
    ))
    mean = float(fold_table[metric].mean())
    fig.add_hline(y=mean, line_dash="dash", line_color=ACCENT_COLOUR,
                  annotation_text=f"mean {mean:.3f}")
    if metric == "directional_accuracy":
        fig.add_hline(y=0.5, line_dash="dot", line_color=NEGATIVE_COLOUR,
                      annotation_text="coin flip")
    fig.update_layout(xaxis_title="Fold")
    return _finish(fig, f"{metric.replace('_', ' ').title()} by fold",
                   metric.replace("_", " ").title())
