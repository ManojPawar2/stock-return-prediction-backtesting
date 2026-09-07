"""Backtest results against the buy-and-hold baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.metrics import compare_to_benchmark, trade_statistics  # noqa: E402
from src.plots import (  # noqa: E402
    drawdown_chart,
    equity_curve_chart,
    positions_chart,
    trade_markers_chart,
)
from src.signals import signal_summary  # noqa: E402
from src.ui import (  # noqa: E402
    get_backtest,
    get_model,
    honest_note,
    page_setup,
    pct,
    require_data,
    show_comparison,
    sidebar,
)

page_setup("Backtest")
cfg = sidebar()

st.title("Backtest")
st.caption(
    "Positions are the signal lagged by one day: a prediction made at the "
    "close of day t can only be traded on day t+1. Costs are charged on "
    "turnover, so reversing a position costs twice what opening one does."
)

prepared = require_data(cfg)
model = get_model(cfg)
result = get_backtest(cfg, model)

test = prepared.split.test
st.caption(
    f"Test period {test.index.min().date()} to {test.index.max().date()} "
    f"({len(test):,} trading days) · {cfg.transaction_cost_bps:.0f} bps "
    f"commission + {cfg.slippage_bps:.0f} bps slippage · threshold "
    f"{cfg.signal_threshold:.4f}"
)

# ---------------------------------------------------------------- headline

strategy_return = result.equity.iloc[-1] - 1
benchmark_return = result.benchmark_equity.iloc[-1] - 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy return", pct(strategy_return),
            pct(strategy_return - benchmark_return))
col2.metric("Buy and hold", pct(benchmark_return))
col3.metric("Max drawdown", pct(result.drawdown.min()),
            pct(result.drawdown.min() - result.benchmark_drawdown.min()))
col4.metric("Costs paid", pct(result.total_cost))

if strategy_return > benchmark_return:
    st.success("The strategy beat buy-and-hold over this period.")
else:
    st.warning(
        "The strategy did not beat buy-and-hold. This is a legitimate and "
        "common research finding, not a broken pipeline."
    )

# ------------------------------------------------------------------ charts

st.plotly_chart(
    equity_curve_chart(result.equity, result.benchmark_equity, cfg.initial_capital),
    use_container_width=True,
)
st.plotly_chart(
    drawdown_chart(result.drawdown, result.benchmark_drawdown),
    use_container_width=True,
)

# ----------------------------------------------------------------- metrics

st.subheader("Strategy versus buy-and-hold")
show_comparison(compare_to_benchmark(
    result.strategy_returns, result.benchmark_returns,
    cfg.risk_free_rate, cfg.trading_days, result.position,
))

honest_note()

# ------------------------------------------------------------------- tabs

tab1, tab2, tab3 = st.tabs(["Trades", "Positions", "Daily detail"])

with tab1:
    stats = trade_statistics(result.trades)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trades", stats["n_trades"])
    col2.metric("Win rate", pct(stats["win_rate"]))
    col3.metric("Average hold", f"{stats['avg_holding_days']:.1f} days")
    col4.metric("Payoff ratio", f"{stats['payoff_ratio']:.2f}")

    if result.n_trades:
        st.plotly_chart(
            trade_markers_chart(test["close"], result.trades),
            use_container_width=True,
        )
        st.dataframe(result.trades.round(4), use_container_width=True, height=340)
    else:
        st.info("The threshold is high enough that no trade was taken.")

with tab2:
    summary = signal_summary(result.signal)
    col1, col2, col3 = st.columns(3)
    col1.metric("Time in market", f"{summary['exposure_pct']:.1f}%")
    col2.metric("Signal changes", summary["n_changes"])
    col3.metric("Average holding", f"{summary['avg_holding_days']:.1f} days")
    st.plotly_chart(
        positions_chart(result.position, test["close"]),
        use_container_width=True,
    )

with tab3:
    frame = result.to_frame()
    st.dataframe(frame.round(6), use_container_width=True, height=420)
    st.download_button(
        "Download backtest CSV",
        frame.to_csv().encode("utf-8"),
        file_name=f"{cfg.ticker}_{cfg.model}_backtest.csv",
        mime="text/csv",
    )
