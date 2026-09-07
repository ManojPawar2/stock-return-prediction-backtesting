"""Stress testing: does the result survive when the assumptions move?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dataset import realised_returns  # noqa: E402
from src.metrics import total_return  # noqa: E402
from src.plots import regime_chart, sweep_chart  # noqa: E402
from src.regime import regime_report, regime_summary_text  # noqa: E402
from src.stress import (  # noqa: E402
    break_even_cost,
    robustness_summary,
    stress_verdict,
    sweep_costs,
    sweep_periods,
    sweep_seeds,
    sweep_thresholds,
    sweep_tickers,
)
from src.ui import (  # noqa: E402
    get_backtest,
    get_model,
    honest_note,
    page_setup,
    pct,
    require_data,
    sidebar,
)

page_setup("Stress Test")
cfg = sidebar()

st.title("Stress Tests & Regimes")
st.caption(
    "One backtest is one number from one configuration. These sweeps vary a "
    "single dimension at a time, so any sensitivity is attributable. Grids "
    "deliberately include settings that should break the strategy."
)

prepared = require_data(cfg)
model = get_model(cfg)
result = get_backtest(cfg, model)

test = prepared.split.test
returns = realised_returns(test)
predictions = model.predictions["test"]
benchmark = total_return(returns)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Cost", "Threshold", "Periods & seeds", "Other tickers", "Regimes"]
)

sweeps: dict[str, pd.DataFrame] = {}

with tab1:
    st.subheader("Transaction cost sensitivity")
    costs = sweep_costs(returns, predictions, cfg)
    sweeps["cost"] = costs

    st.plotly_chart(sweep_chart(costs, "cost_bps", "total_return", benchmark),
                    use_container_width=True)
    st.dataframe(costs.round(4), use_container_width=True, hide_index=True)

    be = break_even_cost(costs, benchmark)
    if pd.isna(be):
        st.error(
            "The strategy never beats buy-and-hold, even at zero cost. Costs "
            "are not what is holding it back — the signal is."
        )
    else:
        st.success(f"The strategy beats buy-and-hold up to **{be:.0f} bps** "
                   "of one-way cost.")
    st.caption(
        "A signal that flips daily at 5 bps round-trip pays roughly 12.6% a "
        "year in friction. Most impressive-looking naive backtests are simply "
        "strategies that forgot to pay for their own trading."
    )

with tab2:
    st.subheader("Signal threshold sensitivity")
    thresholds = sweep_thresholds(returns, predictions, cfg)
    sweeps["threshold"] = thresholds

    st.plotly_chart(
        sweep_chart(thresholds, "threshold", "total_return", benchmark),
        use_container_width=True,
    )
    st.plotly_chart(sweep_chart(thresholds, "threshold", "exposure"),
                    use_container_width=True)
    st.dataframe(thresholds.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Raising the threshold trades less and pays less, but also spends "
        "more time out of the market. If performance depends sharply on one "
        "threshold value, that value has been fitted rather than chosen."
    )

with tab3:
    st.subheader("Sub-period stability")
    periods = sweep_periods(returns, predictions, cfg, 4)
    sweeps["period"] = periods
    st.dataframe(periods.round(4), use_container_width=True, hide_index=True)

    wins = int(periods["beats_buy_and_hold"].sum()) if len(periods) else 0
    st.write(f"The strategy beat buy-and-hold in **{wins} of {len(periods)}** "
             "sub-periods.")
    if len(periods) and wins <= 1:
        st.warning(
            "Winning in one sub-period out of four suggests the headline "
            "result came from a single favourable stretch."
        )

    st.divider()
    st.subheader("Seed stability")
    st.caption("Tree models depend on random initialisation. A wide spread "
               "here means the model is fitting noise.")
    if st.button("Run the seed sweep"):
        with st.spinner("Refitting under several seeds…"):
            seeds = sweep_seeds(prepared.data, cfg, prepared.features, cfg.model)
        st.dataframe(seeds.round(4), use_container_width=True, hide_index=True)
        spread = seeds["total_return"].max() - seeds["total_return"].min()
        st.metric("Return spread across seeds", pct(spread))
        if spread > 0.20:
            st.warning(
                "Returns swing widely with the seed alone, so a large part of "
                "this result is randomness rather than signal."
            )

with tab4:
    st.subheader("Does it generalise to other tickers?")
    st.caption(
        "The same pipeline, unchanged, across a basket. A strategy that only "
        "works on the ticker it was developed against has been fitted to that "
        "ticker's history."
    )
    st.write(f"Universe: {', '.join(cfg.universe)}")
    if st.button("Run the ticker sweep"):
        with st.spinner("Running the full pipeline per ticker…"):
            tickers = sweep_tickers(cfg, model_name=cfg.model)
        if tickers.empty:
            st.error("No ticker completed successfully.")
        else:
            sweeps["ticker"] = tickers
            st.dataframe(tickers.round(4), use_container_width=True, hide_index=True)
            wins = int(tickers["beats_buy_and_hold"].sum())
            st.metric("Tickers beaten", f"{wins} of {len(tickers)}")

with tab5:
    st.subheader("Performance by market regime")
    st.caption(
        "Days are grouped by trailing volatility tercile and by position "
        "relative to the 200-day average. The question is whether the edge is "
        "real or belongs to one environment."
    )
    report = regime_report(test, result.strategy_returns, result.benchmark_returns)

    for name, breakdown in report.items():
        if breakdown is None or breakdown.empty:
            continue
        st.markdown(f"**By {name}**")
        st.dataframe(breakdown.round(4), use_container_width=True)
        if "strategy_total_return" in breakdown.columns:
            st.plotly_chart(regime_chart(breakdown), use_container_width=True)
        st.info(regime_summary_text(breakdown))

# ----------------------------------------------------------------- verdict

st.divider()
st.subheader("Robustness verdict")
st.dataframe(robustness_summary(sweeps).round(4),
             use_container_width=True, hide_index=True)
st.warning(stress_verdict(sweeps))
honest_note()
