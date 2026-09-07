"""Streamlit entry point — the Overview page.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src.plots import price_chart  # noqa: E402
from src.ui import cached_clean, honest_note, page_setup, sidebar  # noqa: E402

page_setup("Overview")
cfg = sidebar(show_model=False, show_costs=False)

st.title("Stock Return Prediction & Strategy Backtesting")
st.caption(
    "An end-to-end quantitative research pipeline: data validation, "
    "leakage-free feature engineering, time-based model validation, and "
    "honest backtesting against a buy-and-hold baseline."
)

# --------------------------------------------------------------------- data

try:
    cleaned, report, log, summary = cached_clean(
        cfg.ticker, cfg.start, cfg.end, cfg.use_cache
    )
except Exception as exc:  # noqa: BLE001 - user-facing boundary
    st.error(f"Could not load **{cfg.ticker}**: {exc}")
    st.caption("Check the ticker symbol and that the date range covers "
               "trading days. The first load needs an internet connection; "
               "afterwards data is served from a local cache.")
    st.stop()

# ------------------------------------------------------------------ summary

st.subheader(f"{cfg.ticker} — {summary['start']} to {summary['end']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Trading days", f"{summary['rows']:,}")
col2.metric("Years of history", f"{summary['years']:.1f}")
col3.metric("Buy-and-hold return", f"{summary['total_return_pct']:.1f}%")
col4.metric("Annualised volatility", f"{summary['annualised_volatility_pct']:.1f}%")

st.plotly_chart(price_chart(cleaned), use_container_width=True)

# --------------------------------------------------------------- validation

left, right = st.columns([3, 2])

with left:
    st.subheader("Data quality")
    st.write(report.summary())
    if report.issues:
        st.dataframe(report.to_frame(), use_container_width=True, hide_index=True)
    else:
        st.success("Every validation check passed.")
    st.caption(f"Cleaning: {log.summary()}")

with right:
    st.subheader("Dataset statistics")
    st.dataframe(
        {"metric": list(summary), "value": [str(v) for v in summary.values()]},
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------------- guide

st.divider()
st.subheader("How this project works")

tab1, tab2, tab3 = st.tabs(["The pipeline", "Why leakage matters", "What to expect"])

with tab1:
    st.markdown(
        """
        1. **Market Data** — daily OHLCV from Yahoo Finance, cached locally.
        2. **EDA** — returns, volatility, volume and their relationships.
        3. **Feature Lab** — 43 features, every one built only from past data.
        4. **Modeling** — Linear, Ridge, Random Forest and XGBoost, trained on
           a chronological split.
        5. **Backtest** — predictions become signals, signals become positions,
           positions pay costs.
        6. **Comparison** — every model against the same buy-and-hold baseline.
        7. **Walk-Forward** — retrain repeatedly and stitch the out-of-sample
           predictions together.
        8. **Stress Test** — sweep costs, thresholds, tickers and periods.
        9. **Research Report** — one-click Markdown summary with the verdict.
        """
    )

with tab2:
    st.markdown(
        """
        The single most important rule in this project:

        > A prediction made at the close of day *t* can only affect the
        > position held on day *t+1*.

        Three mechanisms enforce it:

        - **Features** use only backward-looking windows. A test rebuilds the
          entire feature matrix on truncated history and requires every value
          to be unchanged.
        - **The target** is `close[t+1] / close[t] - 1` — explicitly in the
          future, and the last row is dropped because its future is unknown.
        - **The backtester** applies `position = signal.shift(1)`. A test feeds
          it a signal that peeks at the current bar and proves the lag
          neutralises it.

        Dropping that one `.shift(1)` is the most common way a backtest becomes
        fantasy, and it fails silently — the equity curve simply becomes
        beautiful.
        """
    )

with tab3:
    honest_note()
    st.markdown(
        """
        Concretely, on daily equity data you should expect:

        | Metric | Realistic range | What a "great" value means |
        |---|---|---|
        | R² on next-day returns | −0.05 to +0.02 | Above 0.10 → look for a leak |
        | Directional accuracy | 49–54% | Above 60% → almost certainly a bug |
        | Information coefficient | 0.00–0.05 | Above 0.30 → a leak |
        | Feature vs future correlation | \\|ρ\\| < 0.10 | Above 0.30 → a leak |

        The project is built so that "the strategy did not beat buy-and-hold"
        is a publishable result. A rigorous pipeline that finds only a marginal
        edge is far more credible than a dashboard claiming a Sharpe of 5.
        """
    )

st.divider()
st.caption(
    f"Config fingerprint `{cfg.fingerprint()}` · "
    "Use the sidebar to change ticker and dates, then pick a page on the left. "
    "Not investment advice."
)
