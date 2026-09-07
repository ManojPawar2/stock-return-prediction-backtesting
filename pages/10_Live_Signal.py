"""Live signal and the running forward test.

This page is deliberately **read-only**. It never fetches data and never makes
a prediction — it only displays what the scheduled job has already written.

That separation is the point. A dashboard can only run code while someone has
the browser open, so if predicting happened here the track record would have a
hole on every day nobody visited. The scheduled job runs whether anyone is
watching or not; this page is the window onto its output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.live import PREDICTION_LOG, load_log, scorecard  # noqa: E402
from src.plots import (  # noqa: E402
    BENCHMARK_COLOUR,
    LAYOUT,
    NEGATIVE_COLOUR,
    POSITIVE_COLOUR,
    STRATEGY_COLOUR,
)
from src.ui import page_setup, pct, sidebar  # noqa: E402

page_setup("Live Signal")
cfg = sidebar(show_costs=False)

st.title("Live Signal")
st.caption(
    "Written by a scheduled job that runs after the close each weekday. "
    "This page only reads that log — it never fetches or predicts, so the "
    "record is identical whether or not anyone opens the dashboard."
)

log = load_log(PREDICTION_LOG)

if log.empty:
    st.info(
        "No predictions logged yet. The scheduled job writes the first one "
        "after the next market close.",
        icon="🕒",
    )
    st.code("python cli.py predict --ticker AAPL --model xgboost", language="bash")
    st.caption(f"Log location: `{PREDICTION_LOG}`")
    st.stop()

ticker_log = log[log["ticker"].astype(str) == cfg.ticker]
if ticker_log.empty:
    st.warning(
        f"No live predictions recorded for **{cfg.ticker}** yet. "
        f"Tickers in the log: {', '.join(sorted(log['ticker'].astype(str).unique()))}"
    )
    st.stop()

ticker_log = ticker_log.sort_values("bar_date")
latest = ticker_log.iloc[-1]

# ------------------------------------------------------------ latest signal

st.subheader("Most recent signal")

signal = float(latest["signal"])
direction = "LONG" if signal > 0 else ("SHORT" if signal < 0 else "FLAT")
colour = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}[direction]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Signal", f"{colour} {direction}")
col2.metric("Predicted return", f"{float(latest['prediction']):+.4%}")
col3.metric("From bar closing", str(latest["bar_date"]))
col4.metric("Model", str(latest["model"]))

pending = str(latest["realised_return"]) in ("", "nan") or pd.isna(latest["realised_return"])
if pending:
    st.info(
        "This signal applies to the **next trading session**. Its outcome "
        "will be filled in automatically after that session closes.",
        icon="⏳",
    )
else:
    st.caption(
        f"Outcome already recorded: {float(latest['realised_return']):+.4%} "
        f"on {latest['target_date']}."
    )

st.caption(f"Logged at {latest['logged_at']} UTC · config `{latest['fingerprint']}`")

# --------------------------------------------------------------- scorecard

st.divider()
st.subheader("Live track record")
st.caption(
    "Every row below was written **before** its outcome was known, so unlike "
    "a backtest this cannot have been curve-fitted. Returns here are gross of "
    "trading costs — it scores the signal, not a tradable strategy."
)

card = scorecard(log, cfg.ticker)

if not card["n_scored"]:
    st.info(
        f"{card['n_predictions']} prediction(s) logged, none scored yet. "
        "The first outcome lands after the next close."
    )
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Directional calls", card["n_directional"])
    col2.metric(
        "Hit rate", pct(card["hit_rate"], 1),
        delta=f"{(card['hit_rate'] - 0.5) * 100:+.1f} pts vs coin flip",
    )
    col3.metric("Signal return", pct(card["strategy_return"]))
    col4.metric(
        "Buy and hold", pct(card["buy_and_hold_return"]),
        delta=pct(card["strategy_return"] - card["buy_and_hold_return"]),
        delta_color="off",
    )

    st.caption(
        f"Covering {card['first_date']} to {card['last_date']} · "
        f"{card['n_scored']} scored of {card['n_predictions']} logged"
    )

    if card["n_directional"] < 30:
        st.warning(
            f"Only {card['n_directional']} directional calls so far. At this "
            "sample size the hit rate is dominated by noise — roughly 100 "
            "calls are needed before it means much. Treat it as a record "
            "being built, not a result.",
            icon="⚠️",
        )

    # ------------------------------------------------------ equity chart
    scored = ticker_log[ticker_log["realised_return"].notna()].copy()
    if len(scored) >= 2:
        scored["dt"] = pd.to_datetime(scored["target_date"], errors="coerce")
        scored = scored.dropna(subset=["dt"]).sort_values("dt")
        strategy = (1 + scored["signal"].astype(float)
                    * scored["realised_return"].astype(float)).cumprod()
        benchmark = (1 + scored["realised_return"].astype(float)).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scored["dt"], y=strategy, name="Live signal",
            line=dict(color=STRATEGY_COLOUR, width=2.2),
        ))
        fig.add_trace(go.Scatter(
            x=scored["dt"], y=benchmark, name="Buy and hold",
            line=dict(color=BENCHMARK_COLOUR, width=2, dash="dash"),
        ))
        fig.update_layout(title="Live forward test (gross of costs)",
                          yaxis_title="Growth of 1", height=400, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        hits = scored[scored["correct"].astype(str).isin(["True", "False"])]
        if len(hits):
            fig2 = go.Figure(go.Bar(
                x=hits["dt"],
                y=hits["realised_return"].astype(float),
                marker_color=[
                    POSITIVE_COLOUR if c == "True" else NEGATIVE_COLOUR
                    for c in hits["correct"].astype(str)
                ],
                name="Outcome",
            ))
            fig2.update_layout(
                title="Realised return on days a position was taken "
                      "(green = called correctly)",
                yaxis_title="Return", height=320, **LAYOUT,
            )
            st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------- table

st.divider()
st.subheader("Prediction log")

display = ticker_log.sort_values("bar_date", ascending=False).copy()
st.dataframe(display, use_container_width=True, hide_index=True, height=380)

st.download_button(
    "Download prediction log (CSV)",
    log.to_csv(index=False).encode("utf-8"),
    file_name="live_predictions.csv",
    mime="text/csv",
)

with st.expander("How this runs"):
    st.markdown(
        """
        A GitHub Actions job fires at **20:30 UTC on weekdays** (16:30 New
        York time, half an hour after the close) and runs:

        ```bash
        python cli.py predict --ticker AAPL --model xgboost
        ```

        Each run does three things:

        1. **Updates the cache** — refetches the last 7 days with overlap so
           Yahoo's revisions land, and re-downloads everything if a dividend
           has shifted the adjusted-close basis.
        2. **Backfills outcomes** — finds whichever bar actually came next
           after each pending prediction. No holiday calendar is needed.
        3. **Predicts** — but only if a genuinely new bar has appeared. Re-runs
           and retries are no-ops, so the log can never gain duplicates.

        The job then commits the updated log back to the repository. That
        matters: **the git timestamp proves each prediction was recorded
        before its outcome was known**, which is something no backtest can
        demonstrate.
        """
    )
