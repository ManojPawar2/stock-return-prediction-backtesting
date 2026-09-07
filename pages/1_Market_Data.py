"""Market data page: candlesticks, volume and the raw table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.data_loader import cache_path, is_cached  # noqa: E402
from src.plots import (  # noqa: E402
    candlestick_chart,
    cumulative_returns_chart,
    volume_chart,
)
from src.ui import cached_clean, page_setup, sidebar  # noqa: E402

page_setup("Market Data")
cfg = sidebar(show_model=False, show_costs=False)

st.title("Market Data")
st.caption(
    "Adjusted OHLCV. Open, high and low are rescaled onto the same basis as "
    "the adjusted close, so stock splits do not create fake overnight gaps."
)

cleaned, report, log, summary = cached_clean(
    cfg.ticker, cfg.start, cfg.end, cfg.use_cache
)

window = st.slider(
    "Days to display", 60, len(cleaned), min(500, len(cleaned)), 20,
    help="Charts render the most recent N trading days.",
)
view = cleaned.tail(window)

st.plotly_chart(candlestick_chart(view), use_container_width=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(volume_chart(view), use_container_width=True)
with right:
    st.plotly_chart(
        cumulative_returns_chart(view["close"].pct_change()),
        use_container_width=True,
    )

st.divider()
st.subheader("Historical data")

st.dataframe(
    cleaned.sort_index(ascending=False).round(4),
    use_container_width=True, height=420,
)

col1, col2 = st.columns([1, 3])
with col1:
    st.download_button(
        "Download CSV",
        cleaned.to_csv().encode("utf-8"),
        file_name=f"{cfg.ticker}_{cfg.start}_{cfg.end}.csv",
        mime="text/csv",
    )
with col2:
    cached = is_cached(cfg.ticker, cfg.start, cfg.end)
    source = "Served from local cache" if cached else "Downloaded from Yahoo Finance"
    st.caption(
        f"{source} · `{cache_path(cfg.ticker, cfg.start, cfg.end).name}` · "
        f"{len(cleaned):,} rows"
    )
