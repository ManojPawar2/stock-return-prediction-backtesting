"""Exploratory data analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.features import feature_columns, future_return_correlation  # noqa: E402
from src.plots import (  # noqa: E402
    correlation_heatmap,
    future_correlation_chart,
    price_chart,
    returns_distribution,
    rolling_volatility_chart,
)
from src.ui import cached_clean, page_setup, require_data, sidebar  # noqa: E402

page_setup("EDA")
cfg = sidebar(show_model=False, show_costs=False)

st.title("Exploratory Data Analysis")

cleaned, *_ = cached_clean(cfg.ticker, cfg.start, cfg.end, cfg.use_cache)
returns = cleaned["close"].pct_change().dropna()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean daily return", f"{returns.mean() * 100:.3f}%")
col2.metric("Daily volatility", f"{returns.std() * 100:.2f}%")
col3.metric("Skewness", f"{returns.skew():.2f}")
col4.metric("Excess kurtosis", f"{returns.kurtosis():.2f}")

st.caption(
    "Excess kurtosis well above 0 means fat tails: extreme days happen far "
    "more often than a normal distribution predicts. This is why risk is not "
    "captured by volatility alone."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Price & returns", "Volatility", "Correlations", "Predictive power"]
)

with tab1:
    st.plotly_chart(price_chart(cleaned), use_container_width=True)
    st.plotly_chart(returns_distribution(returns), use_container_width=True)
    st.caption(
        "The histogram rises above the fitted normal curve in the centre and "
        "in the extreme tails, and falls below it in between. That is the "
        "classic fat-tailed shape of equity returns."
    )
    up = (returns > 0).mean()
    st.write(
        f"**Up days:** {up:.1%} · **Down days:** {1 - up:.1%} · "
        f"**Best:** {returns.max():.2%} · **Worst:** {returns.min():.2%}"
    )

with tab2:
    st.plotly_chart(rolling_volatility_chart(returns), use_container_width=True)
    st.caption(
        "Volatility clusters: calm periods follow calm periods and violent "
        "ones follow violent ones. That persistence is one of the few "
        "genuinely predictable properties of markets, which is why volatility "
        "features are included in the model."
    )

with tab3:
    prepared = require_data(cfg)
    features = feature_columns(prepared.data)
    n = st.slider("Features in the heatmap", 5, min(40, len(features)), 20)
    st.plotly_chart(
        correlation_heatmap(prepared.data[features], max_features=n),
        use_container_width=True,
    )
    st.caption(
        "Deep red blocks are near-duplicate features. Financial features are "
        "highly collinear, which is why Ridge is offered alongside plain "
        "linear regression."
    )

with tab4:
    prepared = require_data(cfg)
    corr = future_return_correlation(prepared.data, cfg.target_horizon)
    st.plotly_chart(future_correlation_chart(corr), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Strongest |correlation|", f"{corr.abs().max():.4f}",
                help=f"Feature: {corr.abs().idxmax()}")
    col2.metric("Median |correlation|", f"{corr.abs().median():.4f}")

    if corr.abs().max() > 0.30:
        st.error(
            f"**{corr.abs().idxmax()}** correlates {corr.abs().max():.3f} with "
            "the future return. On daily data that is a leak, not an edge."
        )
    else:
        st.success(
            "Every correlation is small, which is exactly what honest daily "
            "features look like. It also sets the ceiling on what any model "
            "here can achieve — and it is a low ceiling."
        )
