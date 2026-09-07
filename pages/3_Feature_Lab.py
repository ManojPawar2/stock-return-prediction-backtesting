"""Feature inspection and the leakage self-check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.features import (  # noqa: E402
    build_features,
    describe_features,
    feature_columns,
    leakage_report,
)
from src.plots import LAYOUT, STRATEGY_COLOUR  # noqa: E402
from src.ui import cached_clean, page_setup, require_data, sidebar  # noqa: E402

page_setup("Feature Lab")
cfg = sidebar(show_model=False, show_costs=False)

st.title("Feature Lab")
st.caption(
    "Every feature is built from backward-looking windows only. Raw price "
    "levels are computed for plotting but never fed to a model, because a "
    "model trained at one price level cannot extrapolate to another."
)

prepared = require_data(cfg)
features = prepared.features

col1, col2, col3 = st.columns(3)
col1.metric("Features", len(features))
col2.metric("Rows after warm-up", f"{len(prepared.data):,}")
col3.metric("Missing values", int(prepared.data[features].isna().sum().sum()))

tab1, tab2, tab3 = st.tabs(["Explore", "Statistics", "Leakage self-check"])

with tab1:
    default = features.index("ret_1d") if "ret_1d" in features else 0
    chosen = st.selectbox("Feature", features, index=default)
    series = prepared.data[chosen]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series, name=chosen,
        line=dict(color=STRATEGY_COLOUR, width=1.3),
    ))
    fig.update_layout(title=f"{chosen} over time", height=380, **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    hist = go.Figure(go.Histogram(
        x=series, nbinsx=70, marker_color=STRATEGY_COLOUR, opacity=0.8
    ))
    hist.update_layout(title=f"{chosen} distribution", height=330, **LAYOUT)
    st.plotly_chart(hist, use_container_width=True)

    stats = series.describe()
    st.write(
        f"**mean** {stats['mean']:.5f} · **std** {stats['std']:.5f} · "
        f"**min** {stats['min']:.5f} · **max** {stats['max']:.5f}"
    )

with tab2:
    st.dataframe(describe_features(prepared.data).round(5),
                 use_container_width=True, height=520)
    st.caption(
        "A feature with zero standard deviation is a dead column: it carries "
        "no information for any model."
    )

with tab3:
    st.subheader("Correlation with the future")
    st.caption(
        "A feature correlating strongly with tomorrow's return is far more "
        "likely to be a leak than an edge. The threshold below flags anything "
        "suspicious."
    )
    threshold = st.slider("Suspicion threshold", 0.05, 0.60, 0.30, 0.05)
    report = leakage_report(prepared.data, cfg.target_horizon, threshold)

    flagged = report[report["suspicious"]]
    if len(flagged):
        st.error(f"{len(flagged)} feature(s) exceed the threshold:")
        st.dataframe(flagged, use_container_width=True, hide_index=True)
    else:
        st.success(
            f"No feature exceeds the threshold of {threshold:.2f}. The "
            f"strongest is **{report.iloc[0]['feature']}** at "
            f"{report.iloc[0]['corr_with_future']:+.4f}."
        )

    st.dataframe(report.head(25), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Rebuild test")
    st.caption(
        "The definitive check: rebuild the entire feature matrix using only "
        "data up to a chosen date, then compare it against the values "
        "computed from the full history. Any difference is lookahead bias."
    )

    if st.button("Run the rebuild test"):
        cleaned, *_ = cached_clean(cfg.ticker, cfg.start, cfg.end, cfg.use_cache)
        cut = int(len(cleaned) * 0.8)
        with st.spinner("Rebuilding on truncated history…"):
            full = build_features(cleaned, cfg, dropna=False)
            truncated = build_features(cleaned.iloc[: cut + 1], cfg, dropna=False)

        date = cleaned.index[cut]
        offenders = []
        for column in feature_columns(full):
            a, b = full.loc[date, column], truncated.loc[date, column]
            if pd.isna(a) and pd.isna(b):
                continue
            if not np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                offenders.append(column)

        if offenders:
            st.error(
                f"{len(offenders)} feature(s) changed at {date.date()} once "
                f"future data was appended: {offenders}"
            )
        else:
            st.success(
                f"All {len(feature_columns(full))} features at {date.date()} "
                "are identical whether computed from history up to that day "
                "or from the full dataset. No lookahead bias."
            )
