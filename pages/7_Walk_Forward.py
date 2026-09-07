"""Walk-forward validation: retrain repeatedly, always testing on the future."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.metrics import compare_to_benchmark  # noqa: E402
from src.plots import (  # noqa: E402
    drawdown_chart,
    equity_curve_chart,
    feature_importance_chart,
    walkforward_chart,
)
from src.ui import (  # noqa: E402
    honest_note,
    page_setup,
    pct,
    require_data,
    show_comparison,
    sidebar,
)
from src.walkforward import walk_forward_backtest  # noqa: E402

page_setup("Walk-Forward")
cfg = sidebar()

st.title("Walk-Forward Validation")
st.caption(
    "The model is retrained on every fold and always tested on the block that "
    "follows. Every prediction below was therefore made by a model that had "
    "never seen the day it was predicting."
)

prepared = require_data(cfg)

col1, col2, col3 = st.columns(3)
with col1:
    folds = st.number_input("Folds", 2, 12, int(cfg.walk_forward.n_folds))
with col2:
    mode = st.selectbox(
        "Window", ["expanding", "rolling"],
        index=0 if cfg.walk_forward.mode == "expanding" else 1,
        help="Expanding trains on all history so far. Rolling trains on a "
             "fixed recent window, adapting faster but forgetting more.",
    )
with col3:
    min_train = st.number_input(
        "Minimum training days", 250, max(260, len(prepared.data) - 100),
        min(756, max(250, len(prepared.data) // 3)), step=50,
    )

try:
    with st.spinner("Retraining across folds…"):
        result, backtest = walk_forward_backtest(
            prepared.data, cfg, prepared.features, cfg.model,
            n_folds=int(folds), mode=mode, min_train=int(min_train),
        )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------- headline

overall = result.overall_metrics()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Out-of-sample days", f"{overall['n']:,}")
col2.metric("Directional accuracy", pct(overall["directional_accuracy"], 1))
col3.metric("R²", f"{overall['r2']:.4f}")
col4.metric("Information coefficient", f"{overall['information_coefficient']:.4f}")

st.caption(
    f"These figures come from {overall['n']:,} out-of-sample days across "
    f"{result.n_folds} folds — far more evidence than a single hold-out split "
    "provides."
)

# ------------------------------------------------------------------- folds

st.subheader("Per-fold results")
st.dataframe(result.fold_table().round(4), use_container_width=True, hide_index=True)

metric = st.selectbox(
    "Metric by fold",
    ["directional_accuracy", "r2", "rmse", "information_coefficient"],
)
st.plotly_chart(walkforward_chart(result.fold_table(), metric),
                use_container_width=True)

st.subheader("Stability across folds")
stability = result.stability()
st.dataframe(stability.round(4), use_container_width=True)

if "directional_accuracy" in stability.index:
    spread = stability.loc["directional_accuracy", "max"] - \
        stability.loc["directional_accuracy", "min"]
    if spread > 0.10:
        st.warning(
            f"Directional accuracy varies by {spread:.1%} between folds. "
            "An edge that swings that widely is more likely noise than a "
            "durable effect."
        )
    else:
        st.info(
            f"Directional accuracy varies by only {spread:.1%} between folds, "
            "so the model behaves consistently across periods — though "
            "consistently near 50% is still near a coin flip."
        )

# ---------------------------------------------------------------- backtest

st.subheader("Backtest of the stitched predictions")
st.caption("This is the most realistic figure the project produces.")

st.plotly_chart(
    equity_curve_chart(backtest.equity, backtest.benchmark_equity,
                       cfg.initial_capital),
    use_container_width=True,
)
st.plotly_chart(
    drawdown_chart(backtest.drawdown, backtest.benchmark_drawdown),
    use_container_width=True,
)

show_comparison(compare_to_benchmark(
    backtest.strategy_returns, backtest.benchmark_returns,
    cfg.risk_free_rate, cfg.trading_days, backtest.position,
))

with st.expander("Average feature importance across folds"):
    importance = result.average_importance()
    if len(importance):
        st.plotly_chart(feature_importance_chart(importance),
                        use_container_width=True)
        st.caption(
            "Averaging across folds is more trustworthy than any single fit: "
            "a feature that only matters in one fold was probably fitting "
            "that fold's noise."
        )

honest_note()
