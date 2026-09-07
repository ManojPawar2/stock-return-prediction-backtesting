"""All models compared on identical data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.models import MODEL_LABELS, available_models  # noqa: E402
from src.pipeline import compare_models  # noqa: E402
from src.plots import equity_curve_chart, model_comparison_chart  # noqa: E402
from src.ui import honest_note, page_setup, require_data, sidebar  # noqa: E402

page_setup("Comparison")
cfg = sidebar(show_model=False)

st.title("Model Comparison")
st.caption(
    "Every model is trained on identical splits, identical features and an "
    "identical seed, so any difference in this table is attributable to the "
    "model itself rather than the setup."
)

prepared = require_data(cfg)

options = available_models()
chosen = st.multiselect(
    "Models", options, default=options,
    format_func=lambda name: MODEL_LABELS.get(name, name),
)

if not chosen:
    st.info("Select at least one model.")
    st.stop()

with st.spinner("Training every model…"):
    table, models, backtests = compare_models(cfg, prepared, chosen)

# ------------------------------------------------------------------- table

st.subheader("Results")
st.dataframe(
    table.style.format("{:,.4f}", na_rep="—").background_gradient(
        subset=["total_return", "sharpe_ratio"], cmap="RdYlGn",
    ),
    use_container_width=True,
)
st.caption(
    "Buy-and-hold has no prediction-accuracy metrics because it makes no "
    "prediction. Note that the best RMSE and the best strategy return are "
    "usually different models — statistical accuracy and trading performance "
    "are genuinely different objectives."
)

# ------------------------------------------------------------------ charts

metric = st.selectbox(
    "Chart metric",
    ["total_return", "sharpe_ratio", "max_drawdown", "directional_accuracy",
     "rmse", "n_trades"],
)
st.plotly_chart(model_comparison_chart(table, metric), use_container_width=True)

st.subheader("Equity curves")
fig = equity_curve_chart(
    next(iter(backtests.values())).benchmark_equity * 0 + 1, None
)
fig.data = ()
for name, result in backtests.items():
    fig.add_scatter(
        x=result.equity.index, y=result.equity,
        name=MODEL_LABELS.get(name, name), mode="lines",
    )
any_result = next(iter(backtests.values()))
fig.add_scatter(
    x=any_result.benchmark_equity.index, y=any_result.benchmark_equity,
    name="Buy and hold", mode="lines",
    line=dict(dash="dash", width=2.4, color="#94a3b8"),
)
fig.update_layout(title="Growth of 1 unit")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- verdict

strategies = table.drop(index="buy_and_hold", errors="ignore")
if len(strategies):
    best = strategies["total_return"].idxmax()
    benchmark = table.loc["buy_and_hold", "total_return"] \
        if "buy_and_hold" in table.index else float("nan")
    best_return = strategies.loc[best, "total_return"]

    st.subheader("Reading the table")
    if best_return > benchmark:
        st.success(
            f"**{MODEL_LABELS.get(best, best)}** returned {best_return:.1%} "
            f"against buy-and-hold's {benchmark:.1%}. Before treating that as "
            "an edge, check the Walk-Forward and Stress Test pages — a single "
            "split can flatter one model by luck."
        )
    else:
        st.warning(
            f"The best model, **{MODEL_LABELS.get(best, best)}**, returned "
            f"{best_return:.1%} against buy-and-hold's {benchmark:.1%}. "
            "No model beat simply holding the asset over this period."
        )

    if "rmse" in strategies.columns and strategies["rmse"].notna().any():
        best_rmse = strategies["rmse"].idxmin()
        if best_rmse != best:
            st.info(
                f"The lowest RMSE belongs to "
                f"**{MODEL_LABELS.get(best_rmse, best_rmse)}**, but the best "
                f"strategy return belongs to **{MODEL_LABELS.get(best, best)}**. "
                "This is the central lesson of the project: prediction "
                "accuracy and trading performance are not the same thing."
            )

honest_note()
