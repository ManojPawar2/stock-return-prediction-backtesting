"""Model training and evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dataset import TARGET, target_summary  # noqa: E402
from src.evaluation import sanity_flags  # noqa: E402
from src.models import MODEL_LABELS, model_path  # noqa: E402
from src.plots import feature_importance_chart, prediction_scatter  # noqa: E402
from src.ui import get_model, honest_note, page_setup, require_data, sidebar  # noqa: E402

page_setup("Modeling")
cfg = sidebar(show_costs=False)

st.title("Model Training")
st.caption(
    f"{MODEL_LABELS.get(cfg.model, cfg.model)} predicting the "
    f"{cfg.target_horizon}-day forward return. The scaler is fit on the "
    "training rows only, so validation and test statistics never leak into it."
)

prepared = require_data(cfg)

# ------------------------------------------------------------------ splits

st.subheader("Chronological splits")
st.dataframe(prepared.split.summary(), use_container_width=True, hide_index=True)
st.caption(
    "The model learns from the train block, hyperparameters are chosen on "
    "validation, and test is touched once. No random shuffling is used, "
    "because that would let the model train on 2024 and test on 2016."
)

with st.expander("Target definition"):
    summary = target_summary(prepared.data, cfg)
    st.code(
        f"target[t] = close[t + {cfg.target_horizon}] / close[t] - 1",
        language="python",
    )
    # Values are cast to str because the dict mixes numbers with the target
    # kind ("regression"), and Arrow cannot serialise a mixed-type column.
    st.dataframe(
        pd.DataFrame({
            "metric": list(summary),
            "value": [str(v) for v in summary.values()],
        }),
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------------- train

model = get_model(cfg)

st.subheader("Performance by split")
metrics = pd.DataFrame(model.metrics).T
st.dataframe(metrics.round(4), use_container_width=True)

test_metrics = model.metrics.get("test", {})
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test RMSE", f"{test_metrics.get('rmse', float('nan')):.4f}")
col2.metric("Test R²", f"{test_metrics.get('r2', float('nan')):.4f}")
col3.metric("Directional accuracy",
            f"{test_metrics.get('directional_accuracy', float('nan')):.1%}")
col4.metric("Information coefficient",
            f"{test_metrics.get('information_coefficient', float('nan')):.4f}")

flags = sanity_flags(test_metrics)
if flags:
    for flag in flags:
        st.error(flag)
else:
    st.success(
        "No sanity warnings. These values sit inside the range that is "
        "realistic for daily equity returns."
    )

train_r2 = model.metrics.get("train", {}).get("r2")
test_r2 = test_metrics.get("r2")
if train_r2 is not None and test_r2 is not None and train_r2 - test_r2 > 0.15:
    st.warning(
        f"Train R² ({train_r2:.3f}) far exceeds test R² ({test_r2:.3f}). "
        "The model is fitting noise in the training period. That is expected "
        "for boosted trees on daily returns and is exactly why the test split "
        "is kept untouched."
    )

honest_note()

# ------------------------------------------------------------ diagnostics

tab1, tab2, tab3 = st.tabs(["Feature importance", "Predictions", "Save"])

with tab1:
    importance = model.importance()
    st.plotly_chart(feature_importance_chart(importance),
                    use_container_width=True)
    st.caption(
        "For trees this is split-based importance; for linear models it is "
        "the absolute coefficient, comparable across features only because "
        "the inputs were standardised first."
    )
    st.dataframe(importance.head(20).to_frame("importance").round(5),
                 use_container_width=True)

with tab2:
    part = st.radio("Split", ["test", "valid", "train"], horizontal=True)
    frame = getattr(prepared.split, part)
    preds = model.predictions[part]
    st.plotly_chart(prediction_scatter(frame[TARGET], preds),
                    use_container_width=True)
    st.caption(
        "A shapeless cloud is the expected picture. A tight diagonal here "
        "would indicate lookahead bias rather than a good model."
    )
    col1, col2 = st.columns(2)
    col1.metric("Predicted mean", f"{preds.mean():.5f}")
    col2.metric("Predicted std", f"{preds.std():.5f}")

with tab3:
    st.write("Persist the fitted pipeline so this exact model can be reloaded.")
    if st.button("Save model"):
        path = model.pipeline.save(model_path(cfg, model.name))
        st.success(f"Saved to `{path}`")
    st.caption(f"Filename includes the config fingerprint `{cfg.fingerprint()}`, "
               "so models from different configurations never overwrite each other.")
