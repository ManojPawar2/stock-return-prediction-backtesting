"""One-click research report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import EXPERIMENT_LOG  # noqa: E402
from src.metrics import performance_summary  # noqa: E402
from src.pipeline import run_full_study  # noqa: E402
from src.report import log_experiment, save_report, verdict  # noqa: E402
from src.ui import page_setup, sidebar  # noqa: E402

page_setup("Research Report")
cfg = sidebar()

st.title("Research Report")
st.caption(
    "Runs the whole study and writes a Markdown report. The generator states "
    "what the numbers show — including when they show the strategy lost."
)

col1, col2, col3 = st.columns(3)
with col1:
    include_wf = st.checkbox("Include walk-forward", value=True)
with col2:
    include_regimes = st.checkbox("Include regimes", value=True)
with col3:
    include_stress = st.checkbox("Include stress tests", value=True)

st.write(
    f"**Study:** {cfg.ticker} · {cfg.start} to {cfg.end} · model "
    f"`{cfg.model}` · {cfg.transaction_cost_bps:.0f} bps commission + "
    f"{cfg.slippage_bps:.0f} bps slippage · fingerprint `{cfg.fingerprint()}`"
)

if st.button("Generate research report", type="primary"):
    try:
        with st.spinner("Running the full pipeline…"):
            study = run_full_study(
                cfg, cfg.model,
                include_walkforward=include_wf,
                include_regimes=include_regimes,
                include_stress=include_stress,
            )
    except Exception as exc:  # noqa: BLE001 - user-facing boundary
        st.error(f"The study failed: {exc}")
        st.stop()

    st.session_state["report"] = study["report"]

    result = study["backtest"]
    from src.metrics import compare_to_benchmark

    comparison = compare_to_benchmark(
        result.strategy_returns, result.benchmark_returns,
        cfg.risk_free_rate, cfg.trading_days, result.position,
    )
    headline, explanation = verdict(comparison)

    if "did not beat" in headline.lower():
        st.warning(f"**{headline}**\n\n{explanation}")
    else:
        st.success(f"**{headline}**\n\n{explanation}")

    path = save_report(study["report"], cfg, cfg.model)
    log_experiment(
        cfg, cfg.model,
        study["model"].metrics.get("test", {}),
        performance_summary(result.strategy_returns),
    )
    st.session_state["report_path"] = path
    st.caption(f"Saved to `{path}`")

report = st.session_state.get("report")
if report:
    st.download_button(
        "Download report (Markdown)",
        report.encode("utf-8"),
        file_name=f"{cfg.ticker}_{cfg.model}_report.md",
        mime="text/markdown",
    )
    with st.expander("Read the report", expanded=True):
        st.markdown(report)
else:
    st.info("Press the button above to run the study and generate a report.")

# ------------------------------------------------------------ experiment log

st.divider()
st.subheader("Experiment log")
st.caption(
    "Every run is appended here, keyed by config fingerprint, so repeat runs "
    "of one configuration are directly comparable and any divergence is "
    "immediately visible."
)

if EXPERIMENT_LOG.exists():
    log = pd.read_csv(EXPERIMENT_LOG)
    st.dataframe(log.tail(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download experiment log (CSV)",
        log.to_csv(index=False).encode("utf-8"),
        file_name="experiment_log.csv", mime="text/csv",
    )
else:
    st.info("No experiments logged yet.")
