"""Shared Streamlit helpers.

Every page imports its sidebar and its data access from here, so the whole app
runs off one configuration object and one cache.  Without this, each page
would rebuild features independently and could quietly disagree with its
neighbours.

The caching layer is keyed on the primitive parameters rather than on the
``Config`` object, because Streamlit needs a hashable key and a dataclass with
nested sections is not one.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from .config import Config, ensure_dirs, load_config
from .data_loader import describe, load_range
from .models import available_models
from .pipeline import PreparedData, TrainedModel, backtest, prepare, train
from .preprocessing import clean, validate

PAGE_ICON = "📈"


def page_setup(title: str, icon: str = PAGE_ICON) -> None:
    st.set_page_config(page_title=f"{title} · Stock Research",
                       page_icon=icon, layout="wide")
    ensure_dirs()


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading market data…")
def cached_prices(ticker: str, start: str, end: str, use_cache: bool) -> pd.DataFrame:
    # load_range prefers the rolling cache committed to the repository, so a
    # deployed app serves every page without calling Yahoo on a cold start.
    return load_range(ticker, start, end, use_cache)


@st.cache_data(show_spinner="Validating and cleaning…")
def cached_clean(ticker: str, start: str, end: str, use_cache: bool):
    raw = cached_prices(ticker, start, end, use_cache)
    report = validate(raw)
    cleaned, log = clean(raw)
    return cleaned, report, log, describe(cleaned)


@st.cache_data(show_spinner="Building features…")
def cached_prepare(
    ticker: str, start: str, end: str, use_cache: bool,
    train_end: str, valid_end: str, horizon: int, kind: str,
) -> PreparedData:
    cfg = Config(
        ticker=ticker, start=start, end=end, use_cache=use_cache,
        train_end=train_end, valid_end=valid_end,
        target_horizon=horizon, target_kind=kind,
    )
    cleaned, *_ = cached_clean(ticker, start, end, use_cache)
    return prepare(cfg, cleaned)


@st.cache_resource(show_spinner="Training model…")
def cached_train(
    ticker: str, start: str, end: str, use_cache: bool,
    train_end: str, valid_end: str, horizon: int, kind: str,
    model_name: str, seed: int,
) -> TrainedModel:
    cfg = Config(
        ticker=ticker, start=start, end=end, use_cache=use_cache,
        train_end=train_end, valid_end=valid_end,
        target_horizon=horizon, target_kind=kind, model=model_name, seed=seed,
    )
    prepared = cached_prepare(ticker, start, end, use_cache,
                              train_end, valid_end, horizon, kind)
    return train(cfg, prepared, model_name)


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------


def get_prepared(cfg: Config) -> PreparedData:
    return cached_prepare(
        cfg.ticker, cfg.start, cfg.end, cfg.use_cache,
        cfg.train_end, cfg.valid_end, cfg.target_horizon, cfg.target_kind,
    )


def get_model(cfg: Config, model_name: str | None = None) -> TrainedModel:
    return cached_train(
        cfg.ticker, cfg.start, cfg.end, cfg.use_cache,
        cfg.train_end, cfg.valid_end, cfg.target_horizon, cfg.target_kind,
        model_name or cfg.model, cfg.seed,
    )


def get_backtest(cfg: Config, model: TrainedModel, part: str = "test"):
    """Not cached: costs and thresholds are adjusted interactively and the
    backtest is cheap once predictions exist."""
    return backtest(cfg, get_prepared(cfg), model, part)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def sidebar(show_model: bool = True, show_costs: bool = True) -> Config:
    """Render the shared sidebar and return the resulting Config.

    Selections persist in ``st.session_state`` so navigating between pages
    keeps the same study rather than resetting to defaults.
    """
    defaults = st.session_state.get("cfg") or load_config()

    with st.sidebar:
        st.header("Configuration")

        ticker = st.text_input("Ticker", value=defaults.ticker).strip().upper()

        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(
                "Start", value=pd.Timestamp(defaults.start),
                min_value=pd.Timestamp("1990-01-01"),
            )
        with col2:
            end = st.date_input("End", value=pd.Timestamp(defaults.end))

        st.caption("Splits are chronological — never random.")
        col3, col4 = st.columns(2)
        with col3:
            train_end = st.date_input("Train ends", value=pd.Timestamp(defaults.train_end))
        with col4:
            valid_end = st.date_input("Valid ends", value=pd.Timestamp(defaults.valid_end))

        model = defaults.model
        seed = defaults.seed
        if show_model:
            st.divider()
            options = available_models()
            model = st.selectbox(
                "Model", options,
                index=options.index(defaults.model) if defaults.model in options else 0,
            )
            seed = st.number_input("Random seed", value=int(defaults.seed),
                                   step=1, min_value=0)

        threshold = defaults.signal_threshold
        cost = defaults.transaction_cost_bps
        slippage = defaults.slippage_bps
        allow_short = defaults.allow_short
        if show_costs:
            st.divider()
            st.caption("Trading assumptions")
            threshold = st.slider(
                "Signal threshold", 0.0, 0.010,
                float(defaults.signal_threshold), 0.0005, format="%.4f",
                help="Predictions inside ±threshold produce no position. "
                     "Raising it trades less often and pays less in costs.",
            )
            cost = st.slider("Commission (bps)", 0.0, 50.0,
                             float(defaults.transaction_cost_bps), 1.0)
            slippage = st.slider("Slippage (bps)", 0.0, 25.0,
                                 float(defaults.slippage_bps), 1.0)
            allow_short = st.checkbox("Allow short positions",
                                      value=defaults.allow_short)

    cfg = replace(
        defaults,
        ticker=ticker or defaults.ticker,
        start=str(pd.Timestamp(start).date()),
        end=str(pd.Timestamp(end).date()),
        train_end=str(pd.Timestamp(train_end).date()),
        valid_end=str(pd.Timestamp(valid_end).date()),
        model=model,
        seed=int(seed),
        signal_threshold=float(threshold),
        transaction_cost_bps=float(cost),
        slippage_bps=float(slippage),
        allow_short=bool(allow_short),
    )

    try:
        cfg.validate()
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    st.session_state["cfg"] = cfg
    with st.sidebar:
        st.divider()
        st.caption(f"Config fingerprint `{cfg.fingerprint()}`")
    return cfg


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------


def metric_row(metrics: dict[str, tuple[str, str | None]]) -> None:
    """Render a row of ``st.metric`` cards from ``{label: (value, delta)}``."""
    columns = st.columns(len(metrics))
    for column, (label, (value, delta)) in zip(columns, metrics.items()):
        column.metric(label, value, delta)


def pct(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def num(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def show_comparison(comparison: pd.DataFrame) -> None:
    """Strategy vs benchmark table with a red/green difference column."""
    display = comparison.copy()
    from .metrics import PERCENT_METRICS

    for metric in display.index:
        if metric in PERCENT_METRICS:
            display.loc[metric] = display.loc[metric] * 100

    st.dataframe(
        display.style.format("{:,.3f}", na_rep="n/a").map(
            lambda v: (
                "color: #16a34a" if isinstance(v, (int, float)) and v > 0
                else "color: #dc2626" if isinstance(v, (int, float)) and v < 0
                else ""
            ),
            subset=["difference"],
        ),
        use_container_width=True,
    )
    st.caption("Percentage metrics are shown in percent. Green means the "
               "strategy beat buy-and-hold on that measure.")


def honest_note() -> None:
    """The disclaimer that belongs on every results page."""
    st.info(
        "**Reading these numbers.** Daily equity returns are close to "
        "unpredictable. An R² near zero and a directional accuracy in the "
        "49–54% range are the correct, honest outcomes here — not failures. "
        "A backtest describes one asset over one historical window and is "
        "never a prediction about the future.",
        icon="ℹ️",
    )


def require_data(cfg: Config):
    """Load prepared data, showing a readable error instead of a traceback."""
    try:
        return get_prepared(cfg)
    except Exception as exc:  # noqa: BLE001 - user-facing boundary
        st.error(f"Could not prepare data for {cfg.ticker}: {exc}")
        st.caption("Check the ticker symbol, the date range, and that the "
                   "train/validation boundaries fall inside it.")
        st.stop()
