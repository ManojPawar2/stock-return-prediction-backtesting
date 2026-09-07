"""Automated research report.

Writes the study's conclusion to a Markdown file so it is recorded rather than
remembered.  The generator has one editorial rule: **it states what the numbers
show, including when they show the strategy lost.**

:func:`verdict` is where that rule lives.  It compares the strategy to
buy-and-hold on return, risk-adjusted return and drawdown, and writes the
honest sentence — because a report that only ever congratulates the author is
not a research report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS, Config
from .evaluation import sanity_flags
from .metrics import compare_to_benchmark, trade_statistics

logger = logging.getLogger(__name__)


@dataclass
class ReportInputs:
    """Everything the report needs. Optional pieces are simply omitted."""

    cfg: Config
    data_summary: dict
    split_summary: pd.DataFrame
    model_name: str
    model_metrics: dict[str, dict[str, float]]     # split -> metrics
    comparison: pd.DataFrame                        # strategy vs buy-and-hold
    trades: pd.DataFrame
    feature_importance: pd.Series | None = None
    future_correlations: pd.Series | None = None
    regime_breakdown: dict[str, pd.DataFrame] = field(default_factory=dict)
    stress_sweeps: dict[str, pd.DataFrame] = field(default_factory=dict)
    walkforward_table: pd.DataFrame | None = None
    walkforward_stability: pd.DataFrame | None = None
    generated_at: str | None = None


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def verdict(comparison: pd.DataFrame) -> tuple[str, str]:
    """Return ``(headline, explanation)`` describing the honest result."""
    def get(metric: str, column: str = "difference") -> float:
        try:
            return float(comparison.loc[metric, column])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    d_return = get("total_return")
    d_sharpe = get("sharpe_ratio")
    d_dd = get("max_drawdown")          # positive difference = shallower drawdown
    strat_return = get("total_return", "strategy")
    bench_return = get("total_return", "buy_and_hold")

    if np.isnan(d_return):
        return "Inconclusive", "The comparison did not produce usable numbers."

    beat_return = d_return > 0
    beat_sharpe = d_sharpe > 0 if not np.isnan(d_sharpe) else False
    beat_dd = d_dd > 0 if not np.isnan(d_dd) else False

    detail = (
        f"The strategy returned {strat_return:.1%} against buy-and-hold's "
        f"{bench_return:.1%}, a difference of {d_return:+.1%}."
    )

    if beat_return and beat_sharpe:
        headline = "The strategy outperformed buy-and-hold"
        explanation = (
            f"{detail} It also achieved a higher Sharpe ratio "
            f"({d_sharpe:+.2f}). This is a positive result on this data, but "
            "it is a statement about one asset over one historical window, "
            "not evidence of future profitability."
        )
    elif beat_return and not beat_sharpe:
        headline = "The strategy returned more, but took more risk to do it"
        explanation = (
            f"{detail} However its Sharpe ratio was lower ({d_sharpe:+.2f}), "
            "so the extra return came from extra risk rather than skill."
        )
    elif not beat_return and beat_dd:
        headline = "The strategy underperformed on return but reduced drawdown"
        explanation = (
            f"{detail} It did cut the maximum drawdown by "
            f"{abs(d_dd):.1%}, so it was less painful to hold, but staying "
            "invested throughout would have produced more money."
        )
    else:
        headline = "The strategy did not beat buy-and-hold"
        explanation = (
            f"{detail} It did not compensate with better risk-adjusted "
            "returns or a shallower drawdown either. This is a legitimate "
            "research finding: it says the tested features carry no reliable "
            "short-horizon signal for this asset once trading costs are paid."
        )
    return headline, explanation


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def _table(frame: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = True) -> str:
    """Markdown table from a DataFrame, without a tabulate dependency."""
    if frame is None or frame.empty:
        return "_No data._\n"

    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(
                lambda v: "n/a" if pd.isna(v) else floatfmt.format(v)
            )
        else:
            display[col] = display[col].astype(str)

    headers = ([str(display.index.name or "")] if index else []) + [
        str(c) for c in display.columns
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in display.iterrows():
        cells = ([str(idx)] if index else []) + list(row.values)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _kv_table(mapping: dict, floatfmt: str = "{:.4f}") -> str:
    lines = ["| Item | Value |", "|---|---|"]
    for key, value in mapping.items():
        if isinstance(value, float):
            shown = "n/a" if not np.isfinite(value) else floatfmt.format(value)
        else:
            shown = f"{value:,}" if isinstance(value, int) else str(value)
        lines.append(f"| {key.replace('_', ' ').title()} | {shown} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Report body
# --------------------------------------------------------------------------


def build_report(inputs: ReportInputs) -> str:
    """Render the full Markdown research report."""
    cfg = inputs.cfg
    stamp = inputs.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    headline, explanation = verdict(inputs.comparison)

    parts: list[str] = []
    add = parts.append

    # ---- header ---------------------------------------------------------
    add(f"# Research Report — {cfg.ticker}\n")
    add(f"**Generated:** {stamp}  ")
    add(f"**Model:** {inputs.model_name}  ")
    add(f"**Config fingerprint:** `{cfg.fingerprint()}`\n")

    add("## Verdict\n")
    add(f"### {headline}\n")
    add(f"{explanation}\n")

    # ---- setup ----------------------------------------------------------
    add("## 1. Study Design\n")
    add(f"- **Asset:** {cfg.ticker}\n"
        f"- **Period:** {cfg.start} to {cfg.end}\n"
        f"- **Target:** {cfg.target_horizon}-day forward return "
        f"({cfg.target_kind})\n"
        f"- **Signal rule:** long when the predicted return exceeds "
        f"{cfg.signal_threshold:.4f}"
        f"{'; shorting enabled' if cfg.allow_short else '; long-only'}\n"
        f"- **Costs:** {cfg.transaction_cost_bps:.0f} bps commission + "
        f"{cfg.slippage_bps:.0f} bps slippage, charged on turnover\n"
        f"- **Seed:** {cfg.seed}\n")

    add("### Dataset\n")
    add(_kv_table(inputs.data_summary, "{:.2f}"))

    add("### Chronological splits\n")
    add("Splits are strictly ordered in time; no random shuffling is used.\n")
    add(_table(inputs.split_summary, index=False))

    # ---- model ----------------------------------------------------------
    add("## 2. Model Performance\n")
    add("Statistical accuracy of the return forecast, by split.\n")
    add(_table(pd.DataFrame(inputs.model_metrics).T))

    test_metrics = inputs.model_metrics.get("test", {})
    flags = sanity_flags(test_metrics)
    if flags:
        add("\n> **Sanity warnings**\n>\n")
        for flag in flags:
            add(f"> - {flag}\n")
    else:
        add("\nNo sanity warnings: the metrics sit inside the range that is "
            "realistic for daily equity returns. A near-zero or slightly "
            "negative R-squared is the expected, honest outcome here — a high "
            "one would indicate lookahead bias rather than skill.\n")

    if inputs.feature_importance is not None and len(inputs.feature_importance):
        add("\n### Most influential features\n")
        top = inputs.feature_importance.head(12).to_frame("importance")
        top.index.name = "feature"
        add(_table(top))

    if inputs.future_correlations is not None and len(inputs.future_correlations):
        corr = inputs.future_correlations
        add("\n### Correlation with the next day's return\n")
        add(f"Strongest absolute correlation: **{corr.abs().max():.4f}** "
            f"({corr.abs().idxmax()}). Median: **{corr.abs().median():.4f}**.\n\n"
            "Values this small are the normal state of affairs for daily "
            "equity data, and they set the ceiling on what any model here can "
            "achieve. Anything above 0.3 would point to a leak.\n")

    # ---- backtest -------------------------------------------------------
    add("\n## 3. Backtest vs Buy-and-Hold\n")
    add(_table(inputs.comparison))

    stats = trade_statistics(inputs.trades)
    if stats["n_trades"]:
        add("\n### Trade statistics\n")
        add(_kv_table(stats))

    # ---- walk-forward ---------------------------------------------------
    if inputs.walkforward_table is not None and not inputs.walkforward_table.empty:
        add("\n## 4. Walk-Forward Validation\n")
        add("Every prediction below was made by a model retrained only on data "
            "preceding the period it was predicting.\n")
        add(_table(inputs.walkforward_table, index=False))
        if inputs.walkforward_stability is not None:
            add("\n**Stability across folds** — a metric that swings widely "
                "between folds is noise, not an edge.\n")
            add(_table(inputs.walkforward_stability))

    # ---- regimes --------------------------------------------------------
    if inputs.regime_breakdown:
        add("\n## 5. Regime Analysis\n")
        from .regime import regime_summary_text

        for name, frame in inputs.regime_breakdown.items():
            if frame is None or frame.empty:
                continue
            add(f"\n### By {name}\n")
            add(_table(frame))
            add(f"\n{regime_summary_text(frame)}\n")

    # ---- stress ---------------------------------------------------------
    if inputs.stress_sweeps:
        add("\n## 6. Stress Tests\n")
        from .stress import stress_verdict

        for name, frame in inputs.stress_sweeps.items():
            if frame is None or frame.empty:
                continue
            add(f"\n### {name.replace('_', ' ').title()} sweep\n")
            add(_table(frame, index=False))
        add(f"\n**Robustness assessment:** {stress_verdict(inputs.stress_sweeps)}\n")

    # ---- limitations ----------------------------------------------------
    add("\n## 7. Limitations\n")
    add(
        "- A single asset with no portfolio construction or position sizing.\n"
        "- Survivorship bias: the ticker tested is one that still exists.\n"
        "- Execution is assumed at the daily close; real fills differ and "
        "carry market impact.\n"
        "- Costs are modelled as a flat rate; real costs vary with liquidity "
        "and order size.\n"
        "- A backtest describes the past. It is a hypothesis about history, "
        "not a prediction about the future.\n"
    )

    add("\n---\n")
    add("_Generated automatically by the Stock Return Prediction & Strategy "
        "Backtesting pipeline. Not investment advice._\n")

    return "\n".join(parts)


def save_report(
    content: str,
    cfg: Config,
    model_name: str,
    directory: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write the report to ``outputs/reports/`` and return the path."""
    directory = Path(directory) if directory is not None else REPORTS
    directory.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{cfg.ticker}_{model_name}_{stamp}.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote report to %s", path)
    return path


# --------------------------------------------------------------------------
# Experiment log
# --------------------------------------------------------------------------


def log_experiment(
    cfg: Config,
    model_name: str,
    metrics: dict[str, float],
    performance: dict[str, float],
    path: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Append one run to ``outputs/experiment_log.csv``.

    Keyed by the config fingerprint, so two runs of the same configuration are
    directly comparable and any divergence is immediately visible.
    """
    from .config import EXPERIMENT_LOG

    path = Path(path) if path is not None else EXPERIMENT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "fingerprint": cfg.fingerprint(),
        "ticker": cfg.ticker,
        "model": model_name,
        "start": cfg.start,
        "end": cfg.end,
        "train_end": cfg.train_end,
        "valid_end": cfg.valid_end,
        "threshold": cfg.signal_threshold,
        "cost_bps": cfg.transaction_cost_bps,
        "slippage_bps": cfg.slippage_bps,
        "seed": cfg.seed,
        **{f"model_{k}": v for k, v in metrics.items()},
        **{f"strategy_{k}": v for k, v in performance.items()},
    }

    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def assemble_report(
    cfg: Config,
    data_summary: dict,
    split_summary: pd.DataFrame,
    model_name: str,
    model_metrics: dict[str, dict[str, float]],
    backtest,
    **extras,
) -> str:
    """Convenience wrapper: build a report straight from a BacktestResult."""
    comparison = compare_to_benchmark(
        backtest.strategy_returns,
        backtest.benchmark_returns,
        cfg.risk_free_rate,
        cfg.trading_days,
        backtest.position,
    )
    return build_report(ReportInputs(
        cfg=cfg,
        data_summary=data_summary,
        split_summary=split_summary,
        model_name=model_name,
        model_metrics=model_metrics,
        comparison=comparison,
        trades=backtest.trades,
        **extras,
    ))
