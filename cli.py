"""Command-line interface.

Makes the whole study reproducible from one command, which matters for two
reasons: results can be regenerated in CI, and a reviewer can rerun the exact
configuration behind any reported number.

    python cli.py run --ticker AAPL --model xgboost
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ensure_dirs, load_config  # noqa: E402
from src.models import available_models  # noqa: E402


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s  %(message)s",
    )
    # Wide, readable tables in a terminal.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_fetch(cfg, args) -> int:
    from src.pipeline import load

    loaded = load(cfg)
    rule(f"DATA — {cfg.ticker}  {cfg.start} to {cfg.end}")
    for key, value in loaded.summary.items():
        if isinstance(value, float):
            shown = f"{value:,.2f}"
        elif isinstance(value, int):
            shown = f"{value:,}"
        else:
            shown = str(value)
        print(f"  {key.replace('_', ' ').title():28s} {shown}")

    print(f"\nValidation: {loaded.report.summary()}")
    if loaded.report.issues:
        print(loaded.report.to_frame().to_string(index=False))
    print(f"Cleaning:   {loaded.log.summary()}")
    return 0


def cmd_features(cfg, args) -> int:
    from src.features import describe_features, future_return_correlation
    from src.pipeline import load, prepare

    prepared = prepare(cfg, load(cfg).cleaned)
    rule(f"FEATURES — {len(prepared.features)} columns, {len(prepared.data)} rows")
    print(prepared.split.summary().to_string(index=False))

    if args.describe:
        print("\n" + describe_features(prepared.data).round(4).to_string())

    corr = future_return_correlation(prepared.data)
    print("\nStrongest correlations with the next day's return:")
    print(corr.head(10).round(4).to_string())
    print(f"\n  max |corr| {corr.abs().max():.4f}   median |corr| "
          f"{corr.abs().median():.4f}")
    print("  Small values are expected and correct here; anything above 0.30 "
          "would indicate a leak.")
    return 0


def cmd_train(cfg, args) -> int:
    from src.evaluation import format_metrics, sanity_flags
    from src.models import model_path
    from src.pipeline import load, prepare, train

    prepared = prepare(cfg, load(cfg).cleaned)
    model = train(cfg, prepared, args.model)

    rule(f"MODEL — {model.name} on {cfg.ticker}")
    for part, metrics in model.metrics.items():
        print(f"  {part:6s} {format_metrics(metrics)}")

    print("\nTop features:")
    for feature, value in model.importance().head(10).items():
        print(f"  {feature:28s} {value:.4f}")

    flags = sanity_flags(model.metrics.get("test", {}))
    if flags:
        print("\nSANITY WARNINGS:")
        for flag in flags:
            print(f"  ! {flag}")
    else:
        print("\nNo sanity warnings — metrics are in the realistic range for "
              "daily equity returns.")

    if args.save:
        print(f"\nSaved model to {model.pipeline.save(model_path(cfg, model.name))}")
    return 0


def cmd_backtest(cfg, args) -> int:
    from src.metrics import compare_to_benchmark, format_summary, trade_statistics
    from src.pipeline import backtest, load, prepare, train

    prepared = prepare(cfg, load(cfg).cleaned)
    model = train(cfg, prepared, args.model)
    result = backtest(cfg, prepared, model)

    rule(f"BACKTEST — {model.name} on {cfg.ticker} "
         f"({prepared.split.test.index.min().date()} to "
         f"{prepared.split.test.index.max().date()})")

    comparison = compare_to_benchmark(
        result.strategy_returns, result.benchmark_returns,
        cfg.risk_free_rate, cfg.trading_days, result.position,
    )
    print(comparison.round(4).to_string())

    stats = trade_statistics(result.trades)
    print(f"\nTrades: {stats['n_trades']}  |  win rate "
          f"{stats['win_rate']:.1%}  |  average hold "
          f"{stats['avg_holding_days']:.1f} days")
    print(f"Total cost paid: {result.total_cost:.2%} of capital")

    if args.trades and result.n_trades:
        print("\n" + result.trades.head(20).to_string(index=False))
    return 0


def cmd_walkforward(cfg, args) -> int:
    from src.metrics import compare_to_benchmark
    from src.pipeline import load, prepare
    from src.walkforward import walk_forward_backtest

    prepared = prepare(cfg, load(cfg).cleaned)
    result, bt = walk_forward_backtest(
        prepared.data, cfg, prepared.features, args.model,
        n_folds=args.folds, mode=args.mode,
    )

    rule(f"WALK-FORWARD — {result.model_name}, {result.n_folds} {result.mode} folds")
    print(result.fold_table().round(4).to_string(index=False))

    print("\nStability across folds:")
    print(result.stability().round(4).to_string())

    overall = result.overall_metrics()
    print(f"\nStitched out-of-sample: n={overall['n']}  "
          f"R2={overall['r2']:.4f}  "
          f"directional accuracy={overall['directional_accuracy']:.4f}")

    print("\nBacktest of the stitched predictions:")
    print(compare_to_benchmark(
        bt.strategy_returns, bt.benchmark_returns, cfg.risk_free_rate,
        cfg.trading_days, bt.position,
    ).round(4).to_string())
    return 0


def cmd_compare(cfg, args) -> int:
    from src.pipeline import compare_models, load, prepare

    prepared = prepare(cfg, load(cfg).cleaned)
    table, _, _ = compare_models(cfg, prepared)

    rule(f"MODEL COMPARISON — {cfg.ticker} test period")
    print(table.round(4).to_string())
    print("\nEvery model saw identical splits, features and seed, so any "
          "difference is the model itself.")
    return 0


def cmd_stress(cfg, args) -> int:
    from src.dataset import realised_returns
    from src.metrics import total_return
    from src.pipeline import load, prepare, train
    from src.stress import (
        break_even_cost, stress_verdict, sweep_costs, sweep_periods,
        sweep_thresholds, sweep_tickers,
    )

    prepared = prepare(cfg, load(cfg).cleaned)
    model = train(cfg, prepared, args.model)
    test = prepared.split.test
    returns = realised_returns(test)
    preds = model.predictions["test"]
    benchmark = total_return(returns)

    sweeps: dict[str, pd.DataFrame] = {}
    wanted = args.sweep

    if wanted in ("cost", "all"):
        sweeps["cost"] = sweep_costs(returns, preds, cfg)
    if wanted in ("threshold", "all"):
        sweeps["threshold"] = sweep_thresholds(returns, preds, cfg)
    if wanted in ("period", "all"):
        sweeps["period"] = sweep_periods(returns, preds, cfg, 4)
    if wanted in ("ticker", "all"):
        sweeps["ticker"] = sweep_tickers(cfg, model_name=args.model)

    rule(f"STRESS TESTS — {cfg.ticker}  (buy-and-hold {benchmark:.2%})")
    for name, frame in sweeps.items():
        print(f"\n{name.upper()} SWEEP")
        print(frame.round(4).to_string(index=False))
        if name == "cost":
            be = break_even_cost(frame, benchmark)
            print(f"  break-even cost: "
                  f"{'never beats buy-and-hold' if pd.isna(be) else f'{be:.0f} bps'}")

    print(f"\nVERDICT: {stress_verdict(sweeps)}")
    return 0


def cmd_report(cfg, args) -> int:
    from src.pipeline import run_full_study
    from src.report import log_experiment, save_report

    rule(f"FULL STUDY — {cfg.ticker}")
    print("Running: load -> features -> train -> backtest -> walk-forward -> "
          "regimes -> stress -> report ...")

    study = run_full_study(cfg, args.model)
    path = save_report(study["report"], cfg, args.model or cfg.model)

    from src.metrics import performance_summary

    log_experiment(
        cfg, args.model or cfg.model,
        study["model"].metrics.get("test", {}),
        performance_summary(study["backtest"].strategy_returns),
    )

    verdict_line = next(
        (line for line in study["report"].splitlines() if line.startswith("### ")),
        "",
    )
    print(f"\n{verdict_line.replace('### ', 'VERDICT: ')}")
    print(f"Report written to {path}")
    return 0


def cmd_run(cfg, args) -> int:
    """Everything, then the headline numbers."""
    from src.metrics import compare_to_benchmark

    code = cmd_report(cfg, args)
    from src.pipeline import backtest, load, prepare, train

    prepared = prepare(cfg, load(cfg).cleaned)
    model = train(cfg, prepared, args.model)
    result = backtest(cfg, prepared, model)

    rule("HEADLINE")
    comparison = compare_to_benchmark(
        result.strategy_returns, result.benchmark_returns,
        cfg.risk_free_rate, cfg.trading_days, result.position,
    )
    print(comparison.loc[
        ["total_return", "cagr", "sharpe_ratio", "max_drawdown"]
    ].round(4).to_string())
    return code


COMMANDS = {
    "fetch": cmd_fetch,
    "features": cmd_features,
    "train": cmd_train,
    "backtest": cmd_backtest,
    "walkforward": cmd_walkforward,
    "compare": cmd_compare,
    "stress": cmd_stress,
    "report": cmd_report,
    "run": cmd_run,
}


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Stock return prediction and strategy backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py fetch --ticker MSFT\n"
            "  python cli.py train --model random_forest --save\n"
            "  python cli.py backtest --cost-bps 10 --trades\n"
            "  python cli.py walkforward --folds 6 --mode rolling\n"
            "  python cli.py stress --sweep all\n"
            "  python cli.py run --ticker AAPL --model xgboost\n"
        ),
    )
    parser.add_argument("command", choices=sorted(COMMANDS))

    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-end", dest="train_end", default=None)
    parser.add_argument("--valid-end", dest="valid_end", default=None)
    parser.add_argument("--model", default=None, choices=available_models())
    parser.add_argument("--threshold", dest="signal_threshold", type=float, default=None)
    parser.add_argument("--cost-bps", dest="transaction_cost_bps", type=float, default=None)
    parser.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=None)
    parser.add_argument("--allow-short", dest="allow_short", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", default=None)

    parser.add_argument("--folds", type=int, default=None, help="walk-forward folds")
    parser.add_argument("--mode", default=None, choices=["expanding", "rolling"])
    parser.add_argument("--sweep", default="all",
                        choices=["cost", "threshold", "period", "ticker", "all"])
    parser.add_argument("--describe", action="store_true", help="full feature statistics")
    parser.add_argument("--trades", action="store_true", help="print the trade log")
    parser.add_argument("--save", action="store_true", help="persist the trained model")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


CONFIG_FIELDS = (
    "ticker", "start", "end", "train_end", "valid_end", "model",
    "signal_threshold", "transaction_cost_bps", "slippage_bps",
    "allow_short", "seed", "use_cache",
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    ensure_dirs()

    # Only flags the user actually supplied override config.yaml; Config.replace
    # ignores None, so unset flags leave the file's values alone.
    overrides = {f: getattr(args, f, None) for f in CONFIG_FIELDS}

    try:
        cfg = load_config(args.config, **overrides)
    except (ValueError, KeyError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.folds is not None:
        cfg.walk_forward.n_folds = args.folds
    if args.mode is not None:
        cfg.walk_forward.mode = args.mode

    try:
        return COMMANDS[args.command](cfg, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - the CLI is the top-level boundary
        logging.getLogger(__name__).exception("Command failed")
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
