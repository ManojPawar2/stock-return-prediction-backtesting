# Stock Return Prediction & Strategy Backtesting

An end-to-end quantitative research system that asks one honest question:

> **Can historical price and volume data be turned into a predictive model whose trading signals beat a simple buy-and-hold baseline on unseen historical periods — after costs?**

The project is built as a **research pipeline**, not a get-rich system. It is deliberately engineered so that the answer is allowed to be *"no"*. A backtest that says "no" but is leakage-free is worth more than a backtest that says "+400%" and is quietly cheating.

**Stack:** Python · Pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Plotly · Streamlit · pytest

---

## Table of Contents

1. [What This Project Actually Does](#1-what-this-project-actually-does)
2. [Quick Start](#2-quick-start)
3. [Pipeline Overview](#3-pipeline-overview)
4. [The Leakage Contract (read this first)](#4-the-leakage-contract-read-this-first)
5. [Module Reference](#5-module-reference)
6. [Data Layer](#6-data-layer)
7. [Cleaning & Validation](#7-cleaning--validation)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Feature Engineering](#9-feature-engineering)
10. [Target Definition](#10-target-definition)
11. [Models](#11-models)
12. [Time-Based Validation & Walk-Forward](#12-time-based-validation--walk-forward)
13. [Model Evaluation](#13-model-evaluation)
14. [Signal Generation](#14-signal-generation)
15. [Backtesting Engine](#15-backtesting-engine)
16. [Performance Metrics (with formulas)](#16-performance-metrics-with-formulas)
17. [Regime Analysis](#17-regime-analysis)
18. [Stress Testing](#18-stress-testing)
19. [Streamlit Dashboard](#19-streamlit-dashboard)
20. [Command-Line Interface](#20-command-line-interface)
21. [Configuration & Reproducibility](#21-configuration--reproducibility)
22. [Testing Strategy](#22-testing-strategy)
23. [Project Structure](#23-project-structure)
24. [Build Roadmap](#24-build-roadmap)
25. [Honest Expectations & Limitations](#25-honest-expectations--limitations)
26. [What NOT To Do](#26-what-not-to-do)
27. [Interview Preparation](#27-interview-preparation)
28. [Resume Description](#28-resume-description)
29. [Glossary](#29-glossary)

---

## 1. What This Project Actually Does

The system takes a stock ticker and a date range, and runs a full quantitative research loop:

| Stage | What happens | Why it matters |
|-------|--------------|----------------|
| **Ingest** | Pull daily OHLCV from Yahoo Finance, cache to Parquet | Reproducible, offline-capable runs |
| **Validate** | Missing values, duplicates, date monotonicity, non-positive prices, OHLC consistency, gaps | Bad data silently ruins every downstream number |
| **Engineer** | 43 features from lagged returns, moving averages, volatility, volume, RSI/MACD/Bollinger | Every feature uses **only past data** |
| **Target** | Next-day return (regression) or up/down (classification) | Explicitly documented, no ambiguity |
| **Split** | Chronological train / validation / test — never random | Random splits leak the future into the past |
| **Train** | Linear Regression, Ridge, Random Forest, XGBoost | Baseline first, complexity second |
| **Evaluate** | MAE, RMSE, R², directional accuracy, hit rate | ML metrics and trading metrics are *different things* |
| **Signal** | Threshold prediction → position (long-only or long/short) | One clear, explainable rule |
| **Backtest** | Vectorised engine with transaction costs and slippage | Costs are what kill most naive strategies |
| **Compare** | Strategy vs buy-and-hold on equity, drawdown, Sharpe | Without a baseline, any number is meaningless |
| **Stress** | Sweep costs, thresholds, tickers, periods | Robust != lucky on one configuration |
| **Report** | Auto-generated Markdown research report | Conclusions written down, not remembered |

Everything is exposed through both a **Streamlit dashboard** and a **CLI**.

---

## 2. Quick Start

### Requirements
- Python 3.10 or newer
- Internet access for the first data download (afterwards, cached Parquet files are used)

### Install

```bash
git clone https://github.com/ManojPawar2/stock-return-prediction-backtesting.git
cd stock-return-prediction-backtesting

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Run the whole pipeline headless

```bash
python cli.py run --ticker AAPL --start 2015-01-01 --end 2024-12-31 --model xgboost
```

### Run the tests

```bash
pytest -q
```

---

## 3. Pipeline Overview

```text
                     ┌──────────────────────────┐
                     │  Yahoo Finance (yfinance)│
                     │  daily OHLCV             │
                     └────────────┬─────────────┘
                                  │  cached -> data/raw/*.parquet
                                  v
                     ┌──────────────────────────┐
                     │  Validation & Cleaning   │
                     │  missing · dupes · gaps  │
                     │  OHLC sanity · ordering  │
                     └────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    v                           v
        ┌──────────────────────┐    ┌──────────────────────┐
        │ EDA & Visualisation  │    │ Feature Engineering  │
        │ returns · volatility │    │ lags · MA · vol      │
        │ volume · correlation │    │ volume · RSI/MACD/BB │
        └──────────────────────┘    └──────────┬───────────┘
                                               │
                                               v
                                  ┌──────────────────────────┐
                                  │ Target = next-day return │
                                  │ Chronological split      │
                                  │ TRAIN | VALID | TEST     │
                                  └────────────┬─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    v                          v                          v
            Linear / Ridge            Random Forest                  XGBoost
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               v
                                  ┌──────────────────────────┐
                                  │ Evaluation               │
                                  │ MAE RMSE R2 DirAcc       │
                                  └────────────┬─────────────┘
                                               v
                                  ┌──────────────────────────┐
                                  │ Signal: pred > threshold │
                                  │        -> LONG / FLAT    │
                                  └────────────┬─────────────┘
                                               v
                                  ┌──────────────────────────┐
                                  │ Backtester               │
                                  │ position lag · costs     │
                                  │ slippage · trade log     │
                                  └────────────┬─────────────┘
                                               v
                    ┌──────────────────────────┼──────────────────────────┐
                    v                          v                          v
            Strategy metrics          Buy & Hold baseline          Trade statistics
                    └──────────────────────────┼──────────────────────────┘
                                               v
                        Walk-Forward · Regime Analysis · Stress Tests
                                               v
                          Streamlit Dashboard + Markdown Research Report
```

---

## 4. The Leakage Contract (read this first)

This is the single most important section of the project. **Every design decision below exists to prevent lookahead bias.**

### The timeline

```text
       day t-1            day t                day t+1
  ────────┼─────────────────┼────────────────────┼────────
          │                 │                    │
          │            CLOSE(t) known            │
          │                 │                    │
          │        1. build features X_t         │
          │           from data <= t             │
          │                 │                    │
          │        2. model predicts r_{t+1}     │
          │                 │                    │
          │        3. decide position for t+1    │
          │                 ├───────────────────>│
          │                                      │
          │                            4. earn r_{t+1}
          │                               = C(t+1)/C(t) - 1
```

### The three rules, and how they are enforced in code

**Rule 1 — Features may only use information available at or before the close of day `t`.**
Every rolling window is backward-looking: `df["close"].rolling(20).mean()` at index `t` covers `t-19 ... t`. No `center=True`. No `.shift(-n)` anywhere in `features.py`. A unit test asserts that recomputing a feature on a truncated series `df[:t]` gives the identical value at `t`.

**Rule 2 — The target is strictly in the future.**

```python
target[t] = close[t+1] / close[t] - 1     # implemented as close.pct_change().shift(-1)
```

The final row has no target and is dropped from training.

**Rule 3 — A prediction made at close of day `t` can only affect the position held on day `t+1`.**

```python
position = signal.shift(1)                    # the one-bar execution lag
strategy_return = position * daily_return     # both indexed at t+1
```

Forgetting that single `.shift(1)` is the most common way backtests become fantasy. A dedicated test (`test_backtester.py::test_no_lookahead`) feeds the engine a perfect oracle signal and asserts that removing the shift changes the result — proving the lag is actually load-bearing.

### Other leakage sources this project closes

| Leak | How it is avoided |
|------|-------------------|
| Scaling on the full dataset | `StandardScaler` is fit on **train only**, then applied to valid/test |
| Random `train_test_split` | Never used. Splits are chronological by date |
| Tuning on the test set | Hyperparameters selected on the **validation** window; test touched once |
| Dropping NaNs after splitting | Warm-up NaNs from rolling windows are dropped **before** splitting so each split is contiguous |
| Survivorship bias | Acknowledged as a known limitation (single-ticker, still-listed stocks) — see section 25 |
| Adjusted vs raw prices | Returns computed from **Adjusted Close** so splits/dividends do not create fake jumps |

---

## 5. Module Reference

Each module has one job and is independently testable.

### `src/config.py`
Typed configuration loaded from `config.yaml`, overridable from CLI and Streamlit.

```python
@dataclass
class Config:
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    train_end: str = "2021-12-31"
    valid_end: str = "2022-12-31"
    target_horizon: int = 1
    signal_threshold: float = 0.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    allow_short: bool = False
    seed: int = 42
```

### `src/data_loader.py`
```python
load_prices(ticker, start, end, use_cache=True) -> pd.DataFrame
```
Downloads OHLCV via yfinance, flattens yfinance's MultiIndex columns, normalises to lowercase `open/high/low/close/adj_close/volume` with a `DatetimeIndex` named `date`, and caches to `data/raw/{ticker}_{start}_{end}.parquet`.

### `src/preprocessing.py`
```python
validate(df) -> ValidationReport      # non-destructive: reports problems
clean(df)    -> pd.DataFrame          # sorts, de-duplicates, forward-fills gaps, drops invalid rows
```

### `src/features.py`
```python
build_features(df, config) -> pd.DataFrame    # 43 columns, all backward-looking
feature_columns(df) -> list[str]
```

### `src/indicators.py`
`rsi`, `macd`, `bollinger_bands`, `atr`, `obv` — pure functions on a price Series.

### `src/dataset.py`
```python
make_target(df, horizon=1, kind="regression") -> pd.Series
chronological_split(df, train_end, valid_end) -> (train, valid, test)
```

### `src/models.py`
```python
build_model(name, params, seed) -> sklearn-compatible estimator
ModelPipeline.fit(X, y)     # StandardScaler (train-only) + estimator
ModelPipeline.predict(X)
ModelPipeline.feature_importance() -> pd.Series
save(path) / load(path)     # joblib persistence
```

### `src/evaluation.py`
```python
regression_metrics(y_true, y_pred) -> dict     # mae, rmse, r2, directional_accuracy
classification_metrics(y_true, y_pred, y_prob) -> dict
```

### `src/walkforward.py`
```python
walk_forward(df, config, model_name, n_folds, mode="expanding") -> WalkForwardResult
```
Retrains on each fold and stitches out-of-sample predictions into one continuous series.

### `src/signals.py`
```python
generate_signals(predictions, threshold, allow_short) -> pd.Series   # +1 / 0 / -1
```

### `src/backtester.py`
```python
Backtester(cost_bps, slippage_bps).run(returns, signals) -> BacktestResult
```
Returns equity curve, positions, per-day strategy returns, turnover, and a trade log.

### `src/metrics.py`
`total_return`, `cagr`, `annualised_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `summary_table`.

### `src/regime.py`
Labels each day as high/low volatility and bull/bear, then reports strategy performance per regime.

### `src/stress.py`
Parameter sweeps over cost, threshold, ticker, and date range.

### `src/report.py`
Renders a Markdown research report into `outputs/reports/`.

### `src/plots.py`
Plotly figures shared by the dashboard: price, equity curve, drawdown, feature importance, correlation heatmap, return distribution.

---

## 6. Data Layer

### Schema

| Column | Type | Meaning |
|--------|------|---------|
| `date` (index) | datetime64 | Trading day |
| `open` | float | First traded price |
| `high` | float | Session high |
| `low` | float | Session low |
| `close` | float | Last traded price |
| `adj_close` | float | Close adjusted for splits & dividends — **all returns use this** |
| `volume` | int | Shares traded |

### Caching

The first call downloads and writes `data/raw/{TICKER}_{start}_{end}.parquet`. Later calls read the Parquet file, so the project runs offline and produces identical results.

### Multiple tickers

`load_many(tickers, start, end)` returns a dict of DataFrames, used by the multi-stock comparison page and the stress tester.

---

## 7. Cleaning & Validation

`validate()` returns a report — it never silently mutates data.

| Check | Failure condition |
|-------|-------------------|
| Missing values | Any NaN in OHLCV |
| Duplicate dates | Repeated index entries |
| Date ordering | Index not monotonically increasing |
| Non-positive prices | `close <= 0` or `volume < 0` |
| OHLC consistency | `high < low`, or close outside `[low, high]` |
| Infinite values | `np.isinf` anywhere |
| Calendar gaps | Missing business days beyond market holidays |
| Extreme moves | Daily return beyond +/-50% (flagged, not deleted — real crashes exist) |

`clean()` then sorts by date, drops duplicate dates keeping the last, forward-fills short gaps in price (never in volume), and drops rows that remain invalid. Every action is logged so the dashboard can show exactly what was changed.

---

## 8. Exploratory Data Analysis

**Price** — adjusted close with 20/50/200-day moving averages; daily and cumulative returns.

**Returns** — histogram with a fitted normal overlay (markets have fat tails: the histogram should visibly disagree with the normal curve), skewness, excess kurtosis, positive vs negative day counts.

**Volatility** — 20-day rolling annualised volatility; volatility clustering is visible and is a real, exploitable statistical property.

**Volume** — raw volume, 20-day average, and volume/average ratio to spot unusual activity.

**Relationships** — feature correlation heatmap, and the correlation of each feature with the *future* return. Expect these last correlations to be small (|rho| ~ 0.0-0.1). **This is the honest headline of the project:** daily returns are close to unpredictable, and any model must fight for a very thin edge.

---

## 9. Feature Engineering

All features are computed from data up to and including day `t`.

### Return features
`ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, and lagged one-day returns `ret_lag_1 ... ret_lag_5`.

### Moving-average features
`sma_5`, `sma_10`, `sma_20`, `sma_50`, plus **ratios** `close_sma_5_ratio`, `close_sma_20_ratio`, `sma_5_20_ratio`. Ratios rather than raw levels — a raw moving average is non-stationary and lets a tree model memorise price levels instead of learning behaviour.

### Volatility features
`vol_5`, `vol_20`, `vol_60` (rolling std of daily returns, annualised by x sqrt(252)), and `vol_ratio_5_20` as a volatility-regime proxy.

### Volume features
`volume_change`, `volume_sma_20`, `volume_ratio`, `dollar_volume`.

### Range features
`high_low_range` = `(high - low) / close`, `close_position` = `(close - low) / (high - low)`.

### Technical indicators
- **RSI(14)** — `100 - 100/(1 + avg_gain/avg_loss)` over a 14-day Wilder average.
- **MACD** — `EMA(12) - EMA(26)`, signal `EMA(9)` of MACD, plus histogram.
- **Bollinger Bands(20, 2 sigma)** — upper, lower, `%B` position, and bandwidth.
- **ATR(14)** — average true range, normalised by close.

### Calendar features
`day_of_week`, `month` (one-hot). Included to test — and usually reject — calendar effects.

### Stationarity note
Features are ratios, returns, or bounded oscillators. No raw price level is ever fed to a model, because a model trained on 2015 prices would be extrapolating on 2024 prices.

---

## 10. Target Definition

**Primary (regression):**
```python
y[t] = adj_close[t + h] / adj_close[t] - 1        # h = target_horizon, default 1
```

**Secondary (classification):**
```python
y[t] = 1 if adj_close[t+1] > adj_close[t] else 0
```

The classification variant reports ROC-AUC and precision/recall, and is useful for showing that ~52% accuracy can still be a meaningful edge — while ~50% is noise.

---

## 11. Models

| Model | Why it is here |
|-------|----------------|
| **Linear Regression** | The honest baseline. If a complex model cannot beat it, the complexity is not earning its keep |
| **Ridge** | Same, with L2 regularisation — financial features are highly collinear |
| **Random Forest** | Captures non-linear interactions; gives permutation-based feature importance |
| **XGBoost** | Strong gradient-boosted trees; the standard "does more capacity help?" test |

All models share one interface (`ModelPipeline`) so they are swapped by name and compared on identical splits, identical features, and an identical random seed.

### Guardrails
- Trees are shallow (`max_depth` 3-6) and regularised. On daily financial data, a deep unconstrained tree memorises noise perfectly and generalises at chance.
- Feature scaling is fit on train only.
- `n_jobs` and `random_state` fixed for reproducibility.

---

## 12. Time-Based Validation & Walk-Forward

### Fixed chronological split

```text
2015 ─────────────────────── 2021 │ 2022 │ 2023 ────────── 2024
            TRAIN                 │ VALID│        TEST
        model learns here         │ tune │   touched once
```

### Walk-forward validation (the stronger method)

```text
fold 1: TRAIN[2015-2019] -> TEST[2020]
fold 2: TRAIN[2015-2020] -> TEST[2021]      <- expanding window
fold 3: TRAIN[2015-2021] -> TEST[2022]
fold 4: TRAIN[2015-2022] -> TEST[2023]
fold 5: TRAIN[2015-2023] -> TEST[2024]

out-of-sample predictions from every fold are concatenated
into one continuous series and backtested end-to-end
```

Both **expanding** (all history) and **rolling** (fixed-length window) modes are supported. Walk-forward is the closest offline approximation of how a strategy would actually have been run, and it produces far more out-of-sample data than a single split.

---

## 13. Model Evaluation

### Regression
- **MAE** — average absolute error, in return units
- **RMSE** — penalises large misses
- **R²** — *expect this to be near zero or slightly negative on daily returns.* A high R² on daily stock returns almost always means leakage, not skill
- **Directional accuracy** — fraction of days where `sign(pred) == sign(actual)`; the metric that actually matters for a long/flat strategy

### Classification
Accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

### The critical point
A model can have the best RMSE and the worst strategy, because RMSE is dominated by large moves the strategy may never trade. **Both** the statistical metrics and the backtest metrics are always reported side by side, and model selection is done on validation-period strategy performance, not on RMSE alone.

---

## 14. Signal Generation

```python
if prediction >  threshold:  signal = +1   # long
elif prediction < -threshold and allow_short:  signal = -1   # short
else: signal = 0                            # flat
```

Default is **long-only** (`allow_short = False`) because it is simpler to explain and to justify. The threshold is a deliberate friction control: `threshold = 0` trades constantly and pays maximum cost; a higher threshold trades only on stronger convictions. The stress-test page sweeps it to show how sensitive results are to this one number.

---

## 15. Backtesting Engine

Vectorised, with an explicit execution lag.

```python
position   = signal.shift(1).fillna(0)                # decided at t, held at t+1
turnover   = position.diff().abs().fillna(position.abs())
cost       = turnover * (cost_bps + slippage_bps) / 10_000
gross      = position * daily_return
net        = gross - cost
equity     = (1 + net).cumprod()
```

### Capabilities
- Position tracking (long / flat / short)
- Per-trade cost **and** slippage, charged on turnover only
- Full trade log: entry date, exit date, holding period, entry/exit price, trade P&L
- Buy-and-hold baseline computed on the identical date range
- Drawdown series, equity curve, daily returns
- Trade counts, win rate, average win/loss, profit factor

### Why costs are non-negotiable
A signal that flips daily at 5 bps round-trip pays roughly `252 x 0.0005 = 12.6%` per year in friction. Most "profitable" naive backtests are simply strategies that forgot to pay for their own trading.

---

## 16. Performance Metrics (with formulas)

Let `r_t` be daily strategy returns, `N = 252` trading days per year.

| Metric | Formula |
|--------|---------|
| Total return | `prod(1 + r_t) - 1` |
| CAGR | `(final_equity)^(N / n_days) - 1` |
| Annualised volatility | `std(r) * sqrt(N)` |
| Sharpe ratio | `(mean(r) - rf/N) / std(r) * sqrt(N)` |
| Sortino ratio | `(mean(r) - rf/N) / std(r[r<0]) * sqrt(N)` |
| Max drawdown | `min(equity / cummax(equity) - 1)` |
| Calmar ratio | `CAGR / abs(max drawdown)` |
| Win rate | `count(r > 0) / count(r != 0)` |
| Profit factor | `sum(r[r>0]) / abs(sum(r[r<0]))` |
| Exposure | fraction of days with a non-zero position |

Every strategy metric is reported next to the same metric for buy-and-hold. A Sharpe of 0.8 means nothing until you know the benchmark scored 0.9.

---

## 17. Regime Analysis

Days are labelled by 20-day realised volatility (top/bottom tercile -> high/low vol) and by trend (price above/below its 200-day moving average -> bull/bear). The strategy is then scored **within each regime**.

The question this answers: *is the edge real, or is it one lucky regime?* A strategy that only works in 2020's crash is a strategy that works once.

---

## 18. Stress Testing

Sweeps, each rendered as a table plus a sensitivity chart:

- **Transaction cost:** 0, 1, 5, 10, 20, 50 bps — at what cost does the edge vanish?
- **Signal threshold:** 0 to 0.005 — is a single threshold value doing all the work?
- **Ticker:** the same pipeline across a basket of stocks — does it generalise?
- **Date range:** rolling multi-year sub-periods — is performance stable over time?
- **Seed:** several random seeds for tree models — is the result stable or noise?

A strategy that survives only one cell of this grid has not been validated; it has been curve-fitted.

---

## 19. Streamlit Dashboard

Multi-page app. Configuration lives in the sidebar and is shared across pages through `st.session_state`.

| Page | Contents |
|------|----------|
| **Overview** (`app.py`) | Ticker/date selection, dataset summary, validation report, quick price chart, project explanation |
| **1 · Market Data** | Candlestick + volume, raw data table, download to CSV, cache status |
| **2 · EDA** | Returns distribution, rolling volatility, cumulative returns, volume analysis, correlation heatmap, feature vs future-return correlations |
| **3 · Feature Lab** | Full feature table, per-feature time series, distributions, leakage self-check panel |
| **4 · Modeling** | Model picker, hyperparameters, split dates, train button, metrics for train/valid/test, feature importance, predicted vs actual scatter |
| **5 · Backtest** | Equity curve vs buy-and-hold, drawdown, trade markers, trade log, full metrics table, cost/threshold controls |
| **6 · Comparison** | All models trained on identical data — RMSE, MAE, directional accuracy, strategy return, Sharpe, max drawdown in one table |
| **7 · Walk-Forward** | Fold configuration, per-fold metrics, stitched out-of-sample equity curve |
| **8 · Stress Test** | Cost / threshold / ticker / period sweeps with sensitivity charts |
| **9 · Research Report** | One-click auto-generated Markdown report with the honest conclusion, downloadable |

Caching uses `@st.cache_data` on data loading and feature building so page navigation is instant.

---

## 20. Command-Line Interface

```bash
python cli.py fetch     --ticker AAPL --start 2015-01-01 --end 2024-12-31
python cli.py features  --ticker AAPL
python cli.py train     --ticker AAPL --model xgboost
python cli.py backtest  --ticker AAPL --model xgboost --cost-bps 5
python cli.py walkforward --ticker AAPL --model random_forest --folds 5
python cli.py compare   --ticker AAPL
python cli.py stress    --ticker AAPL --sweep cost
python cli.py report    --ticker AAPL --model xgboost
python cli.py run       --ticker AAPL                      # everything, end to end
```

The CLI makes results reproducible in CI and lets the whole study be re-run with one command.

---

## 21. Configuration & Reproducibility

`config.yaml` holds every knob — ticker, dates, split boundaries, feature windows, model hyperparameters, costs, threshold, seed. CLI flags and the Streamlit sidebar override it; nothing is hard-coded in module logic.

Reproducibility guarantees:
- One global `seed` propagated to NumPy and every estimator
- Data cached to Parquet, so results do not change when Yahoo revises history
- Trained models persisted via joblib to `outputs/models/`
- Each run appends a row to `outputs/experiment_log.csv` with config hash, metrics, and timestamp
- `requirements.txt` pins the stack

Running the same config twice produces byte-identical metrics.

---

## 22. Testing Strategy

`pytest` suite covering the parts where a silent bug would invalidate every conclusion:

| Test file | What it proves |
|-----------|----------------|
| `test_data_loader.py` | Schema normalisation, caching round-trip, MultiIndex flattening |
| `test_preprocessing.py` | Every validation check fires on synthetic bad data |
| `test_features.py` | **No lookahead:** feature at `t` computed on `df[:t]` equals the value from the full frame. Known-value checks for RSI, MACD, Bollinger |
| `test_dataset.py` | Target alignment (`y[t]` really is `t+1`'s return), splits are contiguous and non-overlapping |
| `test_backtester.py` | Execution lag is load-bearing; costs reduce returns; a constant long signal reproduces buy-and-hold exactly |
| `test_metrics.py` | Closed-form checks — a known equity curve gives a known Sharpe, CAGR, max drawdown |
| `test_signals.py` | Threshold logic, long-only vs long/short |

The buy-and-hold equivalence test is the strongest one: with a permanently-long signal and zero costs, the engine must reproduce buy-and-hold to floating-point precision. If it does not, the engine is wrong.

---

## 23. Project Structure

```text
stock-return-prediction-backtesting/
│
├── app.py                       # Streamlit entry point (Overview page)
├── cli.py                       # Command-line interface
├── config.yaml                  # Default configuration
├── requirements.txt
├── README.md
├── INTERVIEW_PREP.md            # How to explain this project
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Typed config + YAML loading
│   ├── data_loader.py           # yfinance + Parquet cache
│   ├── preprocessing.py         # Validation + cleaning
│   ├── indicators.py            # RSI, MACD, Bollinger, ATR, OBV
│   ├── features.py              # Feature matrix construction
│   ├── dataset.py               # Target + chronological splits
│   ├── models.py                # Model factory + pipeline + persistence
│   ├── evaluation.py            # Regression/classification metrics
│   ├── walkforward.py           # Expanding / rolling walk-forward
│   ├── signals.py               # Prediction -> position
│   ├── backtester.py            # Vectorised engine + trade log
│   ├── metrics.py               # Performance & risk metrics
│   ├── regime.py                # Volatility / trend regime analysis
│   ├── stress.py                # Parameter sweeps
│   ├── report.py                # Markdown research report
│   └── plots.py                 # Shared Plotly figures
│
├── pages/
│   ├── 1_Market_Data.py
│   ├── 2_EDA.py
│   ├── 3_Feature_Lab.py
│   ├── 4_Modeling.py
│   ├── 5_Backtest.py
│   ├── 6_Comparison.py
│   ├── 7_Walk_Forward.py
│   ├── 8_Stress_Test.py
│   └── 9_Research_Report.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_dataset.py
│   ├── test_signals.py
│   ├── test_backtester.py
│   └── test_metrics.py
│
├── data/
│   ├── raw/                     # Cached Parquet (gitignored)
│   └── processed/
│
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── reports/
│   └── experiment_log.csv
│
└── docs/
    └── original-spec.md         # The original project specification
```

---

## 24. Build Roadmap

All features are complete. Each was built and verified in turn, and committed only after its tests passed.

**Status: 20/20 complete · 459 tests passing · ~10,600 lines**

- [x] **F01** Project scaffold, config system, requirements, gitignore
- [x] **F02** Data loader with Parquet caching and schema normalisation
- [x] **F03** Validation and cleaning layer
- [x] **F04** Technical indicators (RSI, MACD, Bollinger, ATR, OBV)
- [x] **F05** Feature engineering with leakage tests
- [x] **F06** Target construction and chronological splits
- [x] **F07** Model factory, training pipeline, persistence
- [x] **F08** Evaluation metrics
- [x] **F09** Signal generation
- [x] **F10** Backtesting engine with costs, slippage, trade log
- [x] **F11** Performance and risk metrics
- [x] **F12** Walk-forward validation
- [x] **F13** Regime analysis
- [x] **F14** Stress testing
- [x] **F15** Research report generator
- [x] **F16** Plotting layer
- [x] **F17** CLI
- [x] **F18** Streamlit app + all 10 pages
- [x] **F19** Full test suite green
- [x] **F20** Interview preparation document

### What verification actually caught

Each feature was checked against real market data rather than only against
fixtures, which surfaced five genuine defects that unit tests alone would have
missed:

| Defect | Symptom | Cause |
|---|---|---|
| Adjusted/raw price mix | Stochastic %K read −296 on a 0–100 scale; ATR read 5.9% of price | Yahoo adjusts only the close; AAPL's 4:1 split then created a fake gap |
| Non-canonical RSI | Values diverged from Wilder's definition over the first ~40 bars | `ewm(adjust=False)` seeds with the first observation, not the SMA of the first *n* |
| Sharpe of a constant series | Returned 2.4 × 10¹⁶ instead of "undefined" | `sd == 0` never fires — the float std of a repeated constant is ~1e-19 |
| Drawdown ignored day one | A 10% fall on the first day reported zero drawdown | The running peak started at day one's equity instead of at initial capital |
| Parquet cache mismatch | A cached frame compared unequal to a fresh one | `DatetimeIndex.freq` does not survive a Parquet round-trip |

---

## 25. Honest Expectations & Limitations

### What you should expect to see

Daily equity returns are close to a random walk. Realistically:

- **R² on next-day returns:** approximately -0.05 to +0.02. Near zero is the correct, honest result
- **Directional accuracy:** 49-54%. Anything above 60% on daily data should be treated as a bug until proven otherwise
- **After 5 bps costs:** many configurations underperform buy-and-hold, especially in a strong bull market where being flat any day is expensive

**This is a legitimate finding, not a failed project.** Demonstrating that a rigorously-built model finds only a marginal edge is exactly what real quantitative research produces, and it is far more credible than a dashboard claiming a 5.0 Sharpe.

### Known limitations

1. **Single-asset, long-only by default** — no portfolio construction or position sizing
2. **Survivorship bias** — testing on a currently-listed stock ignores companies that failed
3. **Daily close execution assumed** — real fills happen at auction prices with market impact
4. **Fixed cost model** — real costs vary with liquidity, size, and volatility
5. **No corporate-action edge cases** beyond what adjusted close already handles
6. **Backtest != future** — a historical result is a hypothesis about the past, not a prediction
7. **Data source** — Yahoo Finance is free and occasionally revises history; caching mitigates but does not eliminate this

---

## 26. What NOT To Do

- Do **not** randomly split time-series data
- Do **not** use future prices to build present features
- Do **not** evaluate only on the training period
- Do **not** tune hyperparameters on the test period
- Do **not** remove transaction costs to make results look better
- Do **not** pick a model purely on the highest historical return
- Do **not** cherry-pick a favourable ticker or date range
- Do **not** claim the strategy will be profitable in the future

The goal is credible quantitative research practice, not a manufactured backtest.

---

## 27. Interview Preparation

See **[INTERVIEW_PREP.md](INTERVIEW_PREP.md)** for a plain-English explanation of the project, a walkthrough of how every piece works, and a full bank of interview questions with answers — covering data handling, leakage, model choice, validation design, signal construction, backtesting mechanics, risk metrics, and the "why isn't the return higher?" question.

---

## 28. Resume Description

**Stock Return Prediction & Strategy Backtesting** — *Python, Pandas, NumPy, scikit-learn, XGBoost, Streamlit*

- Built an end-to-end quantitative research pipeline over 10 years of daily OHLCV data, engineering 43 leakage-free features (lagged returns, volatility, volume, RSI/MACD/Bollinger) with automated tests proving no lookahead bias.
- Compared Linear Regression, Ridge, Random Forest and XGBoost under chronological and walk-forward validation, evaluating on both statistical error and realised trading performance.
- Developed a vectorised backtesting engine with one-bar execution lag, transaction costs, slippage and full trade logging, benchmarked against buy-and-hold across return, Sharpe, and max-drawdown.
- Delivered a 10-page Streamlit research dashboard with regime analysis, parameter stress-testing and auto-generated research reports; validated by a pytest suite covering leakage, alignment and metric correctness.

---

## 29. Glossary

| Term | Meaning |
|------|---------|
| **OHLCV** | Open, High, Low, Close, Volume — the daily price bar |
| **Adjusted close** | Close price adjusted for splits and dividends |
| **Lookahead bias** | Using information that would not have been available at decision time |
| **Backtest** | Simulating a strategy on historical data |
| **Buy-and-hold** | Buy on day one, hold to the end — the baseline to beat |
| **Drawdown** | Percentage fall from the previous equity peak |
| **Sharpe ratio** | Excess return per unit of volatility |
| **bps** | Basis point — 0.01%. 5 bps = 0.05% |
| **Slippage** | Difference between expected and actual execution price |
| **Turnover** | How much of the portfolio changes hands |
| **Walk-forward** | Repeatedly retrain on the past, test on the next period |
| **Regime** | A market environment (high/low volatility, bull/bear) |

---

*Built as a quantitative research and software engineering portfolio project. Not investment advice.*
