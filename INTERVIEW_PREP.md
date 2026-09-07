# Interview Preparation

How to explain this project, how it actually works, and the questions you will be asked — with answers.

Every number in this document came from an actual run on AAPL, 2015–2024. Nothing is invented. If you quote a figure in an interview, it is real and you can reproduce it with one command.

---

## Table of Contents

1. [The 30-second version](#1-the-30-second-version)
2. [The 2-minute version](#2-the-2-minute-version)
3. [Explaining it to a non-technical person](#3-explaining-it-to-a-non-technical-person)
4. [How it works, step by step](#4-how-it-works-step-by-step)
5. [The results, and how to present them](#5-the-results-and-how-to-present-them)
6. [The single most important concept: leakage](#6-the-single-most-important-concept-leakage)
7. [Interview questions and answers](#7-interview-questions-and-answers)
   - [Data](#data)
   - [Features](#features)
   - [Modelling](#modelling)
   - [Validation](#validation)
   - [Strategy and signals](#strategy-and-signals)
   - [Backtesting](#backtesting)
   - [Metrics](#metrics)
   - [Results and judgement](#results-and-judgement)
   - [Engineering](#engineering)
   - [Hard and adversarial questions](#hard-and-adversarial-questions)
8. [Questions to ask them](#8-questions-to-ask-them)
9. [Traps to avoid](#9-traps-to-avoid)
10. [One-page cheat sheet](#10-one-page-cheat-sheet)

---

## 1. The 30-second version

> I built an end-to-end quantitative research pipeline that tests whether machine learning can predict short-term stock returns well enough to beat buy-and-hold.
>
> It pulls ten years of daily price data, engineers 43 leakage-free features, trains four models under time-based validation, converts predictions into trading signals, and backtests them with realistic transaction costs against a buy-and-hold baseline.
>
> The honest answer is no — the models found only a marginal edge that disappears after costs. The interesting part is the rigour that makes me confident that answer is *correct* rather than a bug: automated tests that prove no future information leaks into any feature, and that the backtester's one-day execution lag is genuinely applied.

**Why this framing works:** you lead with a clear question, a clear method, and an honest result. Interviewers are far more suspicious of "my model got 95% accuracy on stock prices" than of a careful negative finding.

---

## 2. The 2-minute version

Use this when they say "walk me through the project."

**The question.** Can historical price and volume data predict tomorrow's return well enough to trade profitably? This is a real research question, and the honest prior is that it mostly cannot — markets are close to efficient at daily frequency. So I built the pipeline to be capable of proving itself wrong.

**The data.** Daily OHLCV from Yahoo Finance, cached locally as Parquet so results are reproducible. A validation layer runs twelve checks — missing values, duplicate dates, impossible bars where the high is below the low, calendar gaps. Importantly it separates real anomalies from data errors: a −30% day is a crash, not a bad tick, so it is flagged and kept.

**The features.** 43 features: lagged returns, moving-average ratios, volatility measures, volume ratios, and technical indicators like RSI, MACD and Bollinger Bands. Two rules govern all of them. Every one is strictly backward-looking, and every one is stationary — I never feed a raw price or a raw moving average to a model, because a model trained on AAPL at $30 cannot extrapolate to $250. I use ratios instead.

**The models.** Linear Regression, Ridge, Random Forest and XGBoost, all sharing one interface so they compete on identical splits, identical features and an identical seed. Splits are chronological — train on 2015–2021, validate on 2022, test on 2023–2024. Never random, because shuffling time series lets the model train on 2024 and test on 2016.

**The backtest.** Predictions become signals, signals become positions with a one-day execution lag, and positions pay transaction costs on turnover. Everything is compared to buy-and-hold on the same dates.

**The result.** Test-set R² is slightly negative for every model, around −0.02 to −0.06. Directional accuracy is 48–54%. Under walk-forward validation across 1,698 out-of-sample days, directional accuracy is 51.5% with a standard deviation of 2.3% between folds. The best strategy returned 82% against buy-and-hold's 107%, and stress tests show it never beats the benchmark at any cost level, including zero.

**What I would say I learned.** That prediction accuracy and trading performance are different objectives — the model with the lowest RMSE was not the model with the best strategy return. And that most of the engineering effort in quantitative research goes into *not fooling yourself*.

---

## 3. Explaining it to a non-technical person

> Imagine you have ten years of daily records for a stock: what price it opened at, its high, its low, what it closed at, and how many shares traded.
>
> I wrote a program that looks at that history and tries to learn patterns — does a stock that fell yesterday tend to rise today? Does unusually heavy trading say anything about tomorrow?
>
> Then I test it honestly. I let the program learn only from 2015 to 2021, and then ask it to make predictions about 2023 and 2024, which it has never seen. Every day it predicts up or down, and I simulate actually buying and selling on those predictions — including the fees you would really pay.
>
> Finally I compare that to the simplest possible strategy: buy the stock on day one and do nothing.
>
> The result was that doing nothing won. My model returned 82% and buying-and-holding returned 107%. That sounds like a failure, but it is actually the expected answer — if daily stock movements were easy to predict, someone would already have done it. What the project really demonstrates is that I can build the machinery to test an idea rigorously and report the result honestly, rather than accidentally fooling myself into a number that looks good but is wrong.

**The key sentence for a non-technical interviewer:** *"The hard part wasn't building the model. The hard part was making sure the model wasn't secretly cheating."*

---

## 4. How it works, step by step

### Step 1 — Get the data

`src/data_loader.py` downloads daily bars and forces them into one fixed schema. This sounds trivial and is not: the data source returns different column shapes depending on how you call it, and it adjusts only the closing price for stock splits while leaving the open, high and low unadjusted.

**That mattered.** AAPL split 4-for-1 in 2020. Mixing an adjusted close with an unadjusted high made the Stochastic %K indicator read −296 when its range is 0–100, and inflated ATR from about 2% of price to 5.9%. The fix is in `preprocessing.adjust_ohlc`: rescale the whole bar by `adj_close / close`, so every field shares one factor and the bar's internal ordering is preserved exactly.

*This is a great story to tell.* It shows you verify against reality instead of trusting library output.

### Step 2 — Validate and clean

Twelve checks, split into errors and warnings. The design principle is that **inspection and mutation are separate**: `validate()` never changes anything, `clean()` changes things and logs every action.

Why separate them? Because in market data a "problem" is often real. Zero volume means a trading halt. A −30% day means a crash. Those get flagged and kept. Only structurally impossible things get removed — a high below the low, a negative price.

Volume is never forward-filled. A fabricated volume would silently corrupt every volume feature, so those rows are dropped instead.

### Step 3 — Build features

43 features, all backward-looking, all stationary.

The stationarity point is worth understanding well. If you feed a tree model a raw 20-day moving average, it learns rules like "if the average is above 150, predict up" — which is memorising the price level, not learning behaviour. When the price moves outside the training range the model is extrapolating, and trees cannot extrapolate at all. So instead of `sma_20`, the model sees `close / sma_20` — a ratio that hovers around 1.0 regardless of whether the stock trades at $30 or $250.

### Step 4 — Define the target

```python
target[t] = close[t + 1] / close[t] - 1
```

The row at date *t* pairs features known at *t*'s close with the return earned *after* it. The last row has no target and is dropped.

### Step 5 — Split chronologically

```
2015 ─────────────── 2021 │ 2022 │ 2023 ──── 2024
       TRAIN (1703)        │ VALID│  TEST (500)
                           │ (251)│
```

Never `train_test_split`. Never shuffled.

### Step 6 — Train

All four models share `ModelPipeline`, which owns the `StandardScaler`. The scaler is fit on **training rows only**. Fitting it on the whole dataset would leak the test period's mean and variance into training — a quiet leak that inflates every downstream number. A test asserts the scaler's statistics match the training rows exactly *and* differ from the full-sample statistics.

### Step 7 — Generate signals

```python
prediction >  threshold  → +1 (long)
prediction < -threshold  → -1 (short, if enabled)
otherwise                →  0 (flat)
```

Deliberately simple. A complicated rule is hard to explain and easy to overfit. The threshold is one number the stress tester sweeps, so its influence is *measured* rather than assumed.

### Step 8 — Backtest

```python
position = signal.shift(1)                # the execution lag
turnover = position.diff().abs()
cost     = turnover * (commission + slippage) / 10_000
net      = position * returns - cost
equity   = (1 + net).cumprod()
```

Costs are charged on turnover, not on returns. That makes reversing a position (long → short, two units of movement) cost twice what opening one does — which is correct.

### Step 9 — Validate properly with walk-forward

A single split gives one estimate from one period, and that period may simply have been kind. Walk-forward retrains repeatedly:

```
fold 1: train 2015-2018 → test 2018-2019
fold 2: train 2015-2019 → test 2019-2020
...
fold 6: train 2015-2023 → test 2023-2024
```

Stitching the out-of-sample predictions gives 1,698 test days instead of 500.

### Step 10 — Stress and report

Sweep costs, thresholds, seeds, tickers and sub-periods. Score performance within volatility and trend regimes. Then generate a Markdown report that states the verdict plainly, including when the verdict is negative.

---

## 5. The results, and how to present them

### Single chronological split, test period 2023–2024 (500 days)

| Model | RMSE | R² | Directional accuracy | Strategy return | Sharpe | Max drawdown | Trades |
|---|---|---|---|---|---|---|---|
| Linear Regression | 0.0139 | −0.062 | 48.4% | 18.4% | 0.64 | −21.1% | 76 |
| Ridge | 0.0138 | −0.058 | 47.6% | 10.8% | 0.42 | −21.3% | 73 |
| Random Forest | 0.0136 | −0.016 | 52.4% | 69.5% | 1.46 | −19.0% | 24 |
| XGBoost | 0.0136 | −0.021 | 54.0% | 82.5% | 1.75 | −19.2% | 59 |
| **Buy and hold** | — | — | — | **106.5%** | **1.82** | **−16.6%** | 1 |

### Walk-forward, 6 expanding folds, 1,698 out-of-sample days

| Metric | Strategy | Buy and hold |
|---|---|---|
| Total return | 313.3% | 549.2% |
| CAGR | 23.4% | 32.0% |
| Sharpe ratio | 1.01 | 1.06 |
| **Max drawdown** | **−33.4%** | **−38.5%** |
| Win rate | 43.4% | 54.3% |
| Time in market | 63.1% | 100% |

Stitched out-of-sample R² = **−0.051**. Directional accuracy = **51.5%**, standard deviation **2.3%** across folds, ranging 48.6% to 55.3%.

### How to present this

**Do not apologise for it.** Present it as the finding.

> "Across 1,698 out-of-sample days, the model achieved 51.5% directional accuracy. That is a real but tiny edge — and after 7 basis points of round-trip cost, it isn't enough. The strategy returned 313% against buy-and-hold's 549%.
>
> There is one genuine positive: because the strategy is only in the market 63% of the time, it cut the maximum drawdown from 38.5% to 33.4% at a nearly identical Sharpe ratio. So it produced a smoother ride for less money. Whether that trade is worth making depends on the mandate, but it's not a free lunch.
>
> The result I'd defend most strongly is the stress test: the strategy fails to beat the benchmark at *every* cost level, including zero. That tells me the limitation is the signal, not the friction — which is a much more useful conclusion than 'costs ate my returns.'"

### The most interesting single finding

In **low-volatility regimes** the strategy returned **+13.7%** while buy-and-hold **lost −1.9%** — a Sharpe of 1.61 against −0.12. But it won in only 1 of 3 volatility regimes and 0 of 2 trend regimes.

> "That's the kind of result that looks like an edge and probably isn't. Winning in one regime out of five tested is roughly what you'd expect from noise. I report it as a hypothesis worth testing on more assets, not as a finding."

---

## 6. The single most important concept: leakage

If you remember one thing for the interview, remember this.

**Lookahead bias** is using information at decision time that would not actually have been available then. It is the defining failure mode of backtesting, and it is *silent* — the equity curve simply becomes beautiful.

### The timeline

```
      day t                          day t+1
 ───────┼──────────────────────────────┼──────
        │                              │
   CLOSE(t) known                      │
        │                              │
   1. build features from data ≤ t     │
   2. predict r(t+1)                   │
   3. set position for t+1 ────────────►
                                       │
                              4. earn r(t+1)
```

### Three rules, and how each is enforced

**Rule 1 — Features use only data at or before *t*.**
Enforcement: a test rebuilds the entire 43-feature matrix using only history up to a cut-off date and asserts every value at that date is bit-for-bit identical to the value computed from the full dataset. Verified at four points across ten years of real data.

**Rule 2 — The target is strictly future.**
`shift(-1)` is the only negative shift in the entire project, and it appears exactly once, in the label.

**Rule 3 — A prediction at *t* can only trade at *t+1*.**
`position = signal.shift(1)`. This is the one line that matters most.

### How I proved rule 3 is real

Two tests, from opposite directions:

1. Feed the backtester a signal built from *today's* return — same-day information. With the lag applied it returns 7% (a harmless momentum rule). With the lag removed it returns over 100%. This proves the lag actually neutralises information it should.
2. Feed it a genuine *forecast* of tomorrow. Under the lag it becomes a perfect strategy — long on every up day, flat on every down day. This proves signals and returns aren't off by one in the *other* direction.

> "I got the first test backwards when I wrote it, and the failure taught me the distinction between a signal that peeks and a signal that forecasts. I kept both tests, because together they pin the convention from both sides."

That is an excellent thing to say in an interview. It shows tests catching a genuine misunderstanding.

### Six other leaks this project closes

| Leak | Fix |
|---|---|
| Scaler fit on all data | Fit on train only; asserted by test |
| Random train/test split | Chronological only; `train_test_split` never imported |
| Tuning on the test set | Hyperparameters chosen on validation; test touched once |
| Dropping NaNs after splitting | Warm-up rows dropped *before* splitting, so splits stay contiguous |
| Mixing adjusted and raw prices | Whole bar rescaled by one factor |
| Whole-sample regime terciles | A causal expanding-quantile mode is provided, and a test documents that the non-causal labels *do* change when future data arrives |

---

## 7. Interview questions and answers

### Data

**Q: Where did the data come from?**
Yahoo Finance via the `yfinance` library — daily OHLCV for a chosen ticker and date range. It's cached locally as Parquet on first download, so every run afterwards is reproducible and works offline. That matters because Yahoo occasionally revises historical data; caching means my results don't silently change between runs.

**Q: What does OHLCV mean?**
Open, High, Low, Close, Volume — the five numbers summarising one trading day. Open is the first traded price, high and low are the session extremes, close is the last price, volume is shares traded. I also use adjusted close, which corrects for stock splits and dividends.

**Q: Why use adjusted close rather than close?**
Because a 4-for-1 split makes the raw price drop 75% overnight with no economic loss. Using raw closes, my return series would contain a fake −75% day. Adjusted close removes that.

**Q: How did you handle missing data?**
In layers. Validation reports it without changing anything. Cleaning then forward-fills price gaps of at most 3 days, and drops anything longer rather than inventing data. Volume is never forward-filled — a fabricated volume would corrupt every volume feature — so those rows are dropped instead.

**Q: What if the data had an obvious error, like a −30% day?**
I wouldn't assume it's an error. That's a crash. My validator flags moves beyond ±50% as warnings for human review but never deletes them, because deleting real crashes is how you build a backtest that has never seen a bear market. Only structurally impossible bars — high below low, non-positive price — are removed.

**Q: How do you know your data is clean?**
Twelve automated checks. On ten years of AAPL: zero errors, zero warnings, and 92 absent business days — which works out to 9.2 per year, exactly the US market holiday count. The validator recognises that and classifies it as information rather than a problem.

---

### Features

**Q: What features did you build, and why those?**
43, in six families:
- **Lagged returns** (1, 5, 10, 20-day, plus 5 individual lags) — capture momentum and mean reversion.
- **Moving-average ratios** — `close / sma_20` says where the price sits relative to its recent trend.
- **Volatility** (5, 20, 60-day, plus downside deviation) — volatility clusters, which is one of the few genuinely persistent properties of markets.
- **Volume** (change, ratio to average, z-score, OBV) — unusual volume often accompanies information.
- **Intraday range** (high-low range, close position within the day, overnight gap).
- **Technical indicators** — RSI, MACD, Bollinger %B and bandwidth, ATR, Stochastic %K.

**Q: Why ratios instead of the raw moving average?**
Stationarity. A raw moving average is non-stationary — it drifts with the price level. A tree trained on AAPL at $30 would learn thresholds that are meaningless at $250, and trees cannot extrapolate beyond their training range at all. A ratio like `close / sma_20` hovers around 1.0 regardless of price level, so the model learns *behaviour* rather than *level*.

**Q: How do you know none of your features leak?**
A test rebuilds the entire feature matrix on truncated history and requires every value at the cut-off date to be identical to the value from the full dataset. If any feature peeked forward, its value would change once future rows were appended. I ran it at four points across ten years of AAPL: identical every time.

There's also a standing smoke test — the correlation of each feature with the *next day's* return. The strongest is 0.069 and the median is 0.013. Anything above 0.3 would be flagged as a probable leak.

**Q: 0.069 is tiny. Isn't that a problem?**
It's the honest ceiling. Those correlations tell me in advance that no model on these features is going to achieve a high R², because the information simply isn't there. That's useful — it means when I later see R² near zero, I know it's the data, not a bug in my model.

**Q: Which features mattered most?**
It differs by model, which is itself informative. Linear models weight lagged returns most heavily — they're picking up short-horizon mean reversion, and the strongest single correlation with tomorrow is `ret_1d` at −0.069, negative, which is mean reversion. Trees instead weight moving-average ratios and volatility ratios, so they're finding trend and regime information a linear model can't express.

**Q: Did you try feature selection?**
Not aggressively, deliberately. With 43 features and thousands of rows, over-fitting through selection is a real risk — if I select features by test-set performance I've tuned on the test set. I rely on regularisation instead: Ridge for the linear model, shallow depth and large minimum leaf sizes for the trees. Averaging feature importance across walk-forward folds is the more trustworthy signal, since a feature that only matters in one fold was probably fitting that fold's noise.

---

### Modelling

**Q: Why those four models?**
They form a ladder of complexity. Linear Regression is the honest baseline — if something complex can't beat it, the complexity isn't earning its place. Ridge adds L2 regularisation, which matters because financial features are highly collinear. Random Forest captures non-linear interactions. XGBoost is the strongest tree method and answers "does more capacity help?" The answer here was: barely, and not enough.

**Q: Why not a neural network / LSTM / transformer?**
With around 1,700 training rows and a signal-to-noise ratio this low, a deep model would overfit dramatically and I'd have no way to tell whether an improvement was real. XGBoost already overfits visibly here — its training R² is 0.34 against a test R² of −0.02. Adding capacity is the wrong direction. If I were extending this, I'd add *data* — more tickers, cross-sectional features — before adding model complexity.

**Q: How did you tune hyperparameters?**
Conservatively, and on the validation split only. Trees are capped at depth 3–6 with large minimum leaf sizes (50 samples for the Random Forest, `min_child_weight` 20 for XGBoost). On daily financial data an unconstrained tree fits the noise perfectly and generalises at chance. I did not run an exhaustive search, because a large search evaluated on a small validation set is itself a way of overfitting.

**Q: Your XGBoost training R² is 0.34 but test R² is −0.02. Isn't that a broken model?**
It's an overfitting model, and I report it rather than hiding it. It's also the expected behaviour: gradient-boosted trees will always find structure in training noise. The important thing is that this is *visible* — because the test split was never touched during development, the gap is a real measurement rather than something I discovered too late. It's also why I don't select the model on training metrics.

**Q: Why is your R² negative? Doesn't that mean the model is worthless?**
Negative R² means the model predicts worse than always predicting the mean. For daily stock returns that's close to the expected outcome, because the mean is a genuinely hard baseline to beat — daily returns are almost pure noise around a small positive drift. I'd be far more suspicious of a high R². My code has an explicit sanity check that flags any R² above 0.10 on daily data as a probable lookahead bug rather than a success.

---

### Validation

**Q: Why can't you use a random train/test split?**
Two reasons. First, it leaks the future: with a random split the model trains on 2024 data and tests on 2016, so it has effectively seen the answers. Second, financial data is autocorrelated — adjacent days are similar — so random neighbours in the test set are nearly duplicates of training rows, and accuracy is inflated. Time series must be split by time.

**Q: What's walk-forward validation and why is it better?**
Rather than one train/test split, you retrain repeatedly and always test on the block that follows: train 2015–2018 test 2019, train 2015–2019 test 2020, and so on. Two advantages. It mirrors how a strategy would actually have been run — you'd retrain as new data arrived. And it produces far more out-of-sample evidence: 1,698 test days instead of 500.

**Q: Expanding or rolling window?**
I support both. Expanding uses all history, which gives more data but assumes old relationships still hold. Rolling uses a fixed recent window, adapting faster to regime change but forgetting more. On this data expanding performed substantially better, which suggests the model needs more history than a rolling window provides — that's a real finding about the data.

**Q: How do you know walk-forward is actually retraining?**
A test asserts that different folds produce different feature-importance vectors. If the model were fit once and reused, every fold's importance would be identical and later folds would score suspiciously well.

**Q: What does the fold-to-fold variation tell you?**
This is where the interesting judgement lives. Directional accuracy averaged 51.5% but ranged from 48.6% to 55.3% across folds, with a standard deviation of 2.3%. That spread is large relative to the edge itself. If I'd reported only the single best fold I could have claimed 55.3% — which is why reporting the spread, not just the mean, is the honest thing to do.

---

### Strategy and signals

**Q: How does a prediction become a trade?**
A threshold rule. If the predicted return exceeds the threshold, go long. If it's below negative-threshold and shorting is enabled, go short. Otherwise stay flat. Default is long-only, because it's simpler to explain and to justify.

**Q: Why so simple? Couldn't a better rule improve results?**
Almost certainly, on historical data — and that's the problem. Every parameter I add is another thing I could tune until the backtest looks good. One threshold is one number, and my stress test sweeps it across six values so I can *show* how sensitive results are rather than assert they're robust. A complicated rule would make that impossible.

**Q: What does the threshold actually do?**
It's a friction control. At threshold zero the strategy takes a position nearly every day and pays maximum transaction costs. Raising it trades only on high-conviction predictions, cutting costs but also cutting time in the market. On my data, raising it monotonically hurt returns — the reduced exposure cost more than the saved fees.

**Q: Why long-only rather than long/short?**
Simplicity and honesty. Shorting has real-world constraints my backtest doesn't model — borrow costs, availability, unlimited downside. Long-only is a strategy I can defend line by line. The code supports shorting behind a flag, but I don't present short results as the headline.

---

### Backtesting

**Q: What is backtesting?**
Simulating what a strategy would have done on historical data — generating the signals it would have generated, taking the positions it would have taken, and paying the costs it would have paid.

**Q: What's the most common way backtests go wrong?**
Lookahead bias, and the specific instance of it is forgetting the execution lag. If you compute `position * return` without shifting the position by one day, you're trading on information from the close of the same day you're trading — which is impossible. It fails silently: no error, just a beautiful equity curve. I have a dedicated test that proves my lag is load-bearing.

**Q: How did you model transaction costs?**
5 basis points commission plus 2 basis points slippage, charged on **turnover** — the absolute change in position — rather than on returns. So opening a position from flat costs one unit, and reversing from long to short costs two, because the position moves two units. That's correct and it matters.

**Q: Why do costs matter so much?**
Because they compound against you daily. A signal that flips every day at 7 bps one-way pays roughly 250 × 0.0007, about 17% a year, in friction alone. In my actual run, XGBoost paid 8.19% of capital in costs across 59 trades. Most impressive-looking naive backtests are simply strategies that forgot to pay for their own trading.

**Q: How do you know your backtest engine is correct?**
The strongest check I have: with a permanently long position and zero costs, the engine must reproduce buy-and-hold to floating-point precision. If it can't reproduce the simplest possible strategy exactly, nothing else it produces is trustworthy. I also verify that compounded per-trade returns reconcile back to the equity curve.

**Q: What isn't modelled?**
Several things, and I'd say so unprompted. Execution is assumed at the daily close — real fills happen at auction prices with market impact. Costs are a flat rate, whereas real costs vary with liquidity and order size. There's no position sizing or portfolio construction. And there's survivorship bias: AAPL is a company that still exists, and testing on survivors flatters any long-biased strategy.

**Q: Why is a buy-and-hold baseline necessary?**
Because without it, no number means anything. A strategy returning 82% sounds excellent until you learn holding the stock returned 107%. Buy-and-hold is the free alternative that requires no model, no trading and no fees, so it's the bar any strategy has to clear to justify existing.

---

### Metrics

**Q: What is the Sharpe ratio?**
Return per unit of volatility, annualised: `(mean return − risk-free rate) / standard deviation × √252`. It answers "how much return am I getting for the risk I'm taking?" A Sharpe of 1 is decent, 2 is very good. My strategy scored 1.01 under walk-forward and buy-and-hold scored 1.06 — statistically indistinguishable.

**Q: What is maximum drawdown?**
The worst peak-to-trough fall in portfolio value. It's the number that actually determines whether someone can hold a strategy, because a 50% drawdown means watching half your money disappear before any recovery. My strategy's was −33.4% versus buy-and-hold's −38.5%.

**Q: There's a subtlety in how you compute drawdown — what is it?**
The running peak has to start at the initial capital, not at the first day's value. Otherwise a strategy that falls 10% on its very first day reports zero drawdown, because that day is its own running maximum — yet the investor is plainly 10% down on the money they started with. I found this by writing a test that failed.

**Q: Why report Sortino as well as Sharpe?**
Sharpe penalises all volatility, including upside. But a surprise gain isn't a risk. Sortino divides by downside deviation only, so it doesn't punish a strategy for having good days.

**Q: What's the information coefficient?**
The Spearman rank correlation between predictions and outcomes — the standard quant measure of forecast quality. Values around 0.03–0.05 are considered good for daily equity signals, which tells you how thin real edges are. Mine was 0.009 over the stitched out-of-sample period.

**Q: Why does your win rate exclude flat days?**
A day with no position is neither a win nor a loss. Counting flat days as wins would make a strategy that never trades look perfectly consistent.

---

### Results and judgement

**Q: Your strategy lost to buy-and-hold. Why is this project worth showing?**
Because the pipeline is the deliverable, not the return. Anyone can produce a strategy that beats the benchmark on historical data — you just tune until it does, which is curve-fitting. What's hard is building a system rigorous enough that you'd *believe* a positive result if you got one. Mine has automated proof of no leakage, an execution lag verified from both directions, a backtester validated against a closed-form baseline, and stress tests that would expose a fragile result. The negative finding is evidence the machinery works.

**Q: What would make you believe you had found a real edge?**
Four things together. It survives walk-forward across many folds with low variance. It survives transaction costs at realistic levels. It generalises to tickers I didn't develop it on. And it persists across market regimes rather than living in one. My result fails several of those — it wins in 1 of 3 volatility regimes and 0 of 2 trend regimes, and never beats the benchmark at any cost level, including zero.

**Q: What does "it fails even at zero cost" tell you?**
It's the single most useful diagnostic I have. It means the limitation is the *signal*, not friction. If the strategy had beaten the benchmark at 0 bps and lost at 5 bps, the answer would be "reduce turnover." Since it loses even when trading is free, the predictions themselves aren't good enough, and no amount of execution optimisation would fix it.

**Q: Is there anything positive in your results?**
Two things, both qualified. The strategy is only in the market 63% of the time, and it cut maximum drawdown from 38.5% to 33.4% at essentially the same Sharpe — so it delivered a smoother ride for less money, which could suit a drawdown-sensitive mandate. And in low-volatility regimes it returned +13.7% while buy-and-hold lost −1.9%. But that's one regime out of five tested, which is about what noise would produce, so I present it as a hypothesis rather than a finding.

**Q: What would you do next?**
In priority order. Move from one asset to a cross-section — rank many stocks against each other and trade the spread, which is how real equity strategies work and is far more robust than timing a single name. Add data beyond price: fundamentals, earnings dates, sector membership. Predict volatility instead of returns, since volatility is genuinely more predictable. And test on longer horizons — weekly or monthly — where the noise-to-signal ratio is more favourable and costs matter less.

---

### Engineering

**Q: How is the project structured?**
Around 5,000 lines of source in 18 single-purpose modules, 4,000 lines of tests, and a 10-page Streamlit dashboard. Each module does one thing: data loading, validation, indicators, features, dataset assembly, models, evaluation, signals, backtesting, metrics, walk-forward, regimes, stress, reporting, plotting. A `pipeline` module assembles them, and both the CLI and dashboard call it — so the two front-ends can't drift apart and report different numbers from the same configuration.

**Q: How many tests, and what do they cover?**
459. The ones I'd point to are the leakage tests — rebuilding the feature matrix on truncated history — the backtester's execution-lag tests from both directions, the buy-and-hold equivalence test, and the metrics tests that check every formula against a hand-computed closed form. There are also 30 tests that boot every dashboard page through Streamlit's real script runner, because a syntax check proves nothing about runtime.

**Q: Did the tests actually catch anything?**
Several real bugs, and I'd list them:
- Stochastic %K reading −296 and ATR reading 5.9% of price, from mixing adjusted and unadjusted prices across a stock split.
- RSI using a plain EWM rather than Wilder's SMA-seeded recursion — they disagree for the first several dozen bars, exactly the region a short backtest uses.
- Sharpe returning 2.4 × 10¹⁶ instead of "undefined", because I guarded with `sd == 0` and the floating-point standard deviation of a repeated constant is about 1e-19, not zero.
- Drawdown ignoring a first-day loss because the peak started at day one instead of at initial capital.
- Parquet caching silently dropping index frequency metadata, so a cached frame and a fresh one compared unequal.

**Q: How is it reproducible?**
One seed propagated to NumPy and every estimator. Data cached to Parquet so a Yahoo revision can't change results. A SHA-256 fingerprint of the entire configuration, recorded in filenames and appended to an experiment log with every run's metrics. Running the same config twice gives byte-identical numbers.

**Q: Why a CLI as well as a dashboard?**
The dashboard is for exploration; the CLI is for reproducibility. `python cli.py run --ticker AAPL --model xgboost` regenerates the whole study headlessly, which means it can run in CI and a reviewer can reproduce any number I report.

---

### Hard and adversarial questions

**Q: Isn't this just a toy? Real quant funds don't do this.**
Real funds do exactly this, at much larger scale — the difference is breadth of data and sophistication of execution, not the shape of the research loop. Form a hypothesis, prepare data without leakage, validate out-of-sample, account for costs, compare to a benchmark, check robustness. What I'd say is missing is the cross-sectional dimension: real equity strategies rank many assets against each other rather than timing one, and that's the first thing I'd add.

**Q: You only tested one stock. Isn't that a fatal weakness?**
It's a real limitation and I built the tooling to address it — there's a ticker sweep that runs the identical pipeline across a basket. What I'd resist is treating a multi-ticker average as strong evidence, because individual US large-caps are highly correlated, so five tickers is nowhere near five independent tests. Genuinely testing generalisation needs different sectors, market caps, and ideally different countries and eras.

**Q: How do you know you didn't overfit by trying many things and reporting the best?**
That's the right question, and it's the hardest one to answer convincingly for any backtest. Three defences. The test split was touched once, at the end. Model comparison used identical splits, features and seed, so nothing was tuned per model. And the stress tests deliberately include configurations that should break the strategy — a result surviving every configuration would make me suspect a bug rather than celebrate. But I'd concede that with enough iterations, some overfitting is unavoidable, which is exactly why the honest reporting matters.

**Q: If the result is negative, how do I know your pipeline can even detect a positive result?**
Good question — I test that directly. The backtester is fed a perfect oracle signal that knows tomorrow's return, and under the execution lag it produces a perfect strategy, long on every up day. So the machinery *can* turn a real edge into a return. And the leakage detector is verified by planting a deliberate cheat column and asserting it gets flagged. The pipeline is calibrated in both directions.

**Q: Your Random Forest made 24 trades and XGBoost made 59, for similar returns. What does that tell you?**
That Random Forest achieved nearly the same result with far less trading, so it's the better strategy on a cost-adjusted basis even though it returned less on paper — it paid 3.3% of capital in costs against XGBoost's 8.2%. It also had higher exposure, 85% versus 76%. If I had to deploy one, I'd choose Random Forest despite the lower headline return, because its result depends less on cost assumptions being right.

**Q: What's the weakest part of this project?**
The single-asset scope, and I'd say it before being asked. Everything downstream is constrained by it: no portfolio construction, no position sizing, no cross-sectional signal, and survivorship bias I can't eliminate. The second weakest is that daily frequency may simply be the wrong horizon — the noise dominates, and a weekly or monthly target would have a better signal-to-noise ratio and lower cost drag.

**Q: If you had one more week, what would you do?**
Extend to a cross-sectional study across roughly 100 stocks: rank them by predicted return each day, go long the top decile and short the bottom, which neutralises market direction and turns the question into "can I rank stocks?" rather than "can I time one?" That's both more likely to work and a much stronger test, because the market's overall direction stops dominating the result.

---

## 8. Questions to ask them

Good questions signal that you think like a researcher.

- How do you guard against lookahead bias in your own research process — is it tooling, code review, or convention?
- What's your team's standard for deciding a signal is real rather than a fit? A Sharpe threshold, out-of-sample duration, something else?
- How do you handle the tension between a researcher wanting to iterate and the risk of overfitting through iteration?
- What horizon do you typically work at, and how much does transaction cost modelling drive that choice?
- When a research result comes back negative, what happens to it? Is it documented, or does it disappear?

---

## 9. Traps to avoid

**Don't claim the strategy works.** It didn't. Claiming otherwise is the fastest way to lose credibility, and any competent interviewer will probe the numbers.

**Don't say "95% accuracy" about anything.** If a number sounds too good for financial prediction, it is, and saying it signals you don't know the domain.

**Don't confuse prediction accuracy with profitability.** They're different objectives. In my own results, the model with the lowest RMSE was not the model with the best strategy return — be ready to say exactly that.

**Don't hide the overfitting.** XGBoost's train R² of 0.34 against test −0.02 is a real observation. Presenting it shows you understand what you built.

**Don't oversell the drawdown improvement.** Reducing drawdown by being out of the market 37% of the time isn't skill, it's lower exposure. Say so.

**Don't get defensive about the negative result.** Frame it as the finding. "The pipeline is the deliverable" is the line.

**Don't claim more than you tested.** One ticker is one ticker. Say "on AAPL over 2015–2024" rather than "in the market."

---

## 10. One-page cheat sheet

**Pitch:** End-to-end quantitative research pipeline testing whether ML can predict daily stock returns well enough to beat buy-and-hold. Answer: no, and here's the rigour that makes me trust that answer.

**Scale:** 43 features · 4 models · 2,515 days · 459 tests · ~10,600 lines

**Headline numbers**

| | Strategy | Buy and hold |
|---|---|---|
| Return (walk-forward, 1698 days) | 313% | 549% |
| Sharpe | 1.01 | 1.06 |
| Max drawdown | −33.4% | −38.5% |
| Directional accuracy | 51.5% ± 2.3% | — |
| Out-of-sample R² | −0.051 | — |

**The one line that matters:** `position = signal.shift(1)`

**The three leakage rules:** features use only past data · target is strictly future · a prediction at *t* trades at *t+1*

**Best test:** always-long + zero costs must reproduce buy-and-hold exactly

**Best bug story:** mixing adjusted close with unadjusted high across AAPL's 4:1 split drove Stochastic %K to −296 on a 0–100 scale

**Best judgement line:** "It fails to beat the benchmark even at zero cost, so the limitation is the signal, not the friction."

**Key honest framing:** *"The hard part wasn't building the model. The hard part was making sure the model wasn't secretly cheating."*

**Next step if asked:** cross-sectional ranking across ~100 stocks, long the top decile and short the bottom.
