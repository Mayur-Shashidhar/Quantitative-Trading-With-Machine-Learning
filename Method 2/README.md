# Quantitative Trading with Machine Learning

## Overview

This project explores whether historical Volkswagen Frankfurt stock
returns and related supplier-return features can be used to predict
Volkswagen's future return and construct a rule-based trading strategy.

The notebook is organized into eight code cells with Markdown
explanations. It compares three regression models and evaluates their
predictions and trading strategies against a buy-and-hold benchmark.

> **Important:** This notebook is an experimental backtest, not a live
> trading system or evidence of a reliably profitable strategy. Its
> evaluation design has limitations described below.

## Objectives

-   Predict Volkswagen's forward return over a 20-trading-row horizon.
-   Compare Elastic Net, Ridge regression, and Extra Trees regression.
-   Convert predicted returns into long, short, or neutral positions.
-   Include a configurable transaction-cost assumption.
-   Compare strategy returns and risk metrics with buy-and-hold over the
    same sampled evaluation periods.
-   Present results through tables and charts.

## Dataset

The notebook expects the following file to exist in the runtime:

``` text
/content/databaseFrankfurtComplete.csv
```

The CSV must contain at least these columns:

-   `Date` --- observation date.
-   `Close` --- Volkswagen Frankfurt closing price.

The dataset also uses columns matching these naming patterns when
available:

-   `Return_Delta1_*` --- daily-return features for supplier/proxy
    companies.
-   `Return_Delta1_VW` through `Return_Delta28_VW` --- Volkswagen
    return-lag features.

The notebook parses dates, converts the closing price to numeric values,
removes rows missing the date or closing price, sorts chronologically,
and removes duplicate dates.

**Data provenance and field definitions should be documented by the
project owner.** This README does not independently verify the source,
licensing, survivorship characteristics, or exact meaning of every
column in the CSV.

## Methodology

### 1. Target construction

The target is calculated from the closing-price series:

\[ y_t = `\frac{Close_{t+20}}{Close_t}`{=tex} - 1 \]

It represents the return from the current close to the close 20 rows
later. Since the data is ordered by date, these rows are intended to
represent 20 trading observations. The final 20 rows do not have an
observable target and are excluded from the labeled dataset.

### 2. Feature engineering

The feature matrix includes available supplier daily-return columns and
Volkswagen return lags from 1 through 28 days. It also calculates the
following price-derived features:

-   Momentum over 5, 10, 20, and 60 rows.
-   Rolling daily-return volatility over 5, 10, 20, and 60 rows.

Non-numeric values are coerced to missing values, and infinite values
are replaced with missing values. Columns that are entirely missing or
constant are removed. Median imputation is performed inside each model
pipeline.

### 3. Chronological split

The notebook uses a chronological 60/40 split:

-   First 60% of eligible observations: model fitting and threshold
    selection.
-   Remaining 40%: evaluation.

It attempts to purge observations near the split because the
forward-return labels can cross the boundary. The current implementation
calculates this purge using 20 **calendar days**, rather than 20 trading
rows. This is a known methodological limitation.

Trading metrics are calculated using sampled non-overlapping 20-row
periods. The code selects these periods using the row index modulo the
horizon.

### 4. Models

The notebook compares three regressors:

  ------------------------------------------------------------------------
  Model                   Configuration in the     Purpose
                          notebook                 
  ----------------------- ------------------------ -----------------------
  Elastic Net             `alpha=0.01`,            Linear regression with
                          `l1_ratio=0.2`           combined L1/L2
                                                   regularization

  Ridge                   `alpha=10.0`             Linear regression with
                                                   L2 regularization

  Extra Trees             250 trees,               Tree ensemble that can
                          `min_samples_leaf=15`,   represent nonlinear
                          `max_features=0.7`       relationships
  ------------------------------------------------------------------------

Before fitting, the training target is clipped at its 1st and 99th
percentiles. The models use median imputation; Elastic Net and Ridge
also use standardization.

### 5. Trading rules

The notebook evaluates two position modes:

-   **Long-only:** take a long position when the prediction exceeds the
    threshold; otherwise remain neutral.
-   **Long-short:** take a long position when the prediction exceeds the
    threshold, a short position when it is below the negative threshold,
    and otherwise remain neutral.

For each model, candidate thresholds are based on quantiles of the
absolute fitted-period predictions. The notebook selects the threshold
and position mode with the highest approximate annualized Sharpe ratio
on the same periods used to fit the model.

A transaction cost of `0.001` (0.1%) is charged per unit of position
turnover. A move from long to short changes the position by two units
and therefore incurs twice the position turnover of a move from neutral
to long.

### 6. Evaluation metrics

The notebook reports:

**Prediction metrics** - Mean Absolute Error (MAE). - MAE improvement
relative to a zero-return prediction baseline. - Directional accuracy.

**Trading metrics** - Compounded strategy return. - Compounded
buy-and-hold return. - Approximate annualized Sharpe ratio. - Maximum
drawdown. - Percentage of periods with a non-neutral position.

The strategy and buy-and-hold returns are calculated over the same
sampled evaluation periods. These returns are not directly comparable
with another study unless the sample dates, asset, horizon, portfolio
rules, transaction costs, and evaluation procedure are aligned.

## Notebook structure

  Cell   Responsibility
  ------ --------------------------------------------------------------
  1      Imports, configuration, dataset loading, and basic cleaning
  2      Forward target and feature engineering
  3      Chronological 60/40 split and sampled trading periods
  4      Fit Elastic Net, Ridge, and Extra Trees
  5      Select each model's threshold and position mode
  6      Calculate prediction and trading metrics
  7      Display a focused model-comparison table
  8      Generate charts for returns, Sharpe ratio, drawdown, and MAE

## Requirements

The notebook uses Python and the following libraries:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scikit-learn`

Install them in a local environment if required:

``` bash
pip install numpy pandas matplotlib scikit-learn
```

The notebook was written for a notebook environment such as Google
Colab, but the code can be adapted to Jupyter by changing `DATA_PATH` to
the local CSV path.

## How to run

1.  Place `databaseFrankfurtComplete.csv` at
    `/content/databaseFrankfurtComplete.csv`, or update `DATA_PATH` in
    Cell 1.
2.  Open the notebook.
3.  Run the eight code cells in order, following their Markdown
    descriptions.
4.  Review the model comparison table and four charts.
5.  The notebook displays results but does not automatically save a ZIP
    archive.

## Important limitations

The current notebook is a preliminary experiment and should be
interpreted cautiously.

1.  **Threshold-selection bias:** thresholds and position modes are
    selected using predictions on the same observations used to fit each
    model. These are in-sample predictions, so the selected
    configuration may be overly optimistic.
2.  **Purge implementation:** the split purge uses calendar days rather
    than removing the last 20 trading-row labels. It should be corrected
    for a more rigorous time-series evaluation.
3.  **Feature availability:** the dataset's supplier-return columns must
    be audited to confirm that each feature was actually available at
    the time a trade would be placed. Naming conventions alone do not
    prove that a feature is free of look-ahead leakage.
4.  **Simplified execution model:** the backtest assumes a fixed
    turnover cost and does not fully model bid-ask spreads, slippage,
    liquidity, market impact, borrow fees, short-sale constraints, or
    execution timing.
5.  **Single evaluation segment:** one chronological split cannot
    establish stability across market regimes. Walk-forward testing
    across multiple periods would be stronger.
6.  **No guarantee of profitability:** positive historical returns do
    not guarantee future performance. A higher strategy return also does
    not necessarily mean better prediction accuracy.
7.  **Reference-paper comparability:** this implementation uses a 20-row
    target, three models, and a 60/40 split. It is not an exact
    reproduction of a paper that uses different models, data, horizons,
    or rolling evaluation windows.

## Recommended future improvements

-   Replace the single split with rolling or walk-forward evaluation.
-   Use a separate validation segment for model and trading-rule
    selection, then evaluate the final test period only once.
-   Purge training observations by row position to match the
    forward-label horizon.
-   Audit all feature timestamps and definitions for leakage.
-   Add RMSE and (R\^2), and compare against both zero-return and simple
    momentum baselines.
-   Test sensitivity to transaction costs and thresholds.
-   Report the exact evaluation dates and all strategy assumptions.
-   Add tests for target alignment, chronological splits, turnover
    costs, and benchmark consistency.

## Responsible interpretation

This project is intended for educational and research purposes. It
demonstrates a basic machine-learning and backtesting workflow. It
should not be treated as investment advice or a production-ready
automated trading system.
