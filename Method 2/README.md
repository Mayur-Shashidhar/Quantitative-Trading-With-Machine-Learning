# Quantitative Trading with Machine Learning

> An educational machine-learning experiment for forecasting Volkswagen stock returns and evaluating rule-based trading strategies against a buy-and-hold benchmark.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Trading Strategy](#trading-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Notebook Structure](#notebook-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

## Overview

This project investigates whether historical Volkswagen Frankfurt returns and related supplier-return features can help predict Volkswagen's future returns. Predictions from regression models are converted into trading positions and compared with a buy-and-hold benchmark.

The implementation is organized as an eight-cell Jupyter notebook, with Markdown explanations accompanying each code section.

**This is an experimental backtest, not a live trading system or proof of a reliably profitable strategy.**

## Objectives

- Predict Volkswagen's forward return over a 20-row trading horizon.
- Compare Elastic Net, Ridge Regression, and Extra Trees Regression.
- Convert predicted returns into long-only or long-short positions.
- Account for a configurable transaction-cost assumption.
- Compare strategy performance with buy-and-hold over the same sampled evaluation periods.
- Summarize results using tables and visualizations.

## Dataset

The notebook expects the CSV file at:

```text
/content/databaseFrankfurtComplete.csv
```

### Required columns

| Column | Description |
|---|---|
| `Date` | Observation date |
| `Close` | Volkswagen Frankfurt closing price |

### Feature columns

When available, the notebook uses:

- `Return_Delta1_*` columns for daily returns of supplier or proxy companies.
- `Return_Delta1_VW` through `Return_Delta28_VW` for Volkswagen return-lag features.
- Engineered momentum and rolling-volatility features.

The notebook parses dates, converts closing prices to numeric values, removes rows without valid dates or prices, sorts chronologically, and removes duplicate dates.

> **Data provenance:** The source, licensing, and exact definitions of every dataset field should be documented separately. Column names alone do not verify data provenance or prove that every feature was available at the time of a hypothetical trade.

## Dataset Source and Data Preparation

### 1. Original Dataset Source

This project is based on the research paper **[Quantitative Trading with Machine Learning](https://cs229.stanford.edu/proj2021spr/report2/81953230.pdf)**, published as a Stanford CS229 machine learning course project.

The original authors made their replication files publicly available through the following GitHub repository:

- **GitHub repository:** https://github.com/rglawion/cs229_project_report
- **Research paper:** https://cs229.stanford.edu/proj2021spr/report2/81953230.pdf
- **Research poster:** https://cs229.stanford.edu/proj2021spr/poster/81953230.pdf

The original study investigates whether machine learning models can predict future stock returns using historical returns from Volkswagen (VW) and its supply-chain companies. The dataset contains historical financial observations and precomputed return-related features.

### 2. Dataset Selection

The available CSV files included datasets for Frankfurt-listed companies and a broader dataset. For this implementation, we used **`databaseFrankfurtComplete.csv`**, which contains the historical Volkswagen Frankfurt observations alongside a large collection of financial features.

The dataset contains approximately **3,809 rows and 1,292 columns**, covering the period from January 2005 to December 2019. The target stock is identified by the ticker `VOW.F`.

Important columns include:

| Column or feature group | Purpose |
|---|---|
| `Date` | Identifies the observation date |
| `Ticker` | Identifies the stock |
| `Close` | Historical closing price of Volkswagen |
| `Return_Delta1_VW` to `Return_Delta28_VW` | Historical Volkswagen return features |
| Supplier return features | Capture historical return information from related companies |
| `Return_Future_Delta*` features | Precomputed forward-return features supplied with the dataset |

### 3. Conversion and Preprocessing

The original CSV was prepared for use in the Python machine learning pipeline. Rather than downloading a new dataset or collecting stock prices independently, we used the supplied replication data and selected the Frankfurt dataset appropriate for the experiment.

The preparation process involved the following steps:

1. **Load the CSV:** Read `databaseFrankfurtComplete.csv` into a pandas DataFrame.
2. **Inspect the schema:** Examine the dimensions, column names, data types, date coverage, ticker values and missing values.
3. **Select the target stock:** Use the Volkswagen Frankfurt observations identified by `VOW.F`.
4. **Sort chronologically:** Order observations by date to preserve the time-series structure and prevent random shuffling.
5. **Select model features:** Use historical return columns and other permitted numerical features available at prediction time.
6. **Prepare the prediction target:** Use the closing-price series to construct the forward return for the selected prediction horizon.
7. **Engineer additional features:** Calculate rolling momentum and volatility features over 5-, 10-, 20- and 60-observation windows.
8. **Prepare model inputs:** Handle missing feature values using median imputation and apply feature scaling for models that require it.
9. **Split the observations:** Use a chronological 60:40 split for fitting and evaluation, rather than a random train-test split.

The target return is defined as:

\[
y_t = \frac{P_{t+h}}{P_t}-1
\]

where \(P_t\) is the closing price at time \(t\), \(P_{t+h}\) is the closing price after \(h\) trading observations, and \(h\) is the selected prediction horizon.

For example, a 20-observation horizon estimates the percentage price change between the current closing price and the closing price 20 trading observations later.

### 4. Final Dataset Used by the Models

After preparation, the data is represented as a chronological feature matrix \(X\) and a target vector \(y\). These are passed to the machine learning models for return prediction.

The implementation evaluates Elastic Net, Ridge Regression and Extra Trees Regression. Their predictions are subsequently used to construct long-only and long-short trading signals and compare the resulting strategies against a buy-and-hold benchmark.

**Reproducibility note:** The supplied dataset already contains engineered return columns. The project uses these existing features where appropriate and constructs additional features for the selected pipeline. The forward-return target is derived from the closing-price series; precomputed forward-return columns are not automatically assumed to follow the same target convention.

### 5. Reference

Glawion, R. M., *Quantitative Trading with Machine Learning*. Stanford CS229 course project, Spring 2021.

- Paper: https://cs229.stanford.edu/proj2021spr/report2/81953230.pdf
- Replication repository: https://github.com/rglawion/cs229_project_report

## Methodology

### 1. Target construction

The target is the forward return over the next 20 rows:

\[
y_t = \frac{Close_{t+20}}{Close_t} - 1
\]

The dataset is sorted chronologically before the target is created. The final 20 rows do not have a complete forward target and are excluded from supervised learning.

### 2. Feature engineering

The feature set includes available supplier-return features, Volkswagen return lags from 1 to 28 rows, and price-derived features:

- Momentum over 5, 10, 20, and 60 rows.
- Rolling return volatility over 5, 10, 20, and 60 rows.

Non-numeric values are converted to missing values, and infinite values are treated as missing. Entirely missing or constant columns are removed. Median imputation is included in the model pipelines.

### 3. Chronological split

The notebook uses a chronological **60/40 split**:

| Segment | Share | Purpose |
|---|---:|---|
| Fit period | First 60% | Fit models and choose a trading rule |
| Evaluation period | Remaining 40% | Evaluate the selected rules |

Trading performance is calculated over sampled, non-overlapping 20-row periods.

**Known limitation:** The current purge logic uses 20 calendar days around the split rather than purging by 20 trading-row positions. Calendar days and trading observations are not equivalent.

### 4. Model training

The notebook trains three regression models:

| Model | Configuration | Rationale |
|---|---|---|
| Elastic Net | `alpha=0.01`, `l1_ratio=0.2` | Combines L1 and L2 regularization |
| Ridge Regression | `alpha=10.0` | Regularizes a linear regression model |
| Extra Trees Regression | 250 trees, `min_samples_leaf=15`, `max_features=0.7` | Captures nonlinear patterns through an ensemble of randomized trees |

Training targets are clipped at the 1st and 99th percentiles to reduce the influence of extreme values. Median imputation is used for all models; Elastic Net and Ridge also use feature standardization.

### 5. Trading-rule selection

The notebook evaluates two position modes:

- **Long-only:** enter a long position when the prediction exceeds the selected threshold; otherwise hold cash.
- **Long-short:** enter long when the prediction exceeds the threshold, short when it is below the negative threshold, and otherwise remain neutral.

Candidate thresholds are based on quantiles of the absolute predictions from the fit period. The current implementation chooses the threshold and position mode with the highest approximate annualized Sharpe ratio on those same fit-period observations.

**Important:** Because the models were fitted on these observations, these predictions are in-sample. Selecting a trading rule using in-sample predictions can overstate performance.

### 6. Transaction costs

The configured transaction cost is:

```python
TRANSACTION_COST = 0.001  # 0.1% per unit of position turnover
```

Costs are applied when positions change. For example, changing from long (`+1`) to short (`-1`) represents two units of turnover, while changing from neutral (`0`) to long (`+1`) represents one.

This simplified cost model does not fully represent real execution costs.

## Models

### Elastic Net

Linear regression with both L1 and L2 regularization. It can shrink less useful coefficients and help stabilize estimates when features are correlated.

### Ridge Regression

Linear regression with L2 regularization. It penalizes large coefficients and can be useful when many correlated return features are present.

### Extra Trees Regression

An ensemble of randomized decision trees. It can model nonlinear relationships, but may also overfit noisy financial data.

## Evaluation Metrics

### Prediction metrics

| Metric | Interpretation |
|---|---|
| Mean Absolute Error (MAE) | Average absolute difference between predicted and actual returns |
| Zero-baseline MAE | MAE when the predicted return is always zero |
| MAE improvement (%) | Relative MAE improvement compared with the zero-return baseline |
| Directional accuracy (%) | Share of predictions with the correct return direction |

Positive MAE improvement indicates that the model's MAE is lower than the zero-return baseline. Negative improvement indicates worse MAE than the baseline.

### Trading metrics

| Metric | Interpretation |
|---|---|
| Strategy return | Compounded return from the model-driven positions |
| Buy-and-hold return | Compounded return from holding the asset over the sampled periods |
| Sharpe ratio | Approximate risk-adjusted performance |
| Maximum drawdown | Largest peak-to-trough decline in the equity curve |
| Invested percentage | Share of sampled periods with a non-neutral position |

Results should be interpreted jointly. A higher return alone does not establish a better strategy, and a positive historical result does not establish future profitability.

## Notebook Structure

| Cell | Purpose |
|---|---|
| 1 | Imports, configuration, dataset loading, and cleaning |
| 2 | Forward-target construction and feature engineering |
| 3 | Chronological 60/40 split and sampled trading periods |
| 4 | Fit Elastic Net, Ridge, and Extra Trees |
| 5 | Select trading thresholds and position modes |
| 6 | Calculate prediction and portfolio metrics |
| 7 | Display the model-comparison table |
| 8 | Plot returns, Sharpe ratios, drawdowns, and prediction errors |

## Requirements

- Python 3.10 or later recommended
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook or Google Colab

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

1. Ensure `databaseFrankfurtComplete.csv` is available at `/content/databaseFrankfurtComplete.csv`.
2. If using a local Jupyter environment, change `DATA_PATH` in Cell 1 to the correct local path.
3. Open the notebook.
4. Run Cells 1–8 in order.
5. Review the printed metrics table and generated charts.

The notebook does not require a Google upload prompt and does not create ZIP archives. It may export a CSV results table, depending on the final cell configuration.

## Limitations

1. **In-sample threshold selection:** The model is evaluated on fit-period predictions when choosing its trading threshold and position mode. This introduces selection bias.
2. **Calendar-day purge:** The current purge uses calendar days instead of trading-row positions, so it may not fully prevent forward-target overlap.
3. **Feature timing:** Supplier and proxy features must be checked to ensure they were available before the simulated trade. Their names alone cannot establish that.
4. **Simplified execution assumptions:** The backtest does not fully model slippage, bid-ask spreads, market impact, liquidity, borrow fees, or short-sale restrictions.
5. **Single evaluation segment:** A single chronological evaluation period cannot establish robustness across different market regimes.
6. **Model-selection risk:** Comparing multiple models and thresholds can produce a configuration that fits historical noise.
7. **Reference-paper differences:** This notebook uses a 20-row target, three models, and a 60/40 split. It should not be described as an exact reproduction of a paper that uses different models, data, horizons, or rolling evaluation windows.

## Future Improvements

- Introduce a separate chronological validation segment for model and trading-rule selection.
- Keep the final test period untouched until all decisions are fixed.
- Purge observations using trading-row positions to match the forward-label horizon.
- Audit feature timestamps and target alignment.
- Add RMSE and \(R^2\) alongside MAE.
- Compare against zero-return, buy-and-hold, and simple momentum baselines.
- Test performance under different transaction costs and market periods.
- Add automated checks for target alignment, split boundaries, turnover costs, and benchmark consistency.

## Disclaimer

This project is intended for educational and research purposes only. It is not investment advice, and the backtest should not be used as the basis for live trading without substantial additional validation.
