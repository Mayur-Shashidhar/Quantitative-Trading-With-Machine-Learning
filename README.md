# Quantitative Trading With Machine Learning

A machine learning based quantitative trading research project for predicting short-term **Volkswagen AG (VWAGY)** returns using historical market data and publicly available supply-chain and industry proxy information.

## Project Overview

This project investigates whether machine learning can predict short-term future returns of Volkswagen AG's American Depositary Receipt (VWAGY) using historical market information and information from related supply-chain and industry companies.

The project is inspired by the Stanford CS229 **Quantitative Trading With Machine Learning** case study involving Volkswagen. This implementation is a **scaled-down public-data adaptation**, not an exact reproduction of the original research.

## Problem Statement

The objective is to determine whether historical VWAGY information and information from related companies can be used to predict future VWAGY returns. The problem is formulated as regression:

```text
Target_h(t) = Price(t+h) / Price(t) - 1
```

Forecast horizons: **1, 5, 10, 15, 20, and 25 trading days**.

## Dataset

**Target:** `VWAGY`

The final cleaned dataset contains **2,681 observations and 276 input features**, covering **2015-03-31 to 2025-11-24**.

### Supply-Chain / Industry Proxies

| Company | Ticker |
|---|---|
| Continental | `CON.DE` |
| Infineon | `IFX.DE` |
| HELLA | `HLE.DE` |
| BASF | `BAS.DE` |
| BorgWarner | `BWA` |
| Magna | `MGA` |
| Lear | `LEA` |
| Aptiv | `APTV` |
| TSMC | `TSM` |

These are practical publicly available supply-chain or industry proxies, not a verified complete Volkswagen supplier network. TSMC is an indirect semiconductor proxy.

## Feature Engineering

The final feature set contains **276 features**:

- **24 VWAGY technical features**
- **252 supply-chain features**
- 9 proxy companies × 28 lagged daily returns = 252 supply-chain features

The technical features include returns, volatility, absolute returns, momentum, moving-average relationships, and volume-related measures.

## Machine Learning Models

Four regression models are compared:

1. **Elastic Net** — L1/L2 regularized regression suitable for correlated lag features.
2. **Decision Tree** — captures nonlinear relationships.
3. **XGBoost** — gradient-boosted decision trees.
4. **LightGBM** — efficient gradient-boosting framework.

## Rolling Time-Series Evaluation

The final experiment uses a paper-style chronological rolling evaluation rather than a random split.

Each rolling window contains:

```text
Training   → 5 years
Validation → 1 year
Testing    → 1 month
```

The window moves forward monthly and the training period expands as more historical data becomes available.

For each window:

1. All four models are evaluated at all six horizons.
2. Validation RMSE selects the best model-horizon combination.
3. The selected configuration is retrained using available training and validation data.
4. The following month is evaluated as unseen test data.
5. Test predictions are stored as out-of-sample predictions.

## Data Leakage Prevention

The experiment uses:

- Chronological rather than random splitting
- Historical features only
- Future prices only for target construction
- Validation-only model and horizon selection
- Training-only scaling for Elastic Net
- Unseen out-of-sample test periods

## Evaluation Metrics

- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Performance relative to a mean-target baseline
- **Correlation:** Linear relationship between predicted and actual returns

Interpretation of R²:

```text
R² > 0  → Better than the mean baseline
R² ≈ 0  → Little improvement over the mean baseline
R² < 0  → Worse than the mean baseline
```

## Trading Strategy

Predicted returns are converted into simple signals using thresholds of:

```text
0.00%
0.25%
0.50%
1.00%
```

```text
Prediction > +threshold → LONG
Prediction < -threshold → SHORT
Otherwise               → NEUTRAL
```

The resulting strategies are compared with VWAGY Buy and Hold.

# Results

## Aggregate Out-of-Sample Results

| Metric | Result |
|---|---:|
| RMSE | **0.022532** |
| MAE | **0.016674** |
| R² | **-0.084046** |
| Correlation | **-0.052283** |

The aggregate out-of-sample R² is negative, meaning the complete set of rolling predictions did not outperform the simple mean-return baseline.

## Individual Rolling Windows

Several individual windows produced positive R² values, including:

```text
Window 11 → +0.0483
Window 14 → +0.0568
Window 17 → +0.0750
Window 18 → +0.0719
Window 25 → +0.0697
Window 37 → +0.0213
Window 47 → +0.0443
Window 48 → +0.0632
```

This indicates that useful predictive relationships appeared during some market regimes, but they were not stable across the complete evaluation period.

## Why Is Aggregate R² Negative?

Several factors can explain the result:

1. **Financial returns are noisy.** Short-term returns contain substantial unpredictable variation.
2. **Market regimes change.** Relationships between VWAGY and related companies can change over time.
3. **Predictions are concentrated near zero.** The model struggles to capture large positive and negative return movements.
4. **Limited supply-chain information.** Only nine public proxies are used rather than a complete supplier network.
5. **Missing macroeconomic information.** The implementation does not include the broader macroeconomic variables used in the reference methodology.
6. **No single model dominates.** Different models are selected in different rolling windows.

## Buy-and-Hold Benchmark

| Metric | Result |
|---|---:|
| Total Return | **-57.81%** |
| Sharpe Ratio | **-0.4931** |
| Maximum Drawdown | **-69.80%** |

## Key Finding

> The tested technical and supply-chain proxy features produced occasional predictive signal in individual market regimes, but the signal was not stable enough to outperform the mean-return baseline across the complete out-of-sample evaluation period.

This is a valid quantitative research result. The project evaluates the hypothesis using leakage-aware out-of-sample methodology rather than assuming that machine learning must produce a profitable strategy.

## Limitations

### Data

- Publicly available market data only
- Smaller supply-chain proxy universe than the original research
- Proxy companies are not claimed to be an exact supplier list
- Different market calendars
- Yahoo Finance data and ticker availability can change

### Modeling

- Four model families
- Limited hyperparameter tuning
- No statistical significance testing
- Financial returns are highly noisy
- Relationships can change across market regimes

### Trading

The backtest does not fully model slippage, bid-ask spreads, liquidity constraints, borrowing costs, short-sale restrictions, market impact, or capital-allocation constraints. It should therefore not be interpreted as a live trading system.

## Reference Methodology vs This Implementation

This project is inspired by the Volkswagen quantitative-trading case study but is not an exact reproduction.

| Component | Reference Study | This Project |
|---|---|---|
| Target | Volkswagen | VWAGY |
| Data period | Earlier historical period | 2015–2025 |
| Supply-chain universe | Much broader | 9 public proxies |
| Supply-chain lags | 28-day history | 28-day history |
| Macro variables | Included | Not included |
| Models | Elastic Net, Tree, XGBoost, LightGBM | Same four |
| Horizons | Multiple horizons | 1, 5, 10, 15, 20, 25 days |
| Evaluation | Rolling time-series | Rolling time-series |
| Data source | Historical market/research data | Yahoo Finance |

The project should therefore be described as a **paper-inspired, scaled-down public-data implementation**.

## Future Improvements

- Expand the supply-chain company universe
- Add macroeconomic variables
- Add market indices and broader financial indicators
- Systematically tune model hyperparameters
- Test additional ML models
- Perform statistical significance testing
- Test multiple target assets
- Improve execution-cost modelling
- Analyze feature-importance stability
- Test robustness across additional market regimes

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- yfinance
- Google Colab
- Jupyter Notebook

## Running the Project

Open the notebook in Google Colab and run the cells sequentially.

Install dependencies if required:

```bash
pip install yfinance xgboost lightgbm scikit-learn pandas numpy matplotlib
```

The notebook automatically downloads and aligns data, builds features and targets, performs rolling evaluation, compares models, generates out-of-sample predictions, calculates regression metrics, and evaluates trading strategies against Buy and Hold.

## Disclaimer

This project is an academic machine learning and quantitative-finance experiment. It is **not financial advice** and does not constitute a recommendation to buy, sell, or short VWAGY or any other security.

The negative aggregate out-of-sample R² is reported honestly and should not be artificially optimized. The purpose is to evaluate the predictive hypothesis using rigorous time-series methodology.

## Author

**S Mayur**  
B.Tech Computer Science and Engineering  
PES University, Bengaluru
