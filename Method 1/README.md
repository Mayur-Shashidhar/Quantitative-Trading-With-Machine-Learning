# Quantitative Trading With Machine Learning

## Project Overview

This project investigates whether machine learning can predict short-term future returns of Volkswagen AG's American Depositary Receipt (**VWAGY**) using historical VWAGY market information together with publicly available supply-chain and industry proxy companies.

The implementation is **paper-inspired rather than an exact reproduction** of the reference study. It uses a smaller public-data proxy universe and does not include the broader macroeconomic information used in the original methodology.

The project evaluates both:

1. **Return prediction** using multiple machine learning regression models.
2. **Trading performance** by converting predictions into Long, Short, and Neutral positions and comparing them with VWAGY Buy-and-Hold.

---

## Problem Statement

Can historical VWAGY market behaviour and lagged returns from related supply-chain and industry companies provide useful information for predicting future VWAGY returns?

For a forecast horizon \(h\), the target is defined as:

```text
Target_h(t) = Close(t+h) / Close(t) - 1
```

The project evaluates horizons of:

- 1 trading day
- 5 trading days
- 10 trading days
- 15 trading days
- 20 trading days
- 25 trading days

The goal is to evaluate out-of-sample predictive ability rather than assume that machine learning will produce a profitable strategy.

---

## Dataset

### Target Asset

**VWAGY — Volkswagen AG ADR**

Daily market data is downloaded from Yahoo Finance using `yfinance`.

Configured data period:

```text
2015-01-01 → 2026-01-01
```

The final usable period depends on feature construction, missing values, and forward-target availability.

### Supply-Chain / Industry Proxies

The project uses nine publicly traded companies as practical proxies:

| Ticker | Company / Role |
|---|---|
| `CON.DE` | Continental |
| `IFX.DE` | Infineon |
| `HLE.DE` | HELLA |
| `BAS.DE` | BASF |
| `BWA` | BorgWarner |
| `MGA` | Magna |
| `LEA` | Lear |
| `APTV` | Aptiv |
| `TSM` | TSMC — indirect semiconductor proxy |

These are **public supply-chain/industry proxies**, not a claim that they constitute the complete verified Volkswagen supplier network.

---

## Feature Engineering

The final experiment contains **276 input features**.

### VWAGY Technical Features

24 VWAGY technical features are constructed from multiple lookback windows:

- Returns
- Rolling volatility
- Absolute returns
- Momentum

Lookback windows:

```text
3, 5, 10, 20, 30, 60 trading days
```

### Supply-Chain Features

For every proxy company, daily returns are calculated and lagged over the previous 28 trading days.

```text
9 proxies × 28 lags = 252 supply-chain features
```

Therefore:

```text
24 VWAGY features
+ 252 supply-chain features
--------------------------------
= 276 total features
```

### Forward Targets

The model predicts signed future returns for:

```text
1D, 5D, 10D, 15D, 20D, 25D
```

Only historical/current information is used as model input; future prices are used only to construct the target labels.

---

## Data Preparation

The notebook:

1. Downloads VWAGY market data.
2. Downloads the nine proxy datasets.
3. Aligns proxy data to the VWAGY trading calendar.
4. Forward-fills only short proxy gaps using a maximum five-day limit.
5. Constructs VWAGY technical features.
6. Constructs 28 lagged return features for each proxy.
7. Creates forward-return targets.
8. Removes rows that remain incomplete after feature and target construction.
9. Sorts the resulting dataset chronologically.

Short forward-filling is used to handle differences between exchange trading calendars. It is deliberately limited to avoid carrying stale observations indefinitely.

---

## Machine Learning Models

Four regression models are compared:

### Elastic Net

Regularized linear regression combining L1 and L2 penalties. It is useful for a large set of correlated lag features.

Configuration used in the notebook:

```text
alpha = 0.001
l1_ratio = 0.5
max_iter = 10000
random_state = 42
```

### Decision Tree

A nonlinear tree-based regression model capable of capturing feature interactions.

### XGBoost

Gradient-boosted decision trees designed to capture nonlinear relationships and interactions.

### LightGBM

An efficient gradient-boosting framework suitable for large feature sets.

---

## Evaluation Methodology

### Paper-Style Rolling Evaluation

The final notebook uses a chronological rolling methodology rather than a random train-test split.

Each rolling window contains:

```text
5 years  → Training
1 year   → Validation
1 month  → Unseen Test
```

The window moves forward by one month.

The training period is **expanding**, meaning that historical observations remain available as the evaluation progresses.

### Model Selection

For every rolling window:

1. All four models are evaluated.
2. All six forecast horizons are evaluated.
3. Validation RMSE is calculated.
4. The model-horizon combination with the lowest validation RMSE is selected.
5. The selected model is retrained using training + validation data.
6. The following month is predicted as completely unseen out-of-sample data.

The test period is not used for model or horizon selection.

---

## Leakage Prevention

The experiment is designed to avoid time-series leakage.

Key protections include:

- Chronological rather than random evaluation.
- Features use only current and historical information.
- Future prices are used only to construct labels.
- Model and horizon selection use validation data.
- The unseen test month is evaluated only after the model choice is fixed.
- Elastic Net scaling is performed inside a pipeline so the scaler is fitted with the model's training data.

The core principle is:

> **The test data must remain unseen until final evaluation.**

---

## Out-of-Sample Experiment

The current notebook run generated:

```text
Rolling windows: 55
OOS observations: 1,150
```

All 55 rolling windows selected the **1-day forecast horizon**.

The selected model varied across market periods:

| Model | Windows selected |
|---|---:|
| Elastic Net | 27 |
| XGBoost | 18 |
| Decision Tree | 8 |
| LightGBM | 2 |

This variation demonstrates that model performance changes across market regimes.

Several individual windows produced positive test R², including windows 5, 11, 16, 17, 18, 47, 48, and 51. However, positive performance in individual windows does not establish stable predictive power across the complete out-of-sample period.

The notebook calculates the aggregate OOS RMSE, MAE, R², and correlation directly from the 1,150 OOS predictions.

---

## Evaluation Metrics

### Regression Metrics

**RMSE**

Measures the magnitude of prediction error while penalizing larger errors more heavily.

**MAE**

Measures the average absolute prediction error.

**R²**

Measures performance relative to predicting the mean target value.

A negative R² is valid. It means that the model performed worse than the mean-return baseline for the evaluated period.

**Correlation**

Measures the linear relationship between predicted and actual returns.

### Portfolio Metrics

The trading experiment reports:

- Total Return
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Win Rate

Annualization uses 252 trading periods.

---

## Trading Strategy

Model predictions are converted into positions using a prediction threshold.

```text
Prediction > +threshold  → LONG  (+1)
Prediction < -threshold  → SHORT (-1)
Otherwise                → NEUTRAL (0)
```

The notebook evaluates:

```text
0.00%
0.25%
0.50%
1.00%
```

The strategy is compared with a simple VWAGY Buy-and-Hold benchmark.

The notebook also reports the distribution of Long, Short, and Neutral signals for every threshold.

### Important Backtesting Note

This is an academic research backtest, not a live trading system. It does not fully model:

- Slippage
- Bid-ask spreads
- Market impact
- Liquidity constraints
- Borrowing costs
- Real order execution

Therefore, the trading results should not be interpreted as guaranteed real-world performance.

---

## Visualizations

The notebook produces several visual outputs, including:

1. **VWAGY historical closing-price graph**
2. **Actual vs predicted out-of-sample returns**
3. **Rolling-window R²**
4. **Out-of-sample trading equity curves**
5. **Trading signal distribution**

The equity-curve graph uses explicit datetime handling so the x-axis represents the actual OOS trading dates.

---

## Results Interpretation

The project should be interpreted as a quantitative research experiment rather than a claim of profitable forecasting.

If the aggregate OOS R² is negative, the correct interpretation is:

> The tested feature/model configuration did not outperform the mean-return baseline across the complete unseen evaluation period.

This does not mean machine learning cannot predict financial returns in general. It means that the particular information set, proxy universe, models, horizons, and historical period tested here did not produce a stable aggregate predictive signal.

Individual positive-R² windows can occur because market relationships vary across regimes. Such windows should not be selectively reported as evidence of overall success.

---

## Reference Methodology vs This Project

This implementation is **paper-inspired**, not an exact replication.

| Component | Reference-oriented idea | This project |
|---|---|---|
| Target | Volkswagen | VWAGY |
| Market data | Historical market data | Yahoo Finance |
| Supply-chain information | Broad supplier network | 9 public proxies |
| Proxy lags | Historical return lags | 28 return lags |
| Macro variables | Used in reference methodology | Not included |
| Models | Elastic Net, Tree, XGBoost, LightGBM | Same four |
| Forecast horizons | Multiple horizons | 1, 5, 10, 15, 20, 25 days |
| Evaluation | Rolling time-series style | 5Y train / 1Y validation / 1M test |
| Training | Time-aware | Expanding monthly rolling evaluation |

The main limitation is that the original study's broader supplier and macroeconomic information is not fully available in this public-data implementation.

---

## Why the Negative Result Is Valuable

A negative result is still informative.

The experiment demonstrates:

- How to construct a financial prediction dataset.
- How supply-chain proxy information can be incorporated.
- How to create forward-return targets.
- How to compare multiple regression models.
- How to avoid random time-series leakage.
- How to perform rolling out-of-sample evaluation.
- How to convert predictions into trading signals.
- How to compare a model strategy with Buy-and-Hold.

Most importantly, the project demonstrates why strong-looking predictions or individual successful periods are not enough. A signal must remain useful across genuinely unseen market periods.

---

## Limitations

1. The proxy universe is smaller than the complete supplier universe used by the reference study.
2. The proxy companies are practical public proxies, not a verified exact supplier list.
3. Macro-economic variables from the reference methodology are not included.
4. Yahoo Finance data and ticker availability can change over time.
5. Financial relationships can change across market regimes.
6. Hyperparameters are constrained rather than exhaustively optimized.
7. The trading backtest uses simplified execution assumptions.
8. A negative aggregate R² indicates limited predictive power for this tested configuration.

---

## Future Work

Possible extensions include:

- Expanding the supply-chain company universe.
- Adding macroeconomic variables.
- Adding market indices and sector-level variables.
- Systematic hyperparameter optimization.
- Nested or stronger walk-forward validation.
- Statistical significance testing.
- Feature-importance stability analysis.
- Testing additional market regimes and assets.
- More realistic transaction-cost and execution modelling.
- Robustness testing across different feature sets and thresholds.

---

## Technologies Used

- Python
- Google Colab / Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- yfinance

---

## How to Run

### 1. Open the notebook

Open the `.ipynb` file in Google Colab or Jupyter Notebook.

### 2. Install required packages if necessary

```bash
pip install yfinance xgboost lightgbm
```

### 3. Run the notebook sequentially

The notebook is organized into stages:

```text
1. Imports & configuration
2. Data collection
3. Feature engineering
4. Rolling-window generation
5. Model selection & OOS prediction
6. Aggregate OOS evaluation
7. Trading strategy & Buy-and-Hold
```

Because the data is downloaded from Yahoo Finance, results can change slightly between runs due to data revisions, ticker availability, date alignment, and library behaviour.

---

## Project Structure

```text
.
├── Quantitative_Trading_With_Machine_Learning(2).ipynb
└── README.md
```

---

## Academic Disclaimer

This project is an academic machine learning and quantitative-finance experiment. It is not financial advice and does not constitute a recommendation to buy, sell, or short VWAGY or any other security.

Historical backtest results do not guarantee future performance.

---
