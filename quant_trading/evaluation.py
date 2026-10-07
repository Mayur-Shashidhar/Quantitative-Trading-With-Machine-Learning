import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(actual, predicted):
    return {"RMSE": float(np.sqrt(mean_squared_error(actual, predicted))), "MAE": float(mean_absolute_error(actual, predicted)), "R2": float(r2_score(actual, predicted))}

def compare_models(fitted_models, features, target):
    return pd.DataFrame([{"Model": name, **regression_metrics(target, fitted.predict(features))} for name, fitted in fitted_models.items()]).sort_values("RMSE").reset_index(drop=True)

def performance_metrics(returns):
    values = np.asarray(returns, dtype=float); values = values[~np.isnan(values)]
    if len(values) == 0: raise ValueError("Cannot evaluate an empty return series")
    equity = np.cumprod(1 + values); total_return = equity[-1] - 1; annualized_return = (1 + total_return) ** (252 / len(values)) - 1
    annualized_volatility = np.std(values, ddof=1) * np.sqrt(252) if len(values) > 1 else 0.0; sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    return {"Total Return": float(total_return), "Annualized Return": float(annualized_return), "Annualized Volatility": float(annualized_volatility), "Sharpe Ratio": float(sharpe), "Maximum Drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()), "Win Rate": float(np.mean(values > 0))}
