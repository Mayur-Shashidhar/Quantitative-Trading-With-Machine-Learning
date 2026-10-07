from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import performance_metrics


def signals_from_predictions(predictions, threshold: float) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    return np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))


def select_threshold(predictions, actual, thresholds, minimum_active_days: int = 50) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in thresholds:
        signals = signals_from_predictions(predictions, threshold)
        strategy_returns = signals * np.asarray(actual)
        metrics = performance_metrics(strategy_returns)
        rows.append({"Threshold": threshold, "Total Return": metrics["Total Return"], "Annualized Return": metrics["Annualized Return"], "Volatility": metrics["Annualized Volatility"], "Sharpe": metrics["Sharpe Ratio"], "Active Days": int(np.sum(signals != 0))})
    results = pd.DataFrame(rows)
    eligible = results[results["Active Days"] >= minimum_active_days]
    selected = float(eligible.loc[eligible["Sharpe"].idxmax(), "Threshold"]) if len(eligible) else 0.0
    return selected, results


def backtest(predictions, actual, threshold: float, transaction_cost: float = 0.001) -> dict[str, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    signals = signals_from_predictions(predictions, threshold)
    gross = signals * actual
    position_change = np.abs(np.diff(np.concatenate([[0], signals])))
    costs = position_change * transaction_cost
    return {"signals": signals, "gross_returns": gross, "transaction_costs": costs, "net_returns": gross - costs}


def benchmark_buy_and_hold(close_prices) -> np.ndarray:
    return pd.Series(close_prices).pct_change().fillna(0).to_numpy()
