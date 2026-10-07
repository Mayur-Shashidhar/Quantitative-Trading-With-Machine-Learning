from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_FEATURES = ["Return_1D", "Return_5D", "Return_10D", "Return_20D", "Price_MA5_Ratio", "Price_MA20_Ratio", "Price_MA50_Ratio", "Volatility_5D", "Volatility_20D", "Volume_Change", "Volume_MA20", "Volume_Ratio"]

def _clean(frame):
    return frame.replace([np.inf, -np.inf], np.nan).dropna()

def build_baseline_dataset(vw_prices):
    frame = vw_prices.copy(); close, volume = frame["Close"], frame["Volume"]
    for window in (1, 5, 10, 20): frame[f"Return_{window}D"] = close.pct_change(window)
    for window in (5, 10, 20, 50): frame[f"MA_{window}"] = close.rolling(window).mean()
    for window in (5, 20, 50): frame[f"Price_MA{window}_Ratio"] = close / frame[f"MA_{window}"] - 1
    frame["Volatility_5D"] = frame["Return_1D"].rolling(5).std(); frame["Volatility_20D"] = frame["Return_1D"].rolling(20).std()
    frame["Volume_Change"] = volume.pct_change(); frame["Volume_MA20"] = volume.rolling(20).mean(); frame["Volume_Ratio"] = volume / frame["Volume_MA20"]
    frame["Target_Return"] = close.shift(-1) / close - 1; frame = _clean(frame)
    return frame[BASELINE_FEATURES].copy(), frame["Target_Return"].copy()

def build_enhanced_dataset(close_prices, target_ticker="VWAGY", lag_days=28, horizons=(1, 5, 10, 15, 20)):
    target = close_prices[target_ticker]; returns = close_prices.pct_change(); supplier_returns = returns.drop(columns=[target_ticker], errors="ignore")
    supply_columns = {}
    for ticker in supplier_returns.columns:
        for lag in range(1, lag_days + 1):
            supply_columns[f"{ticker}_Return_Lag_{lag}"] = supplier_returns[ticker].shift(lag)
    supply = pd.DataFrame(supply_columns, index=close_prices.index)
    vw = pd.DataFrame(index=close_prices.index); vw["VW_Return_1D"] = target.pct_change()
    for window in (5, 10, 20): vw[f"VW_Return_{window}D"] = target.pct_change(window)
    for window in (5, 20, 50): vw[f"VW_MA{window}_Ratio"] = target / target.rolling(window).mean() - 1
    vw["VW_Volatility_5D"] = target.pct_change().rolling(5).std(); vw["VW_Volatility_20D"] = target.pct_change().rolling(20).std()
    vw["VW_Return_60D"] = target.pct_change(60); vw["VW_Momentum_20D"] = target / target.shift(20) - 1; vw["VW_Momentum_60D"] = target / target.shift(60) - 1
    targets = pd.DataFrame(index=close_prices.index)
    for horizon in horizons: targets[f"Target_{horizon}D"] = target.shift(-horizon) / target - 1
    full = _clean(pd.concat([supply, vw, targets], axis=1)); feature_columns = list(supply.columns) + list(vw.columns)
    return full[feature_columns].copy(), full[targets.columns].copy()

def chronological_split(frame, target, train_fraction=0.60, validation_fraction=0.20):
    train_end = int(len(frame) * train_fraction); validation_end = int(len(frame) * (train_fraction + validation_fraction))
    return {"train": (frame.iloc[:train_end].copy(), target.iloc[:train_end].copy()), "validation": (frame.iloc[train_end:validation_end].copy(), target.iloc[train_end:validation_end].copy()), "test": (frame.iloc[validation_end:].copy(), target.iloc[validation_end:].copy())}
