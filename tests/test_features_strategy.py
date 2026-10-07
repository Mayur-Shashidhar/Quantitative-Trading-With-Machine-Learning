import numpy as np
import pandas as pd

from quant_trading.features import build_baseline_dataset, build_enhanced_dataset, chronological_split
from quant_trading.strategy import backtest, select_threshold


def sample_prices(rows=140):
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    base = np.linspace(10, 15, rows)
    return pd.DataFrame({"Open": base, "High": base + .2, "Low": base - .2, "Close": base, "Volume": np.arange(rows) + 1000}, index=dates)


def test_baseline_and_split_are_chronological():
    x, y = build_baseline_dataset(sample_prices())
    split = chronological_split(x, y)
    assert list(x.columns) == ["Return_1D", "Return_5D", "Return_10D", "Return_20D", "Price_MA5_Ratio", "Price_MA20_Ratio", "Price_MA50_Ratio", "Volatility_5D", "Volatility_20D", "Volume_Change", "Volume_MA20", "Volume_Ratio"]
    assert len(split["train"][0]) + len(split["validation"][0]) + len(split["test"][0]) == len(x)
    assert split["train"][0].index.max() < split["validation"][0].index.min()


def test_enhanced_feature_dimensions_and_target_alignment():
    prices = pd.concat({"VWAGY": sample_prices(), "SUP": sample_prices() + 1}, axis=1)
    closes = prices.xs("Close", axis=1, level=1)
    x, targets = build_enhanced_dataset(closes, horizons=(1, 5))
    assert x.shape[1] == 28 + 12
    assert list(targets.columns) == ["Target_1D", "Target_5D"]
    assert x.index.equals(targets.index)


def test_backtest_charges_open_and_switching_positions():
    result = backtest([1, -1, 0], [0.01, 0.01, 0.0], threshold=0, transaction_cost=0.001)
    np.testing.assert_allclose(result["transaction_costs"], [0.001, 0.002, 0.001])


def test_threshold_selection_uses_only_validation_inputs():
    threshold, table = select_threshold([.003, -.003, 0], [.01, .01, 0], [0, .002], minimum_active_days=1)
    assert threshold in {0.0, 0.002}
    assert len(table) == 2
