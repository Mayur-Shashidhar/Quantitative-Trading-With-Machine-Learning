from .config import ProjectConfig
from .data import align_close_prices, clean_supply_prices, download_prices
from .evaluation import compare_models, performance_metrics, regression_metrics
from .features import build_baseline_dataset, build_enhanced_dataset, chronological_split
from .models import fit_model, model_templates
from .strategy import backtest, benchmark_buy_and_hold, select_threshold
import pandas as pd

def run_experiment(config=None):
    config = config or ProjectConfig(); downloaded = download_prices([config.ticker, *config.supply_chain_tickers], config.start_date, config.end_date); vw_prices = downloaded[config.ticker]; close_prices = clean_supply_prices(align_close_prices(downloaded), config.ticker); templates = model_templates(config.seed)
    baseline_x, baseline_y = build_baseline_dataset(vw_prices); base = chronological_split(baseline_x, baseline_y, config.train_fraction, config.validation_fraction); baseline_models = {n: fit_model(n, t, *base["train"]) for n, t in templates.items()}; baseline_validation = compare_models(baseline_models, *base["validation"])
    enhanced_x, enhanced_targets = build_enhanced_dataset(close_prices, config.ticker, config.supply_lag_days, config.horizons); split = chronological_split(enhanced_x, enhanced_targets, config.train_fraction, config.validation_fraction); rows = []
    for horizon in config.horizons:
        for name, template in templates.items():
            fitted = fit_model(name, template, split["train"][0], split["train"][1][f"Target_{horizon}D"]); rows.append({"Horizon": horizon, "Model": name, **regression_metrics(split["validation"][1][f"Target_{horizon}D"], fitted.predict(split["validation"][0]))})
    horizon_results = pd.DataFrame(rows); best = horizon_results.loc[horizon_results["RMSE"].idxmin()]; horizon, model_name = int(best["Horizon"]), str(best["Model"]); target_name = f"Target_{horizon}D"; train_x = pd.concat([split["train"][0], split["validation"][0]]); train_y = pd.concat([split["train"][1][target_name], split["validation"][1][target_name]]); final_model = fit_model(model_name, templates[model_name], train_x, train_y); test_x, test_y = split["test"]; predictions = final_model.predict(test_x)
    result = {"config": config, "baseline_validation": baseline_validation, "horizon_results": horizon_results, "selected_model": model_name, "selected_horizon": horizon, "final_regression": regression_metrics(test_y[target_name], predictions), "trading": None}
    if horizon == 1:
        validation_model = fit_model(model_name, templates[model_name], split["train"][0], split["train"][1][target_name]); validation_predictions = validation_model.predict(split["validation"][0]); threshold, threshold_results = select_threshold(validation_predictions, split["validation"][1][target_name], config.threshold_grid, config.minimum_active_days); tested = backtest(predictions, test_y[target_name], threshold, config.transaction_cost); buy_hold = benchmark_buy_and_hold(vw_prices["Close"].reindex(test_x.index)); result["trading"] = {"threshold": threshold, "threshold_results": threshold_results, "backtest": tested, "buy_hold_returns": buy_hold, "strategy_metrics": performance_metrics(tested["net_returns"]), "buy_hold_metrics": performance_metrics(buy_hold)}
    return result
