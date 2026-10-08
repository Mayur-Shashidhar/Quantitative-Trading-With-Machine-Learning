# Quantitative Trading With Machine Learning

This repository contains the original notebook/report and a modular Python implementation of the same VWAGY return-prediction experiment.

## Layout

- `quant_trading/data.py` — Yahoo Finance download, column normalization, missing-data handling, and price alignment.
- `quant_trading/features.py` — baseline technical features, supply-chain lag features, targets, and chronological splits.
- `quant_trading/models.py` — the four original regressors and training-only Elastic Net scaling.
- `quant_trading/evaluation.py` — regression and portfolio metrics.
- `quant_trading/strategy.py` — validation threshold selection, signals, transaction costs, and Buy & Hold returns.
- `quant_trading/pipeline.py` — end-to-end orchestration.
- `quant_trading/cli.py` — command-line entry point.
- `Quantitative_Trading_ML_Modular_Colab.ipynb` — Colab-ready presentation notebook with tables, plots, and interpretation.
- `Quantitative_Trading_With_Machine_Learning.ipynb` — original executable narrative, retained for comparison.

## Run

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
quant-trading --start 2015-01-01 --end 2025-12-31
```

Yahoo Finance access is required at runtime. The command downloads the same adjusted daily data used by the notebook and prints the selected configuration and final metrics.

Run the smoke tests with:

```bash
python -m pytest -q
```

## Methodology preserved

The refactor deliberately retains the original 60/20/20 chronological split, feature windows, 28 lagged supplier returns, five forecast horizons, validation-based selection, 0.1% transaction cost, and long/short/neutral signal rule. It does not claim that the reported metrics are reproducible indefinitely: Yahoo Finance data can be revised, ticker availability can change, and model-library versions can alter numerical results.

The original notebook and final report are not modified.
