import argparse

from .config import ProjectConfig
from .pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the modular VWAGY quantitative-trading experiment.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    result = run_experiment(ProjectConfig(start_date=args.start, end_date=args.end))
    print("Selected model:", result["selected_model"])
    print("Selected horizon:", result["selected_horizon"])
    print("Final regression:", result["final_regression"])
    if result["trading"]:
        print("Threshold:", result["trading"]["threshold"])
        print("Strategy:", result["trading"]["strategy_metrics"])
        print("Buy and hold:", result["trading"]["buy_hold_metrics"])


if __name__ == "__main__":
    main()
