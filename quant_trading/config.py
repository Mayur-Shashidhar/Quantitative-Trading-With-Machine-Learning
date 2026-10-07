from dataclasses import dataclass, field


DEFAULT_SUPPLY_CHAIN_TICKERS = (
    "CON.DE", "IFX.DE", "HLE.DE", "BAS.DE", "BWA", "MGA", "LEA", "APTV", "TSM"
)


@dataclass(frozen=True)
class ProjectConfig:
    ticker: str = "VWAGY"
    start_date: str = "2015-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    horizons: tuple[int, ...] = (1, 5, 10, 15, 20)
    supply_chain_tickers: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SUPPLY_CHAIN_TICKERS)
    supply_lag_days: int = 28
    transaction_cost: float = 0.001
    threshold_grid: tuple[float, ...] = (0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025)
    minimum_active_days: int = 50

    def __post_init__(self):
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
