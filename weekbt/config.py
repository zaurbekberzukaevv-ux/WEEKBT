"""Configuration dataclasses for the backtesting framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the similarity-based analog strategy."""

    window_size: int = 20
    horizon: int = 5
    top_n: int = 10
    exclusion_radius: int = 0
    train_months: int = 6
    test_months: int = 6


@dataclass(frozen=True)
class MarketDataConfig:
    """Parameters describing the requested market data universe."""

    symbol: str = "ETH/USDT"
    timeframe: str = "1d"
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class PathConfig:
    """Filesystem locations for cache and reports."""

    cache_dir: Path = Path("data/cache")
    reports_dir: Path = Path("reports")


@dataclass(frozen=True)
class BacktestConfig:
    """Top-level configuration used by the CLI and backtest runner."""

    market: MarketDataConfig
    strategy: StrategyConfig
    paths: PathConfig
