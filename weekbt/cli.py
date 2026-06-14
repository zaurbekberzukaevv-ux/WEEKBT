"""Command-line entrypoint for the backtesting framework."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from weekbt.backtest.rolling_cycle import run_backtest
from weekbt.config import BacktestConfig, MarketDataConfig, PathConfig, StrategyConfig
from weekbt.data.binance_fetcher import BinanceOHLCVFetcher
from weekbt.reporting.exporter import export_cycle_results
from weekbt.time_utils import ensure_utc, inclusive_end_of_day


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run a Binance OHLCV analog-regime backtest.")
    parser.add_argument("--symbol", default="ETH/USDT", help="Binance spot symbol, e.g. ETH/USDT.")
    parser.add_argument("--timeframe", default="1d", help="Binance timeframe, e.g. 1h, 4h, 1d.")
    parser.add_argument("--start-date", default=None, help="UTC start date YYYY-MM-DD. Defaults to 5/4/3 years back.")
    parser.add_argument("--end-date", default=None, help="UTC end date YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--exclusion-radius", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def main() -> None:
    """Download data, run the backtest, and export per-cycle reports."""

    args = build_parser().parse_args()
    end = inclusive_end_of_day(args.end_date) if args.end_date else inclusive_end_of_day(pd.Timestamp.utcnow())
    fetch_start = ensure_utc(args.start_date) if args.start_date else end - pd.DateOffset(years=5)

    fetcher = BinanceOHLCVFetcher()
    ohlcv = fetcher.fetch_with_cache(args.symbol, args.timeframe, fetch_start, end, args.cache_dir)
    if ohlcv.empty:
        raise SystemExit("No OHLCV data returned from Binance for the requested range.")

    effective_start = ensure_utc(args.start_date) if args.start_date else choose_automatic_start(ohlcv, end)
    ohlcv = ohlcv[(ohlcv["timestamp"] >= effective_start) & (ohlcv["timestamp"] <= end)].reset_index(drop=True)
    if ohlcv.empty:
        raise SystemExit("No OHLCV data remains after applying the effective date range.")

    config = BacktestConfig(
        market=MarketDataConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=effective_start.isoformat(),
            end_date=end.isoformat(),
        ),
        strategy=StrategyConfig(
            window_size=args.window_size,
            horizon=args.horizon,
            top_n=args.top_n,
            exclusion_radius=args.exclusion_radius,
        ),
        paths=PathConfig(cache_dir=args.cache_dir, reports_dir=args.reports_dir),
    )

    results = run_backtest(ohlcv, config)
    exported = export_cycle_results(results, args.reports_dir)
    total_predictions = sum(int(result["metrics"]["n_predictions"]) for result in results)

    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Effective range: {effective_start.isoformat()} -> {end.isoformat()}")
    print(f"Cycles exported: {len(exported)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Reports directory: {args.reports_dir}")


def choose_automatic_start(ohlcv: pd.DataFrame, end: pd.Timestamp) -> pd.Timestamp:
    """Choose 5, then 4, then 3 years if enough data exists; otherwise use earliest data."""

    earliest = ensure_utc(ohlcv["timestamp"].min())
    end = ensure_utc(end)
    for years in (5, 4, 3):
        candidate = end - pd.DateOffset(years=years)
        if earliest <= candidate:
            return candidate
    return earliest


if __name__ == "__main__":
    main()
