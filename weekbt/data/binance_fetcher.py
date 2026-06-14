"""Binance OHLCV downloader using ccxt with CSV caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from weekbt.data.ohlcv_storage import OHLCV_COLUMNS, load_ohlcv_csv, save_ohlcv_csv
from weekbt.time_utils import ensure_utc


def sanitize_symbol(symbol: str) -> str:
    """Convert exchange symbols into filesystem-safe names."""

    return symbol.replace("/", "_").replace(":", "_")


def cache_path(cache_dir: Path, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """Build the deterministic cache path for a market data request."""

    start_part = start.strftime("%Y%m%d")
    end_part = end.strftime("%Y%m%d")
    filename = f"{sanitize_symbol(symbol)}_{timeframe}_{start_part}_{end_part}.csv"
    return cache_dir / filename


class BinanceOHLCVFetcher:
    """Download public Binance spot OHLCV candles through ccxt."""

    def __init__(self, exchange: Any | None = None) -> None:
        self.exchange = exchange or ccxt.binance({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch all candles in the inclusive UTC date range."""

        start = ensure_utc(start)
        end = ensure_utc(end)
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list[float]] = []

        while since <= end_ms:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break

            for candle in batch:
                if candle[0] > end_ms:
                    break
                rows.append(candle[:6])

            next_since = int(batch[-1][0]) + 1
            if next_since <= since:
                break
            since = next_since

            if batch[-1][0] >= end_ms:
                break

        if not rows:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].astype(float)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df.loc[:, OHLCV_COLUMNS]

    def fetch_with_cache(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cache_dir: Path,
    ) -> pd.DataFrame:
        """Load OHLCV from CSV cache or download and cache it."""

        path = cache_path(cache_dir, symbol, timeframe, start, end)
        if path.exists():
            return load_ohlcv_csv(path)

        df = self.fetch_ohlcv(symbol, timeframe, start, end)
        save_ohlcv_csv(df, path)
        return df
