"""Provider abstractions and safe baseline implementations."""

import csv
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from .models import Candle, Tick, Timeframe


class ProviderName(StrEnum):
    FREE_API = "free_api"
    CSV = "csv"
    OANDA = "oanda"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    ALPHA_VANTAGE = "alpha_vantage"
    MT5 = "mt5"
    TRADING_VIEW = "trading_view"
    BINANCE = "binance"


class ProviderCapabilities(BaseModel):
    historical: bool = True
    realtime: bool = False
    ticks: bool = False
    supported_timeframes: tuple[Timeframe, ...] = tuple(Timeframe)


class MarketDataProvider(ABC):
    """Port for free, paid, historical, or real-time data providers."""

    provider_name: ProviderName = ProviderName.FREE_API
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    async def candles(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> list[Candle]:
        """Return normalized, chronologically ordered OHLCV candles."""


class InMemoryMarketDataProvider(MarketDataProvider):
    """Deterministic provider for tests, demos, and offline development."""

    provider_name = ProviderName.FREE_API

    def __init__(self, items: list[Candle]) -> None:
        self._items = sorted(items, key=lambda item: item.timestamp)

    async def candles(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> list[Candle]:
        selected = [item for item in self._items if item.symbol == symbol and item.timeframe == timeframe]
        return selected[-limit:]


class CsvHistoricalProvider(MarketDataProvider):
    """Historical OHLCV loader using a documented provider-neutral CSV schema."""

    provider_name = ProviderName.CSV

    def __init__(self, path: Path) -> None:
        self.path = path

    async def candles(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> list[Candle]:
        rows: list[Candle] = []
        with self.path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                rows.append(Candle(symbol=symbol, timeframe=timeframe, timestamp=timestamp, open=float(row["open"]), high=float(row["high"]), low=float(row["low"]), close=float(row["close"]), volume=float(row.get("volume", 0))))
        return sorted(rows, key=lambda item: item.timestamp)[-limit:]


class RealtimeMarketDataProvider(ABC):
    """Port reserved for authenticated streaming providers; no broker coupling."""

    @abstractmethod
    def stream(self, symbol: str) -> AsyncIterator[Tick]:
        """Yield normalized ticks from a future market-data feed."""
