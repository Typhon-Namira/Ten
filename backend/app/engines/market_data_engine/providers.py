"""Provider contracts, capabilities, and deterministic local adapters."""

import csv
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Candle, Tick, Timeframe, canonical_symbol


class ProviderName(StrEnum):
    TWELVE_DATA = "twelve_data"
    ALPHA_VANTAGE = "alpha_vantage"
    FMP = "financial_modeling_prep"
    OANDA = "oanda"
    CSV = "csv"
    MEMORY = "memory"
    POLYGON = "polygon"
    TRADING_VIEW = "trading_view"
    BINANCE = "binance"
    MT5 = "mt5"
    INTERACTIVE_BROKERS = "interactive_brokers"
    DXFEED = "dxfeed"
    DATABENTO = "databento"
    # ---- Keyless public sources (see `docs/MARKET_DATA_ENGINE.md`). LBMA is the authoritative,
    # non-proxy daily gold benchmark. Kraken/OKX are gold-token (PAXG/XAUT) proxy instruments used
    # only for intraday coverage and cross-source validation, never as the sole source. Yahoo
    # Finance and Stooq are kept as fully-implemented but permanently-disabled-by-default adapters
    # because both hosts' robots.txt explicitly disallow automated access — see `adapters.py`.
    LBMA_GOLD_PRICE = "lbma_gold_price"
    KRAKEN = "kraken"
    OKX = "okx"
    YAHOO_FINANCE = "yahoo_finance"
    STOOQ = "stooq"


class ProviderCapabilities(BaseModel):
    historical: bool = True
    realtime_polling: bool = True
    websocket_streaming: bool = False
    tick_data: bool = False
    spread: bool = False
    volume: bool = True
    # `supported_timeframes` already expresses exactly which candle intervals a provider is
    # entitled to — these two cover the concepts it doesn't: a lightweight live-quote endpoint
    # (used for connectivity health checks, independent of any specific candle dataset) and
    # economic-calendar data (not a candle interval at all).
    live_quote: bool = False
    economic_calendar: bool = False
    supported_symbols: tuple[str, ...] = ()
    supported_timeframes: tuple[Timeframe, ...] = tuple(Timeframe)
    maximum_history_candles: int | None = Field(default=None, ge=1)

    def supports(self, symbol: str, timeframe: Timeframe, *, realtime: bool = False) -> bool:
        symbol_supported = not self.supported_symbols or canonical_symbol(symbol) in self.supported_symbols
        mode_supported = self.realtime_polling if realtime else self.historical
        return symbol_supported and timeframe in self.supported_timeframes and mode_supported


class ProviderRequest(BaseModel):
    symbol: str
    timeframe: Timeframe
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=500, ge=1, le=100_000)


class ProviderQuota(BaseModel):
    limit: int | None = Field(default=None, ge=1)
    remaining: int | None = Field(default=None, ge=0)
    reset_at: datetime | None = None


class MarketDataProvider(ABC):
    """Port implemented by every external market-data adapter."""

    provider_name: ProviderName
    capabilities: ProviderCapabilities

    @abstractmethod
    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        """Return normalized candles ordered oldest to newest."""

    @abstractmethod
    async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        """Return the most recent completed normalized candle."""

    async def candles(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> list[Candle]:
        """Backward-compatible historical query."""

        return await self.fetch_history(ProviderRequest(symbol=symbol, timeframe=timeframe, limit=limit))

    async def quota(self) -> ProviderQuota:
        return ProviderQuota()

    async def close(self) -> None:
        """Release provider resources."""
        return None


class StreamingMarketDataProvider(MarketDataProvider, ABC):
    @abstractmethod
    def stream(self, symbol: str, timeframe: Timeframe) -> AsyncIterator[Candle | Tick]:
        """Yield normalized live updates."""


class InMemoryMarketDataProvider(MarketDataProvider):
    """Deterministic local adapter used for tests and explicitly injected datasets."""

    provider_name = ProviderName.MEMORY
    capabilities = ProviderCapabilities(
        historical=True,
        realtime_polling=True,
        spread=True,
        supported_timeframes=tuple(Timeframe),
    )

    def __init__(self, items: Sequence[Candle]) -> None:
        self._items = sorted(items, key=lambda item: item.timestamp)

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        symbol = canonical_symbol(request.symbol)
        selected = [
            item
            for item in self._items
            if canonical_symbol(item.symbol) == symbol
            and item.timeframe == request.timeframe
            and (request.start is None or item.timestamp >= request.start)
            and (request.end is None or item.timestamp <= request.end)
        ]
        return selected[-request.limit :]

    async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        items = await self.fetch_history(ProviderRequest(symbol=symbol, timeframe=timeframe, limit=1))
        if not items:
            raise LookupError(f"No candle available for {symbol}/{timeframe}")
        return items[-1]


class CsvHistoricalProvider(MarketDataProvider):
    """Historical OHLCV loader using the normalized provider-neutral CSV schema."""

    provider_name = ProviderName.CSV
    capabilities = ProviderCapabilities(historical=True, realtime_polling=False, spread=True)

    def __init__(self, path: Path) -> None:
        self.path = path

    async def fetch_history(self, request: ProviderRequest) -> list[Candle]:
        rows: list[Candle] = []
        with self.path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                if request.start and timestamp < request.start:
                    continue
                if request.end and timestamp > request.end:
                    continue
                rows.append(
                    Candle(
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume", 0)),
                        spread=float(row.get("spread", 0)),
                        provider=self.provider_name.value,
                    )
                )
        return sorted(rows, key=lambda item: item.timestamp)[-request.limit :]

    async def fetch_latest(self, symbol: str, timeframe: Timeframe) -> Candle:
        items = await self.fetch_history(ProviderRequest(symbol=symbol, timeframe=timeframe, limit=1))
        if not items:
            raise LookupError(f"No candle available for {symbol}/{timeframe}")
        return items[-1]


RealtimeMarketDataProvider = StreamingMarketDataProvider
