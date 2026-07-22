"""Market-data persistence ports and deterministic in-process implementation."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import HistoricalCandleRecord, RealtimeCandleRecord
from backend.app.storage.batching import bounded_insert_chunks
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import Candle, Timeframe, canonical_symbol


class MarketDataRepository(ABC):
    @abstractmethod
    async def upsert_historical(self, candles: list[Candle]) -> int:
        """Persist normalized historical candles idempotently."""

    @abstractmethod
    async def append_realtime(self, candle: Candle) -> None:
        """Persist one realtime observation."""

    @abstractmethod
    async def history(self, symbol: str, timeframe: Timeframe, start: datetime | None = None, end: datetime | None = None, limit: int = 500) -> list[Candle]:
        """Query a chronological series."""

    @abstractmethod
    async def candle_at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> Candle | None:
        """Return the exact candle visible at a historical timestamp."""

    @abstractmethod
    async def count(self, symbol: str, timeframe: Timeframe) -> int:
        """Return the durable candle count for one series."""


class InMemoryMarketDataRepository(MarketDataRepository):
    def __init__(self) -> None:
        self._historical: dict[tuple[str, Timeframe, datetime], Candle] = {}
        self._realtime: list[Candle] = []
        self._lock = asyncio.Lock()

    async def upsert_historical(self, candles: list[Candle]) -> int:
        async with self._lock:
            for candle in candles:
                self._historical[(canonical_symbol(candle.symbol), candle.timeframe, candle.timestamp)] = candle
        return len(candles)

    async def append_realtime(self, candle: Candle) -> None:
        async with self._lock:
            self._realtime.append(candle)
            self._historical[(canonical_symbol(candle.symbol), candle.timeframe, candle.timestamp)] = candle

    async def history(self, symbol: str, timeframe: Timeframe, start: datetime | None = None, end: datetime | None = None, limit: int = 500) -> list[Candle]:
        normalized = canonical_symbol(symbol)
        async with self._lock:
            selected = [
                candle
                for (item_symbol, item_timeframe, _), candle in self._historical.items()
                if item_symbol == normalized
                and item_timeframe == timeframe
                and (start is None or candle.timestamp >= start)
                and (end is None or candle.timestamp <= end)
            ]
        return sorted(selected, key=lambda item: item.timestamp)[-limit:]

    async def candle_at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> Candle | None:
        candles = await self.history(symbol, timeframe, end=timestamp, limit=1)
        return candles[-1] if candles else None

    async def count(self, symbol: str, timeframe: Timeframe) -> int:
        normalized = canonical_symbol(symbol)
        async with self._lock:
            return sum(1 for item_symbol, item_timeframe, _ in self._historical if item_symbol == normalized and item_timeframe == timeframe)


class SqlAlchemyMarketDataRepository(MarketDataRepository, ScopedSessionRepository):
    """PostgreSQL adapter using bulk conflict-safe writes and indexed range reads."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)
        self._lock = asyncio.Lock()

    @scoped_session
    async def upsert_historical(self, candles: list[Candle]) -> int:
        async with self._lock:
            return await self._upsert_historical(candles)

    async def _upsert_historical(self, candles: list[Candle]) -> int:
        if not candles:
            return 0
        values = [self._values(item) for item in candles]
        try:
            for chunk in bounded_insert_chunks(values):
                statement = insert(HistoricalCandleRecord).values(list(chunk))
                statement = statement.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "timestamp"],
                    set_={name: getattr(statement.excluded, name) for name in ("open", "high", "low", "close", "volume", "spread", "provider", "quality_score", "quality_level", "ingestion_timestamp")},
                )
                await self.session.execute(statement)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return len(candles)

    @scoped_session
    async def append_realtime(self, candle: Candle) -> None:
        async with self._lock:
            self.session.add(
                RealtimeCandleRecord(
                    symbol=candle.symbol,
                    timeframe=candle.timeframe.value,
                    timestamp=candle.timestamp,
                    payload=candle.model_dump(mode="json"),
                    received_at=candle.ingestion_timestamp,
                )
            )
            await self._upsert_historical([candle])

    @scoped_session
    async def history(self, symbol: str, timeframe: Timeframe, start: datetime | None = None, end: datetime | None = None, limit: int = 500) -> list[Candle]:
        statement = select(HistoricalCandleRecord).where(
            HistoricalCandleRecord.symbol == canonical_symbol(symbol),
            HistoricalCandleRecord.timeframe == timeframe.value,
        )
        if start is not None:
            statement = statement.where(HistoricalCandleRecord.timestamp >= start)
        if end is not None:
            statement = statement.where(HistoricalCandleRecord.timestamp <= end)
        statement = statement.order_by(HistoricalCandleRecord.timestamp.desc()).limit(limit)
        async with self._lock:
            records = list((await self.session.scalars(statement)).all())
        return [self._candle(item) for item in reversed(records)]

    async def candle_at(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> Candle | None:
        candles = await self.history(symbol, timeframe, end=timestamp, limit=1)
        return candles[-1] if candles else None

    @scoped_session
    async def count(self, symbol: str, timeframe: Timeframe) -> int:
        query = select(func.count()).select_from(HistoricalCandleRecord).where(
            HistoricalCandleRecord.symbol == canonical_symbol(symbol),
            HistoricalCandleRecord.timeframe == timeframe.value,
        )
        async with self._lock:
            return int((await self.session.scalar(query)) or 0)

    @staticmethod
    def _values(candle: Candle) -> dict[str, object]:
        return {
            "symbol": candle.symbol, "timeframe": candle.timeframe.value, "timestamp": candle.timestamp,
            "open": candle.open, "high": candle.high, "low": candle.low, "close": candle.close,
            "volume": candle.volume, "spread": candle.spread, "provider": candle.provider,
            "quality_score": candle.quality_score, "quality_level": candle.quality_level.value,
            "ingestion_timestamp": candle.ingestion_timestamp,
        }

    @staticmethod
    def _candle(record: HistoricalCandleRecord) -> Candle:
        return Candle(
            symbol=record.symbol, timeframe=Timeframe(record.timeframe), timestamp=record.timestamp,
            open=record.open, high=record.high, low=record.low, close=record.close, volume=record.volume,
            spread=record.spread, provider=record.provider, quality_score=record.quality_score,
            quality_level=record.quality_level, ingestion_timestamp=record.ingestion_timestamp,
        )
