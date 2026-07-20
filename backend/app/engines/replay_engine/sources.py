from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.engines.market_data_engine import Timeframe
from backend.app.storage.models import EconomicCalendarRevisionRecord, HistoricalCandleRecord
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session_stream

from .exceptions import ReplayConfigurationError, ReplayValidationError
from .models import HistoricalEvent, ReplayDatasetReference, ReplayRequest, stable_hash, stable_id


@dataclass(frozen=True)
class HistoricalSourceValidation:
    source_name: str
    source_version: str
    valid: bool
    event_estimate: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class HistoricalEventQuery:
    replay_id: object
    request: ReplayRequest
    after_ordering_key: str | None = None
    batch_size: int = 1000


class HistoricalEventSource(Protocol):
    source_name: str
    source_version: str

    async def validate(self, request: ReplayRequest) -> HistoricalSourceValidation: ...

    def stream(self, query: HistoricalEventQuery) -> AsyncIterator[HistoricalEvent]: ...


class ReplayDatasetRegistry:
    def __init__(self, datasets: Iterable[ReplayDatasetReference] = ()) -> None:
        self._datasets: dict[tuple[str, str], ReplayDatasetReference] = {}
        for dataset in datasets:
            self.register(dataset)

    def register(self, dataset: ReplayDatasetReference) -> None:
        key = (dataset.dataset_id, dataset.dataset_version)
        if key in self._datasets:
            raise ReplayConfigurationError(f"duplicate replay dataset: {dataset.dataset_id}@{dataset.dataset_version}")
        self._datasets[key] = dataset

    def resolve(self, requested: ReplayDatasetReference) -> ReplayDatasetReference:
        key = (requested.dataset_id, requested.dataset_version)
        registered = self._datasets.get(key)
        if registered is None:
            raise ReplayValidationError("replay dataset is not registered")
        if registered.manifest_hash != requested.manifest_hash:
            raise ReplayValidationError("replay dataset manifest does not match registered identity")
        return registered

    def datasets(self) -> tuple[ReplayDatasetReference, ...]:
        return tuple(sorted(self._datasets.values(), key=lambda item: (item.dataset_id, item.dataset_version)))


class HistoricalSourceRegistry:
    def __init__(self, sources: Iterable[HistoricalEventSource] = ()) -> None:
        self._sources: dict[str, HistoricalEventSource] = {}
        for source in sources:
            self.register(source)

    def register(self, source: HistoricalEventSource) -> None:
        if source.source_name in self._sources:
            raise ReplayConfigurationError(f"duplicate historical source: {source.source_name}")
        self._sources[source.source_name] = source

    def resolve(self, names: tuple[str, ...]) -> tuple[HistoricalEventSource, ...]:
        unknown = set(names) - self._sources.keys()
        if unknown:
            raise ReplayConfigurationError(f"unknown historical source: {sorted(unknown)}")
        return tuple(self._sources[name] for name in sorted(names))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources))


class InMemoryHistoricalSource:
    def __init__(self, source_name: str, events: Iterable[HistoricalEvent], source_version: str = "1.0.0") -> None:
        self.source_name = source_name
        self.source_version = source_version
        self._events = tuple(sorted(events, key=lambda event: event.ordering_key()))

    async def validate(self, request: ReplayRequest) -> HistoricalSourceValidation:
        events = tuple(self._visible(request))
        return HistoricalSourceValidation(self.source_name, self.source_version, bool(events), len(events), () if events else ("source_range_empty",))

    async def stream(self, query: HistoricalEventQuery) -> AsyncIterator[HistoricalEvent]:
        for event in self._visible(query.request):
            if query.after_ordering_key is None or event.ordering_key_text() > query.after_ordering_key:
                yield event

    def _visible(self, request: ReplayRequest) -> Iterable[HistoricalEvent]:
        for event in self._events:
            if event.dataset_id != request.dataset.dataset_id or event.dataset_version != request.dataset.dataset_version:
                continue
            if not request.start_at <= event.available_at <= request.end_at:
                continue
            if event.instrument is not None and event.instrument not in request.instruments:
                continue
            if event.timeframe is not None and event.timeframe not in request.timeframes:
                continue
            yield event


class SqlAlchemyHistoricalCandleSource(ScopedSessionRepository):
    source_name = "historical_candles"
    source_version = "1.0.0"

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)

    async def validate(self, request: ReplayRequest) -> HistoricalSourceValidation:
        if request.dataset.source_name != self.source_name:
            return HistoricalSourceValidation(self.source_name, self.source_version, False, warnings=("dataset_source_mismatch",))
        return HistoricalSourceValidation(self.source_name, self.source_version, True, None, ("candle_timestamp_is_open_boundary",))

    @scoped_session_stream
    async def stream(self, query: HistoricalEventQuery) -> AsyncIterator[HistoricalEvent]:
        request = query.request
        for instrument in request.instruments:
            for timeframe_name in request.timeframes:
                timeframe = Timeframe(timeframe_name)
                offset = 0
                while True:
                    statement = (
                        select(HistoricalCandleRecord)
                        .where(
                            HistoricalCandleRecord.symbol == instrument,
                            HistoricalCandleRecord.timeframe == timeframe.value,
                            HistoricalCandleRecord.timestamp + timeframe.duration >= request.start_at,
                            HistoricalCandleRecord.timestamp + timeframe.duration <= request.end_at,
                            HistoricalCandleRecord.ingestion_timestamp <= request.dataset.created_at,
                        )
                        .order_by(HistoricalCandleRecord.timestamp, HistoricalCandleRecord.id)
                        .offset(offset)
                        .limit(query.batch_size)
                    )
                    records = list((await self.session.scalars(statement)).all())
                    if not records:
                        break
                    for record in records:
                        payload = {
                            "open": record.open,
                            "high": record.high,
                            "low": record.low,
                            "close": record.close,
                            "volume": record.volume,
                            "spread": record.spread,
                            "provider": record.provider,
                            "quality_score": record.quality_score,
                            "quality_level": record.quality_level,
                        }
                        available_at = max(record.timestamp + timeframe.duration, record.ingestion_timestamp)
                        payload_hash = stable_hash(payload)
                        event = HistoricalEvent(
                            replay_event_id=stable_id(request.dataset.dataset_id, request.dataset.dataset_version, self.source_name, str(record.id), "market.candle.closed", available_at.isoformat(), instrument, timeframe.value, payload_hash, "1.0"),
                            source_event_id=str(record.id),
                            event_type="market.candle.closed",
                            instrument=instrument,
                            timeframe=timeframe.value,
                            occurred_at=record.timestamp,
                            published_at=available_at,
                            available_at=available_at,
                            source_name=self.source_name,
                            source_version=self.source_version,
                            source_sequence=record.id,
                            priority=20,
                            payload=payload,
                            payload_hash=payload_hash,
                            dataset_id=request.dataset.dataset_id,
                            dataset_version=request.dataset.dataset_version,
                        )
                        if query.after_ordering_key is None or event.ordering_key_text() > query.after_ordering_key:
                            yield event
                    offset += len(records)
                    if len(records) < query.batch_size:
                        break


class SqlAlchemyEconomicRevisionSource(ScopedSessionRepository):
    source_name = "economic_calendar"
    source_version = "1.0.0"

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        ScopedSessionRepository.__init__(self, session_factory)

    async def validate(self, request: ReplayRequest) -> HistoricalSourceValidation:
        return HistoricalSourceValidation(self.source_name, self.source_version, True, None, ("provider_health_history_unavailable",))

    @scoped_session_stream
    async def stream(self, query: HistoricalEventQuery) -> AsyncIterator[HistoricalEvent]:
        request = query.request
        offset = 0
        while True:
            statement = (
                select(EconomicCalendarRevisionRecord)
                .where(EconomicCalendarRevisionRecord.available_at >= request.start_at, EconomicCalendarRevisionRecord.available_at <= request.end_at)
                .order_by(EconomicCalendarRevisionRecord.available_at, EconomicCalendarRevisionRecord.event_id, EconomicCalendarRevisionRecord.revision_number)
                .offset(offset)
                .limit(query.batch_size)
            )
            records = list((await self.session.scalars(statement)).all())
            if not records:
                break
            for record in records:
                payload = {"event_id": str(record.event_id), "revision_number": record.revision_number, "revision_type": record.revision_type, "values": record.payload}
                payload_hash = stable_hash(payload)
                source_id = f"{record.event_id}:{record.revision_number}"
                event = HistoricalEvent(
                    replay_event_id=stable_id(request.dataset.dataset_id, request.dataset.dataset_version, self.source_name, source_id, "economic.revision.published", record.available_at.isoformat(), "", "", payload_hash, "1.0"),
                    source_event_id=source_id,
                    event_type="economic.revision.published",
                    occurred_at=record.available_at,
                    published_at=record.available_at,
                    available_at=record.available_at,
                    source_name=self.source_name,
                    source_version=self.source_version,
                    source_sequence=record.revision_number,
                    priority=30,
                    payload=payload,
                    payload_hash=payload_hash,
                    dataset_id=request.dataset.dataset_id,
                    dataset_version=request.dataset.dataset_version,
                )
                if query.after_ordering_key is None or event.ordering_key_text() > query.after_ordering_key:
                    yield event
            offset += len(records)
            if len(records) < query.batch_size:
                break


def dataset_manifest_hash(dataset_id: str, dataset_version: str, created_at: datetime, source_name: str) -> str:
    return stable_hash({"dataset_id": dataset_id, "dataset_version": dataset_version, "created_at": created_at.isoformat(), "source_name": source_name, "schema_version": "1.0"})
