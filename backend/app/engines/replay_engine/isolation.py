from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from .clock import ReplayClock
from .exceptions import ReplayIsolationError, ReplayPointInTimeError
from .models import HistoricalEvent, ReplayGeneratedEvent

ReplayEvent = HistoricalEvent | ReplayGeneratedEvent
ReplayHandler = Callable[[ReplayEvent], Awaitable[tuple[ReplayGeneratedEvent, ...] | None]]


class ReplayEventBus:
    def __init__(self, replay_id: UUID, dataset_id: str, dataset_version: str, clock: ReplayClock) -> None:
        self.replay_id = replay_id
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.clock = clock
        self._handlers: dict[str, list[ReplayHandler]] = defaultdict(list)
        self._history: list[ReplayEvent] = []
        self._lock = asyncio.Lock()
        self._closed = False

    def subscribe(self, event_type: str, handler: ReplayHandler) -> Callable[[], None]:
        if self._closed:
            raise ReplayIsolationError("replay event bus is closed")
        handlers = self._handlers[event_type]
        if handler in handlers:
            raise ReplayIsolationError("duplicate replay event subscription")
        handlers.append(handler)

        def unsubscribe() -> None:
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: ReplayEvent) -> tuple[ReplayGeneratedEvent, ...]:
        if self._closed:
            raise ReplayIsolationError("replay event bus is closed")
        if isinstance(event, HistoricalEvent):
            if event.dataset_id != self.dataset_id or event.dataset_version != self.dataset_version:
                raise ReplayIsolationError("event dataset does not match replay namespace")
            event_time = event.available_at
        else:
            if event.replay_id != self.replay_id:
                raise ReplayIsolationError("generated event crossed replay namespace")
            event_time = event.virtual_time
        if event_time > self.clock.now():
            raise ReplayPointInTimeError("future event publication is prohibited")
        async with self._lock:
            self._history.append(event)
            handlers = tuple(self._handlers.get(event.event_type, ())) + tuple(self._handlers.get("*", ()))
        generated: list[ReplayGeneratedEvent] = []
        for handler in handlers:
            result = await handler(event)
            if result:
                generated.extend(result)
        return tuple(generated)

    def history(self) -> tuple[ReplayEvent, ...]:
        return tuple(self._history)

    async def close(self) -> None:
        async with self._lock:
            self._handlers.clear()
            self._closed = True


@dataclass(frozen=True)
class ReplayFeature:
    replay_id: UUID
    instrument: str
    timeframe: str
    namespace: str
    as_of: datetime
    engine_name: str
    engine_version: str
    values: dict[str, object]


class ReplayFeatureStore:
    def __init__(self, replay_id: UUID) -> None:
        self.replay_id = replay_id
        self._records: list[ReplayFeature] = []
        self._lock = asyncio.Lock()
        self._closed = False

    async def write(self, feature: ReplayFeature) -> None:
        if self._closed:
            raise ReplayIsolationError("replay Feature Store is closed")
        if feature.replay_id != self.replay_id:
            raise ReplayIsolationError("feature crossed replay namespace")
        if not feature.namespace.startswith(f"replay:{self.replay_id}:"):
            raise ReplayIsolationError("live Feature Store namespace is prohibited")
        async with self._lock:
            self._records.append(feature)

    async def snapshot(self, instrument: str, timeframe: str, as_of: datetime) -> dict[str, ReplayFeature]:
        if as_of.tzinfo is None:
            raise ReplayPointInTimeError("feature query must be point-in-time")
        async with self._lock:
            records = [item for item in self._records if item.instrument == instrument and item.timeframe == timeframe and item.as_of <= as_of]
        result: dict[str, ReplayFeature] = {}
        for item in sorted(records, key=lambda value: (value.as_of, value.namespace, value.engine_name)):
            result[item.namespace] = item
        return result

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
