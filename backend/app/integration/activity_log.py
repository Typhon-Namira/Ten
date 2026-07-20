"""Bounded, subscribable log of every pipeline event, for live dashboards.

Purely observational: it subscribes to the existing typed event bus and never publishes,
mutates engine state, or influences pipeline control flow.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backend.app.events import Event, EventBus


@dataclass(frozen=True)
class ActivityLogEntry:
    id: str
    type: str
    source: str
    occurred_at: datetime
    correlation_id: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "occurred_at": self.occurred_at.isoformat(),
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }


class PipelineActivityLog:
    """Bounded ring buffer of recent events plus live fan-out to subscribed queues (for SSE)."""

    def __init__(self, event_bus: EventBus, *, capacity: int = 500, queue_capacity: int = 500) -> None:
        self._event_bus = event_bus
        self._queue_capacity = queue_capacity
        self._entries: deque[ActivityLogEntry] = deque(maxlen=capacity)
        self._subscribers: dict[int, asyncio.Queue[ActivityLogEntry]] = {}
        self._unsubscribe: Callable[[], None] | None = None

    def start(self) -> None:
        if self._unsubscribe is None:
            self._unsubscribe = self._event_bus.subscribe(Event, self._on_event)

    def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    async def _on_event(self, event: Event) -> None:
        entry = ActivityLogEntry(
            id=str(event.event_id),
            type=type(event).__name__,
            source=event.source,
            occurred_at=event.occurred_at,
            correlation_id=str(event.correlation_id),
            payload=event.payload,
        )
        self._entries.append(entry)
        for queue in list(self._subscribers.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(entry)

    def snapshot(self, limit: int = 200) -> tuple[ActivityLogEntry, ...]:
        values = list(self._entries)
        return tuple(values[-limit:])

    def subscribe(self) -> tuple[int, asyncio.Queue[ActivityLogEntry]]:
        queue: asyncio.Queue[ActivityLogEntry] = asyncio.Queue(maxsize=self._queue_capacity)
        key = id(queue)
        self._subscribers[key] = queue
        return key, queue

    def unsubscribe(self, key: int) -> None:
        self._subscribers.pop(key, None)

    def subscriber_count(self) -> int:
        return len(self._subscribers)
