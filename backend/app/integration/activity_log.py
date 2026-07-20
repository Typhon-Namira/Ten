"""Bounded, subscribable log of every pipeline event, for live dashboards.

Purely observational: it subscribes to the existing typed event bus and never publishes,
mutates engine state, or influences pipeline control flow. It never changes what the event bus
itself delivers to real consumers (repositories, other engines) — only how this
presentation-layer log renders a burst of same-cause events to a human.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from backend.app.events import Event, EventBus

# A burst of same-type events from the same analysis cycle (e.g. one `LiquidityPoolPartiallySwept`
# per swept pool) is genuinely one meaningful state change ("liquidity analysis updated its pool
# lifecycle for N pools"), not N independent facts — collapsing them at this presentation layer is
# what keeps the live log (and its downstream SSE stream) from flooding with near-duplicate lines
# per candle. `MERGE_WINDOW_SECONDS` bounds how long a batch is held open waiting for more of the
# same before it is flushed to subscribers regardless of what (if anything) arrives next.
MERGE_WINDOW_SECONDS = 0.3


@dataclass(frozen=True)
class ActivityLogEntry:
    id: str
    type: str
    source: str
    occurred_at: datetime
    correlation_id: str
    payload: dict[str, Any]
    # >1 means this entry represents a merged burst of `count` same-type, same-correlation events
    # from the same cycle — `payload` carries the most recent one as a representative sample.
    count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "occurred_at": self.occurred_at.isoformat(),
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "count": self.count,
        }


class PipelineActivityLog:
    """Bounded ring buffer of recent events plus live fan-out to subscribed queues (for SSE)."""

    def __init__(self, event_bus: EventBus, *, capacity: int = 500, queue_capacity: int = 500) -> None:
        self._event_bus = event_bus
        self._queue_capacity = queue_capacity
        self._entries: deque[ActivityLogEntry] = deque(maxlen=capacity)
        self._subscribers: dict[int, asyncio.Queue[ActivityLogEntry]] = {}
        self._unsubscribe: Callable[[], None] | None = None
        self._pending: ActivityLogEntry | None = None
        self._flush_timer: asyncio.TimerHandle | None = None

    def start(self) -> None:
        if self._unsubscribe is None:
            self._unsubscribe = self._event_bus.subscribe(Event, self._on_event)

    def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._flush_pending()

    def _cancel_timer(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _schedule_flush(self) -> None:
        self._cancel_timer()
        self._flush_timer = asyncio.get_running_loop().call_later(MERGE_WINDOW_SECONDS, self._flush_pending)

    def _flush_pending(self) -> None:
        self._cancel_timer()
        if self._pending is None:
            return
        entry = self._pending
        self._pending = None
        self._entries.append(entry)
        for queue in list(self._subscribers.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(entry)

    async def _on_event(self, event: Event) -> None:
        entry = ActivityLogEntry(
            id=str(event.event_id),
            type=type(event).__name__,
            source=event.source,
            occurred_at=event.occurred_at,
            correlation_id=str(event.correlation_id),
            payload=event.payload,
        )
        pending = self._pending
        if pending is not None and pending.type == entry.type and pending.correlation_id == entry.correlation_id:
            # Same cause, same kind of change — merge into the open batch instead of starting a
            # new entry. The flush timer was set when the batch opened and is deliberately not
            # reset here, so a long-running burst still flushes within `MERGE_WINDOW_SECONDS`
            # rather than being held open indefinitely by a steady trickle of matching events.
            self._pending = replace(entry, id=pending.id, occurred_at=pending.occurred_at, count=pending.count + 1)
            return
        self._flush_pending()
        self._pending = entry
        self._schedule_flush()

    def snapshot(self, limit: int = 200) -> tuple[ActivityLogEntry, ...]:
        self._flush_pending()
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
