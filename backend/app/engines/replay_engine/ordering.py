from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
import heapq

from .exceptions import ReplayOrderingError
from .models import HistoricalEvent


async def merge_event_sources(sources: Sequence[AsyncIterator[HistoricalEvent]]) -> AsyncIterator[HistoricalEvent]:
    """Deterministic k-way merge using one buffered event per source."""

    heap: list[tuple[tuple[object, ...], int, HistoricalEvent]] = []
    for index, source in enumerate(sources):
        try:
            event = await anext(source)
        except StopAsyncIteration:
            continue
        heapq.heappush(heap, (event.ordering_key(), index, event))
    previous_key: tuple[object, ...] | None = None
    previous_id = None
    while heap:
        key, index, event = heapq.heappop(heap)
        if previous_key is not None and key < previous_key:
            raise ReplayOrderingError("historical source ordering regressed")
        if event.replay_event_id != previous_id:
            yield event
            previous_id = event.replay_event_id
        previous_key = key
        try:
            following = await anext(sources[index])
        except StopAsyncIteration:
            continue
        if following.ordering_key() < key:
            raise ReplayOrderingError("historical source is not internally ordered")
        heapq.heappush(heap, (following.ordering_key(), index, following))


async def timestamp_groups(events: AsyncIterator[HistoricalEvent], limit: int) -> AsyncIterator[tuple[HistoricalEvent, ...]]:
    group: list[HistoricalEvent] = []
    timestamp = None
    async for event in events:
        if timestamp is not None and event.available_at != timestamp:
            yield tuple(group)
            group = []
        timestamp = event.available_at
        group.append(event)
        if len(group) > limit:
            raise ReplayOrderingError("timestamp group exceeds configured limit")
    if group:
        yield tuple(group)
