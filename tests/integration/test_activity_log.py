"""Regression tests for PipelineActivityLog's presentation-layer event merging.

Root cause covered: a single analysis cycle can publish dozens of same-type domain events (one
`LiquidityPoolPartiallySwept` per swept pool, one `LiquidityTargetRankingUpdated` per target) in a
tight loop — genuinely correct on the event bus (each is a distinct object's state change), but
directly mirroring every one of them into the live log/SSE stream floods it with near-duplicate
lines. The activity log must merge a same-type, same-correlation burst into one entry with a
`count`, without ever touching what the event bus itself delivers to real subscribers.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from backend.app.events import Event, InMemoryEventBus
from backend.app.integration.activity_log import MERGE_WINDOW_SECONDS, PipelineActivityLog


class PoolCreated(Event):
    pass


class PoolSwept(Event):
    pass


class CycleCompleted(Event):
    pass


@pytest.mark.asyncio
async def test_same_type_same_correlation_burst_merges_into_one_entry_with_a_count() -> None:
    bus = InMemoryEventBus()
    log = PipelineActivityLog(bus)
    log.start()
    correlation = uuid4()
    for _ in range(5):
        await bus.publish(PoolSwept(correlation_id=correlation, source="liquidity", payload={"pool": "x"}))
    entries = log.snapshot()
    assert len(entries) == 1
    assert entries[0].type == "PoolSwept"
    assert entries[0].count == 5
    log.stop()


@pytest.mark.asyncio
async def test_different_event_types_are_never_merged_even_with_the_same_correlation() -> None:
    bus = InMemoryEventBus()
    log = PipelineActivityLog(bus)
    log.start()
    correlation = uuid4()
    await bus.publish(PoolCreated(correlation_id=correlation, source="liquidity", payload={}))
    await bus.publish(PoolSwept(correlation_id=correlation, source="liquidity", payload={}))
    entries = log.snapshot()
    assert [item.type for item in entries] == ["PoolCreated", "PoolSwept"]
    assert all(item.count == 1 for item in entries)
    log.stop()


@pytest.mark.asyncio
async def test_same_type_different_correlation_is_never_merged() -> None:
    """Two unrelated candle cycles both producing a `PoolSwept` must stay two distinct entries —
    merging is scoped to one cycle's own burst, never across cycles."""
    bus = InMemoryEventBus()
    log = PipelineActivityLog(bus)
    log.start()
    await bus.publish(PoolSwept(correlation_id=uuid4(), source="liquidity", payload={}))
    await bus.publish(PoolSwept(correlation_id=uuid4(), source="liquidity", payload={}))
    entries = log.snapshot()
    assert len(entries) == 2
    assert all(item.count == 1 for item in entries)
    log.stop()


@pytest.mark.asyncio
async def test_an_open_batch_flushes_on_its_own_after_the_merge_window_even_with_nothing_new() -> None:
    """A burst that is the LAST activity for a while (e.g. end of a candle cycle) must still reach
    subscribers promptly — it cannot sit open forever waiting for a next event that triggers the
    flush, since the next real event might be minutes away (next timeframe close)."""
    bus = InMemoryEventBus()
    log = PipelineActivityLog(bus)
    log.start()
    key, queue = log.subscribe()
    await bus.publish(PoolSwept(correlation_id=uuid4(), source="liquidity", payload={}))
    assert queue.empty()  # not flushed yet — still within the merge window
    await asyncio.sleep(MERGE_WINDOW_SECONDS + 0.1)
    entry = await asyncio.wait_for(queue.get(), timeout=1)
    assert entry.type == "PoolSwept" and entry.count == 1
    log.unsubscribe(key)
    log.stop()


@pytest.mark.asyncio
async def test_a_differently_keyed_event_flushes_the_previous_batch_immediately() -> None:
    """Distinct, non-bursty events must not be delayed by the merge window — only an actual same-
    key burst is held open; anything else flushes the prior batch and appears right away."""
    bus = InMemoryEventBus()
    log = PipelineActivityLog(bus)
    log.start()
    key, queue = log.subscribe()
    await bus.publish(PoolSwept(correlation_id=uuid4(), source="liquidity", payload={}))
    await bus.publish(PoolCreated(correlation_id=uuid4(), source="liquidity", payload={}))
    first = await asyncio.wait_for(queue.get(), timeout=1)
    assert first.type == "PoolSwept"
    log.unsubscribe(key)
    log.stop()


@pytest.mark.asyncio
async def test_subscriber_queue_receives_the_merged_entry_not_every_individual_event() -> None:
    bus = InMemoryEventBus()
    log = PipelineActivityLog(bus)
    log.start()
    key, queue = log.subscribe()
    correlation = uuid4()
    for _ in range(10):
        await bus.publish(PoolSwept(correlation_id=correlation, source="liquidity", payload={}))
    await bus.publish(PoolCreated(correlation_id=correlation, source="liquidity", payload={}))
    merged = await asyncio.wait_for(queue.get(), timeout=1)
    assert merged.type == "PoolSwept" and merged.count == 10
    assert queue.qsize() <= 1  # only the PoolCreated (or nothing yet) remains queued
    log.unsubscribe(key)
    log.stop()


@pytest.mark.asyncio
async def test_slow_client_queue_stays_bounded_and_keeps_new_terminal_event() -> None:
    bus = InMemoryEventBus()
    log = PipelineActivityLog(bus, capacity=3, queue_capacity=2)
    log.start()
    key, queue = log.subscribe()
    await bus.publish(PoolCreated(correlation_id=uuid4(), source="liquidity", payload={"n": 1}))
    await bus.publish(PoolSwept(correlation_id=uuid4(), source="liquidity", payload={"n": 2}))
    await bus.publish(CycleCompleted(correlation_id=uuid4(), source="integration", payload={"terminal": True}))
    log.snapshot()  # flush the terminal event

    retained = [queue.get_nowait() for _ in range(queue.qsize())]
    assert queue.qsize() == 0
    assert len(retained) <= 2
    assert retained[-1].type == "CycleCompleted"
    assert log.metrics()["dropped_client_events"] >= 1
    assert log.metrics()["entries"] <= 3
    log.unsubscribe(key)
    log.stop()
