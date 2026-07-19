from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from backend.app.engines.replay_engine import (
    HistoricalEventQuery,
    InMemoryHistoricalSource,
    ReplayClock,
    ReplayEngineRegistration,
    ReplayEventBus,
    ReplayFeature,
    ReplayFeatureStore,
    ReplayGeneratedEvent,
    ReplayIsolationError,
    ReplayPointInTimeError,
    ReplayCompatibilityRegistry,
    merge_event_sources,
    stable_hash,
    stable_id,
    timestamp_groups,
)
from tests.engines.replay_engine.test_replay_engine import NOW, historical_event, replay_request


@pytest.mark.asyncio
async def test_total_order_merge_ties_duplicates_batches_and_groups() -> None:
    first = historical_event(5, 2, priority=20)
    earlier_priority = historical_event(5, 1, priority=10)
    later = historical_event(10, 3)
    source_a = InMemoryHistoricalSource("historical_candles", (first, later))
    source_b = InMemoryHistoricalSource("historical_candles_copy", (earlier_priority, first))
    query = HistoricalEventQuery(uuid4(), replay_request(), batch_size=1)
    merged = [item async for item in merge_event_sources((source_a.stream(query), source_b.stream(query)))]
    assert [item.source_sequence for item in merged] == [1, 2, 3]

    async def stream():
        for item in merged:
            yield item

    groups = [item async for item in timestamp_groups(stream(), 10)]
    assert [len(item) for item in groups] == [2, 1]
    with pytest.raises(Exception, match="timestamp group"):
        _ = [item async for item in timestamp_groups(stream(), 1)]


def test_replay_clock_monotonic_restore_freeze_and_bounds() -> None:
    clock = ReplayClock(NOW, NOW + timedelta(hours=1))
    clock.advance_to(NOW + timedelta(minutes=5))
    assert clock.now() == NOW + timedelta(minutes=5)
    with pytest.raises(ReplayPointInTimeError, match="backward"):
        clock.advance_to(NOW)
    clock.restore(NOW + timedelta(minutes=1))
    clock.freeze()
    with pytest.raises(ReplayPointInTimeError, match="frozen"):
        clock.advance_to(NOW + timedelta(minutes=2))
    clock.unfreeze()
    with pytest.raises(ReplayPointInTimeError, match="exceed"):
        clock.advance_to(NOW + timedelta(hours=2))


@pytest.mark.asyncio
async def test_event_bus_and_feature_store_are_replay_scoped_and_point_in_time() -> None:
    replay_id = uuid4()
    clock = ReplayClock(NOW, NOW + timedelta(hours=1))
    clock.advance_to(NOW + timedelta(minutes=5))
    bus = ReplayEventBus(replay_id, "test-history", "v1", clock)
    observed: list[str] = []

    async def handler(event):
        observed.append(event.event_type)
        return None

    unsubscribe = bus.subscribe("market.candle.closed", handler)
    await bus.publish(historical_event(5, 1))
    unsubscribe()
    assert observed == ["market.candle.closed"] and len(bus.history()) == 1
    with pytest.raises(ReplayPointInTimeError, match="future"):
        await bus.publish(historical_event(10, 2))
    wrong = historical_event(5, 1).model_copy(update={"dataset_id": "other"})
    with pytest.raises(ReplayIsolationError, match="dataset"):
        await bus.publish(wrong)

    store = ReplayFeatureStore(replay_id)
    feature = ReplayFeature(replay_id, "XAUUSD", "M1", f"replay:{replay_id}:smc", NOW + timedelta(minutes=5), "smc", "1.0.0", {"direction": "bullish"})
    await store.write(feature)
    assert (await store.snapshot("XAUUSD", "M1", NOW + timedelta(minutes=5)))[feature.namespace] == feature
    assert await store.snapshot("XAUUSD", "M1", NOW) == {}
    with pytest.raises(ReplayIsolationError, match="namespace"):
        await store.write(ReplayFeature(replay_id, "XAUUSD", "M1", "smc", NOW, "smc", "1.0.0", {}))
    with pytest.raises(ReplayIsolationError, match="crossed"):
        await store.write(ReplayFeature(uuid4(), "XAUUSD", "M1", "replay:x:smc", NOW, "smc", "1.0.0", {}))
    await store.close()
    await bus.close()
    with pytest.raises(ReplayIsolationError, match="closed"):
        await bus.publish(historical_event(5, 1))


def registration(name: str, dependencies: tuple[str, ...] = (), *, safe: bool = True) -> ReplayEngineRegistration:
    return ReplayEngineRegistration(name, "1.0.0", "1.0", safe, safe, safe, safe, safe, required_engine_dependencies=dependencies)


def test_compatibility_registry_validates_dependencies_cycles_and_safety() -> None:
    registry = ReplayCompatibilityRegistry((registration("market_data"), registration("smc", ("market_data",))))
    resolved = registry.resolve(("smc", "market_data"), {"smc": "1.0.0", "market_data": "1.0.0"})
    assert [item.engine_name for item in resolved] == ["market_data", "smc"]
    with pytest.raises(Exception, match="duplicate"):
        registry.register(registration("smc", ("market_data",)))
    with pytest.raises(Exception, match="unknown"):
        registry.resolve(("missing",), {"missing": "1.0.0"})
    with pytest.raises(Exception, match="missing"):
        ReplayCompatibilityRegistry((registration("smc", ("market_data",)),)).resolve(("smc",), {"smc": "1.0.0"})
    with pytest.raises(Exception, match="cycle"):
        ReplayCompatibilityRegistry((registration("a", ("b",)), registration("b", ("a",)))).resolve(("a", "b"), {"a": "1.0.0", "b": "1.0.0"})
    with pytest.raises(Exception, match="not replay-safe"):
        ReplayCompatibilityRegistry((registration("unsafe", safe=False),))


def test_generated_event_identity_is_deterministic() -> None:
    replay_id = uuid4()
    payload = {"fingerprint": "a" * 64}
    event_id = stable_id(replay_id, NOW.isoformat(), "ai_score.completed", "ai_scoring", "1.0.0", "b" * 64, stable_hash(payload), 0)
    event = ReplayGeneratedEvent(event_id=event_id, replay_id=replay_id, virtual_time=NOW, event_type="ai_score.completed", source_engine="ai_scoring", source_engine_version="1.0.0", input_fingerprint="b" * 64, payload=payload)
    assert event.event_id == event_id
    with pytest.raises(Exception, match="identity"):
        ReplayGeneratedEvent.model_validate(event.model_dump() | {"event_id": uuid4()})
