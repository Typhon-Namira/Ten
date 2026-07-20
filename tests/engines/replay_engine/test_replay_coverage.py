from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.replay_engine import (
    HistoricalEventQuery,
    HistoricalSourceRegistry,
    InMemoryHistoricalSource,
    ReplayCheckpoint,
    ReplayCheckpointError,
    ReplayClock,
    ReplayClockAdapter,
    ReplayConfig,
    ReplayConfigurationError,
    ReplayDatasetRegistry,
    ReplayEventBus,
    ReplayFeature,
    ReplayFeatureStore,
    ReplayGeneratedEvent,
    ReplayMode,
    ReplayOutputReference,
    ReplayPointInTimeError,
    ReplayRequest,
    ReplaySourceFilters,
    ReplayStatus,
    ReplayWorker,
    SqlAlchemyEconomicRevisionSource,
    SqlAlchemyHistoricalCandleSource,
    merge_event_sources,
    stable_hash,
    stable_id,
)
from backend.app.engines.replay_engine.coordinator import ReplayProcessingContext
from backend.app.engines.replay_engine.models import ReplaySession, _validate_json, aware
from backend.app.events import Event, InMemoryEventBus
from tests.engines.replay_engine.test_replay_engine import NOW, build_service, dataset, historical_event, replay_request
from tests.engines.replay_engine.test_replay_sql_sources_worker import ScalarResult, db_session
from tests.conftest import FakeSessionFactory


def test_defensive_model_and_clock_invariants() -> None:
    naive = NOW.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        ReplayClock(naive, NOW)
    with pytest.raises(ValueError, match="positive interval"):
        ReplayClock(NOW, NOW)
    clock = ReplayClock(NOW, NOW + timedelta(hours=1))
    with pytest.raises(ReplayPointInTimeError, match="timezone-aware"):
        clock.advance_to(naive)
    clock.freeze()
    with pytest.raises(ReplayPointInTimeError, match="frozen"):
        clock.restore(NOW)
    clock.unfreeze()
    with pytest.raises(ReplayPointInTimeError, match="timezone-aware"):
        clock.restore(naive)
    with pytest.raises(ReplayPointInTimeError, match="outside"):
        clock.restore(NOW + timedelta(days=1))
    adapter = ReplayClockAdapter(clock)
    assert adapter.now() == NOW

    with pytest.raises(ValueError, match="nesting"):
        _validate_json([[[[[[[[[1]]]]]]]]])
    with pytest.raises(ValueError, match="NaN"):
        _validate_json(float("nan"))
    with pytest.raises(ValueError, match="keys"):
        _validate_json({1: "bad"})
    with pytest.raises(ValueError, match="JSON"):
        _validate_json(object())
    _validate_json([1, True, None, "ok", 1.5])

    base = dataset()
    for update, message in (
        ({"instruments": ()}, "instruments"),
        ({"timeframes": ("W1",)}, "timeframes"),
        ({"available_until": base.available_from}, "positive duration"),
        ({"created_at": base.available_from - timedelta(seconds=1)}, "creation"),
    ):
        with pytest.raises(ValidationError, match=message):
            type(base).model_validate(base.model_dump() | update)
    with pytest.raises(ValidationError, match="source filter"):
        replay_request().source_filters.__class__(source_names=tuple(f"source_{index}" for index in range(33)))

    request = replay_request()
    invalid_requests = (
        ({"instruments": ()}, "instruments"),
        ({"timeframes": ("W1",)}, "timeframe"),
        ({"engine_selection": ()}, "engine selection"),
        ({"start_at": base.available_from - timedelta(seconds=1)}, "coverage"),
        ({"instruments": ("EURUSD",)}, "series"),
        ({"engine_versions": {}}, "pinned version"),
        ({"mode": "maximum_speed", "step_unit": "timestamp_group"}, "step unit"),
        ({"metadata": {"x": "y" * 70_000}}, "absolute size"),
    )
    for update, message in invalid_requests:
        with pytest.raises(ValidationError, match=message):
            ReplayRequest.model_validate(request.model_dump() | update)
    with pytest.raises(ValidationError):
        ReplayRequest.model_validate(1)

    large = historical_event(5, 1).model_dump() | {"payload": {"x": "y" * 70_000}}
    large["payload_hash"] = stable_hash(large["payload"])
    large["replay_event_id"] = stable_id("test-history", "v1", "historical_candles", "candle-1", "market.candle.closed", (NOW + timedelta(minutes=5)).isoformat(), "XAUUSD", "M1", large["payload_hash"], "1.0")
    with pytest.raises(ValidationError, match="payload exceeds"):
        type(historical_event(5, 1)).model_validate(large)
    no_sequence = historical_event(5, 1).model_copy(update={"source_sequence": None})
    assert no_sequence.ordering_key()[2] == 2**63 - 1

    session = ReplaySession(
        replay_id=request.replay_id,
        request=request,
        request_fingerprint=request.fingerprint("1.0"),
        status=ReplayStatus.READY,
        created_at=NOW,
        virtual_cursor_at=NOW,
        engine_graph_version="1.0",
        ordering_version="1.0",
        replay_engine_version="1.0.0",
        configuration_hash="a" * 64,
        policy_manifest_hash="b" * 64,
        engine_manifest_hash="c" * 64,
    )
    with pytest.raises(ValidationError, match="outside"):
        ReplaySession.model_validate(session.model_dump() | {"virtual_cursor_at": NOW - timedelta(seconds=1)})
    with pytest.raises(ValidationError, match="progress"):
        ReplaySession.model_validate(session.model_dump() | {"status": "completed", "progress_percent": 99})
    with pytest.raises(ValidationError, match="lease"):
        ReplaySession.model_validate(session.model_dump() | {"lease_expires_at": NOW + timedelta(seconds=1)})


@pytest.mark.asyncio
async def test_registry_source_and_ordering_defensive_paths() -> None:
    registry = ReplayDatasetRegistry((dataset(),))
    with pytest.raises(Exception, match="duplicate"):
        registry.register(dataset())
    with pytest.raises(Exception, match="manifest"):
        registry.resolve(dataset().model_copy(update={"manifest_hash": "1" * 64}))
    sources = HistoricalSourceRegistry((InMemoryHistoricalSource("historical_candles", (historical_event(5, 1),)),))
    with pytest.raises(Exception, match="duplicate"):
        sources.register(InMemoryHistoricalSource("historical_candles", ()))
    with pytest.raises(Exception, match="unknown"):
        sources.resolve(("missing",))

    variants = (
        historical_event(5, 1).model_copy(update={"dataset_id": "other"}),
        historical_event(5, 2).model_copy(update={"available_at": NOW - timedelta(minutes=1)}),
        historical_event(5, 3).model_copy(update={"instrument": "EURUSD"}),
        historical_event(5, 4).model_copy(update={"timeframe": "M5"}),
    )
    source = InMemoryHistoricalSource("historical_candles", variants + (historical_event(5, 5),))
    visible = [item async for item in source.stream(HistoricalEventQuery(uuid4(), replay_request()))]
    assert [item.source_sequence for item in visible] == [5]

    async def empty():
        if False:
            yield historical_event(1, 1)

    assert [item async for item in merge_event_sources((empty(),))] == []

    async def regressing():
        yield historical_event(10, 2)
        yield historical_event(5, 1)

    with pytest.raises(Exception, match="internally ordered"):
        _ = [item async for item in merge_event_sources((regressing(),))]


@pytest.mark.asyncio
async def test_isolation_defensive_paths_and_generated_handler_output() -> None:
    replay_id = uuid4()
    clock = ReplayClock(NOW, NOW + timedelta(hours=1))
    clock.advance_to(NOW + timedelta(minutes=5))
    bus = ReplayEventBus(replay_id, "test-history", "v1", clock)

    async def handler(_):
        payload = {"generated": True}
        fingerprint = stable_hash(payload)
        return (ReplayGeneratedEvent(event_id=stable_id(replay_id, clock.now().isoformat(), "generated.event", "test", "1.0.0", fingerprint, stable_hash(payload), 0), replay_id=replay_id, virtual_time=clock.now(), event_type="generated.event", source_engine="test", source_engine_version="1.0.0", input_fingerprint=fingerprint, payload=payload),)

    bus.subscribe("*", handler)
    with pytest.raises(Exception, match="duplicate"):
        bus.subscribe("*", handler)
    assert len(await bus.publish(historical_event(5, 1))) == 1
    generated = (await bus.publish(historical_event(5, 1)))[0]
    with pytest.raises(Exception, match="crossed"):
        await bus.publish(generated.model_copy(update={"replay_id": uuid4()}))
    await bus.close()
    with pytest.raises(Exception, match="closed"):
        bus.subscribe("*", handler)

    store = ReplayFeatureStore(replay_id)
    await store.close()
    with pytest.raises(Exception, match="closed"):
        await store.write(ReplayFeature(replay_id, "XAUUSD", "M1", f"replay:{replay_id}:x", NOW, "x", "1", {}))
    with pytest.raises(Exception, match="point-in-time"):
        await ReplayFeatureStore(replay_id).snapshot("XAUUSD", "M1", NOW.replace(tzinfo=None))


@pytest.mark.asyncio
async def test_memory_repository_remaining_conflicts() -> None:
    service, repository = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    missing = session.model_copy(update={"replay_id": uuid4(), "row_version": 2})
    with pytest.raises(Exception, match="does not exist"):
        await repository.save_session(missing, 1)
    with pytest.raises(Exception, match="increment"):
        await repository.save_session(session, session.row_version)
    checkpoint = ReplayCheckpoint(checkpoint_id=uuid4(), replay_id=session.replay_id, sequence=1, cursor_at=NOW, last_ordering_key=None, processed_events=0, generated_events=0, semantic_output_hash="0" * 64, state_hash="0" * 64, created_at=NOW, reason="test")
    checkpoint = checkpoint.model_copy(update={"state_hash": checkpoint.calculated_state_hash()})
    await repository.save_checkpoint(checkpoint)
    assert await repository.save_checkpoint(checkpoint) == checkpoint
    wrong_sequence = checkpoint.model_copy(update={"checkpoint_id": uuid4(), "sequence": 3})
    wrong_sequence = wrong_sequence.model_copy(update={"state_hash": wrong_sequence.calculated_state_hash()})
    with pytest.raises(Exception, match="sequence"):
        await repository.save_checkpoint(wrong_sequence)
    with pytest.raises(Exception, match="does not exist"):
        await repository.acquire_lease(uuid4(), "worker", NOW, 30, 1)
    with pytest.raises(Exception, match="version conflict"):
        await repository.acquire_lease(session.replay_id, "worker", NOW, 30, session.row_version - 1)
    with pytest.raises(Exception, match="lost"):
        await repository.renew_lease(session.replay_id, "worker", NOW, 30)
    with pytest.raises(Exception, match="does not own"):
        await repository.release_lease(session.replay_id, "worker")


@pytest.mark.asyncio
async def test_short_sql_source_pages_cover_final_batch() -> None:
    candle = SimpleNamespace(id=1, symbol="XAUUSD", timeframe="M1", timestamp=NOW, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0, spread=0.1, provider="test", quality_score=100.0, quality_level="native", ingestion_timestamp=NOW + timedelta(minutes=1))
    revision = SimpleNamespace(event_id=uuid4(), revision_number=1, revision_type="actual", available_at=NOW + timedelta(minutes=2), payload={})
    db = db_session()
    db.scalars.side_effect = [ScalarResult([candle]), ScalarResult([revision])]
    assert len([item async for item in SqlAlchemyHistoricalCandleSource(FakeSessionFactory(db)).stream(HistoricalEventQuery(uuid4(), replay_request(), batch_size=2))]) == 1
    assert len([item async for item in SqlAlchemyEconomicRevisionSource(FakeSessionFactory(db)).stream(HistoricalEventQuery(uuid4(), replay_request(), batch_size=2))]) == 1


class FailingBus(InMemoryEventBus):
    async def publish(self, event: Event) -> None:
        raise RuntimeError("secret")


@pytest.mark.asyncio
async def test_service_remaining_controls_health_comparison_and_publication() -> None:
    config = ReplayConfig(limits={"max_concurrent_sessions": 1}, worker={"max_concurrency": 1})
    service, repository = await build_service((historical_event(5, 1),), config=config)
    session = await service.create(replay_request())
    with pytest.raises(Exception, match="maximum concurrent"):
        await service.create(replay_request())
    running = await service.command_start(session.replay_id)
    assert await service.command_start(session.replay_id) == running
    assert await service.resume(session.replay_id) == running
    with pytest.raises(Exception, match="step units"):
        await service.step(session.replay_id, 0)

    await service.run(session.replay_id, "worker")
    completed = await service.get(session.replay_id)
    with pytest.raises(Exception, match="terminal"):
        await service.cancel(session.replay_id)
    assert await service.cleanup() == 0

    other_config = ReplayConfig()
    other, other_repository = await build_service((historical_event(5, 1),), config=other_config)
    second = await other.create(replay_request())
    changed = second.model_copy(update={"engine_manifest_hash": "f" * 64, "semantic_output_hash": "e" * 64, "row_version": second.row_version + 1})
    await other_repository.save_session(changed, second.row_version)
    output = ReplayOutputReference(output_id=uuid4(), replay_id=changed.replay_id, output_type="ai_score", source_engine="ai_scoring", source_id="x", fingerprint="d" * 64, as_of=NOW)
    await other_repository.save_outputs((output,))
    comparison = await other.compare(changed.replay_id, changed.replay_id)
    assert comparison.semantic_hash_equal

    no_events_config = ReplayConfig(events={"publish_lifecycle_events": False})
    silent, _ = await build_service((historical_event(5, 1),), config=no_events_config)
    await silent.create(replay_request())
    failing, _ = await build_service((historical_event(5, 1),))
    failing.event_bus = FailingBus()
    await failing.create(replay_request())
    assert failing.metrics.lifecycle_event_publish_failures_total == 1
    await failing.stop()
    assert failing.health()["status"] == "unavailable"
    with pytest.raises(Exception, match="unavailable"):
        await failing.create(replay_request())

    empty_sources_service, _ = await build_service((historical_event(5, 1),))
    empty_sources_service.coordinator.sources = HistoricalSourceRegistry()
    empty_sources_service.repository_mode = "memory"
    health = empty_sources_service.health()
    assert health["status"] == "degraded" and "historical_sources_unavailable" in health["degradation_reasons"]
    empty_sources_service._record_result(completed.model_copy(update={"status": ReplayStatus.CANCELLED}))


@pytest.mark.asyncio
async def test_worker_poll_timeout_failure_continue_and_stop_boundary() -> None:
    service, _ = await build_service((historical_event(5, 1),))
    running = await service.create(replay_request())
    await service.command_start(running.replay_id)
    worker = ReplayWorker(service, ReplayConfig(worker={"poll_interval_seconds": 0.01}), "worker")
    worker._stop.set()
    assert await worker.run_once() == 0
    worker._stop.clear()
    service.run = AsyncMock(side_effect=RuntimeError("worker failure"))
    assert await worker.run_once() == 0
    worker.run_once = AsyncMock(return_value=0)
    worker.start()
    await asyncio.sleep(0.03)
    await worker.stop()
    assert worker.run_once.await_count >= 1


@pytest.mark.asyncio
async def test_coordinator_validation_and_failure_category_helpers() -> None:
    service, _ = await build_service((historical_event(5, 1),))
    coordinator = service.coordinator
    with pytest.raises(Exception, match="must be running"):
        await coordinator.run((await service.create(replay_request())).replay_id, "worker")
    with pytest.raises(Exception, match="does not exist"):
        await coordinator._require_session(uuid4())
    assert coordinator._failure_category(ReplayPointInTimeError("x")).value == "point_in_time_violation"
    assert coordinator._failure_category(ReplayConfigurationError("x")).value == "engine_incompatible"
    assert coordinator._failure_category(ReplayCheckpointError("x")).value == "checkpoint_failure"
    assert coordinator._checkpoint_due((await service.list())[0]) is False

    invalid_config = ReplayConfig(checkpoint={"enabled": False})
    disabled, _ = await build_service((historical_event(5, 1),), config=invalid_config)
    assert disabled.coordinator._checkpoint_due(await disabled.create(replay_request())) is False

    validations = (
        (replay_request().model_copy(update={"instruments": ("XAUUSD",) * 11}), "scope"),
        (replay_request().model_copy(update={"end_at": NOW + timedelta(days=91)}), "duration"),
        (replay_request().model_copy(update={"instruments": ("EURUSD",)}), "not approved"),
        (replay_request().model_copy(update={"source_filters": ReplaySourceFilters(source_names=("unsafe",))}), "source"),
        (replay_request(mode=ReplayMode.ACCELERATED, speed=Decimal("2000")), "speed"),
    )
    for request, message in validations:
        with pytest.raises(Exception, match=message):
            coordinator._validate_request(request)

    naive_coordinator = type(coordinator)(coordinator.repository, coordinator.datasets, coordinator.sources, coordinator.engines, coordinator.config, now=lambda: NOW.replace(tzinfo=None))
    with pytest.raises(Exception, match="operational clock"):
        await naive_coordinator.create(replay_request())


def test_final_identity_and_timezone_guards() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        aware(NOW.replace(tzinfo=None))
    event = historical_event(5, 1)
    with pytest.raises(ValidationError, match="identity mismatch"):
        type(event).model_validate(event.model_dump() | {"replay_event_id": uuid4()})


@pytest.mark.asyncio
async def test_merge_global_regression_guard_with_stateful_key() -> None:
    class StatefulEvent:
        replay_event_id = uuid4()

        def __init__(self, keys):
            self.keys = iter(keys)

        def ordering_key(self):
            return next(self.keys)

    first = StatefulEvent(((2,),))
    following = StatefulEvent(((3,), (1,)))

    async def source():
        yield first
        yield following

    with pytest.raises(Exception, match="regressed"):
        _ = [item async for item in merge_event_sources((source(),))]


@pytest.mark.asyncio
async def test_coordinator_remaining_fail_closed_boundaries() -> None:
    service, repository = await build_service((historical_event(5, 1),))
    coordinator = service.coordinator
    mutable = dataset().model_copy(update={"mutable": True})
    coordinator.datasets = ReplayDatasetRegistry((mutable,))
    with pytest.raises(Exception, match="mutable"):
        await coordinator.create(replay_request().model_copy(update={"dataset": mutable}))
    coordinator.datasets = ReplayDatasetRegistry((dataset(),))
    missing_sources = replay_request().model_copy(update={"source_filters": ReplaySourceFilters(source_names=())})
    with pytest.raises(Exception, match="missing historical sources"):
        await coordinator.create(missing_sources)

    session = await service.create(replay_request())
    running = await service.command_start(session.replay_id)
    clock = ReplayClock(running.request.start_at, running.request.end_at)
    bus = ReplayEventBus(running.replay_id, "test-history", "v1", clock)
    store = ReplayFeatureStore(running.replay_id)
    context = ReplayProcessingContext(running, clock, bus, store)
    with pytest.raises(ReplayPointInTimeError, match="future"):
        await coordinator._process_group(running, (historical_event(5, 1),), context)

    clock.advance_to(NOW + timedelta(minutes=5))
    payload = {"generated": True}
    fingerprint = stable_hash(payload)
    generated = ReplayGeneratedEvent(event_id=stable_id(running.replay_id, clock.now().isoformat(), "generated.event", "test", "1.0.0", fingerprint, stable_hash(payload), 2), replay_id=running.replay_id, virtual_time=clock.now(), event_type="generated.event", source_engine="test", source_engine_version="1.0.0", input_fingerprint=fingerprint, payload=payload, chain_depth=2)
    limited_config = ReplayConfig(processing={"max_chain_depth": 1})
    coordinator.config = limited_config
    with pytest.raises(Exception, match="chain exceeds"):
        await coordinator._drain_generated([generated], context, (), [])
    coordinator.config = ReplayConfig()
    assert await coordinator._drain_generated([generated.model_copy(update={"chain_depth": 0})], context, ("missing",), []) == 1
    with patch.object(ReplayCheckpoint, "calculated_state_hash", side_effect=["a" * 64, "b" * 64]):
        with pytest.raises(Exception, match="construction"):
            await coordinator.checkpoint(running, "test")

    trace_config = ReplayConfig(trace={"enabled": True})
    trace_service, trace_repository = await build_service((historical_event(5, 1),), config=trace_config)
    traced = await trace_service.create(replay_request())
    await trace_service.command_start(traced.replay_id)
    await trace_service.run(traced.replay_id, "trace-worker")
    assert await trace_repository.list_trace(traced.replay_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_status", [ReplayStatus.CANCELLING, ReplayStatus.PAUSING])
async def test_worker_observes_control_state_at_atomic_boundary(requested_status: ReplayStatus) -> None:
    service, repository = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    running = await service.command_start(session.replay_id)
    original = service.coordinator._require_session
    calls = 0

    async def controlled(replay_id):
        nonlocal calls
        calls += 1
        current = await original(replay_id)
        if calls == 2:
            return current.model_copy(update={"status": requested_status})
        return current

    service.coordinator._require_session = controlled
    await service.run(running.replay_id, "worker-control")
    final = await service.get(running.replay_id)
    expected = ReplayStatus.CANCELLED if requested_status == ReplayStatus.CANCELLING else ReplayStatus.PAUSED
    assert final.status == expected


@pytest.mark.asyncio
async def test_worker_lease_loss_and_step_invalid_internal_state() -> None:
    service, repository = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    running = await service.command_start(session.replay_id)
    original = service.coordinator._require_session
    calls = 0

    async def lost(replay_id):
        nonlocal calls
        calls += 1
        current = await original(replay_id)
        if calls == 2:
            return current.model_copy(update={"worker_id": "other"})
        return current

    service.coordinator._require_session = lost
    await service.run(running.replay_id, "worker")
    assert (await service.get(running.replay_id)).status == ReplayStatus.FAILED

    step_service, step_repository = await build_service((historical_event(5, 1),))
    step = await step_service.create(replay_request(mode=ReplayMode.STEP))
    invalid = step.model_copy(update={"status": ReplayStatus.CANCELLING, "row_version": step.row_version + 1})
    await step_repository.save_session(invalid, step.row_version)
    with pytest.raises(Exception, match="not ready"):
        await step_service.step(step.replay_id)


@pytest.mark.asyncio
async def test_comparison_reports_manifest_and_output_divergence() -> None:
    service, repository = await build_service((historical_event(5, 1),))
    left = await service.create(replay_request())
    right = await service.create(replay_request())
    changed = right.model_copy(update={"engine_manifest_hash": "f" * 64, "semantic_output_hash": "e" * 64, "row_version": right.row_version + 1})
    await repository.save_session(changed, right.row_version)
    left_output = ReplayOutputReference(output_id=uuid4(), replay_id=left.replay_id, output_type="ai_score", source_engine="ai_scoring", source_id="left", fingerprint="a" * 64, as_of=NOW)
    right_output = ReplayOutputReference(output_id=uuid4(), replay_id=right.replay_id, output_type="ai_score", source_engine="ai_scoring", source_id="right", fingerprint="b" * 64, as_of=NOW)
    await repository.save_outputs((left_output, right_output))
    comparison = await service.compare(left.replay_id, right.replay_id)
    assert not comparison.comparable
    assert not comparison.semantic_hash_equal
    assert comparison.first_divergence == "output[0]"
