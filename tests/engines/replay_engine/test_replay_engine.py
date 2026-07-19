from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.replay_engine import (
    HistoricalEvent,
    HistoricalSourceRegistry,
    InMemoryHistoricalSource,
    InMemoryReplayRepository,
    ReplayCheckpointPolicy,
    ReplayConfig,
    ReplayCoordinator,
    ReplayDatasetReference,
    ReplayDatasetRegistry,
    ReplayMode,
    ReplayRequest,
    ReplayService,
    ReplaySourceFilters,
    ReplayStatus,
    production_replay_registry,
    stable_hash,
    stable_id,
)
from backend.app.events import InMemoryEventBus

NOW = datetime(2026, 5, 1, 8, tzinfo=UTC)


def dataset(version: str = "v1") -> ReplayDatasetReference:
    return ReplayDatasetReference(
        dataset_id="test-history",
        dataset_version=version,
        source_name="historical_candles",
        created_at=NOW + timedelta(days=2),
        available_from=NOW - timedelta(days=1),
        available_until=NOW + timedelta(days=1),
        manifest_hash=stable_hash({"dataset": version}),
        instruments=("XAUUSD",),
        timeframes=("M1", "M5"),
    )


def historical_event(minute: int, sequence: int, *, event_type: str = "market.candle.closed", priority: int = 20, payload: dict[str, object] | None = None) -> HistoricalEvent:
    value = payload or {"close": 2000 + sequence}
    available_at = NOW + timedelta(minutes=minute)
    payload_hash = stable_hash(value)
    source_id = f"candle-{sequence}"
    return HistoricalEvent(
        replay_event_id=stable_id("test-history", "v1", "historical_candles", source_id, event_type, available_at.isoformat(), "XAUUSD", "M1", payload_hash, "1.0"),
        source_event_id=source_id,
        event_type=event_type,
        instrument="XAUUSD",
        timeframe="M1",
        occurred_at=available_at - timedelta(minutes=1),
        published_at=available_at,
        available_at=available_at,
        source_name="historical_candles",
        source_version="1.0.0",
        source_sequence=sequence,
        priority=priority,
        payload=value,
        payload_hash=payload_hash,
        dataset_id="test-history",
        dataset_version="v1",
    )


def replay_request(*, replay_id=None, mode: ReplayMode = ReplayMode.MAXIMUM_SPEED, speed: Decimal | None = None, checkpoint_every: int = 2) -> ReplayRequest:
    return ReplayRequest(
        replay_id=replay_id or uuid4(),
        name="May regression",
        instruments=("XAUUSD",),
        timeframes=("M1",),
        start_at=NOW,
        end_at=NOW + timedelta(minutes=30),
        mode=mode,
        speed_multiplier=speed,
        dataset=dataset(),
        engine_selection=("market_data",),
        engine_versions={"market_data": "1.0.0"},
        source_filters=ReplaySourceFilters(source_names=("historical_candles",)),
        checkpoint_policy=ReplayCheckpointPolicy(every_events=checkpoint_every, every_virtual_seconds=3600),
    )


async def build_service(events: tuple[HistoricalEvent, ...], *, config: ReplayConfig | None = None, sleeper=None):
    repository = InMemoryReplayRepository()
    source = InMemoryHistoricalSource("historical_candles", events)
    coordinator = ReplayCoordinator(
        repository,
        ReplayDatasetRegistry((dataset(),)),
        HistoricalSourceRegistry((source,)),
        production_replay_registry(),
        config or ReplayConfig(),
        now=lambda: NOW,
        **({"sleeper": sleeper} if sleeper else {}),
    )
    service = ReplayService(repository, coordinator, InMemoryEventBus(), config or ReplayConfig(), repository_mode="postgresql")
    await service.start()
    return service, repository


def test_request_dataset_event_validation_and_fingerprints() -> None:
    request = replay_request()
    assert request.fingerprint("1.0") == request.model_copy(update={"replay_id": uuid4(), "speed_multiplier": None}).fingerprint("1.0")
    assert request.dataset.instruments == ("XAUUSD",)
    legacy = request.model_dump(exclude={"instruments", "timeframes"}) | {"symbol": "xau/usd", "timeframe": "M1"}
    assert ReplayRequest.model_validate(legacy).instruments == ("XAUUSD",)
    with pytest.raises(ValidationError, match="start_at"):
        ReplayRequest.model_validate(request.model_dump() | {"start_at": request.end_at})
    with pytest.raises(ValidationError, match="speed multiplier"):
        ReplayRequest.model_validate(request.model_dump() | {"speed_multiplier": 2})
    with pytest.raises(ValidationError, match="requires a speed multiplier"):
        ReplayRequest.model_validate(request.model_dump() | {"mode": "accelerated", "speed_multiplier": None})
    with pytest.raises(ValidationError, match="payload hash"):
        HistoricalEvent.model_validate(historical_event(1, 1).model_dump() | {"payload_hash": "0" * 64})
    with pytest.raises(ValidationError, match="timestamp semantics"):
        HistoricalEvent.model_validate(historical_event(1, 1).model_dump() | {"published_at": NOW - timedelta(days=1)})


@pytest.mark.asyncio
async def test_complete_replay_is_deterministic_checkpointed_and_isolated() -> None:
    events = (historical_event(5, 2), historical_event(5, 1), historical_event(10, 3))
    service, repository = await build_service(events)
    first = await service.create(replay_request())
    assert first.status == ReplayStatus.READY
    await service.command_start(first.replay_id)
    await service.run(first.replay_id, "worker-a")
    completed = await service.get(first.replay_id)
    assert completed.status == ReplayStatus.COMPLETED
    assert completed.processed_events == 3
    assert completed.progress_percent == Decimal("100")
    assert completed.worker_id is None
    assert completed.semantic_output_hash != "0" * 64
    checkpoints = await repository.list_checkpoints(first.replay_id)
    assert checkpoints and all(item.calculated_state_hash() == item.state_hash for item in checkpoints)
    transitions = await repository.list_transitions(first.replay_id)
    assert [item.to_status for item in transitions][-1] == ReplayStatus.COMPLETED

    second = await service.create(replay_request())
    await service.command_start(second.replay_id)
    await service.run(second.replay_id, "worker-b")
    repeated = await service.get(second.replay_id)
    assert repeated.semantic_output_hash == completed.semantic_output_hash
    comparison = await service.compare(first.replay_id, second.replay_id)
    assert comparison.comparable and comparison.semantic_hash_equal


@pytest.mark.asyncio
async def test_step_pause_resume_cancel_and_idempotent_controls() -> None:
    service, repository = await build_service((historical_event(5, 1), historical_event(10, 2)))
    stepped = await service.create(replay_request(mode=ReplayMode.STEP, checkpoint_every=100))
    assert stepped.request.step_unit is not None
    await service.step(stepped.replay_id)
    paused = await service.get(stepped.replay_id)
    assert paused.status == ReplayStatus.PAUSED and paused.processed_events == 1
    assert await service.pause(paused.replay_id) == paused
    await service.resume(paused.replay_id)
    await service.step(paused.replay_id)
    assert (await service.get(paused.replay_id)).processed_events == 2
    await service.step(paused.replay_id)
    assert (await service.get(paused.replay_id)).status == ReplayStatus.COMPLETED

    cancelled = await service.create(replay_request())
    result = await service.cancel(cancelled.replay_id)
    assert result.status == ReplayStatus.CANCELLED
    assert (await repository.latest_checkpoint(cancelled.replay_id)) is not None
    assert await service.cancel(cancelled.replay_id) == result


@pytest.mark.asyncio
async def test_speed_modes_have_identical_semantics() -> None:
    waits: list[float] = []

    async def sleeper(value: float) -> None:
        waits.append(value)

    events = (historical_event(5, 1), historical_event(10, 2))
    hashes = []
    for mode, speed in ((ReplayMode.MAXIMUM_SPEED, None), (ReplayMode.ACCELERATED, Decimal("100")), (ReplayMode.REAL_TIME, None)):
        service, _ = await build_service(events, sleeper=sleeper)
        session = await service.create(replay_request(mode=mode, speed=speed))
        await service.command_start(session.replay_id)
        await service.run(session.replay_id, f"worker-{mode.value}")
        hashes.append((await service.get(session.replay_id)).semantic_output_hash)
    assert len(set(hashes)) == 1
    assert waits and all(value <= 60 for value in waits)


@pytest.mark.asyncio
async def test_scope_limits_missing_dataset_and_empty_source_fail_closed() -> None:
    strict = ReplayConfig(limits={"max_instruments": 1, "max_timeframes": 1, "max_duration_days": 1, "max_events_per_session": 100, "max_concurrent_sessions": 1, "max_sessions_per_owner": 1, "max_step_units": 1, "max_history_range_days": 1, "max_metadata_bytes": 64}, worker={"max_concurrency": 1})
    service, _ = await build_service((historical_event(5, 1),), config=strict)
    with pytest.raises(Exception, match="metadata"):
        await service.create(replay_request().model_copy(update={"metadata": {"value": "x" * 100}}))
    unknown = replay_request().model_copy(update={"dataset": dataset("missing")})
    with pytest.raises(Exception, match="not registered"):
        await service.create(unknown)

    empty_service, _ = await build_service(())
    with pytest.raises(Exception, match="source validation"):
        await empty_service.create(replay_request())
