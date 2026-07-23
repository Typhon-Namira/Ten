from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.economic_calendar_engine import (
    EconomicCalendarCheckpoint,
    EconomicCalendarConfig,
    EconomicCalendarService,
    FixedClock,
    InMemoryEconomicCalendarRepository,
    InMemoryProvider,
    ProviderConfig,
    ProviderMode,
    SqlAlchemyEconomicCalendarRepository,
    SystemClock,
)
from backend.app.engines.economic_calendar_engine.analyzer import build_snapshot, freshness, instrument_context, revision_between
from backend.app.engines.economic_calendar_engine.models import FreshnessState, ProviderStatus, payload_hash, stable_id
from backend.app.engines.economic_calendar_engine.normalization import normalize_observation
from backend.app.engines.economic_calendar_engine.providers import EconomicCalendarProvider, ProviderFetchRequest, ProviderFetchResult, _datetime
from backend.app.engines.economic_calendar_engine.registration import _build, _execute, register
from backend.app.engines.economic_calendar_engine.repository import _checkpoint_bytes
from tests.conftest import FakeSessionFactory
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.services.engine_factory import EngineFactory
from backend.app.services.pipeline_contracts import PipelineExecutionContext
from backend.app.storage.models import (
    EconomicCalendarEventRecord,
    EconomicCalendarObservationRecord,
    EconomicCalendarSnapshotRecord,
    EconomicCalendarSyncStateRecord,
)

NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)


def raw(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "nfp",
        "name": "Nonfarm Payrolls",
        "category": "employment",
        "country": "US",
        "currency": "USD",
        "importance": "critical",
        "status": "scheduled",
        "scheduled_at": (NOW + timedelta(minutes=5)).isoformat(),
        "available_at": (NOW - timedelta(hours=1)).isoformat(),
        "response_received_at": (NOW - timedelta(hours=1)).isoformat(),
        "ingested_at": (NOW - timedelta(hours=1)).isoformat(),
        "forecast": "180K",
        "previous": "175K",
    }
    value.update(changes)
    return value


def config(mode: ProviderMode = ProviderMode.IN_MEMORY_TEST_PROVIDER) -> EconomicCalendarConfig:
    return EconomicCalendarConfig(providers=(ProviderConfig(name="p", mode=mode, enabled=True),), provider_priority=("p",))


def observation(**changes: object):
    from backend.app.engines.economic_calendar_engine.providers import observation_from_mapping

    return observation_from_mapping("p", "1", raw(**changes), NOW)


def economic_event(**changes: object):
    return normalize_observation(observation(**changes), config())


def test_remaining_model_clock_config_and_freshness_branches() -> None:
    assert SystemClock().now().tzinfo is UTC
    assert FixedClock(NOW).now() == NOW
    assert payload_hash({"b": 2, "a": 1}) == payload_hash({"a": 1, "b": 2})
    cfg = EconomicCalendarConfig()
    assert (cfg.high_impact_pre_minutes, cfg.high_impact_post_minutes, cfg.medium_impact_pre_minutes, cfg.medium_impact_post_minutes) == (30, 30, 15, 15)
    with pytest.raises(ValidationError):
        EconomicCalendarConfig(freshness={"aging_minutes": 10, "stale_minutes": 5, "critical_minutes": 20})
    assert ProviderConfig(name="ok", timezone="UTC").timezone == "UTC"
    base = economic_event()
    with pytest.raises(ValidationError):
        base.model_copy(update={"trading_instruction": True}).__class__.model_validate({**base.model_dump(), "trading_instruction": True})
    with pytest.raises(ValidationError):
        base.__class__.model_validate({**base.model_dump(), "scheduled_at_utc": NOW + timedelta(days=3)})
    with pytest.raises(ValidationError):
        base.__class__.model_validate({**base.model_dump(), "status": "cancelled", "is_cancelled": False})
    with pytest.raises(ValidationError):
        base.__class__.model_validate({**base.model_dump(), "available_at": datetime(2026, 1, 1)})

    def statuses(age: int) -> tuple[ProviderStatus, ...]:
        return (ProviderStatus(provider_name="p", mode=ProviderMode.LIVE_PROVIDER, enabled=True, reachable=True, last_success=NOW - timedelta(minutes=age)),)

    assert freshness(NOW, statuses(40), cfg) == FreshnessState.AGING
    assert freshness(NOW, statuses(130), cfg) == FreshnessState.STALE
    assert freshness(NOW, statuses(1500), cfg) == FreshnessState.CRITICAL
    assert freshness(NOW, (), cfg) == FreshnessState.UNKNOWN


def test_file_import_rejects_oversize_and_invalid_shape(tmp_path) -> None:
    from backend.app.engines.economic_calendar_engine import FileImportProvider, build_providers

    root = tmp_path / "imports"
    root.mkdir()
    path = root / "bad.json"
    path.write_bytes(b" " * 10_000_001)
    with pytest.raises(ValueError, match="size"):
        FileImportProvider("bad", path, import_root=root)
    path.write_text('{"events": "not-a-list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="event objects"):
        FileImportProvider("bad", path, import_root=root)
    fixture = ProviderConfig(name="fixture", mode=ProviderMode.STATIC_FIXTURE, enabled=True)
    assert build_providers((fixture,), fixtures={"fixture": (raw(),)})[0].mode == ProviderMode.STATIC_FIXTURE
    disabled = ProviderConfig(name="live", mode=ProviderMode.LIVE_PROVIDER, enabled=True)
    assert build_providers((disabled,))[0].mode == ProviderMode.DISABLED
    assert build_providers(())[0].mode == ProviderMode.DISABLED
    missing = ProviderConfig(name="file", mode=ProviderMode.FILE_IMPORT, enabled=True)
    with pytest.raises(ValueError, match="file_path"):
        build_providers((missing,), import_root=root)
    good = root / "good.json"
    good.write_text('{"events": []}', encoding="utf-8")
    file_config = ProviderConfig(name="file", mode=ProviderMode.FILE_IMPORT, enabled=True, file_path=str(good))
    assert build_providers((file_config,), import_root=root)[0].mode == ProviderMode.FILE_IMPORT


def test_remaining_revision_and_snapshot_branches() -> None:
    initial = economic_event()
    corrected = initial.model_copy(update={"actual_value": 200000.0, "is_corrected": True})
    assert revision_between(initial, corrected, 2).revision_type.value == "correction"  # type: ignore[union-attr]
    moved = initial.model_copy(update={"scheduled_at_utc": NOW + timedelta(days=1), "is_rescheduled": True})
    assert revision_between(initial, moved, 2).revision_type.value == "reschedule"  # type: ignore[union-attr]
    statuses = (ProviderStatus(provider_name="disabled", mode=ProviderMode.DISABLED, enabled=False),)
    snapshot = build_snapshot((initial,), NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), statuses, EconomicCalendarConfig())
    assert snapshot.degradation.is_degraded and snapshot.freshness == FreshnessState.UNKNOWN
    no_mapping = instrument_context("ZZZ", snapshot, EconomicCalendarConfig())
    assert no_mapping.unavailable_context


def test_no_relevant_event_is_not_reported_as_unavailable_context() -> None:
    """Regression test: `unavailable_context` used to be set whenever a symbol simply had no
    matching-currency event in the window (`not relevant`) — the routine, expected state most of
    the time — which permanently HARD_BLOCKed signal_decision_engine's economic-event rule even
    with a fully healthy, live-syncing provider. It must only reflect genuine unavailability: the
    provider being unreachable, or the calendar sync being stale/never-synced."""
    healthy = (ProviderStatus(provider_name="p", mode=ProviderMode.LIVE_PROVIDER, enabled=True, reachable=True, last_success=NOW),)
    snapshot = build_snapshot((economic_event(),), NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), healthy, EconomicCalendarConfig())
    assert not snapshot.degradation.is_degraded
    assert snapshot.freshness == FreshnessState.FRESH

    irrelevant = instrument_context("EURJPY", snapshot, EconomicCalendarConfig())
    assert irrelevant.relevance_score == 0
    assert irrelevant.unavailable_context == ()

    relevant = instrument_context("XAUUSD", snapshot, EconomicCalendarConfig())
    assert relevant.relevance_score == 1.0
    assert relevant.unavailable_context == ()

    stale = (ProviderStatus(provider_name="p", mode=ProviderMode.LIVE_PROVIDER, enabled=True, reachable=True, last_success=NOW - timedelta(hours=3)),)
    stale_snapshot = build_snapshot((economic_event(),), NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), stale, EconomicCalendarConfig())
    assert stale_snapshot.freshness == FreshnessState.STALE
    assert instrument_context("XAUUSD", stale_snapshot, EconomicCalendarConfig()).unavailable_context


@pytest.mark.asyncio
async def test_provider_base_and_datetime_validation() -> None:
    class Concrete(EconomicCalendarProvider):
        name, version, timezone, mode = "x", "1", "UTC", ProviderMode.IN_MEMORY_TEST_PROVIDER
        from backend.app.engines.economic_calendar_engine.models import ProviderCapabilities

        capabilities = ProviderCapabilities()

        async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
            return ProviderFetchResult(observations=())

        async def health(self) -> ProviderStatus:
            return ProviderStatus(provider_name="x", mode=self.mode, enabled=True)

    provider = Concrete()
    assert await provider.fetch_event("x") is None
    assert not (await provider.fetch_updates(ProviderFetchRequest(start=NOW, end=NOW + timedelta(days=1)), None)).observations
    assert _datetime(NOW) == NOW
    with pytest.raises(ValueError):
        _datetime(datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        _datetime("2026-01-01T00:00:00")


class Scalars:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def all(self) -> list[Any]:
        return self.values

    def first(self) -> Any:
        return self.values[0] if self.values else None


class FakeSession:
    def __init__(self) -> None:
        self.gets: dict[tuple[type[Any], object], Any] = {}
        self.scalar_values: list[list[Any]] = []
        self.executed: list[Any] = []
        self.commits = 0

    async def execute(self, statement: Any) -> None:
        self.executed.append(statement)

    async def commit(self) -> None:
        self.commits += 1

    async def get(self, model: type[Any], key: object) -> Any:
        return self.gets.get((model, key))

    async def scalars(self, statement: Any) -> Scalars:
        self.executed.append(statement)
        return Scalars(self.scalar_values.pop(0) if self.scalar_values else [])


@pytest.mark.asyncio
async def test_sqlalchemy_repository_all_operations() -> None:
    session = FakeSession()
    repository = SqlAlchemyEconomicCalendarRepository(FakeSessionFactory(session))  # type: ignore[arg-type]
    obs = observation()
    item = economic_event()
    revision = revision_between(None, item, 1)
    assert revision
    status = (ProviderStatus(provider_name="p", mode=ProviderMode.LIVE_PROVIDER, enabled=True, reachable=True, last_success=NOW),)
    snapshot = build_snapshot((item,), NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), status, config())
    context = instrument_context("XAUUSD", snapshot, config())
    assert await repository.save_provider_observations(()) == 0
    assert await repository.save_provider_observations((obs,)) == 1
    session.gets[(EconomicCalendarObservationRecord, obs.observation_id)] = SimpleNamespace(payload=obs.model_dump(mode="json"))
    assert await repository.get_provider_observation(obs.observation_id) == obs
    assert await repository.get_provider_observation(uuid4()) is None
    session.scalar_values.append([SimpleNamespace(payload=obs.model_dump(mode="json"))])
    assert await repository.list_provider_observations() == (obs,)
    await repository.save_event(item)
    session.gets[(EconomicCalendarEventRecord, item.event_id)] = SimpleNamespace(payload=item.model_dump(mode="json"))
    assert (await repository.get_event(item.event_id)).event_id == item.event_id  # type: ignore[union-attr]
    assert await repository.get_event(uuid4()) is None
    session.scalar_values.append([SimpleNamespace(payload=item.model_dump(mode="json"))])
    listed = await repository.list_events(NOW - timedelta(days=1), NOW + timedelta(days=1), NOW, 10)
    assert listed[0].event_id == item.event_id
    await repository.save_revision(revision)
    session.scalar_values.append([SimpleNamespace(payload=revision.model_dump(mode="json"))])
    assert await repository.list_revisions(item.event_id) == (revision,)
    session.scalar_values.append([SimpleNamespace(payload=revision.model_dump(mode="json"))])
    assert (await repository.get_event_at_boundary(item.event_id, NOW)).event_id == item.event_id  # type: ignore[union-attr]
    session.scalar_values.append([])
    assert await repository.get_event_at_boundary(uuid4(), NOW) is None
    await repository.save_snapshot(snapshot)
    session.gets[(EconomicCalendarSnapshotRecord, snapshot.snapshot_id)] = SimpleNamespace(payload=snapshot.model_dump(mode="json"))
    assert (await repository.get_snapshot(snapshot.snapshot_id)).snapshot_id == snapshot.snapshot_id  # type: ignore[union-attr]
    assert await repository.get_snapshot(uuid4()) is None
    session.scalar_values.append([SimpleNamespace(payload=snapshot.model_dump(mode="json"))])
    assert (await repository.list_snapshots())[0].snapshot_id == snapshot.snapshot_id
    await repository.save_instrument_context(context)
    session.scalar_values.append([SimpleNamespace(payload=context.model_dump(mode="json"))])
    assert (await repository.get_instrument_context("XAUUSD", NOW)).context_id == context.context_id  # type: ignore[union-attr]
    session.scalar_values.append([])
    assert await repository.get_instrument_context("XAUUSD") is None
    sync = {"cursor": "a", "updated_at": NOW.isoformat()}
    await repository.save_sync_state("p", sync)
    session.gets[(EconomicCalendarSyncStateRecord, "p")] = SimpleNamespace(state=sync)
    assert await repository.load_sync_state("p") == sync
    assert await repository.load_sync_state("none") is None
    state = {"last_successful_sync_at": NOW.isoformat(), "last_provider_cursor": {"p": "a"}, "identity": {}}
    checkpoint = EconomicCalendarCheckpoint(
        checkpoint_id=uuid4(), state_payload=state, payload_hash=sha256(_checkpoint_bytes(state)).hexdigest(), created_at=NOW
    )
    await repository.save_checkpoint(checkpoint)
    with pytest.raises(ValueError):
        await repository.save_checkpoint(checkpoint.model_copy(update={"payload_hash": "bad"}))
    record = SimpleNamespace(
        id=checkpoint.checkpoint_id,
        engine_name="economic_calendar",
        engine_version="1.0.0",
        schema_version="1.0",
        configuration_version="1.0.0",
        normalization_version="1.0.0",
        payload_hash=checkpoint.payload_hash,
        state_payload=state,
        created_at=NOW,
    )
    session.scalar_values.append([record])
    assert await repository.load_checkpoint() is not None
    session.scalar_values.append([])
    assert await repository.load_checkpoint() is None
    session.scalar_values.append([SimpleNamespace(**{**vars(record), "payload_hash": "bad"})])
    with pytest.raises(ValueError):
        await repository.load_checkpoint()
    session.scalar_values.extend([[uuid4()], [], [uuid4()]])
    removed = await repository.prune_history(1, 1, 1)
    assert removed == {"events": 1, "observations": 0, "snapshots": 1}
    assert session.commits > 0 and session.executed


@pytest.mark.asyncio
async def test_in_memory_repository_remaining_filters_and_corrupt_checkpoint() -> None:
    repository = InMemoryEconomicCalendarRepository()
    before = economic_event(available_at=(NOW + timedelta(days=2)).isoformat(), response_received_at=(NOW + timedelta(days=2)).isoformat())
    early = economic_event(id="early", name="Early", scheduled_at=(NOW - timedelta(days=3)).isoformat())
    late = economic_event(id="late", name="Late", scheduled_at=(NOW + timedelta(days=3)).isoformat())
    await repository.save_event(before)
    await repository.save_event(early)
    await repository.save_event(late)
    assert await repository.list_events(NOW - timedelta(days=1), NOW + timedelta(days=1), NOW) == ()
    statuses = (ProviderStatus(provider_name="p", mode=ProviderMode.LIVE_PROVIDER, enabled=True, reachable=True, last_success=NOW),)
    snapshot = build_snapshot((), NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), statuses, config())
    await repository.save_snapshot(snapshot)
    assert (await repository.list_snapshots())[0] == snapshot
    state: dict[str, Any] = {"identity": {}}
    checkpoint = EconomicCalendarCheckpoint(
        checkpoint_id=uuid4(), state_payload=state, payload_hash=sha256(_checkpoint_bytes(state)).hexdigest(), created_at=NOW
    )
    await repository.save_checkpoint(checkpoint)
    object.__setattr__(checkpoint, "payload_hash", "bad")
    with pytest.raises(ValueError, match="integrity"):
        await repository.load_checkpoint()


@pytest.mark.asyncio
async def test_sql_provider_observation_filter_and_boundary_future() -> None:
    session = FakeSession()
    repository = SqlAlchemyEconomicCalendarRepository(FakeSessionFactory(session))  # type: ignore[arg-type]
    obs = observation()
    item = economic_event()
    session.scalar_values.append([SimpleNamespace(payload=obs.model_dump(mode="json"))])
    session.gets[(EconomicCalendarEventRecord, item.event_id)] = SimpleNamespace(payload=item.model_dump(mode="json"))
    assert await repository.list_provider_observations(item.event_id) == (obs,)
    session.scalar_values.append([])
    assert await repository.get_event_at_boundary(item.event_id, NOW - timedelta(days=2)) is None
    released = economic_event(
        actual="200K",
        status="released",
        available_at=(NOW + timedelta(minutes=10)).isoformat(),
        response_received_at=(NOW + timedelta(minutes=10)).isoformat(),
        ingested_at=(NOW + timedelta(minutes=10)).isoformat(),
    )
    initial_revision = revision_between(None, item, 1)
    release_revision = revision_between(item, released, 2)
    assert initial_revision and release_revision
    session.gets[(EconomicCalendarEventRecord, item.event_id)] = SimpleNamespace(payload=released.model_dump(mode="json"))
    session.scalar_values.append(
        [SimpleNamespace(payload=initial_revision.model_dump(mode="json")), SimpleNamespace(payload=release_revision.model_dump(mode="json"))]
    )
    reconstructed = await repository.get_event_at_boundary(item.event_id, NOW)
    assert reconstructed and reconstructed.actual_value is None and reconstructed.revision_count == 1


class FailingProvider(InMemoryProvider):
    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        raise RuntimeError("provider down")

    async def health(self) -> ProviderStatus:
        return ProviderStatus(provider_name=self.name, mode=self.mode, enabled=True, reachable=False, stale=True)


class LiveProvider(InMemoryProvider):
    def __init__(self) -> None:
        super().__init__("p", (raw(),), mode=ProviderMode.LIVE_PROVIDER)


class FailingBus(InMemoryEventBus):
    async def publish(self, event: Any) -> None:
        raise RuntimeError("bus down")


class FailingStore(InMemoryFeatureStore):
    async def write(self, feature: Any) -> None:
        raise RuntimeError("store down")


class FailingCheckpointRepository(InMemoryEconomicCalendarRepository):
    async def save_checkpoint(self, checkpoint: EconomicCalendarCheckpoint) -> None:
        raise RuntimeError("checkpoint down")


@pytest.mark.asyncio
async def test_service_failure_isolation_and_lifecycle() -> None:
    service = EconomicCalendarService(FailingBus(), FailingStore(), config(), providers=(FailingProvider("p", ()),), clock=FixedClock(NOW))
    await service.start()
    snapshot = await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1))
    assert snapshot.degradation.is_degraded
    assert service.metrics.provider_request_failures == 1 and service.metrics.event_publication_failures > 0
    valid = EconomicCalendarService(InMemoryEventBus(), FailingStore(), config(), providers=(InMemoryProvider("p", (raw(),)),), clock=FixedClock(NOW))
    await valid.restore()
    await valid.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1))
    await valid.context("XAUUSD")
    assert valid.metrics.feature_publication_failures == 1
    broken = EconomicCalendarService(
        InMemoryEventBus(), InMemoryFeatureStore(), config(), FailingCheckpointRepository(), (InMemoryProvider("p", (raw(),)),), clock=FixedClock(NOW)
    )
    await broken.restore()
    with pytest.raises(RuntimeError):
        await broken.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1))
    assert broken.metrics.checkpoint_failures == 1 and broken.metrics.synchronization_failures == 1
    live = EconomicCalendarService(
        InMemoryEventBus(), InMemoryFeatureStore(), config(ProviderMode.LIVE_PROVIDER), providers=(LiveProvider(),), clock=FixedClock(NOW)
    )
    await live.start()
    assert live._scheduler is not None
    await live.stop()
    assert live._scheduler is None


@pytest.mark.asyncio
async def test_service_poll_duplicate_publication_same_identity_and_corrupt_restore(monkeypatch) -> None:
    service = EconomicCalendarService(InMemoryEventBus(), InMemoryFeatureStore(), config(), providers=(InMemoryProvider("p", (raw(),)),), clock=FixedClock(NOW))
    await service.restore()
    await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1))
    await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1), boundary=NOW + timedelta(minutes=1))
    assert service.metrics.events_deduplicated >= 1
    assert service.metrics.snapshot()["events_fetched"] == 2
    identifier = stable_id("duplicate-test")
    await service._publish_event(
        __import__("backend.app.engines.economic_calendar_engine.events", fromlist=["EconomicCalendarDegraded"]).EconomicCalendarDegraded, identifier, {}
    )
    count = len(service._published)
    await service._publish_event(
        __import__("backend.app.engines.economic_calendar_engine.events", fromlist=["EconomicCalendarDegraded"]).EconomicCalendarDegraded, identifier, {}
    )
    assert len(service._published) == count

    service.synchronize = AsyncMock(side_effect=RuntimeError("poll"))  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await service._poll()
    assert service.metrics.synchronization_failures >= 1

    class CorruptRepository(InMemoryEconomicCalendarRepository):
        async def load_checkpoint(self):
            state = {"identity": {}}
            return EconomicCalendarCheckpoint(checkpoint_id=uuid4(), state_payload=state, payload_hash="bad", created_at=NOW)

    corrupt = EconomicCalendarService(InMemoryEventBus(), InMemoryFeatureStore(), config(), CorruptRepository(), clock=FixedClock(NOW))
    with pytest.raises(ValueError, match="integrity"):
        await corrupt.restore()


@pytest.mark.asyncio
async def test_calendar_uses_hourly_normal_and_adaptive_high_impact_refresh() -> None:
    near = EconomicCalendarService(
        InMemoryEventBus(), InMemoryFeatureStore(), config(), providers=(InMemoryProvider("p", (raw(),)),), clock=FixedClock(NOW)
    )
    await near.restore()
    near_snapshot = await near.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1))
    assert near.config.processing.polling_interval_seconds == 3600
    assert near._next_sync_delay(near_snapshot, NOW) == near.config.processing.adaptive_refresh_seconds == 300
    assert near.metrics.relevant_upcoming_event is not None

    far = EconomicCalendarService(
        InMemoryEventBus(),
        InMemoryFeatureStore(),
        config(),
        providers=(InMemoryProvider("p", (raw(scheduled_at=(NOW + timedelta(days=1)).isoformat()),)),),
        clock=FixedClock(NOW),
    )
    await far.restore()
    far_snapshot = await far.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=2))
    assert far._next_sync_delay(far_snapshot, NOW) == 3600
    assert far.metrics.relevant_upcoming_event is None


@pytest.mark.asyncio
async def test_recovery_and_replay_failure_branches() -> None:
    repository = InMemoryEconomicCalendarRepository()
    state: dict[str, Any] = {"identity": {}}
    incompatible = EconomicCalendarCheckpoint(
        checkpoint_id=uuid4(), engine_version="2", state_payload=state, payload_hash=sha256(_checkpoint_bytes(state)).hexdigest(), created_at=NOW
    )
    await repository.save_checkpoint(incompatible)
    service = EconomicCalendarService(InMemoryEventBus(), InMemoryFeatureStore(), config(), repository, clock=FixedClock(NOW))
    with pytest.raises(ValueError):
        await service.restore()
    assert service.recovery_state == "failed"

    class ReplayFailure(EconomicCalendarService):
        async def context(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("replay failed")

    replay = ReplayFailure(InMemoryEventBus(), InMemoryFeatureStore(), config(), clock=FixedClock(NOW))
    with pytest.raises(RuntimeError):
        await replay.replay("XAUUSD", NOW)
    assert replay.metrics.replay_failures == 1
    with pytest.raises(ValueError):
        await replay.synchronize(datetime(2026, 1, 1), NOW)


@pytest.mark.asyncio
async def test_registration_contract() -> None:
    from backend.app.engines.market_data_engine import Candle, Timeframe
    from backend.app.features import InMemoryFeatureStore

    factory = EngineFactory()
    register(factory)
    built = _build(None, {})  # type: ignore[arg-type]
    item = economic_event()
    candle = Candle(symbol="XAUUSD", timeframe=Timeframe.M15, timestamp=NOW, open=1, high=2, low=0.5, close=1.5, volume=1, available_at=NOW)
    context = PipelineExecutionContext(correlation_id=uuid4(), candles=[candle], events=[item], feature_store=InMemoryFeatureStore(), now=NOW)
    result = await _execute(built, context)
    assert result.namespace == "economic" and result.confidence_factor == 1.0
