from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.events import NewCandle
from backend.app.events import InMemoryEventBus
from backend.app.integration import CanonicalEventEnvelope, FullSystemIntegrationService, InMemoryIntegrationRepository, IntegrationConfig, IntegrationMode, canonical_hash


class FakeMarketData:
    def __init__(self, candle: Candle) -> None:
        self.candle = candle

    async def history(self, *_: object, **__: object) -> list[Candle]:
        return [self.candle] * 40


class EmptyMarketData(FakeMarketData):
    async def history(self, *_: object, **__: object) -> list[Candle]:
        return []


class FakeAnalysis:
    async def analyze_candles(self, *_: object, **__: object) -> object:
        return SimpleNamespace(id=uuid4(), analysis_timestamp=NOW, engine_version="1.0.0")

    async def analyze(self, *_: object, **__: object) -> object:
        return SimpleNamespace(id=uuid4(), analysis_timestamp=NOW, engine_version="1.0.0")

    async def analyze_snapshot(self, *_: object, **__: object) -> object:
        return SimpleNamespace(snapshot_id=uuid4(), analysis_timestamp=NOW, engine_version="1.0.0")

    async def context(self, *_: object, **__: object) -> object:
        return SimpleNamespace(context_id=uuid4(), as_of=NOW, engine_version="1.0.0")


class FakeScore:
    async def calculate(self, request: object) -> object:
        return SimpleNamespace(snapshot_id=uuid4(), metadata=SimpleNamespace(input_fingerprint="a" * 64), policy_version="1.0.0")


class FakeDecision:
    async def evaluate(self, request: object) -> object:
        return SimpleNamespace(decision_id=uuid4(), input_fingerprint="b" * 64, direction=SimpleNamespace(value="neutral"), state=SimpleNamespace(value="blocked"), confidence_score=50.0, as_of=NOW, valid_until=NOW + timedelta(minutes=15), blockers=(SimpleNamespace(reason_code="analytical_only"),), warnings=(), decision_policy_version="1.0.0")


NOW = datetime(2026, 7, 19, 12, 30, tzinfo=UTC)


def candle() -> Candle:
    return Candle(timestamp=NOW - timedelta(minutes=15), ingestion_timestamp=NOW, symbol="XAU/USD", timeframe=Timeframe.M15, open=3300, high=3305, low=3298, close=3302, volume=100, provider="golden")


def service(bus: InMemoryEventBus, repository: InMemoryIntegrationRepository) -> FullSystemIntegrationService:
    analysis = FakeAnalysis()
    config = IntegrationConfig(worker={"enabled": True, "embedded_api_worker": True})
    return FullSystemIntegrationService(event_bus=bus, repository=repository, config=config, market_data=FakeMarketData(candle()), smc=analysis, liquidity=analysis, volume_profile=analysis, institutional_flow=analysis, market_regime=analysis, economic_calendar=analysis, ai_scoring=FakeScore(), signal_decision=FakeDecision(), repository_mode="postgresql", clock=lambda: NOW)


@pytest.mark.asyncio
async def test_market_event_drives_one_persisted_traceable_signal() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    await coordinator.start()
    event = NewCandle(correlation_id=uuid4(), source="market_data", payload=candle().model_dump(mode="json"))
    await bus.publish(event)
    await bus.publish(event)
    signals = await repository.signals()
    assert len(signals) == 1
    assert signals[0].provider_provenance == ("golden",)
    assert signals[0].analytical_only and not signals[0].trade_execution
    trace = await repository.trace(signals[0].trace_id)
    assert trace and trace[0].output_references[-1] == str(signals[0].operational_signal_id)
    assert coordinator.health()["ready"] is True
    await coordinator.stop()


def test_envelope_semantic_identity_and_mode_isolation() -> None:
    value = candle()
    first = CanonicalEventEnvelope.final_candle(value, uuid4(), NOW)
    second = CanonicalEventEnvelope.final_candle(value, uuid4(), NOW)
    assert first.event_id == second.event_id
    assert first.payload_hash == canonical_hash(first.payload)
    raw = first.model_dump()
    raw.update({"mode": IntegrationMode.REPLAY, "source_name": "live_provider"})
    with pytest.raises(ValueError, match="Replay"):
        CanonicalEventEnvelope.model_validate(raw)


def test_graph_is_exact_and_rejects_cycles() -> None:
    config = IntegrationConfig()
    assert len(config.graph) == 10
    assert {node.engine for node in config.graph} == {"market_data", "smc", "liquidity", "volume_profile", "institutional_flow", "market_regime", "economic_calendar", "ai_scoring", "signal_decision", "replay"}


@pytest.mark.asyncio
async def test_stale_and_rejected_events_fail_closed() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    old = candle().model_copy(update={"timestamp": NOW - timedelta(hours=2), "ingestion_timestamp": NOW - timedelta(hours=1, minutes=45)})
    stale = CanonicalEventEnvelope.final_candle(old, uuid4(), NOW)
    assert await coordinator.process(stale) is None
    rejected_candle = candle().model_copy(update={"quality_score": 10})
    rejected = CanonicalEventEnvelope.final_candle(rejected_candle, uuid4(), NOW)
    assert await coordinator.process(rejected) is None
    assert repository.metrics()["quality_issues"] == 2
    assert await repository.signals() == ()


@pytest.mark.asyncio
async def test_missing_mandatory_evidence_blocks_scoring() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.market_data = EmptyMarketData(candle())
    envelope = CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW)
    assert await coordinator.process(envelope) is None
    assert repository.metrics()["snapshots"] == 1
    assert await repository.signals() == ()
