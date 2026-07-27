import asyncio
from datetime import UTC, datetime, timedelta
import logging
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.events import NewCandle
from backend.app.engines.market_regime_engine import InMemoryMarketRegimeRepository, MarketRegimeService
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.integration import CanonicalEventEnvelope, FullSystemIntegrationService, InMemoryIntegrationRepository, IntegrationConfig, IntegrationMode, MissingIntegrationEventError, OperationalSignal, canonical_hash
from backend.app.integration.stage_tracker import PipelineStageTracker


class FakeMarketData:
    def __init__(self, candle: Candle) -> None:
        self.candle = candle

    async def history(self, *_: object, **__: object) -> list[Candle]:
        return [self.candle] * 40

    async def replay(self, *_: object, **__: object) -> list[Candle]:
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


class DegradedVolumeAnalysis(FakeAnalysis):
    async def analyze(self, *_: object, **__: object) -> object:
        return SimpleNamespace(
            id=uuid4(),
            analysis_timestamp=NOW,
            engine_version="1.0.0",
            status=SimpleNamespace(value="degraded"),
            degraded_reasons=("insufficient_volume_profile_data",),
        )


class FakeScore:
    async def calculate(self, request: object) -> object:
        return SimpleNamespace(snapshot_id=uuid4(), metadata=SimpleNamespace(input_fingerprint="a" * 64), policy_version="1.0.0")


class FakeDecision:
    async def evaluate(self, request: object) -> object:
        return SimpleNamespace(decision_id=uuid4(), input_fingerprint="b" * 64, direction=SimpleNamespace(value="neutral"), state=SimpleNamespace(value="blocked"), confidence_score=50.0, as_of=NOW, valid_until=NOW + timedelta(minutes=15), blockers=(SimpleNamespace(reason_code="analytical_only"),), warnings=(), decision_policy_version="1.0.0")


class FakeEligibleDecision(FakeDecision):
    async def evaluate(self, request: object) -> object:
        return SimpleNamespace(decision_id=uuid4(), input_fingerprint="c" * 64, direction=SimpleNamespace(value="bullish"), state=SimpleNamespace(value="eligible"), confidence_score=82.0, as_of=NOW, valid_until=NOW + timedelta(minutes=15), blockers=(), warnings=(), decision_policy_version="1.0.0")


class FailingDecision(FakeDecision):
    async def evaluate(self, request: object) -> object:
        raise RuntimeError("decision persistence failed")


class UnexpectedShadowCapture:
    async def capture_cycle(self, *_: object, **__: object) -> None:
        raise AssertionError("disabled shadow capture must not be invoked")


class CompleteShadowCapture:
    async def capture_cycle(self, envelope: CanonicalEventEnvelope, *_: object, **__: object) -> object:
        payload = envelope.payload
        return SimpleNamespace(
            state_id=uuid4(),
            trigger_timeframe=payload.timeframe,
            market_data_boundary=payload.close_time,
        )


class FailingQuantForecast:
    async def forecast(self, _: object) -> None:
        raise RuntimeError("shadow model failed")


class SuccessfulQuantForecast:
    async def forecast(self, _: object) -> object:
        return object()


class FailingAIReasoning:
    async def process(self, _state: object, _forecast: object) -> None:
        raise RuntimeError("AI reasoning failed")


class RecordingUnifiedState:
    def __init__(self, stages: list[str]) -> None:
        self.stages = stages

    async def capture_cycle(self, envelope: CanonicalEventEnvelope, *_: object, **__: object) -> object:
        self.stages.append("unified_market_state")
        payload = envelope.payload
        return SimpleNamespace(
            state_id=uuid4(),
            trigger_timeframe=payload.timeframe,
            market_data_boundary=payload.close_time,
        )


class RecordingQuantForecast:
    def __init__(self, stages: list[str]) -> None:
        self.stages = stages

    async def forecast(self, _: object) -> object:
        self.stages.append("quant_forecast")
        return object()


class RecordingAIAndFinalDecision:
    def __init__(self, stages: list[str]) -> None:
        self.stages = stages

    async def process(self, _state: object, _forecast: object) -> None:
        self.stages.extend(("ai_reasoning", "final_decision"))


NOW = datetime(2026, 7, 19, 12, 30, tzinfo=UTC)


def candle(timeframe: Timeframe = Timeframe.M15) -> Candle:
    return Candle(timestamp=NOW - timeframe.duration, ingestion_timestamp=NOW, symbol="XAU/USD", timeframe=timeframe, open=3300, high=3305, low=3298, close=3302, volume=100, provider="golden")


def service(bus: InMemoryEventBus, repository: InMemoryIntegrationRepository) -> FullSystemIntegrationService:
    analysis = FakeAnalysis()
    config = IntegrationConfig(worker={"enabled": True, "embedded_api_worker": True})
    return FullSystemIntegrationService(event_bus=bus, repository=repository, config=config, market_data=FakeMarketData(candle()), smc=analysis, liquidity=analysis, volume_profile=analysis, institutional_flow=analysis, market_regime=analysis, economic_calendar=analysis, ai_scoring=FakeScore(), signal_decision=FakeDecision(), repository_mode="postgresql", clock=lambda: NOW)


@pytest.mark.asyncio
async def test_blocked_market_event_persists_decision_without_false_signal() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    await coordinator.start()
    event = NewCandle(correlation_id=uuid4(), source="market_data", payload=candle().model_dump(mode="json"))
    await bus.publish(event)
    await bus.publish(event)
    assert await repository.signals() == ()
    assert repository.metrics()["snapshots"] == 1
    assert coordinator.health()["ready"] is True
    await coordinator.stop()


@pytest.mark.asyncio
async def test_new_candle_event_reaches_smc_through_the_exact_production_wiring_shape() -> None:
    """Regression test for the "market data healthy but SMC-onward chain silent" investigation.

    `main.py` hardcodes `embedded_api_worker=False` in production — a published `NewCandle` only
    gets enqueued by `_on_candle`; nothing processes it until something else calls
    `process_outbox_once()` (in production, `IntegrationWorker`'s poll loop). Every other test in
    this file uses `embedded_api_worker=True`, which processes inline and would not have caught a
    bug specific to the enqueue-then-drain-later path production actually uses. This test
    reproduces that exact shape: publish on the real event bus, assert nothing has run yet, then
    drain the outbox exactly as `IntegrationWorker.run()` does, and assert SMC (via the snapshot
    it feeds into) actually ran and produced a result — not silence.
    """
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    config = IntegrationConfig(worker={"enabled": True, "embedded_api_worker": False})
    analysis = FakeAnalysis()
    coordinator = FullSystemIntegrationService(
        event_bus=bus, repository=repository, config=config, market_data=FakeMarketData(candle()), smc=analysis, liquidity=analysis,
        volume_profile=analysis, institutional_flow=analysis, market_regime=analysis, economic_calendar=analysis, ai_scoring=FakeScore(),
        signal_decision=FakeDecision(), repository_mode="postgresql", clock=lambda: NOW,
    )
    await coordinator.start()
    event = NewCandle(correlation_id=uuid4(), source="market_data", payload=candle().model_dump(mode="json"))
    await bus.publish(event)
    # Enqueued, but nothing has processed it yet — exactly the production shape.
    assert repository.metrics()["snapshots"] == 0
    assert repository.metrics()["outbox_backlog"] == 1
    processed = await coordinator.process_outbox_once()
    assert processed == 1
    assert repository.metrics()["snapshots"] == 1  # SMC-onward genuinely ran, not silence
    assert repository.metrics()["outbox_backlog"] == 0
    assert coordinator.failures == 0
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


def test_event_id_is_stable_across_repeated_polls_of_the_same_candle() -> None:
    """Regression test for the "market data healthy, SMC-onward chain runs constantly but nothing
    ever progresses" investigation: `event_id` used to be derived from `payload_hash` (a hash of
    the WHOLE payload, including `ingestion_time` — stamped fresh via `datetime.now(UTC)` on every
    single fetch). Two polls of the exact same already-closed candle, seconds apart, therefore
    produced two different `event_id`s, so `repository.processed(event_id)` could never recognize
    the second poll as a duplicate — `process()` re-ran the entire SMC-onward chain from scratch
    on every single poll of an already-processed candle, forever, each time re-triggering a fresh
    `MarketDataService.history()` call (and its full anomaly-validation pass) too. `event_id` must
    depend only on the candle's own identity (provider, instrument, timeframe, open time,
    revision) — not on when any particular fetch happened to observe it."""
    value = candle()
    polled_a_moment_apart = value.model_copy(update={"ingestion_timestamp": value.ingestion_timestamp + timedelta(seconds=4)})
    first = CanonicalEventEnvelope.final_candle(value, uuid4(), NOW)
    second = CanonicalEventEnvelope.final_candle(polled_a_moment_apart, uuid4(), NOW + timedelta(seconds=4))
    assert first.event_id == second.event_id
    assert first.idempotency_key == second.idempotency_key
    # `payload_hash` itself must still faithfully reflect each envelope's own payload — only its
    # role as an INPUT to `event_id` was the bug, not the field's own correctness.
    assert first.payload_hash == canonical_hash(first.payload)
    assert second.payload_hash == canonical_hash(second.payload)
    assert first.payload_hash != second.payload_hash  # the payloads genuinely differ (ingestion_time)

    # A genuinely different candle (different open time, or a different provider) must NOT collide.
    later = value.model_copy(update={"timestamp": value.timestamp + value.timeframe.duration})
    different_provider = value.model_copy(update={"provider": "a-different-provider"})
    later_envelope = CanonicalEventEnvelope.final_candle(later, uuid4(), NOW + timedelta(minutes=15))
    provider_envelope = CanonicalEventEnvelope.final_candle(different_provider, uuid4(), NOW)
    assert later_envelope.event_id != first.event_id
    assert provider_envelope.event_id != first.event_id


@pytest.mark.asyncio
async def test_repeated_polls_of_the_same_candle_never_create_duplicate_snapshots() -> None:
    """End-to-end companion to the `event_id` stability test above, through the real event-bus ->
    enqueue -> outbox-drain path (not the direct `CanonicalEventEnvelope` construction) — two
    `NewCandle` publications for the exact same underlying candle, seconds apart, must result in
    exactly one processed event and one snapshot, not two."""
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    config = IntegrationConfig(worker={"enabled": True, "embedded_api_worker": False})
    analysis = FakeAnalysis()
    coordinator = FullSystemIntegrationService(
        event_bus=bus, repository=repository, config=config, market_data=FakeMarketData(candle()), smc=analysis, liquidity=analysis,
        volume_profile=analysis, institutional_flow=analysis, market_regime=analysis, economic_calendar=analysis, ai_scoring=FakeScore(),
        signal_decision=FakeDecision(), repository_mode="postgresql", clock=lambda: NOW,
    )
    await coordinator.start()
    value = candle()
    first_poll = NewCandle(correlation_id=uuid4(), source="market_data", payload=value.model_dump(mode="json"))
    second_poll = NewCandle(
        correlation_id=uuid4(),
        source="market_data",
        payload=value.model_copy(update={"ingestion_timestamp": value.ingestion_timestamp + timedelta(seconds=4)}).model_dump(mode="json"),
    )
    await bus.publish(first_poll)
    await coordinator.process_outbox_once()
    await bus.publish(second_poll)
    processed_second = await coordinator.process_outbox_once()
    assert processed_second == 0  # nothing new to do — the second poll was correctly recognized as a duplicate
    assert repository.metrics()["events"] == 1
    assert repository.metrics()["snapshots"] == 1
    await coordinator.stop()


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


@pytest.mark.asyncio
async def test_historical_bootstrap_runs_replay_analysis_without_live_signal() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    await coordinator.process_historical_candle(candle())
    await coordinator.process_historical_candle(candle())
    snapshot = await repository.latest_snapshot("XAUUSD", "M15")
    assert snapshot is not None
    assert snapshot.mode == IntegrationMode.REPLAY
    assert repository.metrics()["snapshots"] == 1
    assert await repository.signals() == ()


class OrderingRepository(InMemoryIntegrationRepository):
    def __init__(self) -> None:
        super().__init__()
        self.persistence_order: list[tuple[str, str]] = []

    async def persist_event(self, envelope: CanonicalEventEnvelope) -> None:
        self.persistence_order.append(("parent", envelope.event_id))
        await super().persist_event(envelope)

    async def mark_processed(self, event_id: str) -> None:
        self.persistence_order.append(("processed", event_id))
        await super().mark_processed(event_id)


@pytest.mark.asyncio
async def test_historical_parent_is_persisted_before_processed_marker() -> None:
    repository = OrderingRepository()
    coordinator = service(InMemoryEventBus(), repository)

    await coordinator.process_historical_candle(candle())

    assert [stage for stage, _ in repository.persistence_order] == ["parent", "processed"]
    assert len({event_id for _, event_id in repository.persistence_order}) == 1
    assert repository.metrics()["events"] == 1
    assert repository.metrics()["processed"] == 1


@pytest.mark.asyncio
async def test_missing_parent_cannot_be_marked_processed() -> None:
    repository = InMemoryIntegrationRepository()
    missing_event_id = "f" * 64

    with pytest.raises(MissingIntegrationEventError, match=missing_event_id):
        await repository.mark_processed(missing_event_id)

    assert repository.metrics()["processed"] == 0


class FailFirstSnapshotRepository(InMemoryIntegrationRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_snapshot = True

    async def save_snapshot(self, value: object) -> object:
        if self.fail_next_snapshot:
            self.fail_next_snapshot = False
            raise RuntimeError("simulated snapshot rollback")
        return await super().save_snapshot(value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_failed_historical_cycle_retries_idempotently_without_orphan_marker() -> None:
    repository = FailFirstSnapshotRepository()
    coordinator = service(InMemoryEventBus(), repository)

    with pytest.raises(RuntimeError, match="simulated snapshot rollback"):
        await coordinator.process_historical_candle(candle())

    assert repository.metrics()["events"] == 1
    assert repository.metrics()["processed"] == 0
    assert repository.metrics()["snapshots"] == 0

    await coordinator.process_historical_candle(candle())
    await coordinator.process_historical_candle(candle())

    assert repository.metrics()["events"] == 1
    assert repository.metrics()["processed"] == 1
    assert repository.metrics()["snapshots"] == 1


@pytest.mark.asyncio
async def test_concurrent_historical_duplicate_delivery_is_safe() -> None:
    repository = InMemoryIntegrationRepository()
    coordinator = service(InMemoryEventBus(), repository)

    await asyncio.gather(
        coordinator.process_historical_candle(candle()),
        coordinator.process_historical_candle(candle()),
    )

    assert repository.metrics()["events"] == 1
    assert repository.metrics()["processed"] == 1
    assert repository.metrics()["snapshots"] == 1


@pytest.mark.asyncio
async def test_only_eligible_live_decision_publishes_traceable_scenario() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.signal_decision = FakeEligibleDecision()
    signal = await coordinator.process(CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW))
    assert signal is not None
    assert signal.state == "eligible"
    assert signal.mode == IntegrationMode.LIVE
    assert signal.provider_provenance == ("golden",)
    trace = await repository.trace(signal.trace_id)
    assert trace and trace[0].output_references[-1] == str(signal.operational_signal_id)


class FailingHistoryMarketData(FakeMarketData):
    async def history(self, *_: object, **__: object) -> list[Candle]:
        raise RuntimeError("provider rate limited")


class FailingSaveRepository(InMemoryIntegrationRepository):
    async def save_snapshot(self, value: object) -> object:
        raise RuntimeError("database write failed")


@pytest.mark.asyncio
async def test_market_data_failure_finalizes_the_stage_tracker_cycle_instead_of_freezing_it() -> None:
    """Regression test: `market_data.history()` used to be called before the try/except that
    invokes `tracker.fail_in_flight()`, so a provider failure (e.g. a rate-limit circuit-breaker
    trip) left the cycle permanently stuck at candle_received=success/smc_analysis=waiting
    forever (rendered "running" indefinitely, per `_render()`) instead of reaching a terminal
    failed state — exactly the dashboard symptom of a stage frozen on "running" while later,
    unrelated candles kept completing and publishing their own events."""
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    tracker = PipelineStageTracker()
    coordinator = service(bus, repository)
    coordinator.stage_tracker = tracker
    coordinator.market_data = FailingHistoryMarketData(candle())
    envelope = CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW)
    with pytest.raises(RuntimeError):
        await coordinator.process(envelope)
    cycle = tracker.latest("XAUUSD", "M15")
    assert cycle is not None
    assert cycle["complete"] is True
    stages = {item["key"]: item["status"] for item in cycle["stages"]}
    assert stages["candle_received"] == "success"
    assert stages["smc_analysis"] == "failed"
    assert stages["liquidity_analysis"] == "skipped"
    trace = await repository.trace(envelope.trace_id)
    assert len(trace) == 1
    assert trace[0].status.value == "failed"


@pytest.mark.asyncio
async def test_failed_outbox_item_degrades_health_and_is_not_hot_retried() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.market_data = FailingHistoryMarketData(candle())
    await coordinator.start()
    await bus.publish(NewCandle(correlation_id=uuid4(), source="market_data", payload=candle().model_dump(mode="json")))

    assert coordinator.last_batch_failures == 1
    assert coordinator.health()["status"] == "degraded"
    assert coordinator.health()["ready"] is False
    # Embedded delivery already attempted the row while publishing the event. A failure releases
    # the lease but schedules a bounded retry instead of allowing a tight retry loop.
    assert await coordinator.process_outbox_once() == 0
    assert repository.metrics()["outbox_backlog"] == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_outbox_claim_is_exclusive_until_lease_expiry() -> None:
    repository = InMemoryIntegrationRepository()
    envelope = CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW)
    item = await repository.enqueue(envelope)

    first_claim = await repository.pending(NOW, 10)
    concurrent_claim = await repository.pending(NOW, 10)
    recovered_claim = await repository.pending(NOW + timedelta(minutes=16), 10)

    assert tuple(value.outbox_id for value in first_claim) == (item.outbox_id,)
    assert concurrent_claim == ()
    assert tuple(value.outbox_id for value in recovered_claim) == (item.outbox_id,)

    await repository.complete(item.outbox_id, NOW + timedelta(minutes=16))
    assert await repository.pending(NOW + timedelta(minutes=32), 10) == ()


@pytest.mark.asyncio
async def test_disabled_ai_centric_shadow_path_cannot_change_legacy_production_result() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.unified_market_state = UnexpectedShadowCapture()
    coordinator.ai_centric_shadow_mode = False

    result = await coordinator.process(
        CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW)
    )

    assert result is None
    assert repository.metrics()["snapshots"] == 1
    assert await repository.signals() == ()


@pytest.mark.asyncio
async def test_shadow_capture_failure_is_isolated_from_legacy_production_result() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.unified_market_state = UnexpectedShadowCapture()
    coordinator.ai_centric_shadow_mode = True

    result = await coordinator.process(CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW))

    assert result is None
    assert repository.metrics()["snapshots"] == 1
    assert await repository.signals() == ()
    assert coordinator.failures == 0


@pytest.mark.asyncio
async def test_shadow_forecast_failure_is_isolated_from_scoring_and_publication() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.unified_market_state = CompleteShadowCapture()
    coordinator.quantitative_forecasting = FailingQuantForecast()
    coordinator.ai_centric_shadow_mode = True

    result = await coordinator.process(CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW))

    assert result is None
    assert repository.metrics()["snapshots"] == 1
    assert await repository.signals() == ()
    assert coordinator.failures == 0


@pytest.mark.asyncio
async def test_ai_reasoning_failure_is_isolated_from_scoring_and_publication() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.unified_market_state = CompleteShadowCapture()
    coordinator.quantitative_forecasting = SuccessfulQuantForecast()
    coordinator.ai_reasoning = FailingAIReasoning()
    coordinator.ai_centric_shadow_mode = True

    result = await coordinator.process(
        CanonicalEventEnvelope.final_candle(candle(Timeframe.M5), uuid4(), NOW)
    )

    assert result is None
    assert repository.metrics()["snapshots"] == 1
    assert await repository.signals() == ()
    assert coordinator.failures == 0


@pytest.mark.asyncio
async def test_degraded_volume_evidence_continues_through_unified_state_quant_and_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.volume_profile = DegradedVolumeAnalysis()
    stages: list[str] = []
    coordinator.unified_market_state = RecordingUnifiedState(stages)
    coordinator.quantitative_forecasting = RecordingQuantForecast(stages)
    coordinator.ai_reasoning = RecordingAIAndFinalDecision(stages)
    coordinator.ai_centric_shadow_mode = True

    target_logger = logging.getLogger("backend.app.integration.service")
    events: list[str] = []
    original_info = target_logger.info

    def capture_info(message: object, *args: object, **kwargs: object) -> None:
        events.append(str(message))
        original_info(message, *args, **kwargs)

    monkeypatch.setattr(target_logger, "info", capture_info)
    await coordinator.process(
        CanonicalEventEnvelope.final_candle(candle(Timeframe.M5), uuid4(), NOW)
    )

    assert stages == ["unified_market_state", "quant_forecast", "ai_reasoning", "final_decision"]
    assert repository.metrics()["snapshots"] == 1
    assert coordinator.failures == 0
    assert "ai_reasoning.gate.entered" in events
    assert "ai_reasoning.job.enqueued" in events


@pytest.mark.asyncio
async def test_market_regime_retry_is_idempotent_and_reaches_the_full_shadow_decision_path() -> None:
    class FailFirstDecision(FakeDecision):
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, request: object) -> object:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("fail after market-regime persistence")
            return await super().evaluate(request)

    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    coordinator = service(bus, repository)
    coordinator.market_data = FakeMarketData(candle(Timeframe.M5))
    regime_repository = InMemoryMarketRegimeRepository()
    coordinator.market_regime = MarketRegimeService(
        coordinator.market_data,
        None,
        None,
        None,
        None,
        bus,
        InMemoryFeatureStore(),
        repository=regime_repository,
    )
    stages: list[str] = []
    coordinator.unified_market_state = RecordingUnifiedState(stages)
    coordinator.quantitative_forecasting = RecordingQuantForecast(stages)
    coordinator.ai_reasoning = RecordingAIAndFinalDecision(stages)
    coordinator.ai_centric_shadow_mode = True
    decision = FailFirstDecision()
    coordinator.signal_decision = decision
    envelope = CanonicalEventEnvelope.final_candle(
        candle(Timeframe.M5),
        uuid4(),
        NOW,
    )

    with pytest.raises(RuntimeError, match="fail after market-regime persistence"):
        await coordinator.process(envelope)
    assert repository.metrics()["processed"] == 0

    result = await coordinator.process(envelope)

    assert result is None
    assert len(await regime_repository.list_snapshots("XAUUSD", Timeframe.M5)) == 1
    assert repository.metrics()["processed"] == 1
    assert decision.calls == 2
    assert stages == [
        "unified_market_state",
        "quant_forecast",
        "ai_reasoning",
        "final_decision",
    ] * 2


@pytest.mark.asyncio
async def test_ai_score_success_followed_by_decision_failure_is_terminal_and_traceable() -> None:
    bus, repository = InMemoryEventBus(), InMemoryIntegrationRepository()
    tracker = PipelineStageTracker()
    coordinator = service(bus, repository)
    coordinator.stage_tracker = tracker
    coordinator.signal_decision = FailingDecision()
    envelope = CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW)

    with pytest.raises(RuntimeError, match="decision persistence failed"):
        await coordinator.process(envelope)

    cycle = tracker.latest("XAUUSD", "M15")
    assert cycle is not None and cycle["complete"] is True
    stages = {item["key"]: item["status"] for item in cycle["stages"]}
    assert stages["ai_scoring"] == "success"
    assert stages["scenario_decision"] == "failed"
    assert (await repository.trace(envelope.trace_id))[0].status.value == "failed"


@pytest.mark.asyncio
async def test_snapshot_persistence_failure_finalizes_the_stage_tracker_cycle() -> None:
    """Same defect class as above, at the other unguarded call site: `repository.save_snapshot()`
    used to sit after the try/except block, so a persistence failure left ai_scoring/
    confidence_calculation/scenario_decision stuck "waiting" (rendered "running") forever even
    though every upstream analysis stage had genuinely completed."""
    bus = InMemoryEventBus()
    tracker = PipelineStageTracker()
    coordinator = service(bus, FailingSaveRepository())
    coordinator.stage_tracker = tracker
    envelope = CanonicalEventEnvelope.final_candle(candle(), uuid4(), NOW)
    with pytest.raises(RuntimeError):
        await coordinator.process(envelope)
    cycle = tracker.latest("XAUUSD", "M15")
    assert cycle is not None
    assert cycle["complete"] is True
    stages = {item["key"]: item["status"] for item in cycle["stages"]}
    assert stages["market_regime"] == "success"
    assert stages["ai_scoring"] == "failed"


@pytest.mark.asyncio
async def test_legacy_blocked_signal_is_not_exposed_as_operational_scenario() -> None:
    repository = InMemoryIntegrationRepository()
    eligible = OperationalSignal(
        operational_signal_id=uuid4(), semantic_hash="d" * 64, decision_id=uuid4(), ai_score_id=uuid4(), snapshot_id=uuid4(), trace_id=uuid4(), market_event_id="e" * 64,
        instrument="XAUUSD", timeframe="M15", mode=IntegrationMode.LIVE, direction="neutral", state="blocked", confidence=50, effective_at=NOW,
        expires_at=NOW + timedelta(minutes=15), data_quality_status="valid", provider_provenance=("legacy",), evidence=(), blockers=("legacy_blocked",), created_at=NOW,
        ai_scoring_policy_version="1.0.0", signal_decision_policy_version="1.0.0",
    )
    await repository.save_signal(eligible)
    assert await repository.latest_signal() is None
    assert await repository.signals() == ()
