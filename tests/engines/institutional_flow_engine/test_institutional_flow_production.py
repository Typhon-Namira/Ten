from datetime import UTC, datetime, timedelta
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.app.engines.institutional_flow_engine import (
    AnalysisStatus,
    BaselineInstitutionalFlowAnalyzer,
    CampaignPhase,
    CorrelationGroup,
    EvidenceRole,
    EvidenceSourceEngine,
    EvidenceType,
    FlowDirection,
    FlowPersistenceState,
    FlowState,
    InMemoryInstitutionalFlowRepository,
    InstitutionalFlowConfig,
    InstitutionalFlowContext,
    InstitutionalFlowEvidence,
    InstitutionalFlowService,
    InventoryBehaviorType,
    ProcessingMode,
    SessionType,
    stable_id,
)
from backend.app.engines.institutional_flow_engine.config import ThresholdConfig
from backend.app.engines.institutional_flow_engine.config import MultiTimeframeConfig
from backend.app.engines.institutional_flow_engine.engine import BaselineInstitutionalFlowEngine
from backend.app.engines.institutional_flow_engine.repository import SqlAlchemyInstitutionalFlowRepository
from backend.app.engines.institutional_flow_engine.registration import register
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.engines.market_data_engine import Candle, Timeframe
from tests.conftest import FakeSessionFactory


BASE = datetime(2026, 7, 1, tzinfo=UTC)


def candles(count: int = 20, *, bearish: bool = False, low_efficiency_last: bool = False) -> tuple[Candle, ...]:
    result = []
    price = 3300.0
    for index in range(count):
        direction = -1 if bearish else 1
        open_price = price
        body = 0.05 if low_efficiency_last and index == count - 1 else 0.8
        close = open_price + direction * body
        result.append(
            Candle(
                symbol="XAU/USD",
                timeframe=Timeframe.M15,
                timestamp=BASE + timedelta(minutes=15 * index),
                open=open_price,
                high=max(open_price, close) + 0.4,
                low=min(open_price, close) - 0.4,
                close=close,
                volume=100 + index * (40 if index == count - 1 else 1),
            )
        )
        price = close
    return tuple(result)


def evidence(
    kind: EvidenceType,
    direction: FlowDirection,
    *,
    source: EvidenceSourceEngine = EvidenceSourceEngine.SMC,
    group: CorrelationGroup = CorrelationGroup.STRUCTURE,
    session: SessionType = SessionType.LONDON,
    offset: int = 0,
    strength: float = 0.95,
    quality: float = 0.95,
    role: EvidenceRole = EvidenceRole.SUPPORTING,
    invalidated: bool = False,
) -> InstitutionalFlowEvidence:
    timestamp = BASE + timedelta(hours=2, minutes=offset)
    return InstitutionalFlowEvidence(
        id=stable_id(kind, direction, source, offset, role),
        source_engine=source,
        evidence_type=kind,
        source_object_id=f"{source.value}:{kind.value}:{offset}",
        source_timestamp=timestamp,
        availability_timestamp=timestamp,
        timeframe=Timeframe.M15,
        session=session,
        direction=direction,
        strength=strength,
        confidence=0.95,
        quality=quality,
        role=role,
        correlation_group=group,
        invalidated=invalidated,
        explanation="Typed upstream evidence used as a probabilistic inference input.",
        configuration_version="upstream-v1",
        engine_version="1.0.0",
    )


def permissive() -> InstitutionalFlowConfig:
    return InstitutionalFlowConfig(
        thresholds=ThresholdConfig(
            moderate_participation=0.05,
            high_participation=0.15,
            initiative=0.1,
            responsive=0.1,
            absorption=0.1,
            exhaustion=0.1,
            inventory=0.1,
            conflict=0.35,
            strong_pressure=0.62,
            moderate_pressure=0.2,
        )
    )


def rich(direction: FlowDirection) -> tuple[InstitutionalFlowEvidence, ...]:
    opposite = FlowDirection.BEARISH if direction == FlowDirection.BULLISH else FlowDirection.BULLISH
    return (
        evidence(EvidenceType.DISPLACEMENT, direction, source=EvidenceSourceEngine.SMC, offset=1),
        evidence(EvidenceType.STRUCTURAL_BREAK, direction, source=EvidenceSourceEngine.SMC, offset=2),
        evidence(EvidenceType.LIQUIDITY_EVENT, direction, source=EvidenceSourceEngine.LIQUIDITY, group=CorrelationGroup.LIQUIDITY, offset=3),
        evidence(EvidenceType.VALUE_ACCEPTANCE, direction, source=EvidenceSourceEngine.VOLUME_PROFILE, group=CorrelationGroup.PROFILE, offset=4),
        evidence(EvidenceType.PROFILE_MIGRATION, direction, source=EvidenceSourceEngine.VOLUME_PROFILE, group=CorrelationGroup.PROFILE, offset=5),
        evidence(EvidenceType.VALUE_REJECTION, direction, source=EvidenceSourceEngine.VOLUME_PROFILE, group=CorrelationGroup.PROFILE, session=SessionType.NEW_YORK, offset=6),
        evidence(EvidenceType.REPEATED_TEST, direction, source=EvidenceSourceEngine.LIQUIDITY, group=CorrelationGroup.LIQUIDITY, session=SessionType.NEW_YORK, offset=7),
        evidence(EvidenceType.LIMITED_PROGRESS, direction, source=EvidenceSourceEngine.MARKET_DATA, group=CorrelationGroup.VOLUME, offset=8),
        evidence(EvidenceType.EFFICIENCY_DECLINE, direction, source=EvidenceSourceEngine.MARKET_DATA, group=CorrelationGroup.PRICE_ACTION, offset=9),
        evidence(EvidenceType.STRUCTURAL_FAILURE, opposite, source=EvidenceSourceEngine.SMC, offset=10, strength=0.2, role=EvidenceRole.CONTRADICTING),
    )


def test_domain_validation_and_stable_identity() -> None:
    first = stable_id("x", 1)
    assert first == stable_id("x", 1)
    with pytest.raises(ValidationError, match="timezone-aware"):
        InstitutionalFlowEvidence.model_validate(
            evidence(EvidenceType.DISPLACEMENT, FlowDirection.BULLISH).model_dump()
            | {"source_timestamp": datetime(2026, 1, 1)}
        )
    with pytest.raises(ValidationError, match="availability"):
        InstitutionalFlowEvidence.model_validate(
            evidence(EvidenceType.DISPLACEMENT, FlowDirection.BULLISH).model_dump()
            | {"availability_timestamp": BASE, "source_timestamp": BASE + timedelta(hours=1)}
        )
    with pytest.raises(ValidationError, match="moderate participation"):
        ThresholdConfig(moderate_participation=0.8, high_participation=0.5)


def test_market_evidence_sparse_and_absorption_like_inputs() -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer(permissive())
    assert analyzer.market_evidence(candles(1)) == ()
    generated = analyzer.market_evidence(candles(low_efficiency_last=True), SessionType.LONDON)
    assert {item.evidence_type for item in generated} >= {
        EvidenceType.RANGE_EXPANSION,
        EvidenceType.VOLUME_EXPANSION,
        EvidenceType.DIRECTIONAL_PERSISTENCE,
        EvidenceType.LIMITED_PROGRESS,
        EvidenceType.EFFICIENCY_DECLINE,
    }
    assert all(item.session == SessionType.LONDON for item in generated)


def test_temporal_alignment_deduplication_discount_and_quality() -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer()
    valid = evidence(EvidenceType.DISPLACEMENT, FlowDirection.BULLISH, offset=1)
    future = evidence(EvidenceType.LIQUIDITY_EVENT, FlowDirection.BULLISH, offset=500)
    invalid = evidence(EvidenceType.VALUE_ACCEPTANCE, FlowDirection.BULLISH, offset=2, invalidated=True)
    low_quality = evidence(EvidenceType.VALUE_REJECTION, FlowDirection.BEARISH, offset=3, quality=0.01)
    correlated = tuple(evidence(EvidenceType.STRUCTURAL_BREAK, FlowDirection.BULLISH, offset=20 + index) for index in range(3))
    bundle = analyzer.normalize((valid, valid, future, invalid, low_quality, *correlated), BASE + timedelta(hours=5))
    assert bundle.rejected_future_ids == (future.id,)
    assert set(bundle.rejected_invalid_ids) == {invalid.id, low_quality.id}
    assert bundle.deduplicated_ids == (valid.id,)
    assert bundle.discounted_ids


def test_rich_bullish_snapshot_is_explainable_and_probabilistic() -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer(permissive())
    context = InstitutionalFlowContext(candles(), rich(FlowDirection.BULLISH), SessionType.LONDON, candles()[-1].timestamp, (("smc", "1.0.0"),))
    snapshot = analyzer.analyze_snapshot(context)
    assert snapshot.status == AnalysisStatus.COMPLETE
    assert snapshot.state.participation.level.value in {"moderate", "high"}
    assert snapshot.state.initiative is not None
    assert snapshot.state.responsive is not None
    assert snapshot.state.absorption is not None
    assert snapshot.state.exhaustion is not None
    assert snapshot.state.inventory.behavior in {InventoryBehaviorType.ACCUMULATION, InventoryBehaviorType.AMBIGUOUS}
    assert snapshot.state.cross_session[0].relationship in {"continuation", "reversal", "handoff"}
    assert snapshot.state.confluences
    assert "no participant identity" in snapshot.state.explanation.summary
    assert snapshot.model_dump(mode="json")["configuration_version"] == permissive().version


def test_bearish_snapshot_transition_replay_and_no_lookahead() -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer(permissive())
    first_context = InstitutionalFlowContext(candles(), rich(FlowDirection.BULLISH), analysis_boundary=candles()[-1].timestamp)
    first = analyzer.analyze_snapshot(first_context, ProcessingMode.INCREMENTAL)
    bearish_candles = candles(bearish=True)
    second = analyzer.analyze_snapshot(
        InstitutionalFlowContext(bearish_candles, rich(FlowDirection.BEARISH), analysis_boundary=bearish_candles[-1].timestamp),
        ProcessingMode.INCREMENTAL,
        first,
    )
    assert second.state.persistence.state in {FlowPersistenceState.REVERSING, FlowPersistenceState.PERSISTENT}
    if first.state.pressure.state != second.state.pressure.state:
        assert second.transitions
    complete = candles(20)
    prefix = complete[:12]
    at = prefix[-1].timestamp
    replay_a = analyzer.analyze_snapshot(InstitutionalFlowContext(prefix, analysis_boundary=at), ProcessingMode.REPLAY)
    replay_b = analyzer.analyze_snapshot(InstitutionalFlowContext(complete, analysis_boundary=at), ProcessingMode.REPLAY)
    assert replay_a.id == replay_b.id
    assert replay_a.state.pressure == replay_b.state.pressure


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((1.0, 0.0, 0.0), FlowState.STRONG_BULLISH),
        ((0.6, 0.2, 0.0), FlowState.MODERATE_BULLISH),
        ((0.0, 1.0, 0.0), FlowState.STRONG_BEARISH),
        ((0.2, 0.6, 0.0), FlowState.MODERATE_BEARISH),
        ((0.5, 0.5, 0.0), FlowState.BALANCED),
    ],
)
def test_pressure_states(values: tuple[float, float, float], expected: FlowState) -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer()
    bullish, bearish, neutral = values
    items = []
    if bullish:
        items.append(evidence(EvidenceType.CUSTOM, FlowDirection.BULLISH, strength=bullish))
    if bearish:
        items.append(evidence(EvidenceType.CUSTOM, FlowDirection.BEARISH, strength=bearish, offset=2))
    if neutral:
        items.append(evidence(EvidenceType.CUSTOM, FlowDirection.NEUTRAL, strength=neutral, offset=3))
    assert analyzer._pressure(tuple(items), 1).state == expected


def test_empty_evidence_and_invalid_boundaries() -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer()
    with pytest.raises(ValueError, match="at least one candle"):
        analyzer.analyze_snapshot(InstitutionalFlowContext(()))
    with pytest.raises(ValueError, match="precedes"):
        analyzer.analyze_snapshot(InstitutionalFlowContext(candles(2), analysis_boundary=BASE - timedelta(seconds=1)))
    sparse = analyzer.analyze_snapshot(InstitutionalFlowContext(candles(1)))
    assert sparse.status == AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert sparse.state.pressure.state == FlowState.INDETERMINATE


@pytest.mark.asyncio
async def test_in_memory_repository_idempotency_history_and_checkpoints() -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer()
    repository = InMemoryInstitutionalFlowRepository()
    first = analyzer.analyze_snapshot(InstitutionalFlowContext(candles(5)))
    second = analyzer.analyze_snapshot(InstitutionalFlowContext(candles(6)))
    await repository.save(first)
    await repository.save(first)
    await repository.save(first.model_copy(update={"id": uuid4()}))
    await repository.save(first.model_copy(update={"id": uuid4(), "engine_version": "divergent"}))
    await repository.save(second)
    assert await repository.latest("XAUUSD", Timeframe.M15) == second
    assert await repository.at("XAU/USD", Timeframe.M15, first.analysis_timestamp) == first
    assert await repository.at("EURUSD", Timeframe.M15, first.analysis_timestamp) is None
    assert await repository.checkpoints() == (second,)


def test_legacy_engine_sparse_flat_and_directional_paths() -> None:
    engine = BaselineInstitutionalFlowEngine()
    assert engine.analyze([]).bias.value == "balanced"
    flat = list(candles(2))
    flat[-1] = flat[-1].model_copy(update={"high": flat[-1].close, "low": flat[-1].close, "open": flat[-1].close})
    assert -1 <= engine.analyze(flat).absorption_probability <= 1
    assert engine.analyze(list(candles())).bias.value == "buying"
    assert engine.analyze(list(candles(bearish=True))).bias.value == "selling"


def test_campaign_inventory_and_persistence_branch_matrix() -> None:
    analyzer = BaselineInstitutionalFlowAnalyzer(permissive())
    base = analyzer.analyze_snapshot(InstitutionalFlowContext(candles(), rich(FlowDirection.BULLISH)))
    bearish = analyzer.analyze_snapshot(InstitutionalFlowContext(candles(bearish=True), rich(FlowDirection.BEARISH)))
    pressure_bull = base.state.pressure
    pressure_bear = bearish.state.pressure
    ambiguous_pressure = pressure_bull.model_copy(update={"conflict": 0.9})
    assert analyzer._inventory(rich(FlowDirection.BULLISH), ambiguous_pressure, base.state.absorption).behavior == InventoryBehaviorType.AMBIGUOUS

    inventory = base.state.inventory
    assert analyzer._campaign(inventory, base.state.initiative, None, pressure_bull, base).phase in {CampaignPhase.ACCUMULATION, CampaignPhase.REACCUMULATION}
    distribution = bearish.state.inventory
    markdown_previous = bearish.model_copy(update={"state": bearish.state.model_copy(update={"campaign": bearish.state.campaign.model_copy(update={"phase": CampaignPhase.MARKDOWN})})})
    assert analyzer._campaign(distribution, bearish.state.initiative, None, pressure_bear, markdown_previous).phase in {CampaignPhase.DISTRIBUTION, CampaignPhase.REDISTRIBUTION}
    balanced = inventory.model_copy(update={"behavior": InventoryBehaviorType.BALANCE})
    assert analyzer._campaign(balanced, base.state.initiative, None, pressure_bull, None).phase == CampaignPhase.MARKUP
    assert analyzer._campaign(balanced, bearish.state.initiative, None, pressure_bear, None).phase == CampaignPhase.MARKDOWN
    assert analyzer._campaign(balanced, None, base.state.exhaustion, pressure_bull, None).phase == CampaignPhase.TRANSITION
    ambiguous = inventory.model_copy(update={"behavior": InventoryBehaviorType.AMBIGUOUS})
    assert analyzer._campaign(ambiguous, None, None, pressure_bull, None).phase == CampaignPhase.AMBIGUOUS
    insufficient = inventory.model_copy(update={"behavior": InventoryBehaviorType.INSUFFICIENT})
    assert analyzer._campaign(insufficient, None, None, pressure_bull, None).phase == CampaignPhase.INSUFFICIENT
    assert analyzer._campaign(balanced, None, None, pressure_bull, None).phase == CampaignPhase.PREPARATION

    few = tuple(rich(FlowDirection.BULLISH)[:5])
    strengthening_previous = base.model_copy(update={"state": base.state.model_copy(update={"persistence": base.state.persistence.model_copy(update={"score": 0.1})})})
    weakening_previous = base.model_copy(update={"state": base.state.model_copy(update={"persistence": base.state.persistence.model_copy(update={"score": 0.9})})})
    assert analyzer._persistence(few, pressure_bull, strengthening_previous).state == FlowPersistenceState.STRENGTHENING
    assert analyzer._persistence(few, pressure_bull, weakening_previous).state == FlowPersistenceState.WEAKENING
    assert analyzer._persistence(few, pressure_bull, None).state == FlowPersistenceState.DEVELOPING


def test_snapshot_rejects_availability_after_boundary() -> None:
    snapshot = BaselineInstitutionalFlowAnalyzer().analyze_snapshot(InstitutionalFlowContext(candles(3)))
    with pytest.raises(ValidationError, match="availability cannot exceed"):
        type(snapshot).model_validate(snapshot.model_dump() | {"availability_timestamp": snapshot.analysis_timestamp + timedelta(seconds=1)})


class FakeSessions:
    value = "london"

    def session_at(self, _: datetime) -> SimpleNamespace:
        return SimpleNamespace(value=self.value)


class FakeMarket:
    def __init__(self, values: tuple[Candle, ...]) -> None:
        self.values = values
        self.sessions = FakeSessions()

    async def history(self, *_: object, **__: object) -> list[Candle]:
        return list(self.values)

    async def replay(self, _: str, __: Timeframe, at: datetime, **___: object) -> list[Candle]:
        return [item for item in self.values if item.timestamp <= at]


class EvidenceProvider:
    async def institutional_flow_evidence(self, _: str, __: Timeframe, ___: datetime) -> tuple[InstitutionalFlowEvidence, ...]:
        return rich(FlowDirection.BULLISH)[:1]


@pytest.mark.asyncio
async def test_service_analysis_replay_restore_health_mtf_and_publication() -> None:
    market = FakeMarket(candles())
    repository = InMemoryInstitutionalFlowRepository()
    bus = InMemoryEventBus()
    store = InMemoryFeatureStore()
    provider = EvidenceProvider()
    service = InstitutionalFlowService(market, provider, provider, provider, bus, store, permissive(), repository)
    assert await service.restore() == 0
    assert service.health()["status"] == "degraded"
    result = await service.analyze("XAUUSD", Timeframe.M15)
    assert await service.state("XAUUSD", Timeframe.M15) == result
    assert service.metrics.analyses_completed == 1
    assert service.metrics.snapshot()["analyses_completed"] == 1
    correlation = uuid4()
    await service._publish(result, correlation)
    assert (await store.snapshot(correlation)).features == {}
    replayed = await service.replay("XAUUSD", Timeframe.M15, candles(10)[-1].timestamp)
    assert replayed.processing_mode == ProcessingMode.REPLAY
    mtf = await service.multi_timeframe("XAUUSD", Timeframe.M15)
    assert mtf.maximum_depth == service.config.multi_timeframe.maximum_depth
    assert service.metrics.multi_timeframe_analyses == 1
    restored = InstitutionalFlowService(market, None, None, None, bus, store, permissive(), repository)
    assert await restored.restore() >= 1
    market.sessions.value = "closed"
    assert (await restored.analyze("XAUUSD", Timeframe.M15)).session == SessionType.UNKNOWN


@pytest.mark.asyncio
async def test_service_failure_metrics_and_empty_source() -> None:
    class BrokenRepository(InMemoryInstitutionalFlowRepository):
        async def save(self, snapshot: object) -> None:
            raise RuntimeError("database unavailable")

    empty_service = InstitutionalFlowService(FakeMarket(()), None, None, None, InMemoryEventBus(), InMemoryFeatureStore())
    with pytest.raises(ValueError, match="unavailable"):
        await empty_service.analyze("XAUUSD", Timeframe.M15)
    service = InstitutionalFlowService(FakeMarket(candles()), None, None, None, InMemoryEventBus(), InMemoryFeatureStore(), repository=BrokenRepository())
    with pytest.raises(RuntimeError, match="database"):
        await service.analyze("XAUUSD", Timeframe.M15)
    assert service.metrics.persistence_failures == service.metrics.analyses_failed == 1


@pytest.mark.asyncio
async def test_feature_and_event_publication_failures_are_isolated() -> None:
    class BrokenStore(InMemoryFeatureStore):
        async def write(self, feature: object) -> None:
            raise RuntimeError("feature failure")

    class BrokenBus(InMemoryEventBus):
        async def publish(self, event: object) -> None:
            raise RuntimeError("event failure")

    service = InstitutionalFlowService(FakeMarket(candles()), None, None, None, BrokenBus(), BrokenStore(), permissive())
    result = service.analyzer.analyze_snapshot(InstitutionalFlowContext(candles(), rich(FlowDirection.BULLISH)))
    await service._publish(result, uuid4())
    assert service.metrics.feature_publication_failures == 1
    assert service.metrics.event_publication_failures == 1
    assert InstitutionalFlowService.features(result)["trading_instruction"] is False


@pytest.mark.asyncio
async def test_sqlalchemy_repository_save_queries_and_checkpoint_integrity() -> None:
    snapshot = BaselineInstitutionalFlowAnalyzer().analyze_snapshot(InstitutionalFlowContext(candles(5)))

    class ScalarResult:
        def __init__(self, values: list[object]) -> None:
            self.values = values

        def first(self) -> object | None:
            return self.values[0] if self.values else None

        def all(self) -> list[object]:
            return self.values

    class FakeSession:
        def __init__(self) -> None:
            self.execute = AsyncMock()
            self.commit = AsyncMock()
            self.scalar_values: list[object] = []

        async def scalars(self, _: object) -> ScalarResult:
            return ScalarResult(self.scalar_values)

    session = FakeSession()
    repository = SqlAlchemyInstitutionalFlowRepository(FakeSessionFactory(session))  # type: ignore[arg-type]
    await repository.save(snapshot)
    assert session.execute.await_count == 3
    session.scalar_values = [SimpleNamespace(payload=snapshot.model_dump(mode="json"))]
    assert await repository.latest("XAUUSD", Timeframe.M15) == snapshot
    assert await repository.at("XAUUSD", Timeframe.M15, snapshot.analysis_timestamp) == snapshot
    session.scalar_values = []
    assert await repository.latest("XAUUSD", Timeframe.M15) is None
    from hashlib import sha256

    valid = SimpleNamespace(state_payload=snapshot.model_dump(mode="json"), state_hash=sha256(snapshot.model_dump_json().encode()).hexdigest(), engine_version=snapshot.engine_version)
    bad_hash = SimpleNamespace(state_payload=snapshot.model_dump(mode="json"), state_hash="x" * 64, engine_version=snapshot.engine_version)
    corrupt = SimpleNamespace(state_payload={"bad": True}, state_hash="x" * 64, engine_version=snapshot.engine_version)
    session.scalar_values = [valid, bad_hash, corrupt]
    assert await repository.checkpoints() == (snapshot,)


def test_registration_declares_all_required_dependencies() -> None:
    class Factory:
        def __init__(self) -> None:
            self.args: tuple[object, ...] = ()

        def register(self, *args: object) -> None:
            self.args = args

    factory = Factory()
    register(factory)  # type: ignore[arg-type]
    metadata = factory.args[0]
    assert metadata.dependencies == ("market_data", "smc", "liquidity", "volume_profile")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_registration_builder_executor_and_mtf_rejection_paths() -> None:
    class Factory:
        def __init__(self) -> None:
            self.args: tuple[object, ...] = ()

        def register(self, *args: object) -> None:
            self.args = args

    factory = Factory()
    register(factory)  # type: ignore[arg-type]
    builder, executor = factory.args[1], factory.args[2]
    engine = builder(None, {})  # type: ignore[operator]
    output = await executor(engine, SimpleNamespace(candles=list(candles())))  # type: ignore[operator]
    assert output.namespace == "institutional_flow"

    class FailingMarket(FakeMarket):
        async def history(self, *_: object, **__: object) -> list[Candle]:
            raise RuntimeError("missing timeframe")

    config = InstitutionalFlowConfig(multi_timeframe=MultiTimeframeConfig(hierarchy=("BAD", "M5"), maximum_depth=2))
    service = InstitutionalFlowService(FailingMarket(candles()), None, None, None, InMemoryEventBus(), InMemoryFeatureStore(), config)
    result = await service.multi_timeframe("XAUUSD", Timeframe.M15)
    assert result.direction_by_timeframe == {}


@pytest.mark.asyncio
async def test_existing_public_upstream_snapshots_are_adapted_without_reverse_imports() -> None:
    boundary = candles()[-1].timestamp

    class StateProvider:
        def __init__(self, value: object | None) -> None:
            self.value = value

        async def state(self, *_: object) -> object | None:
            return self.value

    def direction(value: str) -> SimpleNamespace:
        return SimpleNamespace(value=value)
    smc_events = (
        SimpleNamespace(id=uuid4(), timestamp=boundary - timedelta(minutes=2), timeframe=Timeframe.M15, direction=direction("bullish"), displacement_score=90, confidence_score=80, quality_score=70),
        SimpleNamespace(id=uuid4(), timestamp=boundary - timedelta(minutes=1), timeframe=Timeframe.M15, direction=direction("bearish"), displacement_score=10, confidence_score=60, quality_score=80),
        SimpleNamespace(id=uuid4(), timestamp=boundary + timedelta(minutes=1), timeframe=Timeframe.M15, direction=direction("neutral"), displacement_score=10, confidence_score=60, quality_score=80),
    )
    liquidity_events = (
        SimpleNamespace(id=uuid4(), available_at=boundary - timedelta(minutes=2), timeframe=Timeframe.M15, side=direction("buy_side"), confidence_score=80, quality_score=70),
        SimpleNamespace(id=uuid4(), available_at=boundary - timedelta(minutes=1), timeframe=Timeframe.M15, side=direction("sell_side"), confidence_score=80, quality_score=70),
        SimpleNamespace(id=uuid4(), available_at=boundary + timedelta(minutes=1), timeframe=Timeframe.M15, side=direction("unknown"), confidence_score=80, quality_score=70),
    )
    migrations = (
        SimpleNamespace(id=uuid4(), available_at=boundary - timedelta(minutes=3), migration_type=direction("upward"), normalized_change=2, confidence_score=90, quality_score=80),
        SimpleNamespace(id=uuid4(), available_at=boundary - timedelta(minutes=2), migration_type=direction("downward"), normalized_change=-2, confidence_score=90, quality_score=80),
        SimpleNamespace(id=uuid4(), available_at=boundary - timedelta(minutes=1), migration_type=direction("stable"), normalized_change=0, confidence_score=90, quality_score=80),
        SimpleNamespace(id=uuid4(), available_at=boundary + timedelta(minutes=1), migration_type=direction("upward"), normalized_change=2, confidence_score=90, quality_score=80),
    )
    smc = StateProvider(SimpleNamespace(structure_events=smc_events))
    liquidity = StateProvider(SimpleNamespace(events=liquidity_events))
    volume = StateProvider(SimpleNamespace(migrations=migrations, timeframe=Timeframe.M15))
    service = InstitutionalFlowService(FakeMarket(candles()), smc, liquidity, volume, InMemoryEventBus(), InMemoryFeatureStore())
    adapted = await service._upstream_evidence("XAUUSD", Timeframe.M15, boundary)
    assert len(adapted) == 7
    assert {item.source_engine for item in adapted} == {EvidenceSourceEngine.SMC, EvidenceSourceEngine.LIQUIDITY, EvidenceSourceEngine.VOLUME_PROFILE}
    assert {item.direction for item in adapted} >= {FlowDirection.BULLISH, FlowDirection.BEARISH, FlowDirection.NEUTRAL}
    assert all(item.availability_timestamp <= boundary for item in adapted)
    assert service._smc_evidence(None, boundary) == ()
    assert service._liquidity_evidence(None, boundary) == ()
    assert service._volume_profile_evidence(None, boundary) == ()
    clamped = service._evidence(source=EvidenceSourceEngine.SMC, kind=EvidenceType.CUSTOM, source_id="x", timestamp=boundary, timeframe=Timeframe.M15, direction=FlowDirection.INDETERMINATE, strength=2, confidence=-1, quality=2, group=CorrelationGroup.INDEPENDENT, explanation="bounded")
    assert (clamped.strength, clamped.confidence, clamped.quality) == (1, 0, 1)
