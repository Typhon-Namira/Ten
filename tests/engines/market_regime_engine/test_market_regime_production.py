from datetime import UTC, datetime, timedelta
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_regime_engine import (
    AuctionRegime,
    BaselineMarketRegimeAnalyzer,
    BaselineMarketRegimeEngine,
    DominantRegime,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceRole,
    ExpansionRegime,
    InMemoryMarketRegimeRepository,
    InventoryRegime,
    MarketRegimeConfig,
    MarketRegimeCheckpoint,
    MarketRegimeContext,
    MarketRegimeEvidence,
    MarketRegimeService,
    ParticipationRegime,
    ProcessingMode,
    RegimeLifecycle,
    RegimePersistence,
    RegimeTransition,
    SqlAlchemyMarketRegimeRepository,
    StructuralRegime,
    TransitionState,
    TrendMaturity,
    TrendRegime,
    VolatilityRegime,
    stable_id,
)
from backend.app.engines.market_regime_engine.config import DependencyConfig, MultiTimeframeConfig, ThresholdConfig
from backend.app.engines.market_regime_engine.events import MarketRegimeDependencyRecovered, RegimeTransitionInvalidated
from backend.app.engines.market_regime_engine.registration import register
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from tests.conftest import FakeSessionFactory

BASE = datetime(2026, 7, 1, tzinfo=UTC)


def candles(count: int = 40, *, direction: int = 1, widening: bool = False, alternating: bool = False, timeframe: Timeframe = Timeframe.M15) -> tuple[Candle, ...]:
    result = []
    price = 3300.0
    for index in range(count):
        sign = (-1 if index % 2 else 1) if alternating else direction
        body = (0.2 + index * 0.08) if widening else 0.7
        open_price = price
        close = open_price + sign * body
        wick = (0.2 + index * 0.04) if widening else 0.3
        timestamp = BASE + timeframe.duration * index
        result.append(Candle(symbol="XAU/USD", timeframe=timeframe, timestamp=timestamp, ingestion_timestamp=timestamp, open=open_price, high=max(open_price, close) + wick, low=min(open_price, close) - wick, close=close, volume=100 + index, quality_score=95))
        price = close
    return tuple(result)


def evidence(
    family: EvidenceFamily,
    direction: EvidenceDirection,
    *,
    source: str | None = None,
    offset: int = 10,
    strength: float = 0.9,
    group: str | None = None,
    future: bool = False,
    contradicting: bool = False,
    quality: float = 0.9,
    subfamily: str = "observation",
    metadata: dict[str, object] | None = None,
) -> MarketRegimeEvidence:
    event_at = BASE + timedelta(minutes=offset)
    available = event_at + (timedelta(days=5) if future else timedelta())
    boundary = BASE + timedelta(days=1)
    return MarketRegimeEvidence(
        evidence_id=stable_id(family, direction, offset, source, future, contradicting, subfamily),
        source_engine=source or family.value,
        source_engine_version="1.0.0",
        source_object_type=subfamily,
        source_object_id=f"source:{offset}",
        source_snapshot_id="snapshot-upstream",
        symbol="XAUUSD",
        timeframe=Timeframe.M15,
        session="london",
        event_timestamp=event_at,
        available_at=available,
        analysis_boundary=boundary,
        direction=direction,
        role=EvidenceRole.CONTRADICTING if contradicting else EvidenceRole.SUPPORTING,
        family=family,
        subfamily=subfamily,
        raw_strength=strength,
        normalized_strength=min(1, strength),
        source_confidence=0.9,
        source_quality=quality,
        effective_weight=0,
        correlation_group=group or f"{family}:{offset}",
        correlation_discount=1,
        decay_factor=1,
        accepted=not future,
        rejected=future,
        contradicting=contradicting,
        unavailable=future,
        rejection_reason="future" if future else None,
        payload_summary="Time-valid public upstream observation.",
        metadata=metadata or {},
    )


def rich(direction: EvidenceDirection = EvidenceDirection.BULLISH) -> tuple[MarketRegimeEvidence, ...]:
    opposite = EvidenceDirection.BEARISH if direction == EvidenceDirection.BULLISH else EvidenceDirection.BULLISH
    return (
        evidence(EvidenceFamily.STRUCTURE, direction, source="smc", offset=1, subfamily="structural_break"),
        evidence(EvidenceFamily.LIQUIDITY, direction, source="liquidity", offset=2, subfamily="liquidity_event"),
        evidence(EvidenceFamily.VOLUME_PROFILE, direction, source="volume_profile", offset=3, subfamily="profile_migration"),
        evidence(EvidenceFamily.INSTITUTIONAL_FLOW, direction, source="institutional_flow", offset=4, subfamily="directional_pressure", metadata={"campaign": "markup_like" if direction == EvidenceDirection.BULLISH else "markdown_like"}),
        evidence(EvidenceFamily.SESSION, direction, source="market_data", offset=5, subfamily="session_continuation"),
        evidence(EvidenceFamily.STRUCTURE, opposite, source="smc", offset=6, strength=0.2, contradicting=True, subfamily="structural_failure"),
    )


def test_models_config_and_engine_contracts() -> None:
    assert stable_id("x") == stable_id("x")
    assert len({item.value for item in DominantRegime}) == len(DominantRegime)
    with pytest.raises(ValidationError, match="timezone-aware"):
        evidence(EvidenceFamily.STRUCTURE, EvidenceDirection.BULLISH).model_copy(update={"event_timestamp": datetime(2026, 1, 1)}).model_dump()
        MarketRegimeEvidence.model_validate(evidence(EvidenceFamily.STRUCTURE, EvidenceDirection.BULLISH).model_dump() | {"event_timestamp": datetime(2026, 1, 1)})
    with pytest.raises(ValidationError, match="availability"):
        MarketRegimeEvidence.model_validate(evidence(EvidenceFamily.STRUCTURE, EvidenceDirection.BULLISH).model_dump() | {"available_at": BASE, "event_timestamp": BASE + timedelta(hours=1)})
    future = evidence(EvidenceFamily.STRUCTURE, EvidenceDirection.BULLISH, future=True)
    with pytest.raises(ValidationError, match="future evidence"):
        MarketRegimeEvidence.model_validate(future.model_dump() | {"accepted": True})
    with pytest.raises(ValidationError, match="required dependency"):
        DependencyConfig(required=())
    with pytest.raises(ValidationError, match="trend threshold"):
        ThresholdConfig(trend=0.7, strong_trend=0.4)
    with pytest.raises(ValidationError, match="transition watch"):
        ThresholdConfig(transition_watch=0.8, transition_confirm=0.5)
    with pytest.raises(ValidationError, match="low volatility"):
        ThresholdConfig(low_volatility_percentile=0.9, high_volatility_percentile=0.8)
    with pytest.raises(ValidationError, match="unsupported timeframe"):
        MultiTimeframeConfig(hierarchy=("BAD",))
    with pytest.raises(ValidationError, match="repository mode"):
        MarketRegimeConfig(repository_mode="bad")
    with pytest.raises(ValidationError, match="incompatible"):
        MarketRegimeConfig(version="bad")
    engine = BaselineMarketRegimeEngine()
    with pytest.raises(ValueError, match="requires candles"):
        engine.analyze([])
    assert engine.analyze(list(candles())).probabilistic_inference is True
    checkpoint = MarketRegimeCheckpoint(checkpoint_id=uuid4(), engine_name="market_regime", engine_version="1.0.0", schema_version="1.0", configuration_version="market-regime-1.0", algorithm_version="1.0.0", symbol="XAUUSD", timeframe=Timeframe.M15, analysis_boundary=BASE, state_payload={}, payload_hash="x" * 64, created_at=BASE)
    assert checkpoint.analysis_boundary == BASE


def test_no_lookahead_determinism_correlation_and_insufficient_data() -> None:
    analyzer = BaselineMarketRegimeAnalyzer()
    complete = candles()
    boundary = complete[24].timestamp
    future = evidence(EvidenceFamily.VOLUME_PROFILE, EvidenceDirection.BULLISH, future=True)
    correlated = tuple(evidence(EvidenceFamily.STRUCTURE, EvidenceDirection.BULLISH, offset=100 + index, group="same") for index in range(4))
    first = analyzer.analyze_snapshot(MarketRegimeContext(complete[:25], (*correlated, future), boundary))
    second = analyzer.analyze_snapshot(MarketRegimeContext(complete, (*correlated, future), boundary), ProcessingMode.REPLAY)
    assert first.snapshot_id == second.snapshot_id
    assert [item.evidence_id for item in first.evidence] == [item.evidence_id for item in second.evidence]
    assert any(item.unavailable and not item.accepted for item in first.evidence)
    assert any(item.discounted for item in first.evidence)
    sparse = analyzer.analyze_snapshot(MarketRegimeContext(candles(4)))
    assert sparse.dominant_regime == DominantRegime.INSUFFICIENT_DATA
    assert sparse.trend_regime == TrendRegime.INSUFFICIENT_DATA
    assert sparse.volatility_regime == VolatilityRegime.INSUFFICIENT_DATA
    assert sparse.auction_regime == AuctionRegime.INSUFFICIENT_DATA
    assert sparse.expansion_regime == ExpansionRegime.INSUFFICIENT_DATA
    assert sparse.structural_regime == StructuralRegime.INSUFFICIENT_DATA
    assert sparse.participation_regime == ParticipationRegime.INSUFFICIENT_DATA
    assert sparse.inventory_regime == InventoryRegime.INSUFFICIENT_DATA
    assert sparse.persistence == RegimePersistence.INSUFFICIENT_DATA
    assert sparse.trend_maturity == TrendMaturity.INSUFFICIENT_DATA
    with pytest.raises(ValueError, match="analysis boundary"):
        analyzer.analyze_snapshot(MarketRegimeContext(()))


def test_multidimensional_bull_bear_balance_and_degradation() -> None:
    analyzer = BaselineMarketRegimeAnalyzer()
    bull = analyzer.analyze_snapshot(MarketRegimeContext(candles(widening=True), rich()))
    bear = analyzer.analyze_snapshot(MarketRegimeContext(candles(direction=-1, widening=True), rich(EvidenceDirection.BEARISH)))
    balanced = analyzer.analyze_snapshot(MarketRegimeContext(candles(alternating=True), rich(EvidenceDirection.NEUTRAL)))
    degraded = analyzer.analyze_snapshot(MarketRegimeContext(candles(), rich(), missing_dependencies=("volume_profile",), failed_dependencies=("liquidity",)))
    assert bull.directional_bias == EvidenceDirection.BULLISH
    assert bear.directional_bias == EvidenceDirection.BEARISH
    assert balanced.balance_score > 0.5
    assert bull.primary_interpretation and bull.alternative_interpretation
    assert degraded.degradation.is_degraded and degraded.confidence < bull.confidence
    assert all(0 <= value <= 1 for value in (bull.confidence, bull.ambiguity, bull.quality, bull.compression_score, bull.expansion_score))
    with pytest.raises(ValidationError, match="probabilistic"):
        type(bull).model_validate(bull.model_dump() | {"probabilistic_inference": False})
    with pytest.raises(ValidationError, match="historical boundary"):
        type(bull).model_validate(bull.model_dump() | {"historical_boundary": bull.historical_boundary + timedelta(minutes=1)})


@pytest.mark.parametrize(
    ("percentile", "change", "expected"),
    [(0.99, 0.1, VolatilityRegime.VERY_HIGH), (0.85, 0.1, VolatilityRegime.HIGH), (0.01, 0.1, VolatilityRegime.VERY_LOW), (0.1, 0.1, VolatilityRegime.LOW), (0.5, 0.1, VolatilityRegime.NORMAL), (0.5, 0.8, VolatilityRegime.UNSTABLE)],
)
def test_volatility_taxonomy(percentile: float, change: float, expected: VolatilityRegime) -> None:
    assert BaselineMarketRegimeAnalyzer()._volatility(True, percentile, change) == expected


def test_inference_branch_taxonomy() -> None:
    a = BaselineMarketRegimeAnalyzer()
    families = {EvidenceFamily.STRUCTURE, EvidenceFamily.INSTITUTIONAL_FLOW}
    assert a._trend(True, 0.6, 0.7, 0.1, families) == TrendRegime.BULL_TREND
    assert a._trend(True, -0.6, 0.7, 0.1, families) == TrendRegime.BEAR_TREND
    assert a._trend(True, 0, 0, 0.8, families) == TrendRegime.RANGE
    assert a._trend(True, 0.6, 0.7, 0.1, {EvidenceFamily.STRUCTURE}) == TrendRegime.UNCERTAIN
    assert a._trend(True, 0, 0.1, 0.1, families) == TrendRegime.NEUTRAL
    assert a._auction(True, 0.7, 0.1, 0, True) == AuctionRegime.BALANCED_AUCTION
    assert a._auction(True, 0.1, 0.8, 0.8, True) == AuctionRegime.BULLISH_IMBALANCE
    assert a._auction(True, 0.1, 0.8, -0.8, True) == AuctionRegime.BEARISH_IMBALANCE
    assert a._auction(True, 0.1, 0.1, 0, True) == AuctionRegime.MIXED_AUCTION
    assert a._auction(True, 0.1, 0.8, 1, False) == AuctionRegime.UNCERTAIN
    assert a._structure(True, TrendRegime.BULL_TREND, 0) == StructuralRegime.BULLISH_CONTINUATION
    assert a._structure(True, TrendRegime.BEAR_TREND, 0) == StructuralRegime.BEARISH_CONTINUATION
    assert a._structure(True, TrendRegime.RANGE, 0) == StructuralRegime.RANGE_STRUCTURE
    assert a._structure(True, TrendRegime.NEUTRAL, 0) == StructuralRegime.STRUCTURAL_TRANSITION
    assert a._structure(True, TrendRegime.NEUTRAL, 0.8) == StructuralRegime.MIXED_STRUCTURE
    flow = (evidence(EvidenceFamily.INSTITUTIONAL_FLOW, EvidenceDirection.BULLISH),)
    assert a._participation(True, 0.6, flow) in {ParticipationRegime.MODERATE_BULLISH_PARTICIPATION, ParticipationRegime.STRONG_BULLISH_PARTICIPATION}
    assert a._participation(True, -0.6, flow) in {ParticipationRegime.MODERATE_BEARISH_PARTICIPATION, ParticipationRegime.STRONG_BEARISH_PARTICIPATION}
    assert a._participation(True, 0, flow) == ParticipationRegime.NEUTRAL_PARTICIPATION
    assert a._participation(True, 0, ()) == ParticipationRegime.UNCERTAIN
    assert a._participation(True, 0, (flow[0].model_copy(update={"contradicting": True}),)) == ParticipationRegime.CONFLICTED_PARTICIPATION
    assert a._market_evidence(candles(1), BASE) == ()
    assert all(value == 0 for value in a._market_metrics(candles(1)).values())
    assert a._expansion(True, 0.8, 0, None) == ExpansionRegime.COMPRESSION
    prior = a.analyze_snapshot(MarketRegimeContext(candles(), rich()))
    assert a._expansion(True, 0, 0.8, prior.model_copy(update={"expansion_regime": ExpansionRegime.EXPANSION})) == ExpansionRegime.LATE_EXPANSION
    assert a._expansion(True, 0, 0.8, prior.model_copy(update={"expansion_regime": ExpansionRegime.EARLY_EXPANSION})) == ExpansionRegime.EXPANSION
    assert a._expansion(True, 0, 0, prior.model_copy(update={"expansion_score": 0.9})) == ExpansionRegime.DECELERATION


def test_persistence_maturity_transition_lifecycle_and_dominant_branches() -> None:
    a = BaselineMarketRegimeAnalyzer()
    first = a.analyze_snapshot(MarketRegimeContext(candles(), rich()))
    assert a._persistence(True, None, 0.5, 0) == RegimePersistence.TRANSIENT
    assert a._persistence(True, first, -0.5, 0) == RegimePersistence.REVERSING
    assert a._persistence(True, first, first.net_directional_score, 0.8) == RegimePersistence.UNSTABLE
    assert a._persistence(True, first, min(1, abs(first.net_directional_score) + 0.3), 0) in {RegimePersistence.STRENGTHENING, RegimePersistence.STABLE}
    assert a._persistence(True, first, 0, 0) in {RegimePersistence.WEAKENING, RegimePersistence.STABLE}
    assert a._maturity(True, TrendRegime.RANGE, RegimePersistence.STABLE, first, ()) == TrendMaturity.NOT_APPLICABLE
    assert a._maturity(True, TrendRegime.BULL_TREND, RegimePersistence.STABLE, None, ()) == TrendMaturity.EARLY
    assert a._maturity(True, TrendRegime.BULL_TREND, RegimePersistence.WEAKENING, first, ()) == TrendMaturity.WEAKENING
    assert a._maturity(True, TrendRegime.BULL_TREND, RegimePersistence.STABLE, first, (evidence(EvidenceFamily.INSTITUTIONAL_FLOW, EvidenceDirection.BULLISH, subfamily="exhaustion"),)) == TrendMaturity.EXHAUSTION_RISK
    assert a._transition_state(None, DominantRegime.BALANCED, 1) == TransitionState.NONE
    changed = first.model_copy(update={"dominant_regime": DominantRegime.BALANCED, "transition_state": TransitionState.WATCH})
    assert a._transition_state(changed, DominantRegime.TRENDING_BULL, 0.9) == TransitionState.CONFIRMED
    assert a._transition_state(first, DominantRegime.BALANCED, 0.9) == TransitionState.DEVELOPING
    assert a._transition_state(first, DominantRegime.BALANCED, 0.5) == TransitionState.WATCH
    assert a._transition_state(first, DominantRegime.BALANCED, 0.1) == TransitionState.FAILED
    assert a._lifecycle(True, None, RegimePersistence.TRANSIENT, TransitionState.NONE) == RegimeLifecycle.INITIAL
    assert a._lifecycle(True, first, RegimePersistence.WEAKENING, TransitionState.NONE) == RegimeLifecycle.WEAKENING
    assert a._lifecycle(True, first, RegimePersistence.STABLE, TransitionState.NONE) == RegimeLifecycle.MATURE
    assert a._lifecycle(True, first, RegimePersistence.DEVELOPING, TransitionState.NONE) == RegimeLifecycle.DEVELOPING
    assert a._lifecycle(True, first, RegimePersistence.STABLE, TransitionState.WATCH) == RegimeLifecycle.TRANSITIONING
    cases = (
        (TrendRegime.NEUTRAL, VolatilityRegime.NORMAL, AuctionRegime.MIXED_AUCTION, ExpansionRegime.COMPRESSION, InventoryRegime.AMBIGUOUS, 0, DominantRegime.COMPRESSION),
        (TrendRegime.NEUTRAL, VolatilityRegime.NORMAL, AuctionRegime.MIXED_AUCTION, ExpansionRegime.EXPANSION, InventoryRegime.AMBIGUOUS, 1, DominantRegime.EXPANSION_BULL),
        (TrendRegime.BULL_TREND, VolatilityRegime.NORMAL, AuctionRegime.MIXED_AUCTION, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, 1, DominantRegime.TRENDING_BULL),
        (TrendRegime.BEAR_TREND, VolatilityRegime.NORMAL, AuctionRegime.MIXED_AUCTION, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, -1, DominantRegime.TRENDING_BEAR),
        (TrendRegime.NEUTRAL, VolatilityRegime.NORMAL, AuctionRegime.BALANCED_AUCTION, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, 0, DominantRegime.BALANCED),
        (TrendRegime.NEUTRAL, VolatilityRegime.HIGH, AuctionRegime.MIXED_AUCTION, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, 0, DominantRegime.HIGH_VOLATILITY),
        (TrendRegime.NEUTRAL, VolatilityRegime.LOW, AuctionRegime.MIXED_AUCTION, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, 0, DominantRegime.LOW_VOLATILITY),
    )
    for trend, vol, auction, expansion, inventory, net, expected in cases:
        assert a._dominant(True, trend, vol, auction, expansion, inventory, net) == expected
    assert a._dominant(True, TrendRegime.NEUTRAL, VolatilityRegime.NORMAL, AuctionRegime.BULLISH_IMBALANCE, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, 1) == DominantRegime.IMBALANCED_BULL
    assert a._dominant(True, TrendRegime.NEUTRAL, VolatilityRegime.NORMAL, AuctionRegime.BEARISH_IMBALANCE, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, -1) == DominantRegime.IMBALANCED_BEAR
    assert a._dominant(True, TrendRegime.NEUTRAL, VolatilityRegime.NORMAL, AuctionRegime.MIXED_AUCTION, ExpansionRegime.NEUTRAL, InventoryRegime.ACCUMULATION_LIKE, 0) == DominantRegime.ACCUMULATION_LIKE
    assert a._dominant(True, TrendRegime.RANGE, VolatilityRegime.NORMAL, AuctionRegime.MIXED_AUCTION, ExpansionRegime.NEUTRAL, InventoryRegime.AMBIGUOUS, 0) == DominantRegime.RANGING
    assert "structural transition" in a._alternative(DominantRegime.UNCERTAIN, EvidenceDirection.NEUTRAL, 0.1, 0.8)
    mature = first.model_copy(update={"trend_maturity": TrendMaturity.ESTABLISHED})
    assert a._maturity(True, TrendRegime.BULL_TREND, RegimePersistence.STABLE, mature, ()) == TrendMaturity.MATURE
    strengthened_previous = first.model_copy(update={"net_directional_score": 0.05})
    assert a._persistence(True, strengthened_previous, 0.8, 0) == RegimePersistence.STRENGTHENING


@pytest.mark.asyncio
async def test_in_memory_repository_complete_contract_and_checkpoint_integrity() -> None:
    repo = InMemoryMarketRegimeRepository()
    snapshot = BaselineMarketRegimeAnalyzer().analyze_snapshot(MarketRegimeContext(candles(), rich()))
    await repo.save_snapshot(snapshot)
    await repo.save_snapshot(snapshot)
    await repo.save_evidence(snapshot)
    assert await repo.get_latest_snapshot("XAU/USD", Timeframe.M15) == snapshot
    assert await repo.get_snapshot(snapshot.snapshot_id) == snapshot
    assert await repo.list_snapshots("XAUUSD", Timeframe.M15) == (snapshot,)
    assert await repo.list_evidence("XAUUSD", Timeframe.M15)
    transition = RegimeTransition(transition_id=uuid4(), symbol="XAUUSD", timeframe=Timeframe.M15, from_regime=DominantRegime.BALANCED, to_regime=DominantRegime.TRENDING_BULL, started_at=BASE, state=TransitionState.WATCH, confidence=0.5, ambiguity=0.5, reasoning_summary="Probabilistic transition watch.")
    await repo.save_transition(transition)
    await repo.save_transition(transition)
    assert await repo.get_transition(transition.transition_id) == transition
    assert await repo.list_transitions("XAUUSD", Timeframe.M15) == (transition,)
    await repo.save_checkpoint(snapshot)
    assert await repo.load_checkpoint("XAUUSD", Timeframe.M15) == snapshot
    assert await repo.checkpoints() == (snapshot,)
    assert await repo.prune_history("XAUUSD", Timeframe.M15, 0) == 1
    assert await repo.get_latest_snapshot("XAUUSD", Timeframe.M15) is None
    repo._checkpoints[("XAUUSD", Timeframe.M15)] = (snapshot, "x" * 64)
    with pytest.raises(ValueError, match="integrity"):
        await repo.load_checkpoint("XAUUSD", Timeframe.M15)
    assert await repo.load_checkpoint("EURUSD", Timeframe.M15) is None


class FakeMarket:
    def __init__(self, values: tuple[Candle, ...]) -> None:
        self.values = values
        self.sessions = SimpleNamespace(session_at=lambda _: SimpleNamespace(value="london"))

    async def history(self, *_: object, **__: object) -> list[Candle]:
        return list(self.values)

    async def replay(self, _: str, __: Timeframe, at: datetime, limit: int = 500) -> list[Candle]:
        return list(item for item in self.values if item.timestamp <= at)[-limit:]


@pytest.mark.asyncio
async def test_service_lifecycle_replay_incremental_features_events_failure_isolation() -> None:
    bus = InMemoryEventBus()
    store = InMemoryFeatureStore()
    service = MarketRegimeService(FakeMarket(candles()), None, None, None, None, bus, store)
    assert await service.restore() == 0
    snapshot = await service.analyze_snapshot("XAUUSD", Timeframe.M15)
    same = await service.update_incremental("XAUUSD", Timeframe.M15)
    replayed = await service.replay("XAUUSD", Timeframe.M15, candles()[-1].timestamp)
    assert snapshot.snapshot_id == same.snapshot_id == replayed.snapshot_id
    assert service.features(snapshot)["trading_instruction"] is False
    assert service.health()["status"] == "degraded"
    assert await service.state("XAUUSD", Timeframe.M15) == snapshot
    assert await service.history("XAUUSD", Timeframe.M15)
    assert (await service.recover("XAUUSD", Timeframe.M15)).snapshot_id == snapshot.snapshot_id  # type: ignore[union-attr]
    assert bus.history()
    assert service.metrics.snapshot()["analysis_count"] >= 3

    class BrokenStore(InMemoryFeatureStore):
        async def write(self, feature: object) -> None:
            raise RuntimeError("feature")

    class BrokenBus(InMemoryEventBus):
        async def publish(self, event: object) -> None:
            raise RuntimeError("event")

    broken = MarketRegimeService(FakeMarket(candles()), None, None, None, None, BrokenBus(), BrokenStore())
    await broken._publish(snapshot, None, uuid4())
    assert broken.metrics.feature_publication_failures == broken.metrics.event_publication_failures == 1


@pytest.mark.asyncio
async def test_service_empty_market_failures_recovery_and_mtf() -> None:
    service = MarketRegimeService(FakeMarket(()), None, None, None, None, InMemoryEventBus(), InMemoryFeatureStore())
    with pytest.raises(ValueError, match="Market Data"):
        await service.analyze_snapshot("XAUUSD", Timeframe.M15)
    with pytest.raises(ValueError, match="Market Data"):
        await service.replay("XAUUSD", Timeframe.M15, BASE)
    assert service.metrics.replay_failures == 1

    healthy = MarketRegimeService(FakeMarket(candles()), None, None, None, None, InMemoryEventBus(), InMemoryFeatureStore(), MarketRegimeConfig(multi_timeframe=MultiTimeframeConfig(hierarchy=("M15", "W1"), maximum_depth=2)))
    mtf = await healthy.multi_timeframe("XAUUSD", Timeframe.M15)
    assert mtf.included_timeframes == ("M15",)
    assert mtf.unavailable_timeframes == ("W1",)
    snapshot = await healthy.analyze_snapshot("XAUUSD", Timeframe.M15)
    incompatible = snapshot.model_copy(update={"engine_version": "bad"})
    await healthy.repository.save_checkpoint(incompatible)
    with pytest.raises(ValueError, match="incompatible"):
        await healthy.recover("XAUUSD", Timeframe.M15)

    class BrokenRepo(InMemoryMarketRegimeRepository):
        async def checkpoints(self) -> tuple[object, ...]:
            raise RuntimeError("bad checkpoint")

    failed = MarketRegimeService(FakeMarket(candles()), None, None, None, None, InMemoryEventBus(), InMemoryFeatureStore(), repository=BrokenRepo())
    with pytest.raises(RuntimeError):
        await failed.restore()
    assert failed.recovery_state == "failed"


@pytest.mark.asyncio
async def test_upstream_adaptation_sessions_restore_transitions_and_event_matrix() -> None:
    from backend.app.engines.institutional_flow_engine import BaselineInstitutionalFlowAnalyzer, InstitutionalFlowContext

    flow_snapshot = BaselineInstitutionalFlowAnalyzer().analyze_snapshot(InstitutionalFlowContext(candles()))

    class Flow:
        async def state(self, *_: object) -> object | None:
            return None

        async def replay(self, *_: object) -> object:
            return flow_snapshot

    class Provider:
        def __init__(self, value: object | None = None, broken: bool = False) -> None:
            self.value, self.broken = value, broken

        async def state(self, *_: object) -> object | None:
            if self.broken:
                raise RuntimeError("optional upstream failure")
            return self.value

    service = MarketRegimeService(FakeMarket(candles()), Provider(None), Provider(broken=True), Provider(object()), Flow(), InMemoryEventBus(), InMemoryFeatureStore())
    values, missing, failed, session = await service._upstream("XAUUSD", Timeframe.M15, candles()[-1].timestamp)
    assert values and "smc" in missing and "liquidity" in failed
    assert session.session_alignment == "no_prior_session_context"
    bearish_pressure = flow_snapshot.state.pressure.model_copy(update={"net_pressure": -0.8})
    bearish_flow = flow_snapshot.model_copy(update={"state": flow_snapshot.state.model_copy(update={"pressure": bearish_pressure})})
    assert service._flow_evidence(bearish_flow, bearish_flow.analysis_timestamp)[-1].direction == EvidenceDirection.BEARISH
    class BrokenFlow:
        async def state(self, *_: object) -> object:
            raise RuntimeError("flow")

    failing_flow = MarketRegimeService(FakeMarket(candles()), None, None, None, BrokenFlow(), InMemoryEventBus(), InMemoryFeatureStore())
    _, _, flow_failed, _ = await failing_flow._upstream("XAUUSD", Timeframe.M15, candles()[-1].timestamp)
    assert "institutional_flow" in flow_failed

    session_item = SimpleNamespace(previous_session=SimpleNamespace(value="asia"), current_session=SimpleNamespace(value="london"), relationship="handoff", strength=0.8, confidence=0.7)
    session_snapshot = SimpleNamespace(state=SimpleNamespace(cross_session=(session_item,)), session=SimpleNamespace(value="london"))
    assert service._session(session_snapshot).handoff_score == 0.8

    snapshot = BaselineMarketRegimeAnalyzer().analyze_snapshot(MarketRegimeContext(candles(), rich()))
    await service.repository.save_checkpoint(snapshot)
    assert await service.restore() == 1
    transitioning = snapshot.model_copy(update={"previous_dominant_regime": DominantRegime.BALANCED, "dominant_regime": DominantRegime.TRENDING_BULL, "transition_state": TransitionState.WATCH, "transition_started_at": snapshot.analysis_timestamp})
    assert service._transition(transitioning) is not None

    previous = snapshot.model_copy(update={"snapshot_id": uuid4(), "dominant_regime": DominantRegime.BALANCED})
    eventful = transitioning.model_copy(update={"snapshot_id": uuid4(), "trend_regime": TrendRegime.BEAR_TREND, "volatility_regime": VolatilityRegime.HIGH, "auction_regime": AuctionRegime.BEARISH_IMBALANCE, "expansion_regime": ExpansionRegime.COMPRESSION, "lifecycle": RegimeLifecycle.WEAKENING, "trend_maturity": TrendMaturity.EXHAUSTION_RISK, "multi_timeframe": snapshot.multi_timeframe.model_copy(update={"conflict_score": 0.8}), "cross_session": snapshot.cross_session.model_copy(update={"handoff_score": 0.8})})
    await service._publish(eventful, previous, uuid4())
    expansion = eventful.model_copy(update={"snapshot_id": uuid4(), "expansion_regime": ExpansionRegime.EXPANSION, "transition_state": TransitionState.CONFIRMED})
    await service._publish(expansion, previous, uuid4())
    failed_transition = eventful.model_copy(update={"snapshot_id": uuid4(), "transition_state": TransitionState.FAILED})
    await service._publish(failed_transition, previous, uuid4())
    await service._publish(failed_transition, previous, uuid4())
    assert len(service.event_bus.history()) > 10


@pytest.mark.asyncio
async def test_analysis_persistence_failure_and_historical_state_and_mtf_replay() -> None:
    class BrokenSave(InMemoryMarketRegimeRepository):
        async def save_snapshot(self, snapshot: object) -> None:
            raise RuntimeError("persistence")

    broken = MarketRegimeService(FakeMarket(candles()), None, None, None, None, InMemoryEventBus(), InMemoryFeatureStore(), repository=BrokenSave())
    with pytest.raises(RuntimeError, match="persistence"):
        await broken.analyze_context(MarketRegimeContext(candles(), rich()))
    assert broken.metrics.failed_analysis_count == 1

    service = MarketRegimeService(FakeMarket(candles()), None, None, None, None, InMemoryEventBus(), InMemoryFeatureStore(), MarketRegimeConfig(multi_timeframe=MultiTimeframeConfig(hierarchy=("M15",), maximum_depth=1)))
    first = await service.analyze_snapshot("XAUUSD", Timeframe.M15)
    assert await service.state("XAUUSD", Timeframe.M15, first.analysis_timestamp) == first
    assert await service.state("XAUUSD", Timeframe.M15, BASE - timedelta(days=1)) is None
    mtf = await service.multi_timeframe("XAUUSD", Timeframe.M15, first.analysis_timestamp)
    assert mtf.included_timeframes == ("M15",)

    class FailingMarket(FakeMarket):
        async def history(self, *_: object, **__: object) -> list[Candle]:
            raise RuntimeError("missing")

    mtf_failure = MarketRegimeService(FailingMarket(candles()), None, None, None, None, InMemoryEventBus(), InMemoryFeatureStore(), MarketRegimeConfig(multi_timeframe=MultiTimeframeConfig(hierarchy=("M15",), maximum_depth=1)))
    assert (await mtf_failure.multi_timeframe("XAUUSD", Timeframe.M15)).unavailable_timeframes == ("M15",)

    future = first.model_copy(update={"analysis_timestamp": first.analysis_timestamp + timedelta(minutes=15), "historical_boundary": first.historical_boundary + timedelta(minutes=15)})
    service.replay = AsyncMock(return_value=future)  # type: ignore[method-assign]
    assert (await service.multi_timeframe("XAUUSD", Timeframe.M15, first.analysis_timestamp)).unavailable_timeframes == ("M15",)

    transition = RegimeTransition(transition_id=uuid4(), symbol="XAUUSD", timeframe=Timeframe.M15, from_regime=DominantRegime.BALANCED, to_regime=DominantRegime.TRENDING_BULL, started_at=BASE, state=TransitionState.WATCH, confidence=0.5, ambiguity=0.5, reasoning_summary="watch")
    save_transition = MarketRegimeService(FakeMarket(candles()), None, None, None, None, InMemoryEventBus(), InMemoryFeatureStore())
    save_transition._transition = lambda _: transition  # type: ignore[method-assign]
    await save_transition.analyze_context(MarketRegimeContext(candles(), rich()))
    assert await save_transition.repository.get_transition(transition.transition_id) == transition

    no_analysis_health = MarketRegimeService(FakeMarket(candles()), object(), object(), object(), object(), InMemoryEventBus(), InMemoryFeatureStore(), repository_mode="sqlalchemy")
    assert "no_analysis_completed" in no_analysis_health.health()["degradation_reasons"]


@pytest.mark.asyncio
async def test_sqlalchemy_repository_contract_with_fake_session() -> None:
    snapshot = BaselineMarketRegimeAnalyzer().analyze_snapshot(MarketRegimeContext(candles(), rich()))
    transition = RegimeTransition(transition_id=uuid4(), symbol="XAUUSD", timeframe=Timeframe.M15, from_regime=DominantRegime.BALANCED, to_regime=DominantRegime.TRENDING_BULL, started_at=BASE, state=TransitionState.WATCH, confidence=0.5, ambiguity=0.5, reasoning_summary="watch")

    class ScalarResult:
        def __init__(self, values: list[object]) -> None: self.values = values
        def first(self) -> object | None: return self.values[0] if self.values else None
        def all(self) -> list[object]: return self.values

    class Session:
        def __init__(self) -> None:
            self.execute = AsyncMock()
            self.commit = AsyncMock()
            self.values: list[object] = []
            self.one: object | None = None

        async def scalars(self, _: object) -> ScalarResult: return ScalarResult(self.values)
        async def get(self, _: object, __: object) -> object | None: return self.one

    session = Session()
    repo = SqlAlchemyMarketRegimeRepository(FakeSessionFactory(session))  # type: ignore[arg-type]
    await repo.save_snapshot(snapshot)
    await repo.save_evidence(snapshot)
    await repo.save_transition(transition)
    await repo.save_checkpoint(snapshot)
    session.values = [SimpleNamespace(payload=snapshot.model_dump(mode="json"))]
    assert await repo.get_latest_snapshot("XAUUSD", Timeframe.M15) == snapshot
    assert await repo.list_snapshots("XAUUSD", Timeframe.M15) == (snapshot,)
    session.one = SimpleNamespace(payload=snapshot.model_dump(mode="json"))
    assert await repo.get_snapshot(snapshot.snapshot_id) == snapshot
    session.one = SimpleNamespace(payload=transition.model_dump(mode="json"))
    assert await repo.get_transition(transition.transition_id) == transition
    session.values = [SimpleNamespace(payload=transition.model_dump(mode="json"))]
    assert await repo.list_transitions("XAUUSD", Timeframe.M15) == (transition,)
    session.values = [SimpleNamespace(payload=item.model_dump(mode="json")) for item in snapshot.evidence]
    assert await repo.list_evidence("XAUUSD", Timeframe.M15)
    digest = sha256(snapshot.model_dump_json().encode()).hexdigest()
    session.values = [SimpleNamespace(state_payload=snapshot.model_dump(mode="json"), payload_hash=digest, analysis_boundary=snapshot.analysis_timestamp)]
    assert await repo.load_checkpoint("XAUUSD", Timeframe.M15) == snapshot
    assert await repo.checkpoints() == (snapshot,)
    session.values = [SimpleNamespace(state_payload=snapshot.model_dump(mode="json"), payload_hash="x" * 64, analysis_boundary=snapshot.analysis_timestamp)]
    with pytest.raises(ValueError, match="integrity"):
        await repo.load_checkpoint("XAUUSD", Timeframe.M15)
    session.values = [SimpleNamespace(state_payload={"bad": True}, payload_hash="x" * 64), SimpleNamespace(state_payload=snapshot.model_dump(mode="json"), payload_hash="x" * 64)]
    assert await repo.checkpoints() == ()
    session.values = []
    assert await repo.load_checkpoint("XAUUSD", Timeframe.M15) is None
    assert await repo.get_latest_snapshot("XAUUSD", Timeframe.M15) is None
    session.one = None
    assert await repo.get_snapshot(uuid4()) is None
    session.values = [uuid4(), uuid4()]
    assert await repo.prune_history("XAUUSD", Timeframe.M15, 2) == 2


@pytest.mark.asyncio
async def test_registration_metadata_builder_executor_and_event_types() -> None:
    class Factory:
        def __init__(self) -> None: self.args: tuple[object, ...] = ()
        def register(self, *args: object) -> None: self.args = args

    factory = Factory()
    register(factory)  # type: ignore[arg-type]
    metadata, builder, executor = factory.args
    assert metadata.dependencies == ("market_data", "smc", "liquidity", "volume_profile", "institutional_flow")  # type: ignore[attr-defined]
    engine = builder(None, {})  # type: ignore[operator]
    output = await executor(engine, SimpleNamespace(candles=list(candles())))  # type: ignore[operator]
    assert output.namespace == "market_regime"
    assert issubclass(MarketRegimeDependencyRecovered, object)
    assert issubclass(RegimeTransitionInvalidated, object)


def test_migration_and_security_boundaries() -> None:
    from pathlib import Path

    root = Path(__file__).parents[3]
    sql = (root / "migrations" / "20260718_market_regime_v1.sql").read_text()
    for table in ("market_regime_snapshots", "market_regime_evidence", "market_regime_transitions", "market_regime_checkpoints"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    package = "\n".join(path.read_text() for path in (root / "backend/app/engines/market_regime_engine").glob("*.py"))
    for prohibited in ("place_order", "broker_connection", "take_profit_price", "expected_profit"):
        assert prohibited not in package
