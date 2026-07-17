from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.smc_engine import (
    AnalysisStatus,
    BaselineSMCAnalyzer,
    InMemorySMCRepository,
    ProcessingMode,
    SMCConfig,
    SMCService,
    SqlAlchemySMCRepository,
    ConfirmationState,
    StructureLeg,
    StructureDirection,
    StructureEventType,
    StructureScope,
    SwingPoint,
    SwingType,
)
from backend.app.engines.smc_engine.context import CandleContext
from backend.app.engines.smc_engine.exceptions import InvalidSMCInput
from backend.app.engines.smc_engine.structure import StructureAnalyzer
from backend.app.engines.smc_engine.swing import SwingDetector
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore


def _candles(prices: list[tuple[float, float, float, float]], *, quality: float = 100.0, symbol: str = "XAU/USD") -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [Candle(symbol=symbol, timeframe=Timeframe.M15, timestamp=start + timedelta(minutes=15 * index), open=row[0], high=row[1], low=row[2], close=row[3], quality_score=quality) for index, row in enumerate(prices)]


def test_config_is_versioned_and_validated() -> None:
    config = SMCConfig()
    assert config.version == SMCConfig().version
    assert config.swing_window == 2 and config.minimum_displacement_ratio == 0.6 and config.equal_level_tolerance > 0
    with pytest.raises(ValidationError):
        SMCConfig.model_validate({"swing": {"left_window": 4, "right_window": 4}, "processing": {"minimum_history": 5}})


def test_context_rejects_corrupt_boundaries_and_tracks_quality() -> None:
    candles = _candles([(10, 11, 9, 10), (10, 12, 9, 11), (11, 13, 10, 12), (12, 12.5, 8, 9), (9, 14, 8.5, 13)], quality=55)
    context = CandleContext.build(candles, SMCConfig())
    assert context.degraded and context.average_true_range > 0
    assert context.candle_id(0).startswith("XAUUSD:M15")
    with pytest.raises(InvalidSMCInput, match="duplicate"):
        CandleContext.build([*candles, candles[-1]], SMCConfig())
    with pytest.raises(InvalidSMCInput, match="chronological"):
        CandleContext.build(list(reversed(candles)), SMCConfig())
    mixed = candles.copy()
    mixed[-1] = mixed[-1].model_copy(update={"symbol": "EUR/USD"})
    with pytest.raises(InvalidSMCInput, match="share symbol"):
        CandleContext.build(mixed, SMCConfig())
    with pytest.raises(InvalidSMCInput, match="requires"):
        CandleContext.build([], SMCConfig())


def test_swing_confirmation_is_no_lookahead_and_stable() -> None:
    candles = _candles([(10, 11, 9, 10), (10, 12, 9.5, 11), (11, 15, 10, 12), (12, 13, 9, 10), (10, 12, 8, 9), (9, 13, 8.5, 12), (12, 14, 10, 13)])
    detector = SwingDetector(SMCConfig())
    partial = detector.detect(CandleContext.build(candles[:4], SMCConfig()))
    full = detector.detect(CandleContext.build(candles, SMCConfig()))
    high = next(item for item in full if item.candle_index == 2)
    assert not partial
    assert high.confirmed_at == candles[4].timestamp
    assert high.price >= candles[2].high
    assert full == detector.detect(CandleContext.build(candles, SMCConfig()))
    assert not detector._significant(CandleContext.build(candles, SMCConfig()), high, high.candle_index)


def _swing(context: CandleContext, index: int, high: bool, confirmed_index: int, scope: StructureScope = StructureScope.INTERNAL) -> SwingPoint:
    swing_type = SwingType.INTERNAL_HIGH if high else SwingType.INTERNAL_LOW
    return SwingPoint(id=uuid4(), symbol=context.symbol, timeframe=context.timeframe, timestamp=context.candles[index].timestamp, candle_index=index, price=context.candles[index].high if high else context.candles[index].low, swing_type=swing_type, scope=scope, confirmation_state=ConfirmationState.CONFIRMED, left_window=1, right_window=1, strength=80, confidence_score=85, quality_score=100, source_candle_ids=(context.candle_id(index),), detected_at=context.candles[index].timestamp, confirmed_at=context.candles[confirmed_index].timestamp)


def test_structure_transitions_distinguish_bos_choch_and_mss() -> None:
    analyzer = StructureAnalyzer(SMCConfig())
    assert analyzer._transition(StructureDirection.NEUTRAL, StructureDirection.BULLISH, 0, StructureScope.INTERNAL) == (StructureEventType.BOS, StructureDirection.BULLISH)
    assert analyzer._transition(StructureDirection.BULLISH, StructureDirection.BEARISH, 10, StructureScope.INTERNAL) == (StructureEventType.CHOCH, StructureDirection.TRANSITIONAL)
    assert analyzer._transition(StructureDirection.BULLISH, StructureDirection.BEARISH, 90, StructureScope.EXTERNAL) == (StructureEventType.MSS, StructureDirection.BEARISH)


def test_structure_analysis_builds_legs_and_prevents_duplicate_breaks() -> None:
    candles = _candles([(10, 11, 9.5, 10), (10, 12, 9.8, 11), (11, 11.5, 10, 10.5), (10.5, 13.5, 10, 13), (13, 13.2, 9.5, 10), (10, 11, 9.7, 10.2), (10.2, 10.5, 7.5, 8)])
    context = CandleContext.build(candles, SMCConfig())
    swings = (_swing(context, 1, True, 2), _swing(context, 4, False, 5))
    state, legs, events = StructureAnalyzer(SMCConfig()).analyze(context, swings)
    assert [item.event_type for item in events] == [StructureEventType.BOS, StructureEventType.CHOCH]
    assert len(legs) == 1 and legs[0].high >= legs[0].low
    assert state.last_bos_id == events[0].id and state.last_choch_id == events[1].id


def test_structure_confirmation_modes_and_model_invariant() -> None:
    candle = _candles([(10, 12, 9, 10.5)])[0]
    close = StructureAnalyzer(SMCConfig())
    assert close._break(candle, 11, StructureDirection.BULLISH, 1)[0] is False
    wick = StructureAnalyzer(SMCConfig.model_validate({"structure": {"confirmation_method": "wick"}}))
    assert wick._break(candle, 11, StructureDirection.BULLISH, 1)[0] is True
    hybrid = StructureAnalyzer(SMCConfig.model_validate({"structure": {"confirmation_method": "hybrid"}}))
    assert hybrid._break(candle, 11, StructureDirection.BEARISH, 1)[0] is True
    with pytest.raises(ValidationError, match="high cannot"):
        StructureLeg(id=uuid4(), symbol="XAUUSD", timeframe=Timeframe.M15, start_swing_id=uuid4(), end_swing_id=uuid4(), direction=StructureDirection.BULLISH, scope=StructureScope.INTERNAL, high=10, low=11, magnitude=1, duration_seconds=60, displacement_score=50, confirmation_state=ConfirmationState.CONFIRMED, confidence_score=80)


def test_analysis_is_deterministic_degraded_and_bootstrapped(candles: list[Candle]) -> None:
    analyzer = BaselineSMCAnalyzer()
    first = analyzer.analyze_snapshot(candles)
    second = analyzer.analyze_snapshot(candles)
    assert first == second
    assert first.structure_events[-1].event_type == StructureEventType.BOS
    assert first.structure_state.current_direction == StructureDirection.BULLISH
    degraded = analyzer.analyze_snapshot([item.model_copy(update={"quality_score": 40}) for item in candles])
    assert degraded.status == AnalysisStatus.DEGRADED_INPUT
    assert degraded.structure_events[-1].confidence_score < first.structure_events[-1].confidence_score
    assert analyzer.analyze_snapshot([]).status == AnalysisStatus.INSUFFICIENT_HISTORY
    bearish = _candles([(10, 11, 9, 10), (10, 10.5, 8.8, 9.5), (9.5, 10, 8.5, 9), (9, 9.5, 8, 8.5), (8.5, 9, 7, 7.5)])
    assert analyzer.analyze_snapshot(bearish).structure_state.current_direction == StructureDirection.BEARISH


@pytest.mark.asyncio
async def test_repository_time_travel_is_idempotent(candles: list[Candle]) -> None:
    repository = InMemorySMCRepository()
    snapshot = BaselineSMCAnalyzer().analyze_snapshot(candles, ProcessingMode.REPLAY)
    await repository.save(snapshot)
    await repository.save(snapshot)
    assert await repository.latest("XAU/USD", Timeframe.M15) == snapshot
    assert await repository.at("XAUUSD", Timeframe.M15, snapshot.analysis_timestamp) == snapshot
    assert await repository.at("XAUUSD", Timeframe.M15, snapshot.analysis_timestamp - timedelta(seconds=1)) is None


@pytest.mark.asyncio
async def test_service_publishes_typed_features_and_events_once(candles: list[Candle]) -> None:
    bus = InMemoryEventBus()
    store = InMemoryFeatureStore()
    service = SMCService(cast(Any, object()), bus, store)
    snapshot = await service.analyze_candles(candles)
    event_count = len(bus.history())
    await service.analyze_candles(candles)
    assert len(bus.history()) == event_count
    assert service.metrics.bos_count == 2
    assert "current_structure_direction" in service.features(snapshot)
    assert service.health()["status"] == "healthy"


@pytest.mark.asyncio
async def test_service_market_boundary_replay_state_and_recalculation(candles: list[Candle]) -> None:
    market = SimpleNamespace(history=AsyncMock(return_value=candles), replay=AsyncMock(return_value=candles))
    service = SMCService(cast(Any, market), InMemoryEventBus(), InMemoryFeatureStore())
    historical = await service.analyze("XAU/USD", Timeframe.M15)
    replay = await service.replay("XAU/USD", Timeframe.M15, candles[-1].timestamp)
    assert replay.processing_mode == ProcessingMode.REPLAY
    assert await service.state("XAU/USD", Timeframe.M15) == replay
    assert await service.state("XAU/USD", Timeframe.M15, historical.analysis_timestamp) is not None
    rebuilt = await service.bounded_recalculate("XAU/USD", Timeframe.M15, candles[-1].timestamp)
    assert rebuilt.processing_mode == ProcessingMode.REBUILD


class _Scalars:
    def __init__(self, value: object | None) -> None:
        self.value = value

    def first(self) -> object | None:
        return self.value

    def all(self) -> list[object]:
        return [self.value] if self.value is not None else []


class _Session:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload
        self.execute = AsyncMock()
        self.commit = AsyncMock()

    async def scalars(self, _statement: object) -> _Scalars:
        return _Scalars(SimpleNamespace(payload=self.payload, state_payload=self.payload) if self.payload else None)


@pytest.mark.asyncio
async def test_sql_repository_conflict_safe_write_and_reads(candles: list[Candle]) -> None:
    snapshot = BaselineSMCAnalyzer().analyze_snapshot(candles)
    session = _Session(snapshot.model_dump(mode="json"))
    repository = SqlAlchemySMCRepository(cast(Any, session))
    await repository.save(snapshot)
    assert session.execute.await_count == 3 and session.commit.await_count == 1
    assert await repository.latest("XAU/USD", Timeframe.M15) == snapshot
    assert await repository.at("XAU/USD", Timeframe.M15, snapshot.analysis_timestamp) == snapshot
    assert await repository.checkpoints() == (snapshot,)
    empty = SqlAlchemySMCRepository(cast(Any, _Session()))
    assert await empty.latest("XAU/USD", Timeframe.M15) is None
