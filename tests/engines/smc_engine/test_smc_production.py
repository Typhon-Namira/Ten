from datetime import UTC, datetime, timedelta
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.smc_engine import BaselineSMCAnalyzer, ConfirmationState, InMemorySMCRepository, LifecycleState, SMCConfig, SMCResult, SMCService, SMCZone, StructureDirection, StructureScope, SwingPoint, SwingType, ZoneType
from backend.app.engines.smc_engine.advanced import AdvancedSMCAnalyzer
from backend.app.engines.smc_engine.context import CandleContext
from backend.app.engines.smc_engine.liquidity_contract import ExternalLiquidityEvidence
from backend.app.engines.smc_engine.models import utc_from
from backend.app.engines.smc_engine.registration import _execute
from backend.app.engines.smc_engine.structure import StructureAnalyzer
from backend.app.services.pipeline_contracts import PipelineExecutionContext
from pydantic import ValidationError
from uuid import uuid4
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore


def candles(rows: list[tuple[float, float, float, float, float]], timeframe: Timeframe = Timeframe.M15) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [Candle(symbol="XAU/USD", timeframe=timeframe, timestamp=start + timeframe.duration * index, open=o, high=h, low=low, close=c, volume=v) for index, (o, h, low, c, v) in enumerate(rows)]


def test_displacement_impulse_volume_and_invalidation() -> None:
    data = candles([(100, 101, 99, 100, 10), (100, 105, 99.8, 104.8, 30), (104.8, 110, 104, 109.5, 40), (109.5, 110, 106, 107, 15), (107, 108, 98, 99, 50)])
    snapshot = BaselineSMCAnalyzer().analyze_snapshot(data)
    bullish = next(item for item in snapshot.displacements if item.direction == StructureDirection.BULLISH)
    assert len(bullish.source_candle_ids) == 2
    assert bullish.atr_normalized_impulse > 1
    assert bullish.lifecycle_state == LifecycleState.INVALIDATED
    assert bullish.invalidated_at == data[-1].timestamp


def test_fvg_split_mitigation_inversion_and_void_lifecycle() -> None:
    data = candles([(100, 101, 99, 100, 10), (100, 102, 99, 101, 10), (103, 106, 103, 105.5, 40), (105.5, 106, 102, 103, 20), (103, 104, 100, 100.5, 30)])
    snapshot = BaselineSMCAnalyzer().analyze_snapshot(data)
    types = {item.zone_type for item in snapshot.zones}
    assert ZoneType.BULLISH_FVG in types
    assert ZoneType.BEARISH_INVERSION_FVG in types
    parent = next(item for item in snapshot.zones if item.zone_type == ZoneType.BULLISH_FVG)
    assert parent.lifecycle_state in {LifecycleState.SUPERSEDED, LifecycleState.INVALIDATED}
    assert any(item.parent_zone_id for item in snapshot.zones)


def test_order_block_is_structurally_triggered(candles: list[Candle]) -> None:
    snapshot = BaselineSMCAnalyzer().analyze_snapshot(candles)
    order_blocks = [item for item in snapshot.zones if item.zone_type in {ZoneType.BULLISH_ORDER_BLOCK, ZoneType.BEARISH_ORDER_BLOCK}]
    assert order_blocks
    assert all(item.trigger_event_id is not None for item in order_blocks)
    assert all(item.upper_price >= item.lower_price for item in order_blocks)


def test_local_liquidity_and_nested_dealing_ranges() -> None:
    data = candles([(10, 11, 9, 10, 10), (10, 12, 9.5, 11, 20), (11, 15, 10, 14, 30), (14, 14.5, 9, 10, 15), (10, 11, 6, 7, 40), (7, 10, 6.5, 9, 20), (9, 13, 8, 12, 25)])
    snapshot = BaselineSMCAnalyzer().analyze_snapshot(data)
    assert snapshot.liquidity_references
    assert all(item.available_at >= item.timestamp for item in snapshot.liquidity_references)
    assert snapshot.dealing_ranges
    dealing_range = snapshot.dealing_ranges[-1]
    assert dealing_range.range_low < dealing_range.equilibrium < dealing_range.range_high
    assert dealing_range.ote_low < dealing_range.ote_high
    assert BaselineSMCAnalyzer().analyze(data).premium_discount_position == "premium"
    discounted = [*data[:-1], data[-1].model_copy(update={"close": 9.0})]
    assert BaselineSMCAnalyzer().analyze(discounted).premium_discount_position == "discount"
    external = ExternalLiquidityEvidence(id="sweep-1", symbol="XAUUSD", timeframe=Timeframe.M15, event_type="sweep", price=15, occurred_at=data[-1].timestamp, available_at=data[-1].timestamp, confidence_score=80, source_version="1.0")
    assert external.available_at == data[-1].timestamp


class Market:
    def __init__(self, source: list[Candle]) -> None:
        self.source = source

    async def history(self, symbol: str, timeframe: Timeframe, **_: object) -> list[Candle]:
        return [item.model_copy(update={"timeframe": timeframe, "timestamp": self.source[0].timestamp + timeframe.duration * index}) for index, item in enumerate(self.source)]

    async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, **kwargs: object) -> list[Candle]:
        return [item for item in await self.history(symbol, timeframe, **kwargs) if item.timestamp <= timestamp]


@pytest.mark.asyncio
async def test_mtf_no_future_data_and_restart_recovery(candles: list[Candle]) -> None:
    repository = InMemorySMCRepository()
    market = Market(candles)
    service = SMCService(cast(Any, market), InMemoryEventBus(), InMemoryFeatureStore(), repository=repository)
    snapshot = await service.analyze("XAU/USD", Timeframe.M15)
    restarted = SMCService(cast(Any, market), InMemoryEventBus(), InMemoryFeatureStore(), repository=repository)
    assert await restarted.restore() == 1
    assert await restarted.state("XAU/USD", Timeframe.M15) == snapshot
    context = await restarted.multi_timeframe("XAU/USD", Timeframe.M15, candles[-1].timestamp)
    assert context.analyzed_through <= candles[-1].timestamp
    assert {"W1", "MN1"} <= context.directions.keys()
    assert context.reasoning_metadata["no_future_candles"] is True


@pytest.mark.asyncio
async def test_dedicated_liquidity_contract_filters_future_evidence(candles: list[Candle]) -> None:
    visible = ExternalLiquidityEvidence(id="visible", symbol="XAUUSD", timeframe=Timeframe.M15, event_type="sweep", price=2650, occurred_at=candles[-1].timestamp, available_at=candles[-1].timestamp, confidence_score=90, source_version="1")
    future = visible.model_copy(update={"id": "future", "available_at": candles[-1].timestamp + timedelta(minutes=15)})
    reader = SimpleNamespace(evidence=AsyncMock(return_value=(visible, future)))
    service = SMCService(cast(Any, Market(candles)), InMemoryEventBus(), InMemoryFeatureStore(), liquidity_reader=cast(Any, reader))
    snapshot = await service.analyze_candles(candles)
    assert [item.external_sweep_id for item in snapshot.liquidity_references if item.external_sweep_id] == ["visible"]


@pytest.mark.asyncio
async def test_mtf_insufficient_failover_calendar_and_full_publication() -> None:
    class FailingMarket:
        async def history(self, symbol: str, timeframe: Timeframe, **_: object) -> list[Candle]:
            if timeframe == Timeframe.D1:
                return []
            raise RuntimeError("series unavailable")

        async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, **_: object) -> list[Candle]:
            raise RuntimeError("series unavailable")

    bus = InMemoryEventBus()
    service = SMCService(cast(Any, FailingMarket()), bus, InMemoryFeatureStore())
    mtf = await service.multi_timeframe("XAUUSD", Timeframe.M15)
    assert mtf.conflict_state.value == "insufficient" and mtf.alignment_score == 0
    assert service._calendar_directions([]) == {}
    calendar = candles([(10, 11, 9, 10, 1), (10, 12, 9, 11, 1), (11, 12, 9, 10, 1)], Timeframe.D1)
    calendar[1] = calendar[1].model_copy(update={"timestamp": calendar[0].timestamp + timedelta(days=8)})
    calendar[2] = calendar[2].model_copy(update={"timestamp": calendar[0].timestamp + timedelta(days=40)})
    assert service._calendar_directions(calendar)["W1"] == StructureDirection.BEARISH
    scenario = candles([(10, 11, 9, 10, 10), (10, 12, 9.5, 11, 20), (11, 15, 10, 14, 30), (14, 14.5, 9, 10, 15), (10, 11, 6, 7, 40), (7, 10, 6.5, 9, 20), (9, 13, 8, 12, 25)])
    scenario = [item.model_copy(update={"quality_score": 40}) for item in scenario]
    publishing_bus = InMemoryEventBus()
    publishing = SMCService(cast(Any, Market(scenario)), publishing_bus, InMemoryFeatureStore())
    snapshot = await publishing.analyze_candles(scenario)
    assert snapshot.dealing_ranges
    gap_snapshot = BaselineSMCAnalyzer().analyze_snapshot(candles([(100, 101, 99, 100, 10), (100, 102, 99, 101, 10), (103, 106, 103, 105.5, 40), (105.5, 106, 102, 103, 20), (103, 104, 100, 100.5, 30)]))
    versioned = snapshot.model_copy(update={"id": uuid4(), "zones": (gap_snapshot.zones[0].model_copy(update={"version": 2}),)})
    await publishing._publish(versioned, uuid4())
    names = {type(item).__name__ for item in publishing_bus.history()}
    assert {"SwingConfirmed", "SMCObjectLifecycleChanged", "DealingRangeUpdated", "SMCInputDegraded"} <= names


def test_batch_replay_prefix_convergence_and_stress() -> None:
    rows: list[tuple[float, float, float, float, float]] = []
    price = 100.0
    for index in range(2000):
        move = 1.8 if index % 11 < 6 else -1.5
        close = max(10.0, price + move)
        rows.append((price, max(price, close) + 0.4, min(price, close) - 0.4, close, float(100 + index % 17)))
        price = close
    data = candles(rows, Timeframe.M1)
    analyzer = BaselineSMCAnalyzer()
    started = perf_counter()
    full = analyzer.analyze_snapshot(data)
    elapsed = perf_counter() - started
    prefix = analyzer.analyze_snapshot(data[:-1])
    assert elapsed < 5
    assert all(item.confirmation_timestamp <= full.analysis_timestamp for item in full.zones)
    assert all(item.confirmation_timestamp <= prefix.analysis_timestamp for item in prefix.zones)
    assert analyzer.analyze_snapshot(data) == full


def test_configuration_zone_and_timestamp_invariants() -> None:
    with pytest.raises(ValidationError, match="strong displacement"):
        SMCConfig.model_validate({"displacement": {"minimum_atr_impulse": 2, "strong_atr_impulse": 1}})
    with pytest.raises(ValidationError, match="OTE"):
        SMCConfig.model_validate({"dealing_range": {"ote_low_ratio": 0.8, "ote_high_ratio": 0.7}})
    base = dict(id=uuid4(), zone_type=ZoneType.BULLISH_FVG, symbol="XAUUSD", timeframe=Timeframe.M15, direction=StructureDirection.BULLISH, scope=StructureScope.INTERNAL, origin_timestamp=datetime.now(UTC), confirmation_timestamp=datetime.now(UTC), source_candle_ids=("c",), lifecycle_state=LifecycleState.ACTIVE, confidence_score=80, quality_score=90)
    with pytest.raises(ValidationError, match="upper"):
        SMCZone(**base, upper_price=10, lower_price=11, midpoint=10.5)
    with pytest.raises(ValidationError, match="midpoint"):
        SMCZone(**base, upper_price=12, lower_price=10, midpoint=13)
    with pytest.raises(ValueError, match="timezone"):
        utc_from(datetime(2026, 1, 1))
    assert utc_from(datetime(2026, 1, 1, tzinfo=UTC)).tzinfo == UTC


def test_zone_merge_duplicate_void_and_mitigation_branches() -> None:
    data = candles([(100, 101, 99, 100, 10), (100, 104, 99.8, 103.8, 60), (105, 110, 105, 109.5, 100), (109.5, 110, 102, 103, 20), (103, 104, 100, 101, 20)])
    config = SMCConfig.model_validate({"displacement": {"strong_atr_impulse": 0.8}})
    context = CandleContext.build(data, config)
    advanced = AdvancedSMCAnalyzer(config)
    snapshot = BaselineSMCAnalyzer(config).analyze_snapshot(data)
    assert any(item.zone_type == ZoneType.LIQUIDITY_VOID for item in snapshot.zones)
    zones: list[SMCZone] = []
    active: dict[object, tuple[int, int]] = {}
    generated: set[object] = set()
    advanced._add_zone(context, zones, active, generated, 2, ZoneType.BULLISH_FVG, StructureDirection.BULLISH, 101, 103, (0, 2), None)
    advanced._add_zone(context, zones, active, generated, 3, ZoneType.BULLISH_FVG, StructureDirection.BULLISH, 102, 104, (1, 3), None)
    assert zones[0].lifecycle_state == LifecycleState.SUPERSEDED
    count = len(zones)
    advanced._add_zone(context, zones, active, generated, 3, ZoneType.LIQUIDITY_VOID, StructureDirection.BULLISH, 101, 102, (1, 3), None)
    advanced._add_zone(context, zones, active, generated, 3, ZoneType.LIQUIDITY_VOID, StructureDirection.BULLISH, 101, 102, (1, 3), None)
    advanced._add_zone(context, zones, active, generated, 3, ZoneType.BULLISH_FVG, StructureDirection.BULLISH, 104, 104, (1, 3), None)
    assert len(zones) == count + 1
    same, derivatives = advanced._advance_zone(context, 2, 2, zones[-1])
    assert same == zones[-1] and not derivatives
    order_block = zones[-1].model_copy(update={"zone_type": ZoneType.BULLISH_ORDER_BLOCK, "lower_price": 100.0, "upper_price": 104.0, "midpoint": 102.0})
    updated, children = advanced._advance_zone(context, 3, 1, order_block)
    assert updated.lifecycle_state == LifecycleState.PARTIALLY_MITIGATED
    assert children[0].zone_type == ZoneType.BULLISH_MITIGATION_BLOCK
    bounded = AdvancedSMCAnalyzer(SMCConfig.model_validate({"processing": {"maximum_active_objects": 100}}))
    bounded_zones: list[SMCZone] = []
    bounded_active: dict[object, tuple[int, int]] = {}
    bounded_generated: set[object] = set()
    for item in range(101):
        bounded._add_zone(context, bounded_zones, bounded_active, bounded_generated, 2, ZoneType.LIQUIDITY_VOID, StructureDirection.BULLISH, 100 + item * 2, 101 + item * 2, (0, 2), None)
    assert len(bounded_active) == 100 and bounded_zones[0].lifecycle_state == LifecycleState.EXPIRED


def _point(data: list[Candle], index: int, high: bool) -> SwingPoint:
    return SwingPoint(id=uuid4(), symbol="XAUUSD", timeframe=Timeframe.M15, timestamp=data[index].timestamp, candle_index=index, price=data[index].high if high else data[index].low, swing_type=SwingType.INTERNAL_HIGH if high else SwingType.INTERNAL_LOW, scope=StructureScope.INTERNAL, confirmation_state=ConfirmationState.CONFIRMED, left_window=1, right_window=1, strength=70, confidence_score=80, quality_score=100, source_candle_ids=(str(index),), detected_at=data[index].timestamp, confirmed_at=data[min(index + 1, len(data) - 1)].timestamp)


def test_defensive_order_block_and_range_paths(candles: list[Candle]) -> None:
    context = CandleContext.build(candles, SMCConfig())
    advanced = AdvancedSMCAnalyzer(SMCConfig())
    snapshot = BaselineSMCAnalyzer().analyze_snapshot(candles)
    event = snapshot.structure_events[-1]
    displacement = snapshot.displacements[-1]
    advanced._order_block(context, [], {}, set(), len(candles) - 1, event, None)
    advanced._order_block(context, [], {}, set(), len(candles) - 1, event, displacement.model_copy(update={"confidence_score": 0.0}))
    bullish = [item.model_copy(update={"open": item.low, "close": item.high}) for item in candles]
    bullish_context = CandleContext.build(bullish, SMCConfig())
    advanced._order_block(bullish_context, [], {}, set(), len(bullish) - 1, event, displacement)
    strict_body = AdvancedSMCAnalyzer(SMCConfig.model_validate({"order_block": {"minimum_body_ratio": 0.99}}))
    strict_body._order_block(context, [], {}, set(), len(candles) - 1, event, displacement)
    volume_required = AdvancedSMCAnalyzer(SMCConfig.model_validate({"order_block": {"require_volume_confirmation": True}}))
    volume_required._order_block(context, [], {}, set(), len(candles) - 1, event, displacement.model_copy(update={"volume_ratio": None}))
    high1, high2 = _point(candles, 1, True), _point(candles, 3, True)
    assert not advanced.dealing_ranges(context, (high1, high2))
    assert not StructureAnalyzer(SMCConfig())._legs(context, (high1, high2))
    inverted_low = _point(candles, 3, False).model_copy(update={"price": high1.price + 1})
    assert not advanced.dealing_ranges(context, (high1, inverted_low))


@pytest.mark.asyncio
async def test_registration_rejects_analyzer_without_snapshot(candles: list[Candle]) -> None:
    engine = SimpleNamespace(analyze=lambda _: SMCResult())
    context = PipelineExecutionContext(correlation_id=uuid4(), candles=candles, events=[], feature_store=InMemoryFeatureStore())
    with pytest.raises(RuntimeError, match="snapshot"):
        await _execute(cast(Any, engine), context)
