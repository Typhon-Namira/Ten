"""Explicit internal/external structure state machine and structural events."""

from datetime import datetime

from backend.app.engines.market_data_engine import Candle

from .config import SMCConfig
from .context import CandleContext
from .models import (
    ConfirmationMethod,
    ConfirmationState,
    Evidence,
    MarketStructureState,
    StructureDirection,
    StructureEvent,
    StructureEventType,
    StructureLeg,
    StructureScope,
    SwingPoint,
    stable_id,
)


class StructureAnalyzer:
    def __init__(self, config: SMCConfig) -> None:
        self.config = config

    def analyze(self, context: CandleContext, swings: tuple[SwingPoint, ...]) -> tuple[MarketStructureState, tuple[StructureLeg, ...], tuple[StructureEvent, ...]]:
        high: dict[StructureScope, SwingPoint | None] = {StructureScope.INTERNAL: None, StructureScope.EXTERNAL: None}
        low: dict[StructureScope, SwingPoint | None] = {StructureScope.INTERNAL: None, StructureScope.EXTERNAL: None}
        directions = {StructureScope.INTERNAL: StructureDirection.NEUTRAL, StructureScope.EXTERNAL: StructureDirection.NEUTRAL}
        events: list[StructureEvent] = []
        broken: set[object] = set()
        available: dict[datetime, list[SwingPoint]] = {}
        for swing in swings:
            if swing.confirmed_at is not None:
                available.setdefault(swing.confirmed_at, []).append(swing)
        for index, candle in enumerate(context.candles):
            for swing in available.get(candle.timestamp, []):
                target = high if "high" in swing.swing_type.value else low
                target[swing.scope] = swing
            for scope in (StructureScope.INTERNAL, StructureScope.EXTERNAL):
                for direction, level in ((StructureDirection.BULLISH, high[scope]), (StructureDirection.BEARISH, low[scope])):
                    if level is None or level.id in broken or candle.timestamp <= (level.confirmed_at or level.timestamp):
                        continue
                    confirmed, close_confirmed, wick_confirmed, distance = self._break(candle, level.price, direction, context.atr_at(index))
                    if not confirmed:
                        continue
                    previous = directions[scope]
                    displacement = self.displacement(candle, context.atr_at(index))
                    event_type, resulting = self._transition(previous, direction, displacement, scope)
                    event = self._event(context, candle, level, scope, direction, previous, resulting, event_type, distance, displacement, close_confirmed, wick_confirmed)
                    events.append(event)
                    broken.add(level.id)
                    directions[scope] = resulting
        current = directions[StructureScope.EXTERNAL] if directions[StructureScope.EXTERNAL] != StructureDirection.NEUTRAL else directions[StructureScope.INTERNAL]
        legs = self._legs(context, swings)
        active_high = high[StructureScope.EXTERNAL] or high[StructureScope.INTERNAL]
        active_low = low[StructureScope.EXTERNAL] or low[StructureScope.INTERNAL]
        state = MarketStructureState(
            symbol=context.symbol,
            timeframe=context.timeframe,
            current_direction=current,
            previous_direction=events[-1].previous_direction if events else StructureDirection.NEUTRAL,
            internal_direction=directions[StructureScope.INTERNAL],
            external_direction=directions[StructureScope.EXTERNAL],
            active_swing_high_id=active_high.id if active_high else None,
            active_swing_low_id=active_low.id if active_low else None,
            protected_high_id=active_high.id if current == StructureDirection.BEARISH and active_high else None,
            protected_low_id=active_low.id if current == StructureDirection.BULLISH and active_low else None,
            last_bos_id=next((item.id for item in reversed(events) if item.event_type == StructureEventType.BOS), None),
            last_choch_id=next((item.id for item in reversed(events) if item.event_type == StructureEventType.CHOCH), None),
            last_mss_id=next((item.id for item in reversed(events) if item.event_type == StructureEventType.MSS), None),
            state_version=len(events),
            last_processed_candle=context.candles[-1].timestamp,
            updated_at=context.candles[-1].timestamp,
        )
        return state, legs, tuple(events)

    def _break(self, candle: Candle, level: float, direction: StructureDirection, atr: float) -> tuple[bool, bool, bool, float]:
        close_distance = candle.close - level if direction == StructureDirection.BULLISH else level - candle.close
        wick_distance = candle.high - level if direction == StructureDirection.BULLISH else level - candle.low
        required = max(self.config.structure.minimum_break_distance, atr * self.config.structure.atr_break_multiplier)
        close_confirmed = close_distance >= required
        wick_confirmed = wick_distance >= required
        method = self.config.structure.confirmation_method
        confirmed = close_confirmed if method == ConfirmationMethod.CLOSE else wick_confirmed if method == ConfirmationMethod.WICK else close_confirmed and wick_confirmed
        return confirmed, close_confirmed, wick_confirmed, max(0.0, close_distance if close_confirmed else wick_distance)

    def displacement(self, candle: Candle, atr: float) -> float:
        candle_range = max(candle.high - candle.low, 1e-12)
        body_ratio = abs(candle.close - candle.open) / candle_range
        range_ratio = candle_range / max(atr, 1e-12)
        return min(100.0, body_ratio * 60.0 + min(2.0, range_ratio) / 2.0 * 40.0)

    def _transition(self, previous: StructureDirection, break_direction: StructureDirection, displacement: float, scope: StructureScope) -> tuple[StructureEventType, StructureDirection]:
        if previous in (StructureDirection.NEUTRAL, break_direction):
            return StructureEventType.BOS, break_direction
        strong = displacement >= self.config.structure.mss_displacement_score and (scope == StructureScope.EXTERNAL or not self.config.structure.require_protected_level_for_mss)
        return (StructureEventType.MSS, break_direction) if strong else (StructureEventType.CHOCH, StructureDirection.TRANSITIONAL)

    def _event(self, context: CandleContext, candle: Candle, level: SwingPoint, scope: StructureScope, direction: StructureDirection, previous: StructureDirection, resulting: StructureDirection, event_type: StructureEventType, distance: float, displacement: float, close_confirmed: bool, wick_confirmed: bool) -> StructureEvent:
        confidence = min(100.0, level.confidence_score * 0.35 + displacement * 0.35 + level.quality_score * 0.3)
        evidence = (
            Evidence(code="level_break", description="configured structural level was broken", value=distance, threshold=self.config.structure.minimum_break_distance),
            Evidence(code="displacement", description="confirmation candle displacement score", value=displacement, threshold=self.config.structure.mss_displacement_score, passed=displacement >= self.config.structure.mss_displacement_score),
            Evidence(code="source_quality", description="minimum source candle quality", value=level.quality_score, threshold=self.config.processing.minimum_input_quality, passed=level.quality_score >= self.config.processing.minimum_input_quality),
        )
        return StructureEvent(
            id=stable_id("structure-event", context.symbol, context.timeframe, event_type.value, level.id, candle.timestamp.isoformat(), self.config.version),
            event_type=event_type, symbol=context.symbol, timeframe=context.timeframe, scope=scope, direction=direction,
            timestamp=candle.timestamp, broken_level=level.price, broken_swing_id=level.id, confirmation_candle_id=f"{context.symbol}:{context.timeframe.value}:{candle.timestamp.isoformat()}",
            confirmation_method=self.config.structure.confirmation_method, close_confirmed=close_confirmed, wick_confirmed=wick_confirmed, previous_direction=previous, resulting_direction=resulting,
            break_distance=distance, displacement_score=displacement, confidence_score=confidence, quality_score=level.quality_score, evidence=evidence,
            invalidation_metadata={"rule": "opposing confirmed break of protected structural level"}, algorithm_version=self.config.algorithm_version, created_at=candle.timestamp,
        )

    def _legs(self, context: CandleContext, swings: tuple[SwingPoint, ...]) -> tuple[StructureLeg, ...]:
        legs: list[StructureLeg] = []
        chronological = sorted(swings, key=lambda item: item.timestamp)
        for start, end in zip(chronological, chronological[1:], strict=False):
            if ("high" in start.swing_type.value) == ("high" in end.swing_type.value):
                continue
            direction = StructureDirection.BULLISH if end.price > start.price else StructureDirection.BEARISH
            magnitude = abs(end.price - start.price)
            atr = max(context.average_true_range, 1e-12)
            legs.append(StructureLeg(id=stable_id("leg", context.symbol, context.timeframe, start.id, end.id), symbol=context.symbol, timeframe=context.timeframe, start_swing_id=start.id, end_swing_id=end.id, direction=direction, scope=end.scope, high=max(start.price, end.price), low=min(start.price, end.price), magnitude=magnitude, duration_seconds=(end.timestamp - start.timestamp).total_seconds(), displacement_score=min(100.0, magnitude / atr * 25.0), confirmation_state=ConfirmationState.CONFIRMED, confidence_score=min(start.confidence_score, end.confidence_score)))
        return tuple(legs)
