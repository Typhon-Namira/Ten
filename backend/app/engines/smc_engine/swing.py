"""Volatility-aware, no-lookahead pivot detection."""

from .config import SMCConfig
from .context import CandleContext
from .models import ConfirmationState, StructureScope, SwingPoint, SwingType, stable_id


class SwingDetector:
    def __init__(self, config: SMCConfig) -> None:
        self.config = config

    def detect(self, context: CandleContext) -> tuple[SwingPoint, ...]:
        left = self.config.swing.left_window
        right = self.config.swing.right_window
        confirmed: list[SwingPoint] = []
        last_by_side: dict[str, int] = {}
        for index in range(left, len(context.candles) - right):
            candle = context.candles[index]
            before = context.candles[index - left : index]
            after = context.candles[index + 1 : index + right + 1]
            high = all(candle.high > item.high for item in before) and all(candle.high >= item.high for item in after)
            low = all(candle.low < item.low for item in before) and all(candle.low <= item.low for item in after)
            if high:
                point = self._point(context, index, True)
                if self._significant(context, point, last_by_side.get("high")):
                    confirmed.append(point)
                    last_by_side["high"] = index
            if low:
                point = self._point(context, index, False)
                if self._significant(context, point, last_by_side.get("low")):
                    confirmed.append(point)
                    last_by_side["low"] = index
        return tuple(sorted(confirmed, key=lambda item: (item.confirmed_at or item.timestamp, item.timestamp, item.swing_type.value)))

    def _significant(self, context: CandleContext, point: SwingPoint, previous_index: int | None) -> bool:
        if previous_index is None:
            return True
        if point.candle_index - previous_index < self.config.swing.minimum_separation:
            return False
        previous = context.candles[previous_index]
        excursion = abs(point.price - (previous.high if "high" in point.swing_type.value else previous.low))
        required = max(self.config.swing.minimum_excursion, context.atr_at(point.candle_index) * self.config.swing.atr_excursion_multiplier)
        return excursion >= required

    def _point(self, context: CandleContext, index: int, high: bool) -> SwingPoint:
        candle = context.candles[index]
        left = self.config.swing.left_window
        right = self.config.swing.right_window
        confirmation_index = index + right
        neighborhood = context.candles[index - left : confirmation_index + 1]
        price = candle.high if high else candle.low
        opposite = min(item.low for item in neighborhood) if high else max(item.high for item in neighborhood)
        excursion = abs(price - opposite)
        atr = max(context.atr_at(index), 1e-12)
        strength = min(100.0, 40.0 + excursion / atr * 20.0 + (left + right) * 2.5)
        scope = StructureScope.EXTERNAL if strength >= self.config.swing.external_min_strength else StructureScope.INTERNAL
        swing_type = (
            SwingType.EXTERNAL_HIGH if high and scope == StructureScope.EXTERNAL else
            SwingType.INTERNAL_HIGH if high else
            SwingType.EXTERNAL_LOW if scope == StructureScope.EXTERNAL else
            SwingType.INTERNAL_LOW
        )
        quality = min(item.quality_score for item in neighborhood)
        confidence = min(100.0, strength * 0.7 + quality * 0.3)
        confirmed_at = context.candles[confirmation_index].timestamp
        return SwingPoint(
            id=stable_id("swing", context.symbol, context.timeframe, candle.timestamp.isoformat(), swing_type.value, self.config.version),
            symbol=context.symbol,
            timeframe=context.timeframe,
            timestamp=candle.timestamp,
            candle_index=index,
            price=price,
            swing_type=swing_type,
            scope=scope,
            confirmation_state=ConfirmationState.CONFIRMED,
            left_window=left,
            right_window=right,
            strength=strength,
            confidence_score=confidence,
            quality_score=quality,
            source_candle_ids=tuple(context.candle_id(item) for item in range(index - left, confirmation_index + 1)),
            detected_at=candle.timestamp,
            confirmed_at=confirmed_at,
        )
