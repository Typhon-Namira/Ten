"""One-pass replay-safe SMC displacement, zone, and range analysis."""

from collections import defaultdict, deque
from statistics import fmean

from .config import SMCConfig
from .context import CandleContext
from .models import (
    DealingRange,
    Displacement,
    DisplacementStrength,
    Evidence,
    LifecycleState,
    LiquidityReferenceType,
    SMCZone,
    StructureDirection,
    StructureEvent,
    StructureLiquidityReference,
    StructureScope,
    SwingPoint,
    ZoneType,
    stable_id,
)


_TERMINAL_ZONE_STATES = frozenset(
    {LifecycleState.MITIGATED, LifecycleState.INVALIDATED, LifecycleState.EXPIRED, LifecycleState.BROKEN, LifecycleState.SUPERSEDED, LifecycleState.ARCHIVED}
)


class AdvancedSMCAnalyzer:
    """Deterministic production analyzers sharing the candle availability boundary."""

    def __init__(self, config: SMCConfig) -> None:
        self.config = config

    def analyze(
        self,
        context: CandleContext,
        swings: tuple[SwingPoint, ...],
        events: tuple[StructureEvent, ...],
    ) -> tuple[tuple[Displacement, ...], tuple[SMCZone, ...], tuple[StructureLiquidityReference, ...], tuple[DealingRange, ...]]:
        displacements = self.displacements(context)
        zones = self.zones(context, events, displacements)
        references = self.liquidity_references(context, swings, events, zones)
        ranges = self.dealing_ranges(context, swings)
        return displacements, zones, references, ranges

    def displacements(self, context: CandleContext) -> tuple[Displacement, ...]:
        output: list[Displacement] = []
        volumes: deque[float] = deque(maxlen=self.config.displacement.volume_lookback)
        for index, candle in enumerate(context.candles):
            candle_range = max(candle.high - candle.low, 1e-12)
            body = abs(candle.close - candle.open)
            body_ratio = body / candle_range
            atr_ratio = candle_range / max(context.atr_at(index), 1e-12)
            efficiency = body / candle_range
            average_volume = fmean(volumes) if volumes and any(volumes) else 0.0
            volume_ratio = candle.volume / average_volume if average_volume > 0 else None
            volumes.append(candle.volume)
            cfg = self.config.displacement
            if atr_ratio < cfg.minimum_atr_impulse or body_ratio < cfg.minimum_body_ratio or efficiency < cfg.minimum_efficiency:
                continue
            direction = StructureDirection.BULLISH if candle.close > candle.open else StructureDirection.BEARISH
            strong = atr_ratio >= cfg.strong_atr_impulse and (volume_ratio is None or volume_ratio >= cfg.volume_confirmation_ratio)
            strength = DisplacementStrength.STRONG if strong else DisplacementStrength.WEAK
            volume_score = 70.0 if volume_ratio is None else min(100.0, volume_ratio / cfg.volume_confirmation_ratio * 100.0)
            confidence = min(100.0, atr_ratio / cfg.strong_atr_impulse * 35.0 + body_ratio * 30.0 + efficiency * 20.0 + volume_score * 0.15)
            evidence = (
                Evidence(code="atr_impulse", description="true range normalized by rolling ATR", value=atr_ratio, threshold=cfg.minimum_atr_impulse),
                Evidence(code="body_ratio", description="real body divided by candle range", value=body_ratio, threshold=cfg.minimum_body_ratio),
                Evidence(code="directional_efficiency", description="directional body efficiency", value=efficiency, threshold=cfg.minimum_efficiency),
                Evidence(code="volume_confirmation", description="volume relative to available rolling mean", value=volume_ratio, threshold=cfg.volume_confirmation_ratio, passed=volume_ratio is None or volume_ratio >= cfg.volume_confirmation_ratio),
            )
            detected = Displacement(
                id=stable_id("displacement", context.symbol, context.timeframe, candle.timestamp.isoformat(), self.config.version), symbol=context.symbol,
                timeframe=context.timeframe, direction=direction, strength=strength, start_timestamp=candle.timestamp, end_timestamp=candle.timestamp,
                source_candle_ids=(context.candle_id(index),), atr_normalized_impulse=atr_ratio, body_ratio=body_ratio, directional_efficiency=efficiency,
                volume_ratio=volume_ratio, lifecycle_state=LifecycleState.CONFIRMED, confidence_score=confidence,
                quality_score=candle.quality_score, evidence=evidence, algorithm_version=self.config.algorithm_version,
            )
            if output and index > 0 and output[-1].end_timestamp == context.candles[index - 1].timestamp and output[-1].direction == direction:
                previous = output.pop()
                count = len(previous.source_candle_ids) + 1
                detected = detected.model_copy(update={"id": stable_id("displacement-impulse", context.symbol, context.timeframe, previous.start_timestamp.isoformat(), candle.timestamp.isoformat(), self.config.version), "start_timestamp": previous.start_timestamp, "source_candle_ids": (*previous.source_candle_ids, context.candle_id(index)), "atr_normalized_impulse": previous.atr_normalized_impulse + atr_ratio, "body_ratio": (previous.body_ratio * (count - 1) + body_ratio) / count, "directional_efficiency": (previous.directional_efficiency * (count - 1) + efficiency) / count, "strength": DisplacementStrength.STRONG if previous.strength == DisplacementStrength.STRONG or strong else DisplacementStrength.WEAK, "confidence_score": min(100.0, max(previous.confidence_score, confidence) + count * 2.0), "quality_score": min(previous.quality_score, candle.quality_score), "version": previous.version + 1})
            output.append(detected)
        index_by_time = {item.timestamp: index for index, item in enumerate(context.candles)}
        for position, item in enumerate(output):
            origin_index = index_by_time[item.start_timestamp]
            origin = context.candles[origin_index]
            end = min(len(context.candles), index_by_time[item.end_timestamp] + self.config.displacement.invalidation_candles + 1)
            future = context.candles[index_by_time[item.end_timestamp] + 1 : end]
            invalidator = next((candidate for candidate in future if (candidate.close < origin.low if item.direction == StructureDirection.BULLISH else candidate.close > origin.high)), None)
            if invalidator:
                output[position] = item.model_copy(update={"lifecycle_state": LifecycleState.INVALIDATED, "invalidated_at": invalidator.timestamp, "version": item.version + 1})
        return tuple(output)

    def zones(self, context: CandleContext, events: tuple[StructureEvent, ...], displacements: tuple[Displacement, ...]) -> tuple[SMCZone, ...]:
        by_time = {item.end_timestamp: item for item in displacements}
        events_by_time: dict[object, list[StructureEvent]] = defaultdict(list)
        for event in events:
            events_by_time[event.timestamp].append(event)
        zones: list[SMCZone] = []
        active: dict[object, tuple[int, int]] = {}
        generated: set[object] = set()
        for index, candle in enumerate(context.candles):
            for zone_id, (origin_index, position) in list(active.items()):
                zone = zones[position]
                updated, derivatives = self._advance_zone(context, index, origin_index, zone)
                zones[position] = updated
                if updated.lifecycle_state in (LifecycleState.MITIGATED, LifecycleState.INVALIDATED, LifecycleState.EXPIRED, LifecycleState.BROKEN, LifecycleState.SUPERSEDED):
                    active.pop(zone_id)
                else:
                    active[zone_id] = (origin_index, position)
                for derivative in derivatives:
                    if derivative.id not in generated:
                        zones.append(derivative)
                        generated.add(derivative.id)
                        active[derivative.id] = (index, len(zones) - 1)
            if index >= 2:
                left = context.candles[index - 2]
                if candle.low > left.high:
                    self._add_zone(context, zones, active, generated, index, ZoneType.BULLISH_FVG, StructureDirection.BULLISH, left.high, candle.low, (index - 2, index), None)
                elif candle.high < left.low:
                    self._add_zone(context, zones, active, generated, index, ZoneType.BEARISH_FVG, StructureDirection.BEARISH, candle.high, left.low, (index - 2, index), None)
            displacement = by_time.get(candle.timestamp)
            if displacement and displacement.strength == DisplacementStrength.STRONG and len(displacement.source_candle_ids) >= self.config.imbalance.void_minimum_candles and index > 0:
                previous = context.candles[index - 1]
                low, high = sorted((previous.close, candle.open))
                if high > low:
                    self._add_zone(context, zones, active, generated, index, ZoneType.LIQUIDITY_VOID, displacement.direction, low, high, (index - 1, index), displacement.id)
            for event in events_by_time.get(candle.timestamp, []):
                self._order_block(context, zones, active, generated, index, event, by_time.get(candle.timestamp))
        return self._prune_zones(zones, context)

    def _prune_zones(self, zones: list[SMCZone], context: CandleContext) -> tuple[SMCZone, ...]:
        # `zones` accumulates every zone object created anywhere in this window's replay (slots
        # are mutated in place, never removed), so a zone that lives for hundreds of candles used
        # to drag every other long-dead zone from earlier in the window along with it into the
        # persisted snapshot/evidence payload. A non-terminal zone is always kept; a terminal one
        # is kept only if it terminated within `evidence_retention_candles` of the window's end.
        # Full zone-version history remains available via `smc_objects` (a separate append-only
        # audit table) regardless of what a cycle's snapshot embeds.
        retention = self.config.processing.evidence_retention_candles
        cutoff = context.candles[max(0, len(context.candles) - retention)].timestamp
        kept = [
            zone
            for zone in zones
            if zone.lifecycle_state not in _TERMINAL_ZONE_STATES
            or (zone.invalidation_timestamp or zone.expiration_timestamp or zone.mitigation_timestamp or zone.confirmation_timestamp) >= cutoff
        ]
        return tuple(kept[-self.config.processing.maximum_active_objects :])

    def _add_zone(self, context: CandleContext, zones: list[SMCZone], active: dict[object, tuple[int, int]], generated: set[object], index: int, zone_type: ZoneType, direction: StructureDirection, low: float, high: float, source_indices: tuple[int, ...], trigger: object | None, parent: object | None = None) -> None:
        size = high - low
        if size <= 0 or size < max(self.config.imbalance.minimum_size, context.atr_at(index) * self.config.imbalance.minimum_atr_size):
            return
        fvg_types = {ZoneType.BULLISH_FVG, ZoneType.BEARISH_FVG, ZoneType.BULLISH_INVERSION_FVG, ZoneType.BEARISH_INVERSION_FVG}
        if zone_type in fvg_types:
            tolerance = context.atr_at(index) * self.config.imbalance.merge_tolerance_atr
            for active_id, (_, position) in list(active.items()):
                existing = zones[position]
                if existing.zone_type == zone_type and low <= existing.upper_price + tolerance and high >= existing.lower_price - tolerance:
                    low, high = min(low, existing.lower_price), max(high, existing.upper_price)
                    parent = existing.id
                    zones[position] = existing.model_copy(update={"lifecycle_state": LifecycleState.SUPERSEDED, "version": existing.version + 1})
                    active.pop(active_id)
                    break
        identifier = stable_id("zone", context.symbol, context.timeframe, zone_type.value, context.candles[index].timestamp.isoformat(), low, high, trigger, parent)
        if identifier in generated:
            return
        if len(active) >= self.config.processing.maximum_active_objects:
            oldest_id, (_, oldest_position) = min(active.items(), key=lambda item: item[1][0])
            zones[oldest_position] = zones[oldest_position].model_copy(update={"lifecycle_state": LifecycleState.EXPIRED, "expiration_timestamp": context.candles[index].timestamp, "version": zones[oldest_position].version + 1})
            active.pop(oldest_id)
        scope = StructureScope.EXTERNAL if size >= context.atr_at(index) else StructureScope.INTERNAL
        quality = min(context.candles[item].quality_score for item in source_indices)
        confidence = min(100.0, 45.0 + size / max(context.atr_at(index), 1e-12) * 25.0 + quality * 0.3)
        zone = SMCZone(
            id=identifier, zone_type=zone_type, symbol=context.symbol, timeframe=context.timeframe, direction=direction, scope=scope,
            origin_timestamp=context.candles[source_indices[0]].timestamp, confirmation_timestamp=context.candles[index].timestamp,
            upper_price=high, lower_price=low, midpoint=(high + low) / 2, source_candle_ids=tuple(context.candle_id(item) for item in source_indices),
            trigger_event_id=trigger if hasattr(trigger, "hex") else None, parent_zone_id=parent if hasattr(parent, "hex") else None,
            lifecycle_state=LifecycleState.ACTIVE, confidence_score=confidence, quality_score=quality,
            evidence=(Evidence(code="zone_size", description="zone size normalized by ATR", value=size / max(context.atr_at(index), 1e-12), threshold=self.config.imbalance.minimum_atr_size),),
            algorithm_version=self.config.algorithm_version,
        )
        zones.append(zone)
        generated.add(identifier)
        active[identifier] = (index, len(zones) - 1)

    def _advance_zone(self, context: CandleContext, index: int, origin_index: int, zone: SMCZone) -> tuple[SMCZone, tuple[SMCZone, ...]]:
        if index <= origin_index:
            return zone, ()
        candle = context.candles[index]
        age = index - origin_index
        expiration = self.config.order_block.expiration_candles if "block" in zone.zone_type.value or "breaker" in zone.zone_type.value else self.config.imbalance.expiration_candles
        if age > expiration:
            return zone.model_copy(update={"lifecycle_state": LifecycleState.EXPIRED, "expiration_timestamp": candle.timestamp, "version": zone.version + 1}), ()
        bullish = zone.direction == StructureDirection.BULLISH
        invalidated = candle.close < zone.lower_price if bullish else candle.close > zone.upper_price
        if invalidated:
            inversion_types = {ZoneType.BULLISH_FVG: ZoneType.BEARISH_INVERSION_FVG, ZoneType.BEARISH_FVG: ZoneType.BULLISH_INVERSION_FVG, ZoneType.BULLISH_ORDER_BLOCK: ZoneType.BEARISH_BREAKER, ZoneType.BEARISH_ORDER_BLOCK: ZoneType.BULLISH_BREAKER}
            updated = zone.model_copy(update={"lifecycle_state": LifecycleState.INVALIDATED, "invalidation_timestamp": candle.timestamp, "invalidation_reason": "close beyond distal boundary", "version": zone.version + 1})
            target = inversion_types.get(zone.zone_type)
            if target is None:
                return updated, ()
            derivative = zone.model_copy(update={"id": stable_id("zone-conversion", context.symbol, context.timeframe, zone.id, target.value, candle.timestamp.isoformat()), "zone_type": target, "direction": StructureDirection.BEARISH if bullish else StructureDirection.BULLISH, "origin_timestamp": candle.timestamp, "confirmation_timestamp": candle.timestamp, "parent_zone_id": zone.id, "lifecycle_state": LifecycleState.ACTIVE, "fill_percentage": 0.0, "mitigation_percentage": 0.0, "first_touch_timestamp": None, "mitigation_timestamp": None, "invalidation_timestamp": None, "invalidation_reason": None, "version": 1})
            return updated, (derivative,)
        touched = candle.low <= zone.upper_price and candle.high >= zone.lower_price
        if not touched:
            decay = self.config.imbalance.time_decay_per_candle
            return zone.model_copy(update={"confidence_score": max(0.0, zone.confidence_score - decay), "version": zone.version + 1}), ()
        penetration = zone.upper_price - candle.low if bullish else candle.high - zone.lower_price
        fill = min(100.0, max(zone.fill_percentage, penetration / max(zone.upper_price - zone.lower_price, 1e-12) * 100.0))
        threshold = self.config.order_block.mitigation_threshold if "block" in zone.zone_type.value or "breaker" in zone.zone_type.value else self.config.imbalance.mitigation_threshold
        state = LifecycleState.MITIGATED if fill >= threshold else LifecycleState.PARTIALLY_MITIGATED
        updated = zone.model_copy(update={"lifecycle_state": state, "fill_percentage": fill, "mitigation_percentage": fill, "first_touch_timestamp": zone.first_touch_timestamp or candle.timestamp, "mitigation_timestamp": candle.timestamp if state == LifecycleState.MITIGATED else None, "version": zone.version + 1})
        derivatives: tuple[SMCZone, ...] = ()
        fvg_types = {ZoneType.BULLISH_FVG, ZoneType.BEARISH_FVG, ZoneType.BULLISH_INVERSION_FVG, ZoneType.BEARISH_INVERSION_FVG}
        if state == LifecycleState.PARTIALLY_MITIGATED and zone.zone_type in fvg_types:
            remaining_low = zone.lower_price if bullish else max(zone.lower_price, candle.high)
            remaining_high = min(zone.upper_price, candle.low) if bullish else zone.upper_price
            if remaining_high > remaining_low:
                updated = updated.model_copy(update={"lifecycle_state": LifecycleState.SUPERSEDED})
                derivatives = (updated.model_copy(update={"id": stable_id("gap-split", context.symbol, context.timeframe, zone.id, candle.timestamp.isoformat(), remaining_low, remaining_high), "lower_price": remaining_low, "upper_price": remaining_high, "midpoint": (remaining_low + remaining_high) / 2, "origin_timestamp": candle.timestamp, "confirmation_timestamp": candle.timestamp, "parent_zone_id": zone.id, "lifecycle_state": LifecycleState.ACTIVE, "fill_percentage": 0.0, "mitigation_percentage": 0.0, "first_touch_timestamp": None, "version": 1}),)
        if state == LifecycleState.PARTIALLY_MITIGATED and zone.zone_type in (ZoneType.BULLISH_ORDER_BLOCK, ZoneType.BEARISH_ORDER_BLOCK):
            kind = ZoneType.BULLISH_MITIGATION_BLOCK if bullish else ZoneType.BEARISH_MITIGATION_BLOCK
            derivatives = (updated.model_copy(update={"id": stable_id("mitigation-block", context.symbol, context.timeframe, zone.id, candle.timestamp.isoformat()), "zone_type": kind, "parent_zone_id": zone.id, "origin_timestamp": candle.timestamp, "confirmation_timestamp": candle.timestamp, "fill_percentage": 0.0, "mitigation_percentage": 0.0, "first_touch_timestamp": None, "version": 1}),)
        return updated, derivatives

    def _order_block(self, context: CandleContext, zones: list[SMCZone], active: dict[object, tuple[int, int]], generated: set[object], index: int, event: StructureEvent, displacement: Displacement | None) -> None:
        if displacement is None or displacement.confidence_score < self.config.order_block.minimum_displacement_score:
            return
        bullish = event.direction == StructureDirection.BULLISH
        start = max(0, index - self.config.order_block.lookback)
        origin = next((item for item in range(index - 1, start - 1, -1) if (context.candles[item].close < context.candles[item].open) == bullish), None)
        if origin is None:
            return
        candle = context.candles[origin]
        body_ratio = abs(candle.close - candle.open) / max(candle.high - candle.low, 1e-12)
        if body_ratio < self.config.order_block.minimum_body_ratio:
            return
        if self.config.order_block.require_volume_confirmation and not displacement.volume_ratio:
            return
        low, high = (min(candle.open, candle.close), max(candle.open, candle.close)) if self.config.order_block.refine_to_body else (candle.low, candle.high)
        kind = ZoneType.BULLISH_ORDER_BLOCK if bullish else ZoneType.BEARISH_ORDER_BLOCK
        before = len(zones)
        self._add_zone(context, zones, active, generated, index, kind, event.direction, low, high, (origin, index), event.id)
        if len(zones) > before:
            created = zones[-1]
            institutional = displacement.strength == DisplacementStrength.STRONG and (displacement.volume_ratio or 0) >= self.config.displacement.volume_confirmation_ratio
            zones[-1] = created.model_copy(update={"evidence": (*created.evidence, Evidence(code="order_block_validation", description="last opposing candle validated by structural break and displacement", value=displacement.confidence_score, threshold=self.config.order_block.minimum_displacement_score), Evidence(code="institutional_order_block", description="strong displacement with confirmed relative volume", value=displacement.volume_ratio, threshold=self.config.displacement.volume_confirmation_ratio, passed=institutional))})
            active[created.id] = (active[created.id][0], len(zones) - 1)

    def liquidity_references(self, context: CandleContext, swings: tuple[SwingPoint, ...], events: tuple[StructureEvent, ...], zones: tuple[SMCZone, ...]) -> tuple[StructureLiquidityReference, ...]:
        references: list[StructureLiquidityReference] = []
        for swing in swings:
            kind = LiquidityReferenceType.SWING_HIGH if "high" in swing.swing_type.value else LiquidityReferenceType.SWING_LOW
            direction = StructureDirection.BEARISH if kind == LiquidityReferenceType.SWING_HIGH else StructureDirection.BULLISH
            references.append(StructureLiquidityReference(id=stable_id("local-liquidity", context.symbol, context.timeframe, swing.id), symbol=context.symbol, timeframe=context.timeframe, reference_type=kind, direction=direction, price=swing.price, timestamp=swing.timestamp, available_at=swing.confirmed_at or swing.timestamp, source_swing_id=swing.id, confidence_score=swing.confidence_score, evidence=(Evidence(code="confirmed_swing", description="structure-local liquidity reference from confirmed swing", value=swing.strength),), algorithm_version=self.config.algorithm_version))
        swing_by_id = {item.id: item for item in swings}
        for event in events:
            broken_swing = swing_by_id.get(event.broken_swing_id)
            if broken_swing:
                references.append(StructureLiquidityReference(id=stable_id("inducement", context.symbol, context.timeframe, event.id, broken_swing.id), symbol=context.symbol, timeframe=context.timeframe, reference_type=LiquidityReferenceType.INDUCEMENT, direction=event.direction, price=broken_swing.price, timestamp=broken_swing.timestamp, available_at=event.timestamp, source_swing_id=broken_swing.id, source_object_id=event.id, confidence_score=min(event.confidence_score, broken_swing.confidence_score), evidence=(Evidence(code="structural_break_reference", description="broken swing used as local inducement evidence", value=event.event_type.value),), algorithm_version=self.config.algorithm_version))
        return tuple(references[-self.config.processing.maximum_active_objects :])

    def dealing_ranges(self, context: CandleContext, swings: tuple[SwingPoint, ...]) -> tuple[DealingRange, ...]:
        output: list[DealingRange] = []
        ordered = sorted(swings, key=lambda item: item.timestamp)
        for first, second in zip(ordered, ordered[1:], strict=False):
            if ("high" in first.swing_type.value) == ("high" in second.swing_type.value):
                continue
            high_swing = first if "high" in first.swing_type.value else second
            low_swing = second if high_swing is first else first
            high, low = high_swing.price, low_swing.price
            if high <= low:
                continue
            direction = StructureDirection.BULLISH if second.price > first.price else StructureDirection.BEARISH
            span = high - low
            cfg = self.config.dealing_range
            ote_values = (low + span * cfg.ote_low_ratio, low + span * cfg.ote_high_ratio) if direction == StructureDirection.BULLISH else (high - span * cfg.ote_high_ratio, high - span * cfg.ote_low_ratio)
            golden_values = (low + span * cfg.golden_low_ratio, low + span * cfg.golden_high_ratio) if direction == StructureDirection.BULLISH else (high - span * cfg.golden_high_ratio, high - span * cfg.golden_low_ratio)
            output.append(DealingRange(id=stable_id("dealing-range", context.symbol, context.timeframe, first.id, second.id), symbol=context.symbol, timeframe=context.timeframe, direction=direction, scope=StructureScope.NESTED if first.scope != second.scope else second.scope, range_high=high, range_low=low, equilibrium=low + span * 0.5, premium_boundary=low + span * cfg.premium_ratio, discount_boundary=low + span * cfg.discount_ratio, ote_low=ote_values[0], ote_high=ote_values[1], golden_zone_low=golden_values[0], golden_zone_high=golden_values[1], source_swing_high_id=high_swing.id, source_swing_low_id=low_swing.id, start_timestamp=first.timestamp, end_timestamp=second.timestamp, lifecycle_state=LifecycleState.ACTIVE, confidence_score=min(first.confidence_score, second.confidence_score)))
        return tuple(output[-self.config.dealing_range.maximum_ranges :])
