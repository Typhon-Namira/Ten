from abc import ABC
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import logging
from math import isfinite, sqrt
from statistics import fmean

from backend.app.engines.common import AnalysisEngine
from backend.app.engines.market_data_engine import Candle, MarketSession, Timeframe
from backend.app.engines.market_data_engine.sessions import MarketSessionEngine

from .config import VolumeProfileConfig
from .contracts import VolumeProfileContext
from .models import (
    AnalysisStatus,
    DataQualityLevel,
    HighVolumeNode,
    LowVolumeNode,
    NodeType,
    PointOfControl,
    PriceGridMethod,
    PriceNode,
    ProcessingMode,
    ProfileConfluence,
    ProfileLifecycleState,
    ProfileLifecycleTransition,
    ProfileMigration,
    ProfileMigrationType,
    ProfileShape,
    ProfileShapeType,
    ProfileSkipReason,
    ProfileStatus,
    ProfileType,
    SessionType,
    SkippedProfilePeriod,
    ValueArea,
    ValueAreaMethod,
    VolumeAllocationMethod,
    VolumeDataQuality,
    VolumeGap,
    VolumeProfile,
    VolumeProfileAnalysisSnapshot,
    VolumeProfileBucket,
    VolumeProfileEvidence,
    VolumeProfileResult,
    VolumeShelf,
    VolumeSourceType,
    stable_id,
)

logger = logging.getLogger(__name__)


class VolumeProfileAnalyzer(AnalysisEngine[list[Candle], VolumeProfileResult], ABC):
    """Provider-neutral deterministic volume-at-price contract."""


class BaselineVolumeProfileAnalyzer(VolumeProfileAnalyzer):
    name = "volume_profile"
    version = "1.0.0"

    def __init__(self, config: VolumeProfileConfig | None = None, sessions: MarketSessionEngine | None = None) -> None:
        self.config = config or VolumeProfileConfig()
        self.sessions = sessions or MarketSessionEngine()

    def analyze(self, data: list[Candle]) -> VolumeProfileResult:
        snapshot = self.analyze_snapshot(VolumeProfileContext(tuple(data)))
        profile = next((x for x in snapshot.profiles if x.profile_type == ProfileType.FIXED_RANGE), None)
        if profile is None:
            return VolumeProfileResult(observations=["No usable positive volume supplied."], snapshot=snapshot)
        nodes = [
            PriceNode(price=x.midpoint, volume=x.volume, kind="HVN" if any(x.id in n.bucket_ids for n in profile.hvns) else "LVN") for x in profile.buckets
        ]
        return VolumeProfileResult(
            poc=profile.poc.price if profile.poc else None,
            vah=profile.value_area.vah if profile.value_area else None,
            val=profile.value_area.val if profile.value_area else None,
            total_volume=profile.total_volume,
            nodes=nodes,
            profile_type=profile.profile_type.value,
            snapshot=snapshot,
        )

    def analyze_snapshot(self, context: VolumeProfileContext, mode: ProcessingMode = ProcessingMode.HISTORICAL) -> VolumeProfileAnalysisSnapshot:
        candles = sorted(context.candles, key=lambda x: x.timestamp)
        boundary = candles[-1].timestamp if candles else datetime(1970, 1, 1, tzinfo=UTC)
        quality = self._quality(candles, context.volume_source_type)
        status = AnalysisStatus.COMPLETE
        profiles: list[VolumeProfile] = []
        skipped_periods: list[SkippedProfilePeriod] = []
        if len(candles) < self.config.processing.minimum_candles:
            status = AnalysisStatus.INSUFFICIENT_HISTORY
            skipped_periods.append(
                self._skipped_period(
                    candles,
                    context,
                    ProfileType.FIXED_RANGE,
                    "analysis_window",
                    ProfileSkipReason.INSUFFICIENT_VOLUME_PROFILE_DATA,
                    boundary,
                )
            )
        elif quality.usable_volume_ratio == 0:
            status = AnalysisStatus.DEGRADED
            skipped_periods.append(
                self._skipped_period(
                    candles,
                    context,
                    ProfileType.FIXED_RANGE,
                    "analysis_window",
                    ProfileSkipReason.INSUFFICIENT_VOLUME_PROFILE_DATA,
                    boundary,
                )
            )
        else:
            bounded = candles[-self.config.processing.maximum_candles :]
            fixed = self._profile(bounded, ProfileType.FIXED_RANGE, context, completed=True)
            developing = self._profile(bounded, ProfileType.DEVELOPING, context, completed=False)
            if fixed is not None:
                profiles.append(fixed)
            if developing is not None:
                profiles.append(developing)
            for profile_type in (ProfileType.SESSION, ProfileType.DAILY, ProfileType.WEEKLY, ProfileType.MONTHLY):
                period_profiles, period_skips = self._period_profiles(bounded, context, profile_type)
                profiles.extend(period_profiles)
                skipped_periods.extend(period_skips)
            completed = [x for x in profiles if x.status == ProfileStatus.COMPLETED]
            if completed:
                composite = self._profile(
                    bounded,
                    ProfileType.COMPOSITE,
                    context,
                    completed=True,
                    constituent_ids=tuple(x.id for x in completed[-self.config.processing.maximum_composite_profiles :]),
                )
                if composite is not None:
                    profiles.append(composite)
            for anchor in context.anchors:
                anchored = [x for x in bounded if x.timestamp >= anchor.availability_timestamp and anchor.availability_timestamp <= boundary]
                if anchored:
                    profile = self._profile(anchored, ProfileType.ANCHORED, context, completed=False, anchor=anchor)
                    if profile is not None:
                        profiles.append(profile)
                    else:
                        skipped_periods.append(
                            self._skipped_period(
                                anchored,
                                context,
                                ProfileType.ANCHORED,
                                str(anchor.id),
                                ProfileSkipReason.INSUFFICIENT_VOLUME_PROFILE_DATA,
                                boundary,
                            )
                        )
            profiles = self._tested_references(profiles, bounded)
            if skipped_periods:
                status = AnalysisStatus.DEGRADED
        migrations = self._migrations(profiles)
        confluences = self._confluences(profiles, context)
        transition_items = []
        for profile in profiles:
            initial = ProfileLifecycleState.DEVELOPING if profile.status == ProfileStatus.DEVELOPING else ProfileLifecycleState.COMPLETED
            transition_items.append(
                ProfileLifecycleTransition(
                    id=stable_id("transition", profile.id, initial),
                    profile_id=profile.id,
                    previous=ProfileLifecycleState.INITIALIZED,
                    current=initial,
                    available_at=profile.availability_timestamp,
                    reason="source boundary evaluated",
                )
            )
            if profile.lifecycle_state == ProfileLifecycleState.TESTED:
                test_times = [
                    x
                    for x in (
                        profile.poc.first_test_at if profile.poc else None,
                        profile.value_area.first_test_at if profile.value_area else None,
                        *(node.first_test_at for node in (*profile.hvns, *profile.lvns)),
                    )
                    if x is not None
                ]
                transition_items.append(
                    ProfileLifecycleTransition(
                        id=stable_id("transition", profile.id, "tested"),
                        profile_id=profile.id,
                        previous=ProfileLifecycleState.COMPLETED,
                        current=ProfileLifecycleState.TESTED,
                        available_at=min(test_times),
                        reason="later visible price intersected a completed profile reference",
                    )
                )
        transitions = tuple(transition_items)
        snapshot_id = stable_id(
            "snapshot",
            candles[0].symbol if candles else context.instrument,
            candles[0].timeframe if candles else context.requested_timeframe or Timeframe.M15,
            boundary,
            mode,
            context.volume_source_type,
            tuple(x.id for x in context.anchors),
            context.liquidity_source_ids,
            self.config.version,
        )
        return VolumeProfileAnalysisSnapshot(
            id=snapshot_id,
            symbol=candles[0].symbol.replace("/", "") if candles else context.instrument.replace("/", "").upper() or "UNKNOWN",
            timeframe=candles[0].timeframe if candles else context.requested_timeframe or Timeframe.M15,
            analysis_timestamp=boundary,
            processing_mode=mode,
            status=status,
            profiles=tuple(profiles),
            developing=tuple(x for x in profiles if x.status == ProfileStatus.DEVELOPING),
            completed=tuple(x for x in profiles if x.status == ProfileStatus.COMPLETED),
            migrations=migrations,
            confluences=confluences,
            lifecycle_transitions=transitions,
            volume_data_quality=quality,
            confidence_summary={"overall": fmean([x.confidence_score for x in profiles]) if profiles else 0.0},
            quality_summary={"market_data": quality.score, "volume_source": self._source_score(context.volume_source_type)},
            degraded_reasons=tuple(dict.fromkeys(item.reason for item in skipped_periods)),
            skipped_periods=tuple(skipped_periods),
            configuration_version=self.config.version,
            engine_version=self.version,
            market_data_boundary=f"{len(candles)}:{boundary.isoformat()}",
            created_at=boundary,
        )

    @staticmethod
    def _is_usable_candle(candle: Candle) -> bool:
        prices = (candle.open, candle.high, candle.low, candle.close)
        return (
            isfinite(candle.volume)
            and candle.volume > 0
            and all(isfinite(value) and value > 0 for value in prices)
            and candle.low <= min(candle.open, candle.close)
            and candle.high >= max(candle.open, candle.close)
            and candle.low <= candle.high
        )

    def _quality(self, candles: list[Candle], source: VolumeSourceType) -> VolumeDataQuality:
        missing = sum(x.volume == 0 for x in candles)
        usable = sum(self._is_usable_candle(x) for x in candles)
        invalid = len(candles) - missing - usable
        ratio = usable / len(candles) if candles else 0.0
        score = min(self._source_score(source), fmean([x.quality_score for x in candles]) if candles else 0.0) * ratio
        level = (
            DataQualityLevel.HIGH
            if score >= 80
            else DataQualityLevel.MEDIUM
            if score >= 60
            else DataQualityLevel.LOW
            if score > 0
            else DataQualityLevel.UNUSABLE
        )
        limitations = []
        if source == VolumeSourceType.TICK:
            limitations.append("tick volume is not centralized exchange volume")
        if source == VolumeSourceType.UNKNOWN:
            limitations.append("volume semantics are unknown")
        if missing:
            limitations.append("zero-volume observations were excluded")
        if invalid:
            limitations.append("malformed volume or OHLC observations were excluded")
        return VolumeDataQuality(
            source_type=source,
            quality_level=level,
            score=score,
            usable_volume_ratio=ratio,
            missing_observations=missing,
            invalid_observations=invalid,
            limitations=tuple(limitations),
        )

    def _skipped_period(
        self,
        candles: list[Candle],
        context: VolumeProfileContext,
        kind: ProfileType,
        period_key: str,
        reason: ProfileSkipReason,
        boundary: datetime,
    ) -> SkippedProfilePeriod:
        usable_count = sum(self._is_usable_candle(candle) for candle in candles)
        symbol = candles[0].symbol.replace("/", "") if candles else context.instrument.replace("/", "").upper()
        timeframe = candles[0].timeframe if candles else context.requested_timeframe or Timeframe.M15
        item = SkippedProfilePeriod(
            symbol=symbol or "UNKNOWN",
            timeframe=timeframe,
            profile_type=kind,
            period_key=period_key,
            reason=reason,
            input_count=len(candles),
            usable_count=usable_count,
            analysis_boundary=boundary,
        )
        logger.warning(
            "volume_profile.period.skipped",
            extra={
                "volume_profile_symbol": item.symbol,
                "volume_profile_timeframe": item.timeframe.value,
                "volume_profile_boundary": item.analysis_boundary.isoformat(),
                "volume_profile_input_count": item.input_count,
                "volume_profile_usable_count": item.usable_count,
                "volume_profile_skipped_period": item.period_key,
                "volume_profile_profile_type": item.profile_type.value,
                "volume_profile_skip_reason": item.reason.value,
            },
        )
        return item

    @staticmethod
    def _source_score(source: VolumeSourceType) -> float:
        return {
            VolumeSourceType.EXCHANGE: 100.0,
            VolumeSourceType.BROKER: 80.0,
            VolumeSourceType.TICK: 70.0,
            VolumeSourceType.SYNTHETIC: 40.0,
            VolumeSourceType.UNKNOWN: 50.0,
            VolumeSourceType.MISSING: 0.0,
        }[source]

    def _row_size(self, candles: list[Candle], context: VolumeProfileContext) -> tuple[Decimal, PriceGridMethod]:
        cfg = self.config.price_grid
        low, high = min(x.low for x in candles), max(x.high for x in candles)
        span = max(high - low, cfg.tick_size)
        method = PriceGridMethod(cfg.method)
        if method == PriceGridMethod.TICK:
            raw = context.tick_size or cfg.tick_size
        elif method == PriceGridMethod.FIXED:
            raw = cfg.fixed_increment
        elif method == PriceGridMethod.PERCENTAGE:
            raw = fmean([x.close for x in candles]) * cfg.percentage
        elif method == PriceGridMethod.ATR:
            raw = fmean([x.high - x.low for x in candles]) * cfg.atr_multiplier
        elif method == PriceGridMethod.AUTO:
            raw = span / min(max(int(sqrt(len(candles)) * 4), cfg.minimum_bins), cfg.maximum_bins)
        else:
            raw = span / cfg.rows
        tick = Decimal(str(context.tick_size or cfg.tick_size))
        aligned = (Decimal(str(max(raw, float(tick)))) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        bins = int((Decimal(str(high)) - Decimal(str(low))) / aligned) + 1
        if bins > cfg.maximum_bins:
            aligned = (Decimal(str(span)) / Decimal(cfg.maximum_bins) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        return aligned, method

    def _grid(self, candles: list[Candle], context: VolumeProfileContext) -> tuple[Decimal, Decimal, int, PriceGridMethod]:
        row, method = self._row_size(candles, context)
        low = min(Decimal(str(x.low)) for x in candles)
        high = max(Decimal(str(x.high)) for x in candles)
        base = (low / row).to_integral_value(rounding=ROUND_FLOOR) * row
        top = (high / row).to_integral_value(rounding=ROUND_CEILING) * row
        count = max(self.config.price_grid.minimum_bins, int((top - base) / row))
        count = min(count, self.config.price_grid.maximum_bins)
        return base, row, count, method

    def _allocate(self, candles: list[Candle], context: VolumeProfileContext) -> tuple[tuple[VolumeProfileBucket, ...], float, PriceGridMethod]:
        if not candles:
            return (), 0.0, PriceGridMethod(self.config.price_grid.method)
        base, row, count, grid_method = self._grid(candles, context)
        volumes = [0.0] * count
        buys = [0.0] * count
        sources = [0] * count
        method = VolumeAllocationMethod(self.config.allocation.method)
        for candle in candles:
            low_index = max(0, min(count - 1, int((Decimal(str(candle.low)) - base) / row)))
            high_index = max(0, min(count - 1, int((Decimal(str(candle.high)) - base) / row)))
            if method in {VolumeAllocationMethod.CLOSE, VolumeAllocationMethod.TYPICAL_PRICE}:
                price = candle.close if method == VolumeAllocationMethod.CLOSE else (candle.high + candle.low + candle.close) / 3
                indexes = [max(0, min(count - 1, int((Decimal(str(price)) - base) / row)))]
                weights = [1.0]
            elif method == VolumeAllocationMethod.BODY_WICK and candle.high > candle.low:
                body_low, body_high = sorted((candle.open, candle.close))
                body = [
                    i for i in range(low_index, high_index + 1) if float(base + row * Decimal(i + 1)) > body_low and float(base + row * Decimal(i)) <= body_high
                ]
                wick = [i for i in range(low_index, high_index + 1) if i not in body]
                body_weight = self.config.allocation.body_weight if body else 0.0
                wick_weight = self.config.allocation.wick_weight if wick else 0.0
                total_weight = body_weight + wick_weight
                indexes = body + wick
                weights = [body_weight / total_weight / len(body)] * len(body) + [wick_weight / total_weight / len(wick)] * len(wick)
            else:
                indexes = list(range(low_index, high_index + 1))
                weights = [1 / len(indexes)] * len(indexes)
            buy_ratio = 0.5
            if self.config.allocation.directional_approximation and candle.high > candle.low:
                buy_ratio = min(1.0, max(0.0, (candle.close - candle.low) / (candle.high - candle.low)))
            for index, weight in zip(indexes, weights, strict=True):
                allocated = candle.volume * weight
                volumes[index] += allocated
                buys[index] += allocated * buy_ratio
                sources[index] += 1
        included = sum(volumes)
        buckets = tuple(
            VolumeProfileBucket(
                id=stable_id("bucket", candles[0].symbol, candles[0].timeframe, base, row, index),
                index=index,
                lower=float(base + row * index),
                upper=float(base + row * (index + 1)),
                midpoint=float(base + row * (Decimal(index) + Decimal("0.5"))),
                volume=volume,
                estimated_buy_volume=buys[index],
                estimated_sell_volume=volume - buys[index],
                source_count=sources[index],
                upper_inclusive=index == count - 1,
            )
            for index, volume in enumerate(volumes)
        )
        return buckets, included, grid_method

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percentile))] if ordered else 0.0

    def _poc(self, buckets: tuple[VolumeProfileBucket, ...], total: float, profile_key: object) -> PointOfControl:
        mean_price = sum(x.midpoint * x.volume for x in buckets) / total
        chosen = min(buckets, key=lambda x: (-x.volume, abs(x.midpoint - mean_price), x.index))
        tied = sum(x.volume == chosen.volume for x in buckets)
        confidence = max(40.0, 100.0 - (tied - 1) * 15)
        return PointOfControl(
            id=stable_id("poc", profile_key, chosen.id),
            bucket_id=chosen.id,
            price=chosen.midpoint,
            volume=chosen.volume,
            volume_percent=chosen.volume / total,
            confidence_score=confidence,
        )

    def _value_area(self, buckets: tuple[VolumeProfileBucket, ...], poc: PointOfControl, total: float, profile_key: object) -> ValueArea:
        index = next(i for i, x in enumerate(buckets) if x.id == poc.bucket_id)
        included = {index}
        volume = buckets[index].volume
        target = total * self.config.value_area_percent
        lower = upper = index
        while volume + 1e-12 < target and len(included) < len(buckets):
            down = buckets[lower - 1].volume if lower > 0 else -1.0
            up = buckets[upper + 1].volume if upper + 1 < len(buckets) else -1.0
            if up > down:
                upper += 1
                selected = upper
            else:
                lower -= 1
                selected = lower
            included.add(selected)
            volume += buckets[selected].volume
        achieved = volume / total
        ids = tuple(buckets[i].id for i in sorted(included))
        return ValueArea(
            id=stable_id("value-area", profile_key, ids),
            method=ValueAreaMethod.POC_EXPANSION,
            target_percent=self.config.value_area_percent,
            achieved_percent=achieved,
            val=buckets[lower].lower,
            vah=buckets[upper].upper,
            included_bucket_ids=ids,
            included_volume=volume,
            overshoot_percent=max(0.0, achieved - self.config.value_area_percent),
            confidence_score=95,
        )

    def _nodes(self, buckets: tuple[VolumeProfileBucket, ...], profile_key: object) -> tuple[tuple[HighVolumeNode, ...], tuple[LowVolumeNode, ...]]:
        values = [x.volume for x in buckets]
        high = self._percentile(values, self.config.nodes.high_percentile)
        low = self._percentile(values, self.config.nodes.low_percentile)
        maximum = max(values, default=0)
        hvn_indexes = [
            i for i, x in enumerate(values) if x >= high and x > 0 and x >= (values[i - 1] if i else 0) and x >= (values[i + 1] if i + 1 < len(values) else 0)
        ]
        lvn_indexes = [i for i, x in enumerate(values) if x <= low and 0 < i < len(values) - 1 and x <= values[i - 1] and x <= values[i + 1]]
        return (
            tuple(HighVolumeNode.model_validate(self._node(NodeType.HVN, [i], buckets, profile_key, maximum)) for i in hvn_indexes),
            tuple(LowVolumeNode.model_validate(self._node(NodeType.LVN, [i], buckets, profile_key, maximum)) for i in lvn_indexes),
        )

    @staticmethod
    def _node(
        kind: NodeType, indexes: list[int], buckets: tuple[VolumeProfileBucket, ...], profile_key: object, maximum: float
    ) -> HighVolumeNode | LowVolumeNode:
        selected = [buckets[i] for i in indexes]
        peak = max(selected, key=lambda x: x.volume) if kind == NodeType.HVN else min(selected, key=lambda x: x.volume)
        values = [x.volume for x in selected]
        cls = HighVolumeNode if kind == NodeType.HVN else LowVolumeNode
        prominence = (peak.volume / maximum) if maximum else 0.0
        return cls(
            id=stable_id(kind, profile_key, *(x.id for x in selected)),
            lower=selected[0].lower,
            upper=selected[-1].upper,
            peak_price=peak.midpoint,
            total_volume=sum(values),
            mean_volume=fmean(values),
            prominence=prominence,
            bucket_ids=tuple(x.id for x in selected),
            confidence_score=min(100, 50 + prominence * 50),
            quality_score=90,
        )

    def _shelves_gaps(
        self, buckets: tuple[VolumeProfileBucket, ...], poc: PointOfControl, value_area: ValueArea, profile_key: object
    ) -> tuple[tuple[VolumeShelf, ...], tuple[VolumeGap, ...]]:
        mean = fmean([x.volume for x in buckets])
        shelf_indexes = [i for i, x in enumerate(buckets) if x.volume >= mean]
        shelves = []
        for group in self._groups(shelf_indexes):
            if len(group) >= self.config.nodes.shelf_minimum_width:
                selected = [buckets[i] for i in group]
                peak = max(selected, key=lambda x: x.volume)
                shelves.append(
                    VolumeShelf(
                        id=stable_id("shelf", profile_key, *group),
                        lower=selected[0].lower,
                        upper=selected[-1].upper,
                        peak_price=peak.midpoint,
                        mean_bucket_volume=fmean(x.volume for x in selected),
                        total_volume=sum(x.volume for x in selected),
                        width_bins=len(group),
                        prominence=peak.volume / mean if mean else 0,
                        contains_poc=any(x.id == poc.bucket_id for x in selected),
                        overlaps_value_area=selected[-1].upper >= value_area.val and selected[0].lower <= value_area.vah,
                        confidence_score=80,
                    )
                )
        gap_indexes = [
            i
            for i, x in enumerate(buckets)
            if 0 < i < len(buckets) - 1 and x.volume <= mean * self.config.nodes.gap_maximum_ratio and buckets[i - 1].volume > 0 and buckets[i + 1].volume > 0
        ]
        gaps = []
        for group in self._groups(gap_indexes):
            selected = [buckets[i] for i in group]
            strength = min(buckets[group[0] - 1].volume, buckets[group[-1] + 1].volume) / mean if mean else 0
            gaps.append(
                VolumeGap(
                    id=stable_id("gap", profile_key, *group),
                    lower=selected[0].lower,
                    upper=selected[-1].upper,
                    width_bins=len(group),
                    surrounding_strength=strength,
                    caused_by_missing_data=False,
                    confidence_score=min(100, 60 + strength * 10),
                    quality_score=90,
                )
            )
        return tuple(shelves), tuple(gaps)

    @staticmethod
    def _groups(indexes: list[int]) -> list[list[int]]:
        groups: list[list[int]] = []
        for index in indexes:
            if not groups or index > groups[-1][-1] + 1:
                groups.append([index])
            else:
                groups[-1].append(index)
        return groups

    def _shape(self, buckets: tuple[VolumeProfileBucket, ...], poc: PointOfControl, hvns: tuple[HighVolumeNode, ...], profile_key: object) -> ProfileShape:
        weights = [x.volume for x in buckets]
        total = sum(weights)
        mean = sum(x.midpoint * x.volume for x in buckets) / total
        variance = sum(x.volume * (x.midpoint - mean) ** 2 for x in buckets) / total
        sigma = sqrt(variance) if variance else 0
        skew = sum(x.volume * (x.midpoint - mean) ** 3 for x in buckets) / total / sigma**3 if sigma else 0
        kurtosis = sum(x.volume * (x.midpoint - mean) ** 4 for x in buckets) / total / sigma**4 - 3 if sigma else 0
        location = poc.bucket_id == buckets[0].id and 0.0 or next(i for i, x in enumerate(buckets) if x.id == poc.bucket_id) / max(1, len(buckets) - 1)
        concentration = max(weights) / total
        modes = len(hvns)
        elongation = len([x for x in buckets if x.volume > 0]) / max(1, len(buckets))
        candidates: list[ProfileShapeType]
        if modes >= 3:
            candidates = [ProfileShapeType.MULTIMODAL, ProfileShapeType.DOUBLE_DISTRIBUTION]
        elif modes == 2:
            candidates = [ProfileShapeType.DOUBLE_DISTRIBUTION, ProfileShapeType.MULTIMODAL]
        elif abs(skew) < 0.35 and 0.3 <= location <= 0.7:
            candidates = [ProfileShapeType.D_SHAPED, ProfileShapeType.UNDEFINED]
        elif skew < -0.45 and location > 0.55:
            candidates = [ProfileShapeType.P_SHAPED, ProfileShapeType.TREND]
        elif skew > 0.45 and location < 0.45:
            candidates = [ProfileShapeType.B_SHAPED, ProfileShapeType.TREND]
        elif elongation > 0.85 and concentration < 0.15:
            candidates = [ProfileShapeType.THIN, ProfileShapeType.TREND]
        elif abs(skew) > 0.7:
            candidates = [ProfileShapeType.TREND, ProfileShapeType.UNDEFINED]
        else:
            candidates = [ProfileShapeType.UNDEFINED, ProfileShapeType.D_SHAPED]
        confidence = min(95.0, 45 + abs(skew) * 20 + min(modes, 3) * 10)
        return ProfileShape(
            id=stable_id("shape", profile_key, candidates[0]),
            shape_type=candidates[0],
            alternative=candidates[1],
            features={
                "skewness": skew,
                "excess_kurtosis": kurtosis,
                "poc_location": location,
                "concentration": concentration,
                "mode_count": float(modes),
                "elongation": elongation,
            },
            conflicting_evidence=("allocation is candle-derived",),
            confidence_score=confidence,
            configuration_version=self.config.version,
        )

    def _profile(
        self,
        candles: list[Candle],
        kind: ProfileType,
        context: VolumeProfileContext,
        *,
        completed: bool,
        constituent_ids: tuple[object, ...] = (),
        anchor: object | None = None,
        session: SessionType | None = None,
    ) -> VolumeProfile | None:
        usable = [x for x in candles if self._is_usable_candle(x)]
        if not usable:
            return None
        buckets, included, grid_method = self._allocate(usable, context)
        if not buckets or included <= 0:
            return None
        source_total = sum(x.volume for x in candles if isfinite(x.volume) and x.volume >= 0)
        key = stable_id("logical-profile", candles[0].symbol, candles[0].timeframe, kind, candles[0].timestamp, getattr(anchor, "id", None), constituent_ids)
        poc = self._poc(buckets, included, key)
        value_area = self._value_area(buckets, poc, included, key)
        hvns, lvns = self._nodes(buckets, key)
        shelves, gaps = self._shelves_gaps(buckets, poc, value_area, key)
        shape = self._shape(buckets, poc, hvns, key)
        status = ProfileStatus.COMPLETED if completed else ProfileStatus.DEVELOPING
        lifecycle = ProfileLifecycleState.COMPLETED if completed else ProfileLifecycleState.DEVELOPING
        identifier = stable_id("profile-version", key, candles[-1].timestamp, included, self.config.version)
        quality = self._quality(candles, context.volume_source_type)
        return VolumeProfile(
            id=identifier,
            logical_id=key,
            symbol=candles[0].symbol.replace("/", ""),
            timeframe=candles[0].timeframe,
            instrument=context.instrument,
            profile_type=kind,
            session=session,
            status=status,
            lifecycle_state=lifecycle,
            start_timestamp=candles[0].timestamp,
            end_timestamp=candles[-1].timestamp,
            availability_timestamp=candles[-1].timestamp,
            completion_timestamp=candles[-1].timestamp if completed else None,
            source_candle_count=len(candles),
            source_first_timestamp=candles[0].timestamp,
            source_last_timestamp=candles[-1].timestamp,
            volume_source_type=context.volume_source_type,
            allocation_method=VolumeAllocationMethod(self.config.allocation.method),
            price_grid_method=grid_method,
            row_size=buckets[0].upper - buckets[0].lower,
            bucket_count=len(buckets),
            total_volume=source_total,
            included_volume=included,
            excluded_volume=max(0.0, source_total - included),
            buckets=buckets,
            poc=poc,
            value_area=value_area,
            hvns=hvns,
            lvns=lvns,
            shelves=shelves,
            gaps=gaps,
            shape=shape,
            anchor=anchor,
            constituent_profile_ids=tuple(constituent_ids),
            evidence=(
                VolumeProfileEvidence(
                    code="volume_conservation",
                    passed=abs(sum(x.volume for x in buckets) - included) < 1e-7,
                    value=included,
                    threshold=source_total,
                    explanation="Allocated bucket volume equals included source volume.",
                ),
            ),
            volume_data_quality=quality,
            confidence_score=min(quality.score, fmean([poc.confidence_score, shape.confidence_score])),
            quality_score=quality.score,
            configuration_version=self.config.version,
            engine_version=self.version,
            analysis_boundary=candles[-1].timestamp,
            version=len(candles),
        )

    def _period_profiles(
        self, candles: list[Candle], context: VolumeProfileContext, kind: ProfileType
    ) -> tuple[list[VolumeProfile], list[SkippedProfilePeriod]]:
        groups: dict[object, list[Candle]] = defaultdict(list)
        key: object
        for candle in candles:
            if kind == ProfileType.SESSION:
                session = self.sessions.session_at(candle.timestamp)
                if session in {MarketSession.CLOSED, MarketSession.WEEKEND, MarketSession.HOLIDAY}:
                    continue
                key = (candle.timestamp.date(), session.value)
            elif kind == ProfileType.DAILY:
                key = candle.timestamp.date()
            elif kind == ProfileType.WEEKLY:
                key = candle.timestamp.isocalendar()[:2]
            else:
                key = (candle.timestamp.year, candle.timestamp.month)
            groups[key].append(candle)
        ordered = sorted(groups.items(), key=lambda x: x[1][0].timestamp)
        result: list[VolumeProfile] = []
        skipped: list[SkippedProfilePeriod] = []
        boundary = candles[-1].timestamp if candles else datetime(1970, 1, 1, tzinfo=UTC)
        if not ordered:
            skipped.append(
                self._skipped_period(candles, context, kind, "no_eligible_period", ProfileSkipReason.EMPTY_PROFILE_PERIOD, boundary)
            )
            return result, skipped
        for index, (key, values) in enumerate(ordered[-3:]):
            profile_session: SessionType | None = None
            if kind == ProfileType.SESSION:
                session_name = str(key[1])  # type: ignore[index]
                profile_session = SessionType.OVERLAP if session_name == "london_new_york_overlap" else SessionType(session_name)
            profile = self._profile(values, kind, context, completed=index < len(ordered[-3:]) - 1, session=profile_session)
            if profile is None:
                skipped.append(
                    self._skipped_period(
                        values,
                        context,
                        kind,
                        str(key),
                        ProfileSkipReason.INSUFFICIENT_VOLUME_PROFILE_DATA,
                        boundary,
                    )
                )
                continue
            result.append(profile)
        return result, skipped

    @staticmethod
    def _tested_references(profiles: list[VolumeProfile], candles: list[Candle]) -> list[VolumeProfile]:
        result = []
        for profile in profiles:
            future = [x for x in candles if profile.completion_timestamp is not None and x.timestamp > profile.completion_timestamp]
            if not future:
                result.append(profile)
                continue
            tested_at: list[datetime] = []
            poc, area = profile.poc, profile.value_area
            if poc:
                hits = [x.timestamp for x in future if x.low <= poc.price <= x.high]
                if hits:
                    poc = poc.model_copy(update={"tested": True, "first_test_at": hits[0], "test_count": len(hits)})
                    tested_at.append(hits[0])
            if area:
                vah_hits = [x.timestamp for x in future if x.low <= area.vah <= x.high]
                val_hits = [x.timestamp for x in future if x.low <= area.val <= x.high]
                hits = sorted([*vah_hits, *val_hits])
                if hits:
                    area = area.model_copy(
                        update={"vah_tested": bool(vah_hits), "val_tested": bool(val_hits), "first_test_at": hits[0], "test_count": len(hits)}
                    )
                    tested_at.append(hits[0])

            def nodes(
                values: tuple[HighVolumeNode, ...] | tuple[LowVolumeNode, ...],
                visible: list[Candle] = future,
                test_times: list[datetime] = tested_at,
            ) -> tuple[HighVolumeNode | LowVolumeNode, ...]:
                updated = []
                for node in values:
                    hits = [x.timestamp for x in visible if x.low <= node.peak_price <= x.high]
                    updated.append(
                        node.model_copy(
                            update={
                                "tested": bool(hits),
                                "first_test_at": hits[0] if hits else None,
                                "test_count": len(hits),
                                "lifecycle_state": ProfileLifecycleState.TESTED if hits else node.lifecycle_state,
                            }
                        )
                    )
                    if hits:
                        test_times.append(hits[0])
                return tuple(updated)

            result.append(
                profile.model_copy(
                    update={
                        "poc": poc,
                        "value_area": area,
                        "hvns": nodes(profile.hvns),
                        "lvns": nodes(profile.lvns),
                        "lifecycle_state": ProfileLifecycleState.TESTED if tested_at else profile.lifecycle_state,
                    }
                )
            )
        return result

    @staticmethod
    def _migrations(profiles: list[VolumeProfile]) -> tuple[ProfileMigration, ...]:
        result = []
        by_type: dict[ProfileType, list[VolumeProfile]] = defaultdict(list)
        for profile in profiles:
            by_type[profile.profile_type].append(profile)
        for values in by_type.values():
            for previous, current in zip(values, values[1:], strict=False):
                if not previous.poc or not current.poc or not previous.value_area or not current.value_area:
                    continue
                change = current.poc.price - previous.poc.price
                normalized = change / current.row_size
                migration = (
                    ProfileMigrationType.STABLE if abs(normalized) < 0.5 else ProfileMigrationType.UPWARD if change > 0 else ProfileMigrationType.DOWNWARD
                )
                result.append(
                    ProfileMigration(
                        id=stable_id("migration", previous.id, current.id),
                        previous_profile_id=previous.id,
                        current_profile_id=current.id,
                        migration_type=migration,
                        poc_change=change,
                        vah_change=current.value_area.vah - previous.value_area.vah,
                        val_change=current.value_area.val - previous.value_area.val,
                        normalized_change=normalized,
                        bucket_change=round(normalized),
                        elapsed_seconds=max(0, (current.availability_timestamp - previous.availability_timestamp).total_seconds()),
                        available_at=current.availability_timestamp,
                        confidence_score=min(previous.confidence_score, current.confidence_score),
                        quality_score=min(previous.quality_score, current.quality_score),
                    )
                )
        return tuple(result)

    @staticmethod
    def _confluences(profiles: list[VolumeProfile], context: VolumeProfileContext) -> tuple[ProfileConfluence, ...]:
        result = []
        pocs = [x for x in profiles if x.poc]
        for i, first in enumerate(pocs):
            first_poc = first.poc
            assert first_poc is not None
            matches = [x for x in pocs[i + 1 :] if x.poc and abs(x.poc.price - first_poc.price) <= max(x.row_size, first.row_size)]
            if matches:
                all_profiles = [first, *matches]
                ids = (
                    tuple(str(x.poc.id) for x in all_profiles if x.poc)
                    + context.liquidity_source_ids
                    + tuple(x.id for x in context.smc.levels if x.available_at <= first.analysis_boundary)
                    if context.smc
                    else tuple(str(x.poc.id) for x in all_profiles if x.poc) + context.liquidity_source_ids
                )
                sources = tuple(
                    dict.fromkeys(
                        ["volume_profile", *("liquidity" for _ in context.liquidity_source_ids), *("smc" for _ in (context.smc.levels if context.smc else ()))]
                    )
                )
                adjustment = 1 / max(1, len(all_profiles))
                prices = [x.poc.price for x in all_profiles if x.poc is not None]
                result.append(
                    ProfileConfluence(
                        id=stable_id("confluence", *ids),
                        price=fmean(prices),
                        lower=min(price - profile.row_size / 2 for price, profile in zip(prices, all_profiles, strict=True)),
                        upper=max(price + profile.row_size / 2 for price, profile in zip(prices, all_profiles, strict=True)),
                        source_object_ids=tuple(str(x) for x in ids),
                        source_types=sources,
                        timeframe_count=len({x.timeframe for x in all_profiles}),
                        source_diversity=len(sources),
                        correlation_adjustment=adjustment,
                        confidence_score=min(100, 55 + 10 * len(sources)),
                        quality_score=min(x.quality_score for x in all_profiles),
                    )
                )
                break
        return tuple(result)
