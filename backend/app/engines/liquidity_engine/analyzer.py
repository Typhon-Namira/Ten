"""Pure, deterministic and replay-safe liquidity analysis."""

from abc import ABC
from collections import defaultdict
from datetime import UTC, datetime
from math import floor
from statistics import fmean, pstdev

from backend.app.engines.common import AnalysisEngine
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.engines.market_data_engine.sessions import MarketSessionEngine

from .config import LiquidityConfig
from .contracts import LiquidityContext
from .models import (
    AnalysisStatus,
    EqualLevelCluster,
    FalseBreak,
    LiquidityAnalysisSnapshot,
    LiquidityConfluence,
    LiquidityEvent,
    LiquidityEventType,
    LiquidityEvidence,
    LiquidityGrab,
    LiquidityLevel,
    LiquidityLevelType,
    LiquidityLifecycleState,
    LiquidityMapBand,
    LiquidityPool,
    LiquidityPoolType,
    LiquidityRaid,
    LiquidityResult,
    LiquidityScope,
    LiquiditySide,
    LiquiditySource,
    LiquidityState,
    LiquiditySweep,
    LiquidityTarget,
    MultiTimeframeLiquidityContext,
    ProcessingMode,
    ReferenceLiquidityLevel,
    SessionLiquidityRange,
    SessionType,
    StopHunt,
    SweepClassification,
    TargetStatus,
    stable_id,
)


class LiquidityAnalyzer(AnalysisEngine[LiquidityContext | list[Candle], LiquidityResult], ABC):
    """Domain contract independent of FastAPI, SQLAlchemy, and providers."""


class BaselineLiquidityAnalyzer(LiquidityAnalyzer):
    name = "liquidity"
    version = "1.0.0"

    def __init__(self, config: LiquidityConfig | None = None, sessions: MarketSessionEngine | None = None) -> None:
        self.config = config or LiquidityConfig()
        self.sessions = sessions or MarketSessionEngine()

    def analyze(self, data: LiquidityContext | list[Candle]) -> LiquidityResult:
        context = data if isinstance(data, LiquidityContext) else LiquidityContext(tuple(data))
        snapshot = self.analyze_snapshot(context)
        if snapshot.status == AnalysisStatus.INSUFFICIENT_HISTORY:
            return LiquidityResult(observations=["Insufficient candles supplied."], snapshot=snapshot)
        price = snapshot.state.latest_price
        buys = [p.upper_bound for p in snapshot.pools if p.side == LiquiditySide.BUY_SIDE and p.upper_bound >= price and _active(p)]
        sells = [p.lower_bound for p in snapshot.pools if p.side == LiquiditySide.SELL_SIDE and p.lower_bound <= price and _active(p)]
        return LiquidityResult(
            levels=list(snapshot.levels),
            nearest_buy_side=min(buys, default=None),
            nearest_sell_side=max(sells, default=None),
            active_session=snapshot.sessions[-1].session.value if snapshot.sessions else "unknown",
            snapshot=snapshot,
        )

    def analyze_snapshot(self, context: LiquidityContext, mode: ProcessingMode = ProcessingMode.HISTORICAL) -> LiquidityAnalysisSnapshot:
        candles = sorted(context.candles, key=lambda c: c.timestamp)[-self.config.processing.maximum_candles :]
        if not candles:
            return self._insufficient("UNKNOWN", Timeframe.M1, datetime(1970, 1, 1, tzinfo=UTC), 1, mode, 0)
        boundary = candles[-1].timestamp
        candles = [c for c in candles if c.timestamp <= boundary]
        symbol, timeframe = canonical_symbol(candles[-1].symbol), candles[-1].timeframe
        if len(candles) < self.config.processing.minimum_history:
            return self._insufficient(symbol, timeframe, boundary, candles[-1].close, mode, len(candles))
        atr = _atr(candles)
        equal = self._equal(candles, context, atr)
        references = self._references(candles, atr)
        sessions = self._sessions(candles)
        levels: list[LiquidityLevel] = [
            *equal,
            *references,
            *self._session_levels(sessions, candles, atr),
            *self._structural(context, boundary, atr),
            *self._round(candles, atr),
        ]
        levels = list({x.id: x for x in levels}.values())
        pools = self._pools(levels, candles[-1].close, boundary)
        inducements = self._inducements(levels, pools, candles[-1].close, boundary)
        if inducements:
            levels.extend(inducements)
            pools = self._pools(levels, candles[-1].close, boundary)
        events, sweeps, grabs, raids, hunts, false_breaks, pools = self._lifecycle(pools, candles, atr)
        confluences = self._confluences(pools, atr)
        targets = self._targets(pools, candles[-1].close, atr, boundary)
        ranks = {x.pool_id: x.rank for x in targets}
        pools = [x.model_copy(update={"target_rank": ranks.get(x.id)}) for x in pools]
        bands = tuple(self._band(x, candles[-1].close, boundary) for x in pools)
        state = LiquidityState(
            symbol=symbol,
            timeframe=timeframe,
            latest_price=candles[-1].close,
            active_pool_ids=tuple(x.id for x in pools if _active(x)),
            consumed_pool_ids=tuple(x.id for x in pools if x.lifecycle_state == LiquidityLifecycleState.CONSUMED),
            last_event_id=events[-1].id if events else None,
            updated_at=boundary,
        )
        quality = fmean(c.quality_score for c in candles)
        degraded = ("market_data_quality_below_threshold",) if quality < self.config.processing.minimum_input_quality else ()
        mtf = MultiTimeframeLiquidityContext(
            symbol=symbol,
            requested_timeframe=timeframe,
            pools_by_timeframe={timeframe.value: tuple(x.id for x in pools)},
            confluence_score=fmean(x.confidence_score for x in confluences) if confluences else 0,
            conflict_count=0,
            analyzed_through=boundary,
            maximum_depth=self.config.multi_timeframe.maximum_depth,
        )
        confidence = fmean(x.confidence_score for x in pools) if pools else 0
        return LiquidityAnalysisSnapshot(
            id=stable_id("snapshot", symbol, timeframe, boundary, self.config.version, mode),
            symbol=symbol,
            timeframe=timeframe,
            analysis_timestamp=boundary,
            market_data_boundary=f"{boundary.isoformat()}:{len(candles)}",
            processing_mode=mode,
            status=AnalysisStatus.DEGRADED if degraded else AnalysisStatus.COMPLETE,
            state=state,
            levels=tuple(levels),
            equal_levels=tuple(equal),
            pools=tuple(pools),
            events=tuple(events),
            sweeps=tuple(sweeps),
            grabs=tuple(grabs),
            raids=tuple(raids),
            stop_hunts=tuple(hunts),
            false_breaks=tuple(false_breaks),
            sessions=tuple(sessions),
            reference_levels=tuple(references),
            inducements=tuple(inducements),
            confluences=tuple(confluences),
            targets=tuple(targets),
            map_bands=bands,
            multi_timeframe=mtf,
            confidence_summary={"overall": confidence},
            quality_summary={"market_data": quality},
            degraded_reasons=degraded,
            configuration_version=self.config.version,
            created_at=boundary,
        )

    def _insufficient(self, symbol: str, timeframe: Timeframe, at: datetime, price: float, mode: ProcessingMode, count: int) -> LiquidityAnalysisSnapshot:
        state = LiquidityState(symbol=symbol, timeframe=timeframe, latest_price=price, updated_at=at)
        return LiquidityAnalysisSnapshot(
            id=stable_id("snapshot", symbol, timeframe, at, count),
            symbol=symbol,
            timeframe=timeframe,
            analysis_timestamp=at,
            market_data_boundary=f"{at.isoformat()}:{count}",
            processing_mode=mode,
            status=AnalysisStatus.INSUFFICIENT_HISTORY,
            state=state,
            configuration_version=self.config.version,
            created_at=at,
        )

    def _tolerance(self, price: float, atr: float) -> float:
        c = self.config.tolerances
        return max(c.absolute, c.ticks * c.tick_size, atr * c.atr_multiplier, price * c.percentage)

    def _equal(self, candles: list[Candle], context: LiquidityContext, atr: float) -> list[EqualLevelCluster]:
        if not self.config.equal_levels.enabled:
            return []
        candidates: dict[LiquiditySide, list[tuple[float, datetime, str, LiquidityScope, float, float]]] = defaultdict(list)
        if context.smc:
            for x in context.smc.levels:
                if x.available_at <= candles[-1].timestamp:
                    side = LiquiditySide.BUY_SIDE if "high" in x.kind else LiquiditySide.SELL_SIDE
                    scope = LiquidityScope.EXTERNAL if "external" in x.scope else LiquidityScope.INTERNAL
                    candidates[side].append((x.price, x.available_at, x.id, scope, x.confidence_score, x.quality_score))
        if self.config.equal_levels.allow_micro_candles:
            step = self.config.equal_levels.minimum_separation_candles
            for i, c in enumerate(candles[:-1]):
                if i % step == 0:
                    candidates[LiquiditySide.BUY_SIDE].append((c.high, c.timestamp, f"micro-high:{i}", LiquidityScope.INTERNAL, 50, c.quality_score))
                    candidates[LiquiditySide.SELL_SIDE].append((c.low, c.timestamp, f"micro-low:{i}", LiquidityScope.INTERNAL, 50, c.quality_score))
        result = []
        for side, values in candidates.items():
            groups: list[list[tuple[float, datetime, str, LiquidityScope, float, float]]] = []
            group_sums: list[float] = []
            for value in sorted(values):
                if groups and abs(value[0] - group_sums[-1] / len(groups[-1])) <= self._tolerance(value[0], atr) * self.config.equal_levels.merge_multiplier:
                    groups[-1].append(value)
                    group_sums[-1] += value[0]
                else:
                    groups.append([value])
                    group_sums.append(value[0])
            for group in groups:
                if len(group) < self.config.equal_levels.minimum_touches:
                    continue
                mean, deviation = fmean(x[0] for x in group), pstdev(x[0] for x in group)
                accepted = [x for x in group if deviation == 0 or abs(x[0] - mean) / deviation <= self.config.equal_levels.outlier_zscore]
                if len(accepted) < self.config.equal_levels.minimum_touches:
                    continue
                price, available = fmean(x[0] for x in accepted), max(x[1] for x in accepted)
                tolerance = self._tolerance(price, atr)
                source = LiquiditySource.SMC_SWING if any(not x[2].startswith("micro") for x in accepted) else LiquiditySource.MICRO_CANDLE
                score = min(100.0, 45 + len(accepted) * 10 + (10 if source == LiquiditySource.SMC_SWING else 0))
                kind = LiquidityLevelType.EQUAL_HIGH if side == LiquiditySide.BUY_SIDE else LiquidityLevelType.EQUAL_LOW
                result.append(
                    EqualLevelCluster(
                        id=stable_id("equal", canonical_symbol(candles[-1].symbol), candles[-1].timeframe, side, round(price, 8), *(x[2] for x in accepted)),
                        symbol=canonical_symbol(candles[-1].symbol),
                        timeframe=candles[-1].timeframe,
                        level_type=kind,
                        scope=LiquidityScope.EXTERNAL if any(x[3] == LiquidityScope.EXTERNAL for x in accepted) else LiquidityScope.INTERNAL,
                        side=side,
                        price=price,
                        lower_bound=price - tolerance,
                        upper_bound=price + tolerance,
                        source_timestamps=tuple(x[1] for x in accepted),
                        created_at=min(x[1] for x in accepted),
                        available_at=available,
                        confirmation_at=available,
                        confidence_score=score,
                        quality_score=fmean(x[5] for x in accepted),
                        strength_score=score,
                        freshness_score=90,
                        touch_count=len(accepted),
                        source=source,
                        evidence=(LiquidityEvidence(code="minimum_touches", observed_value=len(accepted), threshold=self.config.equal_levels.minimum_touches),),
                        source_object_ids=tuple(x[2] for x in accepted),
                        configuration_version=self.config.version,
                        analysis_boundary=candles[-1].timestamp,
                        member_prices=tuple(x[0] for x in accepted),
                        tolerance_used=tolerance,
                        temporal_separation_seconds=(available - min(x[1] for x in accepted)).total_seconds(),
                        outliers_rejected=len(group) - len(accepted),
                    )
                )
        return result

    def _structural(self, context: LiquidityContext, boundary: datetime, atr: float) -> list[LiquidityLevel]:
        result = []
        for x in context.smc.levels if context.smc else ():
            if x.available_at > boundary:
                continue
            side = LiquiditySide.BUY_SIDE if "high" in x.kind else LiquiditySide.SELL_SIDE
            price, tolerance = x.price, self._tolerance(x.price, atr)
            result.append(
                LiquidityLevel(
                    id=stable_id("smc-level", x.symbol, x.timeframe, x.id),
                    symbol=canonical_symbol(x.symbol),
                    timeframe=x.timeframe,
                    level_type=LiquidityLevelType.SWING_HIGH if side == LiquiditySide.BUY_SIDE else LiquidityLevelType.SWING_LOW,
                    scope=LiquidityScope.EXTERNAL if "external" in x.scope else LiquidityScope.INTERNAL,
                    side=side,
                    price=price,
                    lower_bound=price - tolerance,
                    upper_bound=price + tolerance,
                    source_timestamps=(x.occurred_at,),
                    created_at=x.occurred_at,
                    available_at=x.available_at,
                    confirmation_at=x.available_at,
                    confidence_score=x.confidence_score,
                    quality_score=x.quality_score,
                    strength_score=x.confidence_score,
                    freshness_score=90,
                    source=LiquiditySource.SMC_SWING,
                    source_object_ids=(x.id,),
                    configuration_version=self.config.version,
                    analysis_boundary=boundary,
                )
            )
        return result

    def _references(self, candles: list[Candle], atr: float) -> list[ReferenceLiquidityLevel]:
        groups: dict[tuple[str, object], list[Candle]] = {}
        for c in candles:
            iso = c.timestamp.isocalendar()
            for key in (("day", c.timestamp.date()), ("week", (iso.year, iso.week)), ("month", (c.timestamp.year, c.timestamp.month))):
                groups.setdefault(key, []).append(c)
        settings = {
            "day": (self.config.references.previous_day, LiquidityLevelType.PREVIOUS_DAY_HIGH, LiquidityLevelType.PREVIOUS_DAY_LOW),
            "week": (self.config.references.previous_week, LiquidityLevelType.PREVIOUS_WEEK_HIGH, LiquidityLevelType.PREVIOUS_WEEK_LOW),
            "month": (self.config.references.previous_month, LiquidityLevelType.PREVIOUS_MONTH_HIGH, LiquidityLevelType.PREVIOUS_MONTH_LOW),
        }
        result = []
        for period, (enabled, high_kind, low_kind) in settings.items():
            ordered = sorted((key, value) for key, value in groups.items() if key[0] == period)
            if not enabled or len(ordered) < 2:
                continue
            source, available = ordered[-2][1], ordered[-1][1][0].timestamp
            for side, kind, price in (
                (LiquiditySide.BUY_SIDE, high_kind, max(x.high for x in source)),
                (LiquiditySide.SELL_SIDE, low_kind, min(x.low for x in source)),
            ):
                tolerance = self._tolerance(price, atr)
                result.append(
                    ReferenceLiquidityLevel(
                        id=stable_id("reference", canonical_symbol(candles[-1].symbol), candles[-1].timeframe, kind, source[0].timestamp),
                        symbol=canonical_symbol(candles[-1].symbol),
                        timeframe=candles[-1].timeframe,
                        level_type=kind,
                        scope=LiquidityScope.PERIOD,
                        side=side,
                        price=price,
                        lower_bound=price - tolerance,
                        upper_bound=price + tolerance,
                        source_timestamps=tuple(x.timestamp for x in source),
                        created_at=source[-1].timestamp,
                        available_at=available,
                        confirmation_at=available,
                        confidence_score=80,
                        quality_score=fmean(x.quality_score for x in source),
                        strength_score=80,
                        freshness_score=85,
                        source=LiquiditySource.PERIOD,
                        configuration_version=self.config.version,
                        analysis_boundary=candles[-1].timestamp,
                        source_period_start=source[0].timestamp,
                        source_period_end=source[-1].timestamp,
                    )
                )
        if self.config.references.current_periods:
            current = (
                ("day", LiquidityLevelType.CURRENT_PERIOD_HIGH, LiquidityLevelType.CURRENT_PERIOD_LOW),
                ("week", LiquidityLevelType.CURRENT_PERIOD_HIGH, LiquidityLevelType.CURRENT_PERIOD_LOW),
                ("month", LiquidityLevelType.CURRENT_PERIOD_HIGH, LiquidityLevelType.CURRENT_PERIOD_LOW),
            )
            for period, high_kind, low_kind in current:
                ordered = sorted((key, value) for key, value in groups.items() if key[0] == period)
                source = ordered[-1][1]
                for side, kind, price in (
                    (LiquiditySide.BUY_SIDE, high_kind, max(x.high for x in source)),
                    (LiquiditySide.SELL_SIDE, low_kind, min(x.low for x in source)),
                ):
                    tolerance = self._tolerance(price, atr)
                    result.append(
                        ReferenceLiquidityLevel(
                            id=stable_id("current-reference", canonical_symbol(candles[-1].symbol), candles[-1].timeframe, period, kind, source[0].timestamp),
                            symbol=canonical_symbol(candles[-1].symbol),
                            timeframe=candles[-1].timeframe,
                            level_type=kind,
                            scope=LiquidityScope.PERIOD,
                            side=side,
                            price=price,
                            lower_bound=max(1e-12, price - tolerance),
                            upper_bound=price + tolerance,
                            source_timestamps=tuple(x.timestamp for x in source),
                            created_at=source[0].timestamp,
                            available_at=source[-1].timestamp,
                            confirmation_at=source[-1].timestamp,
                            confidence_score=60,
                            quality_score=fmean(x.quality_score for x in source),
                            strength_score=55,
                            freshness_score=100,
                            source=LiquiditySource.PERIOD,
                            evidence=(LiquidityEvidence(code="developing_period", observed_value=period),),
                            configuration_version=self.config.version,
                            analysis_boundary=candles[-1].timestamp,
                            source_period_start=source[0].timestamp,
                            source_period_end=source[-1].timestamp,
                        )
                    )
        return result

    def _session_levels(self, sessions: list[SessionLiquidityRange], candles: list[Candle], atr: float) -> list[LiquidityLevel]:
        result = []
        for session in sessions:
            for side, kind, price in (
                (LiquiditySide.BUY_SIDE, LiquidityLevelType.SESSION_HIGH, session.high),
                (LiquiditySide.SELL_SIDE, LiquidityLevelType.SESSION_LOW, session.low),
            ):
                tolerance = self._tolerance(price, atr)
                result.append(
                    LiquidityLevel(
                        id=stable_id("session-level", session.symbol, session.timeframe, session.id, side),
                        symbol=session.symbol,
                        timeframe=session.timeframe,
                        level_type=kind,
                        scope=LiquidityScope.SESSION,
                        side=side,
                        price=price,
                        lower_bound=max(1e-12, price - tolerance),
                        upper_bound=price + tolerance,
                        source_timestamps=(session.opened_at, session.available_at),
                        created_at=session.opened_at,
                        available_at=session.available_at,
                        confirmation_at=session.available_at if session.completed else None,
                        confidence_score=session.confidence_score,
                        quality_score=session.quality_score,
                        strength_score=70 if session.completed else 50,
                        freshness_score=100,
                        source=LiquiditySource.SESSION,
                        source_object_ids=(str(session.id),),
                        configuration_version=self.config.version,
                        analysis_boundary=candles[-1].timestamp,
                    )
                )
        return result

    def _round(self, candles: list[Candle], atr: float) -> list[LiquidityLevel]:
        if not self.config.round_numbers.enabled:
            return []
        symbol, timeframe, boundary, price = canonical_symbol(candles[-1].symbol), candles[-1].timeframe, candles[-1].timestamp, candles[-1].close
        increment = self.config.round_numbers.increments.get(symbol, self.config.round_numbers.increments["DEFAULT"]) / self.config.round_numbers.minor_divisor
        base = floor(price / increment) * increment
        result = []
        for value in (base - increment, base, base + increment, base + 2 * increment):
            if value <= 0 or abs(value - price) > atr * self.config.round_numbers.maximum_distance_atr:
                continue
            side, tolerance = (LiquiditySide.BUY_SIDE if value >= price else LiquiditySide.SELL_SIDE), max(self.config.tolerances.tick_size, increment * 0.01)
            result.append(
                LiquidityLevel(
                    id=stable_id("round", symbol, timeframe, value),
                    symbol=symbol,
                    timeframe=timeframe,
                    level_type=LiquidityLevelType.ROUND_NUMBER,
                    scope=LiquidityScope.INTERNAL,
                    side=side,
                    price=value,
                    lower_bound=value - tolerance,
                    upper_bound=value + tolerance,
                    source_timestamps=(boundary,),
                    created_at=boundary,
                    available_at=boundary,
                    confirmation_at=boundary,
                    confidence_score=self.config.round_numbers.confidence_cap,
                    quality_score=candles[-1].quality_score,
                    strength_score=45,
                    freshness_score=100,
                    source=LiquiditySource.ROUND_NUMBER,
                    configuration_version=self.config.version,
                    analysis_boundary=boundary,
                )
            )
        return result

    def _pools(self, levels: list[LiquidityLevel], price: float, boundary: datetime) -> list[LiquidityPool]:
        types = {
            LiquiditySource.SMC_SWING: LiquidityPoolType.STRUCTURAL,
            LiquiditySource.PERIOD: LiquidityPoolType.REFERENCE,
            LiquiditySource.ROUND_NUMBER: LiquidityPoolType.PSYCHOLOGICAL,
        }
        raw = [
            LiquidityPool(
                id=stable_id("pool", x.symbol, x.timeframe, x.id),
                symbol=x.symbol,
                timeframe=x.timeframe,
                pool_type=types.get(x.source, LiquidityPoolType.EQUAL_LEVEL),
                scope=x.scope,
                side=x.side,
                lower_bound=x.lower_bound,
                upper_bound=x.upper_bound,
                constituent_level_ids=(x.id,),
                touch_count=x.touch_count,
                created_at=x.created_at,
                available_at=x.available_at,
                lifecycle_state=LiquidityLifecycleState.ACTIVE,
                confidence_score=x.confidence_score,
                quality_score=x.quality_score,
                strength_score=x.strength_score,
                freshness_score=x.freshness_score,
                distance_from_price=(x.price - price) / price * 100,
                evidence=x.evidence,
                configuration_version=self.config.version,
                analysis_boundary=boundary,
            )
            for x in levels[-self.config.pools.maximum_active :]
        ]
        merged: list[LiquidityPool] = []
        for pool in sorted(raw, key=lambda item: (item.side.value, item.lower_bound, str(item.id))):
            prior = merged[-1] if merged else None
            if prior and prior.side == pool.side and prior.scope == pool.scope and pool.lower_bound <= prior.upper_bound:
                constituents = tuple(dict.fromkeys((*prior.constituent_level_ids, *pool.constituent_level_ids)))
                merged[-1] = prior.model_copy(
                    update={
                        "id": stable_id("composite-pool", pool.symbol, pool.timeframe, *constituents),
                        "pool_type": LiquidityPoolType.COMPOSITE,
                        "lower_bound": min(prior.lower_bound, pool.lower_bound),
                        "upper_bound": max(prior.upper_bound, pool.upper_bound),
                        "constituent_level_ids": constituents,
                        "touch_count": prior.touch_count + pool.touch_count,
                        "confidence_score": min(100, (prior.confidence_score + pool.confidence_score) / 2 + 5),
                        "quality_score": min(prior.quality_score, pool.quality_score),
                        "strength_score": min(100, max(prior.strength_score, pool.strength_score) + 5),
                    }
                )
            else:
                merged.append(pool)
        return merged

    def _inducements(self, levels: list[LiquidityLevel], pools: list[LiquidityPool], price: float, boundary: datetime) -> list[LiquidityLevel]:
        result = []
        by_id = {item.id: item for item in levels}
        for external in (item for item in pools if item.scope == LiquidityScope.EXTERNAL):
            center = (external.lower_bound + external.upper_bound) / 2
            candidates = [
                item
                for item in pools
                if item.scope == LiquidityScope.INTERNAL
                and item.side == external.side
                and (
                    (price < (item.lower_bound + item.upper_bound) / 2 < center)
                    if item.side == LiquiditySide.BUY_SIDE
                    else (center < (item.lower_bound + item.upper_bound) / 2 < price)
                )
            ]
            if not candidates:
                continue
            candidate = min(candidates, key=lambda item: abs((item.lower_bound + item.upper_bound) / 2 - price))
            source = by_id[candidate.constituent_level_ids[0]]
            result.append(
                source.model_copy(
                    update={
                        "id": stable_id("inducement", source.symbol, source.timeframe, source.id, external.id),
                        "level_type": LiquidityLevelType.INDUCEMENT,
                        "source_object_ids": (*source.source_object_ids, str(external.id)),
                        "confidence_score": min(75, source.confidence_score),
                        "strength_score": min(70, source.strength_score),
                        "evidence": (
                            *source.evidence,
                            LiquidityEvidence(code="larger_external_target_beyond_minor_liquidity", source_ids=(str(external.id),), passed=True),
                        ),
                        "analysis_boundary": boundary,
                    }
                )
            )
        return result

    def _lifecycle(
        self, pools: list[LiquidityPool], candles: list[Candle], atr: float
    ) -> tuple[list[LiquidityEvent], list[LiquiditySweep], list[LiquidityGrab], list[LiquidityRaid], list[StopHunt], list[FalseBreak], list[LiquidityPool]]:
        events: list[LiquidityEvent] = []
        sweeps: list[LiquiditySweep] = []
        grabs: list[LiquidityGrab] = []
        hunts: list[StopHunt] = []
        false_breaks: list[FalseBreak] = []
        updated = []
        for pool in pools:
            later = [c for c in candles if c.timestamp > pool.available_at]
            penetrated = [c for c in later if (c.high > pool.upper_bound if pool.side == LiquiditySide.BUY_SIDE else c.low < pool.lower_bound)]
            if not penetrated:
                age = sum(c.timestamp > pool.available_at for c in candles)
                touched = next((c for c in later if c.high >= pool.lower_bound and c.low <= pool.upper_bound), None)
                distance = min((min(abs(c.close - pool.lower_bound), abs(c.close - pool.upper_bound)) for c in later), default=float("inf"))
                if age >= self.config.pools.expiration_candles:
                    state = LiquidityLifecycleState.EXPIRED
                elif touched:
                    state = LiquidityLifecycleState.TOUCHED
                elif distance <= atr * self.config.pools.approach_atr:
                    state = LiquidityLifecycleState.APPROACHED
                else:
                    state = LiquidityLifecycleState.ACTIVE
                updated.append(pool.model_copy(update={"lifecycle_state": state, "version": pool.version + (state != LiquidityLifecycleState.ACTIVE)}))
                continue
            candle = penetrated[0]
            distance = candle.high - pool.upper_bound if pool.side == LiquiditySide.BUY_SIDE else pool.lower_bound - candle.low
            reclaimed = candle.close < pool.lower_bound if pool.side == LiquiditySide.BUY_SIDE else candle.close > pool.upper_bound
            band_width = max(pool.upper_bound - pool.lower_bound, self.config.tolerances.tick_size)
            partial = distance < band_width and not reclaimed
            state = LiquidityLifecycleState.SWEPT if reclaimed else LiquidityLifecycleState.PARTIALLY_SWEPT if partial else LiquidityLifecycleState.CONSUMED
            confidence = min(100.0, pool.confidence_score * 0.6 + min(30, distance / max(atr, 1e-12) * 20) + (20 if reclaimed else 5))
            penetration_percentage = distance / max(pool.upper_bound - pool.lower_bound, self.config.tolerances.tick_size) * 100
            common = dict(
                symbol=pool.symbol,
                timeframe=pool.timeframe,
                pool_id=pool.id,
                side=pool.side,
                occurred_at=candle.timestamp,
                available_at=candle.timestamp,
                price=candle.close,
                lifecycle_from=LiquidityLifecycleState.ACTIVE,
                lifecycle_to=state,
                confidence_score=confidence,
                quality_score=candle.quality_score,
                evidence=(LiquidityEvidence(code="pool_penetration", observed_value=distance),),
                configuration_version=self.config.version,
                classification=SweepClassification.WICK_ONLY if reclaimed else SweepClassification.CLOSE_THROUGH,
                penetration_distance=distance,
                penetration_percentage=penetration_percentage,
                wick_penetration=distance,
                close_penetration=0,
                time_outside_seconds=0,
                reclaim_timestamp=candle.timestamp if reclaimed else None,
                reclaim_strength=confidence if reclaimed else 0,
            )
            sweep = LiquiditySweep(id=stable_id("sweep", pool.symbol, pool.timeframe, pool.id, candle.timestamp), event_type=LiquidityEventType.SWEEP, **common)
            sweeps.append(sweep)
            events.append(sweep)
            if reclaimed:
                grab = LiquidityGrab(
                    id=stable_id("grab", pool.symbol, pool.timeframe, pool.id, candle.timestamp),
                    event_type=LiquidityEventType.GRAB,
                    rejection_candles=1,
                    **common,
                )
                grabs.append(grab)
                events.append(grab)
                if confidence >= self.config.sweeps.stop_hunt_minimum_confidence:
                    hunt = StopHunt(
                        id=stable_id("hunt", pool.symbol, pool.timeframe, pool.id, candle.timestamp), event_type=LiquidityEventType.STOP_HUNT, **common
                    )
                    hunts.append(hunt)
                    events.append(hunt)
            else:
                following = later[later.index(candle) + 1 : later.index(candle) + 1 + self.config.sweeps.reclaim_candles]
                reclaimed_at = next(
                    (c for c in following if (c.close < pool.lower_bound if pool.side == LiquiditySide.BUY_SIDE else c.close > pool.upper_bound)), None
                )
                if reclaimed_at:
                    payload = dict(
                        common, available_at=reclaimed_at.timestamp, reclaim_timestamp=reclaimed_at.timestamp, lifecycle_to=LiquidityLifecycleState.RECLAIMED
                    )
                    false = FalseBreak(
                        id=stable_id("false", pool.symbol, pool.timeframe, pool.id, reclaimed_at.timestamp),
                        event_type=LiquidityEventType.FALSE_BREAK,
                        held_outside_candles=following.index(reclaimed_at) + 1,
                        **payload,
                    )
                    false_breaks.append(false)
                    events.append(false)
                    state = LiquidityLifecycleState.RECLAIMED
            updated.append(pool.model_copy(update={"lifecycle_state": state, "sweep_percentage": min(100, penetration_percentage), "version": 2}))
        raids: list[LiquidityRaid] = []
        grouped: dict[datetime, list[LiquiditySweep]] = defaultdict(list)
        for sweep in sweeps:
            grouped[sweep.occurred_at].append(sweep)
        for at, items in grouped.items():
            if len(items) >= self.config.sweeps.raid_minimum_pools:
                first = items[0]
                payload = first.model_dump(exclude={"id", "event_type", "classification"})
                raid = LiquidityRaid(
                    id=stable_id("raid", first.symbol, first.timeframe, at, *(x.pool_id for x in items)),
                    event_type=LiquidityEventType.RAID,
                    classification=SweepClassification.FULL,
                    consumed_pool_ids=tuple(x.pool_id for x in items),
                    **payload,
                )
                raids.append(raid)
                events.append(raid)
        return events, sweeps, grabs, raids, hunts, false_breaks, updated

    def _sessions(self, candles: list[Candle]) -> list[SessionLiquidityRange]:
        groups: list[tuple[SessionType, list[Candle]]] = []
        for candle in candles:
            session = SessionType(self.sessions.session_at(candle.timestamp).value)
            if groups and groups[-1][0] == session:
                groups[-1][1].append(candle)
            else:
                groups.append((session, [candle]))
        result = []
        for index, (session, items) in enumerate(groups):
            if session in {SessionType.CLOSED, SessionType.WEEKEND, SessionType.HOLIDAY}:
                continue
            high, low = max(x.high for x in items), min(x.low for x in items)
            result.append(
                SessionLiquidityRange(
                    id=stable_id("session", canonical_symbol(items[0].symbol), items[0].timeframe, session, items[0].timestamp),
                    symbol=canonical_symbol(items[0].symbol),
                    timeframe=items[0].timeframe,
                    session=session,
                    opened_at=items[0].timestamp,
                    available_at=items[-1].timestamp,
                    high=high,
                    low=low,
                    midpoint=(high + low) / 2,
                    completed=index < len(groups) - 1,
                    source_candle_ids=tuple(x.timestamp.isoformat() for x in items),
                    confidence_score=75,
                    quality_score=fmean(x.quality_score for x in items),
                    configuration_version=self.config.version,
                )
            )
        return result

    def _confluences(self, pools: list[LiquidityPool], atr: float) -> list[LiquidityConfluence]:
        buckets: dict[tuple[LiquiditySide, int], list[LiquidityPool]] = defaultdict(list)
        width = max(atr * 0.1, self.config.tolerances.tick_size)
        for pool in pools:
            buckets[(pool.side, floor(((pool.lower_bound + pool.upper_bound) / 2) / width))].append(pool)
        result = []
        for items in buckets.values():
            sources = {x.pool_type for x in items}
            if len(items) >= 2 and len(sources) >= 2:
                first = items[0]
                confidence = min(100, fmean(x.confidence_score for x in items) + (len(sources) - 1) * 8)
                result.append(
                    LiquidityConfluence(
                        id=stable_id("confluence", first.symbol, first.timeframe, *(x.id for x in items)),
                        symbol=first.symbol,
                        timeframe=first.timeframe,
                        lower_bound=min(x.lower_bound for x in items),
                        upper_bound=max(x.upper_bound for x in items),
                        contributing_source_ids=tuple(str(x.id) for x in items),
                        source_diversity=len(sources),
                        timeframe_diversity=1,
                        agreement_score=confidence,
                        confidence_score=confidence,
                        quality_score=fmean(x.quality_score for x in items),
                        lifecycle_state=LiquidityLifecycleState.ACTIVE,
                        available_at=max(x.available_at for x in items),
                        configuration_version=self.config.version,
                    )
                )
        return result

    def _targets(self, pools: list[LiquidityPool], price: float, atr: float, boundary: datetime) -> list[LiquidityTarget]:
        active = [x for x in pools if _active(x)]
        scored = []
        for pool in active:
            distance = abs((pool.lower_bound + pool.upper_bound) / 2 - price)
            intervening = sum(1 for x in active if x.side == pool.side and abs((x.lower_bound + x.upper_bound) / 2 - price) < distance)
            access = max(0, 100 - distance / max(atr, 1e-12) * 5 - intervening * 10)
            scope = 100 if pool.scope == LiquidityScope.EXTERNAL else 60
            priority = (
                access * self.config.ranking.distance_weight
                + pool.strength_score * self.config.ranking.strength_weight
                + pool.freshness_score * self.config.ranking.freshness_weight
                + scope * self.config.ranking.scope_weight
                + pool.quality_score * self.config.ranking.quality_weight
            )
            scored.append((priority, pool, distance, intervening, access))
        scored.sort(key=lambda x: (-x[0], str(x[1].id)))
        return [
            LiquidityTarget(
                id=stable_id("target", p.symbol, p.timeframe, p.id, boundary),
                pool_id=p.id,
                symbol=p.symbol,
                timeframe=p.timeframe,
                side=p.side,
                rank=i,
                relative_distance=d / price * 100,
                strength_score=p.strength_score,
                accessibility_score=a,
                path_obstruction_score=min(100, n * 20),
                intermediate_pool_count=n,
                confidence_score=min(100, score),
                status=TargetStatus.ACTIVE,
                invalidation_condition="pool becomes terminal",
                available_at=boundary,
                configuration_version=self.config.version,
            )
            for i, (score, p, d, n, a) in enumerate(scored, 1)
        ]

    @staticmethod
    def _band(pool: LiquidityPool, price: float, boundary: datetime) -> LiquidityMapBand:
        return LiquidityMapBand(
            lower_bound=pool.lower_bound,
            upper_bound=pool.upper_bound,
            side=pool.side,
            inferred_density=min(100, pool.touch_count * 20),
            source_count=len(pool.constituent_level_ids),
            weighted_strength=pool.strength_score,
            timeframe_composition=(pool.timeframe.value,),
            active=_active(pool),
            distance_from_price=((pool.lower_bound + pool.upper_bound) / 2 - price) / price * 100,
            confidence_score=pool.confidence_score,
            age_seconds=max(0, (boundary - pool.available_at).total_seconds()),
        )


def _atr(candles: list[Candle]) -> float:
    pairs = list(zip(candles[-15:-1], candles[-14:], strict=False))
    return (
        fmean(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)) for p, c in pairs) if pairs else max(candles[-1].high - candles[-1].low, 1e-12)
    )


def _active(pool: LiquidityPool) -> bool:
    return pool.lifecycle_state not in {
        LiquidityLifecycleState.CONSUMED,
        LiquidityLifecycleState.INVALIDATED,
        LiquidityLifecycleState.EXPIRED,
        LiquidityLifecycleState.ARCHIVED,
    }
