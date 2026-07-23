from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from backend.app.core.bounded import BoundedSet
from backend.app.engines.market_data_engine import MarketDataService, Timeframe
from backend.app.engines.smc_engine.liquidity_contract import SMCLiquidityReader
from backend.app.events import Event, EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .analyzer import BaselineLiquidityAnalyzer
from .config import LiquidityConfig
from .contracts import LiquidityContext
from .events import (
    EqualHighClusterConfirmed,
    EqualLowClusterConfirmed,
    FalseBreakConfirmed,
    LiquidityAnalysisUpdated,
    LiquidityCheckpointRecovered,
    LiquidityConfluenceUpdated,
    LiquidityGrabDetected,
    LiquidityInputDegraded,
    LiquidityPoolApproached,
    LiquidityPoolConsumed,
    LiquidityPoolCreated,
    LiquidityPoolExpired,
    LiquidityPoolPartiallySwept,
    LiquidityPoolSwept,
    LiquidityPoolTouched,
    LiquidityRaidDetected,
    LiquidityReplayCompleted,
    LiquidityTargetRankingUpdated,
    ReferenceLiquidityCreated,
    SessionLiquidityUpdated,
    StopHuntClassified,
)
from .models import AnalysisStatus, LiquidityAnalysisSnapshot, LiquidityLifecycleState, MultiTimeframeLiquidityContext, ProcessingMode, stable_id
from .repository import InMemoryLiquidityRepository, LiquidityRepository


class LiquidityMetrics:
    def __init__(self) -> None:
        self.analyses_completed = 0
        self.candles_processed = 0
        self.smc_references_consumed = 0
        self.equal_high_clusters = 0
        self.equal_low_clusters = 0
        self.pools_created = 0
        self.pools_active = 0
        self.pools_swept = 0
        self.pools_consumed = 0
        self.grabs = 0
        self.raids = 0
        self.stop_hunts = 0
        self.false_breaks = 0
        self.reclaims = 0
        self.session_updates = 0
        self.reference_levels = 0
        self.targets_ranked = 0
        self.degraded_input_analyses = 0
        self.replay_runs = 0
        self.checkpoint_recoveries = 0
        self.persistence_failures = 0
        self.event_publication_failures = 0
        self.average_analysis_latency_ms = 0.0
        self.latest_successful_analysis_timestamp: str | None = None

    def snapshot(self) -> dict[str, int | float | str | None]:
        return dict(vars(self))


class LiquidityService:
    def __init__(
        self,
        market_data: MarketDataService,
        smc: SMCLiquidityReader,
        event_bus: EventBus,
        feature_store: FeatureStore,
        config: LiquidityConfig | None = None,
        repository: LiquidityRepository | None = None,
        repository_mode: str = "memory",
    ) -> None:
        self.market_data = market_data
        self.smc = smc
        self.event_bus = event_bus
        self.feature_store = feature_store
        self.config = config or LiquidityConfig()
        self.repository = repository or InMemoryLiquidityRepository()
        self.repository_mode = repository_mode
        self.analyzer = BaselineLiquidityAnalyzer(self.config, market_data.sessions)
        self.metrics = LiquidityMetrics()
        self._published = BoundedSet[UUID](10_000)
        self._recovered: dict[tuple[str, Timeframe], LiquidityAnalysisSnapshot] = {}
        self.recovery_status = "not_attempted"

    async def restore(self) -> int:
        items = await self.repository.checkpoints()
        self._recovered = {(x.symbol, x.timeframe): x for x in items}
        self.metrics.checkpoint_recoveries += len(items)
        self.recovery_status = "recovered" if items else "clean_start"
        if items:
            await self.event_bus.publish(LiquidityCheckpointRecovered(correlation_id=uuid4(), source="liquidity", payload={"count": len(items)}))
        return len(items)

    async def analyze(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
        mode: ProcessingMode = ProcessingMode.HISTORICAL,
        correlation_id: UUID | None = None,
    ) -> LiquidityAnalysisSnapshot:
        candles = await self.market_data.history(symbol, timeframe, start=start, end=end, limit=min(limit, self.config.processing.maximum_candles))
        smc = await self.smc.liquidity_context(symbol, timeframe, candles[-1].timestamp) if candles else None
        return await self.analyze_context(LiquidityContext(tuple(candles), smc), mode, correlation_id)

    async def analyze_context(
        self, context: LiquidityContext, mode: ProcessingMode = ProcessingMode.HISTORICAL, correlation_id: UUID | None = None
    ) -> LiquidityAnalysisSnapshot:
        started = perf_counter()
        snapshot = self.analyzer.analyze_snapshot(context, mode)
        try:
            await self.repository.save(snapshot)
        except Exception:
            self.metrics.persistence_failures += 1
            raise
        latency = (perf_counter() - started) * 1000
        self._record(snapshot, len(context.candles), len(context.smc.levels) if context.smc else 0, latency)
        await self._publish(snapshot, correlation_id or uuid4())
        return snapshot

    async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, limit: int = 500) -> LiquidityAnalysisSnapshot:
        candles = await self.market_data.replay(symbol, timeframe, timestamp, limit=limit)
        smc = await self.smc.liquidity_context(symbol, timeframe, timestamp)
        snapshot = await self.analyze_context(LiquidityContext(tuple(candles), smc), ProcessingMode.REPLAY)
        self.metrics.replay_runs += 1
        await self.event_bus.publish(LiquidityReplayCompleted(correlation_id=uuid4(), source="liquidity", payload={"snapshot_id": str(snapshot.id)}))
        return snapshot

    async def state(self, symbol: str, timeframe: Timeframe, at: datetime | None = None) -> LiquidityAnalysisSnapshot | None:
        return await self.repository.at(symbol, timeframe, at) if at else await self.repository.latest(symbol, timeframe)

    async def multi_timeframe(self, symbol: str, timeframe: Timeframe, at: datetime | None = None, limit: int = 500) -> MultiTimeframeLiquidityContext:
        names = self.config.multi_timeframe.hierarchy[: self.config.multi_timeframe.maximum_depth]
        pools: dict[str, tuple[UUID, ...]] = {}
        through = at
        confidence: list[float] = []
        for name in names:
            try:
                item = Timeframe(name)
            except ValueError:
                pools[name] = ()
                continue
            try:
                snapshot = await self.replay(symbol, item, at, limit) if at else await self.analyze(symbol, item, limit=limit)
            except Exception:
                continue
            pools[name] = tuple(
                pool.id
                for pool in snapshot.pools
                if pool.lifecycle_state not in {LiquidityLifecycleState.CONSUMED, LiquidityLifecycleState.INVALIDATED, LiquidityLifecycleState.EXPIRED}
            )
            confidence.extend(pool.confidence_score for pool in snapshot.pools)
            through = snapshot.analysis_timestamp if through is None else min(through, snapshot.analysis_timestamp)
        ids = [identifier for values in pools.values() for identifier in values]
        context = MultiTimeframeLiquidityContext(
            symbol=symbol,
            requested_timeframe=timeframe,
            pools_by_timeframe=pools,
            nested_pool_ids=tuple(dict.fromkeys(ids)),
            confluence_score=sum(confidence) / len(confidence) if confidence else 0,
            conflict_count=0,
            analyzed_through=through or datetime(1970, 1, 1, tzinfo=UTC),
            maximum_depth=self.config.multi_timeframe.maximum_depth,
        )
        return context

    def health(self) -> dict[str, object]:
        degraded = self.repository_mode == "memory" and self.config.persistence.required_in_production
        return {
            "status": "degraded" if degraded else "healthy",
            "engine_version": self.analyzer.version,
            "configuration_version": self.config.version,
            "repository_mode": self.repository_mode,
            "database_status": "available" if self.repository_mode == "sqlalchemy" else "unavailable",
            "checkpoint_recovery_status": self.recovery_status,
            "market_data_dependency": "configured",
            "smc_contract_status": "configured",
            "last_analysis_timestamp": self.metrics.latest_successful_analysis_timestamp,
            "degraded_reasons": ["ephemeral_persistence"] if degraded else [],
            "checked_at": datetime.now(UTC),
        }

    def _record(self, s: LiquidityAnalysisSnapshot, candles: int, refs: int, latency: float) -> None:
        m = self.metrics
        old = m.analyses_completed
        m.analyses_completed += 1
        m.candles_processed += candles
        m.smc_references_consumed += refs
        m.equal_high_clusters += sum(x.side.value == "buy_side" for x in s.equal_levels)
        m.equal_low_clusters += sum(x.side.value == "sell_side" for x in s.equal_levels)
        m.pools_created += len(s.pools)
        m.pools_active = len(s.state.active_pool_ids)
        m.pools_swept += len(s.sweeps)
        m.pools_consumed += len(s.state.consumed_pool_ids)
        m.grabs += len(s.grabs)
        m.raids += len(s.raids)
        m.stop_hunts += len(s.stop_hunts)
        m.false_breaks += len(s.false_breaks)
        m.reclaims += sum(x.reclaim_timestamp is not None for x in s.sweeps)
        m.session_updates += len(s.sessions)
        m.reference_levels += len(s.reference_levels)
        m.targets_ranked += len(s.targets)
        m.degraded_input_analyses += s.status == AnalysisStatus.DEGRADED
        m.average_analysis_latency_ms = (m.average_analysis_latency_ms * old + latency) / m.analyses_completed
        m.latest_successful_analysis_timestamp = s.analysis_timestamp.isoformat()

    async def _publish(self, s: LiquidityAnalysisSnapshot, correlation_id: UUID) -> None:
        if s.id in self._published:
            return
        await self.feature_store.write(
            FeatureRecord(
                correlation_id=correlation_id,
                namespace="liquidity",
                engine_name="liquidity",
                engine_version=s.engine_version,
                compatibility_version="1.0",
                values=self.features(s),
            )
        )
        event = LiquidityInputDegraded if s.status == AnalysisStatus.DEGRADED else LiquidityAnalysisUpdated
        try:
            common = {
                "symbol": s.symbol,
                "timeframe": s.timeframe.value,
                "event_timestamp": s.analysis_timestamp.isoformat(),
                "availability_timestamp": s.analysis_timestamp.isoformat(),
                "configuration_version": s.configuration_version,
                "engine_version": s.engine_version,
            }
            publications: list[tuple[type[Event], Any]] = []
            publications.extend((EqualHighClusterConfirmed if item.side.value == "buy_side" else EqualLowClusterConfirmed, item) for item in s.equal_levels)
            pool_events = {
                LiquidityLifecycleState.APPROACHED: LiquidityPoolApproached,
                LiquidityLifecycleState.TOUCHED: LiquidityPoolTouched,
                LiquidityLifecycleState.PARTIALLY_SWEPT: LiquidityPoolPartiallySwept,
                LiquidityLifecycleState.SWEPT: LiquidityPoolSwept,
                LiquidityLifecycleState.CONSUMED: LiquidityPoolConsumed,
                LiquidityLifecycleState.EXPIRED: LiquidityPoolExpired,
            }
            publications.extend((pool_events.get(item.lifecycle_state, LiquidityPoolCreated), item) for item in s.pools)
            for event_class, items in (
                (LiquidityGrabDetected, s.grabs),
                (LiquidityRaidDetected, s.raids),
                (StopHuntClassified, s.stop_hunts),
                (FalseBreakConfirmed, s.false_breaks),
                (SessionLiquidityUpdated, s.sessions),
                (ReferenceLiquidityCreated, s.reference_levels),
                (LiquidityConfluenceUpdated, s.confluences),
                (LiquidityTargetRankingUpdated, s.targets),
            ):
                publications.extend((event_class, item) for item in items)
            for publication_class, item in publications:
                payload = {
                    **common,
                    "source_object_ids": [str(item.id)],
                    "confidence": getattr(item, "confidence_score", 0),
                    "object": item.model_dump(mode="json"),
                }
                await self.event_bus.publish(
                    publication_class(
                        event_id=stable_id("liquidity-event", s.symbol, s.timeframe, publication_class.__name__, item.id),
                        correlation_id=correlation_id,
                        source="liquidity",
                        payload=payload,
                    )
                )
            await self.event_bus.publish(
                event(
                    event_id=stable_id("liquidity-event", s.symbol, s.timeframe, event.__name__, s.id),
                    correlation_id=correlation_id,
                    source="liquidity",
                    payload={**common, "snapshot_id": str(s.id), "status": s.status.value},
                )
            )
        except Exception:
            self.metrics.event_publication_failures += 1
        self._published.add(s.id)

    @staticmethod
    def features(s: LiquidityAnalysisSnapshot) -> dict[str, object]:
        buys = [x for x in s.targets if x.side.value == "buy_side"]
        sells = [x for x in s.targets if x.side.value == "sell_side"]
        buy_pools = [x for x in s.pools if x.side.value == "buy_side"]
        sell_pools = [x for x in s.pools if x.side.value == "sell_side"]
        return {
            "nearest_buy_side_liquidity": buys[0].model_dump(mode="json") if buys else None,
            "nearest_sell_side_liquidity": sells[0].model_dump(mode="json") if sells else None,
            "active_equal_highs": [x.model_dump(mode="json") for x in s.equal_levels if x.side.value == "buy_side"],
            "active_equal_lows": [x.model_dump(mode="json") for x in s.equal_levels if x.side.value == "sell_side"],
            "reference_levels": [x.model_dump(mode="json") for x in s.reference_levels],
            "session_ranges": [x.model_dump(mode="json") for x in s.sessions],
            "latest_sweep": s.sweeps[-1].model_dump(mode="json") if s.sweeps else None,
            "latest_raid": s.raids[-1].model_dump(mode="json") if s.raids else None,
            "latest_reclaim": next((x.model_dump(mode="json") for x in reversed(s.sweeps) if x.reclaim_timestamp), None),
            "strongest_buy_side_pool": max(buy_pools, key=lambda x: x.strength_score).model_dump(mode="json") if buy_pools else None,
            "strongest_sell_side_pool": max(sell_pools, key=lambda x: x.strength_score).model_dump(mode="json") if sell_pools else None,
            "active_inducement_references": [x.model_dump(mode="json") for x in s.inducements],
            "confluences": [x.model_dump(mode="json") for x in s.confluences],
            "target_rankings": [x.model_dump(mode="json") for x in s.targets],
            "liquidity_map": [x.model_dump(mode="json") for x in s.map_bands],
            "liquidity_density_above": sum(x.inferred_density for x in s.map_bands if x.side.value == "buy_side"),
            "liquidity_density_below": sum(x.inferred_density for x in s.map_bands if x.side.value == "sell_side"),
            "path_obstruction_scores": {str(x.pool_id): x.path_obstruction_score for x in s.targets},
            "multi_timeframe_confluence": s.multi_timeframe.model_dump(mode="json") if s.multi_timeframe else None,
            "confidence": s.confidence_summary.get("overall", 0),
            "data_quality": s.quality_summary.get("market_data", 0),
            "analytical_timestamp": s.analysis_timestamp.isoformat(),
            "configuration_version": s.configuration_version,
            "engine_version": s.engine_version,
            "snapshot_id": str(s.id),
        }
