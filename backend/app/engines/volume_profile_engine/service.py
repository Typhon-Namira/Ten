from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from backend.app.engines.market_data_engine import MarketDataService, Timeframe
from backend.app.events import Event, EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .analyzer import BaselineVolumeProfileAnalyzer
from .config import VolumeProfileConfig
from .contracts import VolumeProfileContext
from .events import (
    AnchoredProfileCreated,
    CompositeProfileCompleted,
    HighVolumeNodeDetected,
    LowVolumeNodeDetected,
    PointOfControlMigrated,
    ProfileShapeChanged,
    ValueAreaChanged,
    VolumeGapDetected,
    VolumeProfileAnalysisUpdated,
    VolumeProfileCheckpointRecovered,
    VolumeProfileCompleted,
    VolumeProfileDegraded,
    VolumeProfileDeveloping,
    VolumeProfileReplayCompleted,
    VolumeShelfDetected,
)
from .models import AnalysisStatus, MultiTimeframeVolumeProfileContext, ProcessingMode, ProfileType, VolumeProfileAnalysisSnapshot, VolumeSourceType, stable_id
from .repository import InMemoryVolumeProfileRepository, VolumeProfileRepository


class VolumeProfileMetrics:
    def __init__(self) -> None:
        self.analyses_completed = 0
        self.candles_processed = 0
        self.source_volume_processed = 0.0
        self.missing_volume_observations = 0
        self.profiles_initialized = 0
        self.profiles_completed = 0
        self.profiles_degraded = 0
        self.buckets_generated = 0
        self.hvns_detected = 0
        self.lvns_detected = 0
        self.shelves_detected = 0
        self.gaps_detected = 0
        self.migration_events = 0
        self.confluences = 0
        self.checkpoint_recoveries = 0
        self.replay_runs = 0
        self.persistence_failures = 0
        self.event_publication_failures = 0
        self.feature_publication_failures = 0
        self.average_analysis_latency_ms = 0.0
        self.maximum_bucket_count = 0
        self.latest_successful_analysis_timestamp: str | None = None

    def snapshot(self) -> dict[str, int | float | str | None]:
        return dict(vars(self))


class VolumeProfileService:
    def __init__(
        self,
        market_data: MarketDataService,
        smc: Any,
        liquidity: Any,
        event_bus: EventBus,
        feature_store: FeatureStore,
        config: VolumeProfileConfig | None = None,
        repository: VolumeProfileRepository | None = None,
        repository_mode: str = "memory",
    ) -> None:
        self.market_data, self.smc, self.liquidity = market_data, smc, liquidity
        self.event_bus, self.feature_store = event_bus, feature_store
        self.config = config or VolumeProfileConfig()
        self.repository = repository or InMemoryVolumeProfileRepository()
        self.repository_mode = repository_mode
        self.analyzer = BaselineVolumeProfileAnalyzer(self.config, market_data.sessions)
        self.metrics = VolumeProfileMetrics()
        self._published: set[UUID] = set()
        self._recovered: dict[tuple[str, Timeframe], VolumeProfileAnalysisSnapshot] = {}
        self.recovery_status = "not_attempted"

    async def restore(self) -> int:
        items = tuple(
            item
            for item in await self.repository.checkpoints()
            if item.configuration_version == self.config.version and item.engine_version == self.analyzer.version
        )
        self._recovered = {(x.symbol, x.timeframe): x for x in items}
        self.metrics.checkpoint_recoveries += len(items)
        self.recovery_status = "recovered" if items else "clean_start"
        if items:
            await self.event_bus.publish(VolumeProfileCheckpointRecovered(correlation_id=uuid4(), source="volume_profile", payload={"count": len(items)}))
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
    ) -> VolumeProfileAnalysisSnapshot:
        limit = min(limit, self.config.processing.maximum_candles)
        candles = await self.market_data.history(symbol, timeframe, start=start, end=end, limit=limit)
        smc = await self.smc.liquidity_context(symbol, timeframe, candles[-1].timestamp) if candles and self.smc else None
        liquidity_ids: tuple[str, ...] = ()
        if candles and self.liquidity:
            state = await self.liquidity.state(symbol, timeframe, candles[-1].timestamp)
            liquidity_ids = tuple(str(x.id) for x in state.pools) if state else ()
        source = VolumeSourceType(self.config.default_volume_source)
        return await self.analyze_context(VolumeProfileContext(tuple(candles), source, symbol, None, smc, liquidity_ids), mode, correlation_id)

    async def analyze_context(
        self, context: VolumeProfileContext, mode: ProcessingMode = ProcessingMode.HISTORICAL, correlation_id: UUID | None = None
    ) -> VolumeProfileAnalysisSnapshot:
        started = perf_counter()
        snapshot = self.analyzer.analyze_snapshot(context, mode)
        try:
            await self.repository.save(snapshot)
        except Exception:
            self.metrics.persistence_failures += 1
            raise
        self._record(snapshot, context, (perf_counter() - started) * 1000)
        await self._publish(snapshot, correlation_id or uuid4())
        return snapshot

    async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, limit: int = 500) -> VolumeProfileAnalysisSnapshot:
        candles = await self.market_data.replay(symbol, timeframe, timestamp, limit=min(limit, self.config.processing.maximum_candles))
        snapshot = await self.analyze_context(
            VolumeProfileContext(tuple(candles), VolumeSourceType(self.config.default_volume_source), symbol), ProcessingMode.REPLAY
        )
        self.metrics.replay_runs += 1
        await self.event_bus.publish(VolumeProfileReplayCompleted(correlation_id=uuid4(), source="volume_profile", payload={"snapshot_id": str(snapshot.id)}))
        return snapshot

    async def state(self, symbol: str, timeframe: Timeframe, at: datetime | None = None) -> VolumeProfileAnalysisSnapshot | None:
        return await self.repository.at(symbol, timeframe, at) if at else await self.repository.latest(symbol, timeframe)

    async def multi_timeframe(self, symbol: str, timeframe: Timeframe, at: datetime | None = None, limit: int = 500) -> MultiTimeframeVolumeProfileContext:
        profiles: dict[str, tuple[UUID, ...]] = {}
        through = at
        for name in self.config.multi_timeframe.hierarchy[: self.config.multi_timeframe.maximum_depth]:
            try:
                item = Timeframe(name)
            except ValueError:
                profiles[name] = ()
                continue
            try:
                snapshot = await self.replay(symbol, item, at, limit) if at else await self.analyze(symbol, item, limit=limit)
            except Exception:
                continue
            profiles[name] = tuple(x.id for x in snapshot.completed)
            through = snapshot.analysis_timestamp if through is None else min(through, snapshot.analysis_timestamp)
        return MultiTimeframeVolumeProfileContext(
            symbol=symbol,
            requested_timeframe=timeframe,
            profile_ids_by_timeframe=profiles,
            analyzed_through=through or datetime(1970, 1, 1, tzinfo=UTC),
            maximum_depth=self.config.multi_timeframe.maximum_depth,
        )

    def health(self) -> dict[str, object]:
        reasons = []
        if self.repository_mode == "memory" and self.config.persistence.required_in_production:
            reasons.append("ephemeral_persistence")
        if self.metrics.latest_successful_analysis_timestamp is None:
            reasons.append("volume_source_not_yet_observed")
        if self.config.default_volume_source in {"unknown", "missing", "synthetic"}:
            reasons.append("volume_source_semantics_low_confidence")
        return {
            "status": "degraded" if reasons else "healthy",
            "engine_version": self.analyzer.version,
            "configuration_version": self.config.version,
            "repository_mode": self.repository_mode,
            "database_status": "available" if self.repository_mode == "sqlalchemy" else "unavailable",
            "checkpoint_recovery_status": self.recovery_status,
            "market_data_dependency": "configured",
            "smc_contract_status": "configured" if self.smc else "optional_unavailable",
            "liquidity_contract_status": "configured" if self.liquidity else "optional_unavailable",
            "volume_source_status": self.config.default_volume_source,
            "last_analysis_timestamp": self.metrics.latest_successful_analysis_timestamp,
            "degraded_reasons": reasons,
            "checked_at": datetime.now(UTC),
        }

    def _record(self, snapshot: VolumeProfileAnalysisSnapshot, context: VolumeProfileContext, latency: float) -> None:
        m, old = self.metrics, self.metrics.analyses_completed
        m.analyses_completed += 1
        m.candles_processed += len(context.candles)
        m.source_volume_processed += sum(x.volume for x in context.candles)
        m.missing_volume_observations += snapshot.volume_data_quality.missing_observations
        m.profiles_initialized += len(snapshot.profiles)
        m.profiles_completed += len(snapshot.completed)
        m.profiles_degraded += snapshot.status == AnalysisStatus.DEGRADED
        m.buckets_generated += sum(x.bucket_count for x in snapshot.profiles)
        m.hvns_detected += sum(len(x.hvns) for x in snapshot.profiles)
        m.lvns_detected += sum(len(x.lvns) for x in snapshot.profiles)
        m.shelves_detected += sum(len(x.shelves) for x in snapshot.profiles)
        m.gaps_detected += sum(len(x.gaps) for x in snapshot.profiles)
        m.migration_events += len(snapshot.migrations)
        m.confluences += len(snapshot.confluences)
        m.maximum_bucket_count = max(m.maximum_bucket_count, *(x.bucket_count for x in snapshot.profiles), 0)
        m.average_analysis_latency_ms = (m.average_analysis_latency_ms * old + latency) / m.analyses_completed
        m.latest_successful_analysis_timestamp = snapshot.analysis_timestamp.isoformat()

    async def _publish(self, snapshot: VolumeProfileAnalysisSnapshot, correlation_id: UUID) -> None:
        if snapshot.id in self._published:
            return
        try:
            await self.feature_store.write(
                FeatureRecord(
                    correlation_id=correlation_id,
                    namespace="volume_profile",
                    engine_name="volume_profile",
                    engine_version=snapshot.engine_version,
                    compatibility_version="1.0",
                    values=self.features(snapshot),
                )
            )
        except Exception:
            self.metrics.feature_publication_failures += 1
        try:
            for profile in snapshot.profiles:
                event_class: type[Event] = VolumeProfileCompleted if profile.status.value == "completed" else VolumeProfileDeveloping
                if profile.profile_type == ProfileType.ANCHORED:
                    event_class = AnchoredProfileCreated
                elif profile.profile_type == ProfileType.COMPOSITE:
                    event_class = CompositeProfileCompleted
                await self.event_bus.publish(
                    event_class(
                        event_id=stable_id("event", event_class.__name__, profile.id),
                        correlation_id=correlation_id,
                        source="volume_profile",
                        payload={
                            "profile_id": str(profile.id),
                            "profile_type": profile.profile_type.value,
                            "availability_timestamp": profile.availability_timestamp.isoformat(),
                            "configuration_version": profile.configuration_version,
                            "engine_version": profile.engine_version,
                        },
                    )
                )
                for cls, values in (
                    (HighVolumeNodeDetected, profile.hvns),
                    (LowVolumeNodeDetected, profile.lvns),
                    (VolumeShelfDetected, profile.shelves),
                    (VolumeGapDetected, profile.gaps),
                ):
                    for value in values:
                        await self.event_bus.publish(
                            cls(
                                event_id=stable_id("event", cls.__name__, value.id),
                                correlation_id=correlation_id,
                                source="volume_profile",
                                payload={"profile_id": str(profile.id), "source_object_ids": [str(value.id)]},
                            )
                        )
                if profile.shape:
                    await self.event_bus.publish(
                        ProfileShapeChanged(
                            event_id=stable_id("event", "shape", profile.shape.id),
                            correlation_id=correlation_id,
                            source="volume_profile",
                            payload={"profile_id": str(profile.id), "shape": profile.shape.shape_type.value},
                        )
                    )
                if profile.value_area:
                    await self.event_bus.publish(
                        ValueAreaChanged(
                            event_id=stable_id("event", "va", profile.value_area.id),
                            correlation_id=correlation_id,
                            source="volume_profile",
                            payload={"profile_id": str(profile.id)},
                        )
                    )
            for migration in snapshot.migrations:
                await self.event_bus.publish(
                    PointOfControlMigrated(
                        event_id=stable_id("event", "migration", migration.id),
                        correlation_id=correlation_id,
                        source="volume_profile",
                        payload={"source_object_ids": [str(migration.previous_profile_id), str(migration.current_profile_id)]},
                    )
                )
            final = VolumeProfileDegraded if snapshot.status == AnalysisStatus.DEGRADED else VolumeProfileAnalysisUpdated
            await self.event_bus.publish(
                final(
                    event_id=stable_id("event", final.__name__, snapshot.id),
                    correlation_id=correlation_id,
                    source="volume_profile",
                    payload={"snapshot_id": str(snapshot.id), "status": snapshot.status.value},
                )
            )
        except Exception:
            self.metrics.event_publication_failures += 1
        self._published.add(snapshot.id)

    @staticmethod
    def features(snapshot: VolumeProfileAnalysisSnapshot) -> dict[str, object]:
        current = snapshot.developing[-1] if snapshot.developing else (snapshot.profiles[0] if snapshot.profiles else None)
        completed = [x for x in snapshot.completed if x.profile_type == ProfileType.SESSION]
        periods = {
            kind.value: next((x for x in reversed(snapshot.completed) if x.profile_type == kind), None)
            for kind in (ProfileType.DAILY, ProfileType.WEEKLY, ProfileType.MONTHLY)
        }
        hvns = [n for x in snapshot.profiles for n in x.hvns]
        lvns = [n for x in snapshot.profiles for n in x.lvns]
        price = current.poc.price if current and current.poc else 0
        return {
            "developing_poc": current.poc.model_dump(mode="json") if current and current.poc else None,
            "developing_value_area": current.value_area.model_dump(mode="json") if current and current.value_area else None,
            "completed_session": completed[-1].model_dump(mode="json") if completed else None,
            "previous_day": periods["daily"].model_dump(mode="json") if periods["daily"] else None,
            "previous_week": periods["weekly"].model_dump(mode="json") if periods["weekly"] else None,
            "previous_month": periods["monthly"].model_dump(mode="json") if periods["monthly"] else None,
            "nearest_hvn_above": next((x.model_dump(mode="json") for x in sorted(hvns, key=lambda n: n.peak_price) if x.peak_price >= price), None),
            "nearest_hvn_below": next((x.model_dump(mode="json") for x in sorted(hvns, key=lambda n: -n.peak_price) if x.peak_price <= price), None),
            "nearest_lvn_above": next((x.model_dump(mode="json") for x in sorted(lvns, key=lambda n: n.peak_price) if x.peak_price >= price), None),
            "nearest_lvn_below": next((x.model_dump(mode="json") for x in sorted(lvns, key=lambda n: -n.peak_price) if x.peak_price <= price), None),
            "active_shelves": [x.model_dump(mode="json") for p in snapshot.profiles for x in p.shelves],
            "active_gaps": [x.model_dump(mode="json") for p in snapshot.profiles for x in p.gaps],
            "profile_shape": current.shape.model_dump(mode="json") if current and current.shape else None,
            "poc_migration": snapshot.migrations[-1].model_dump(mode="json") if snapshot.migrations else None,
            "profile_range": {"low": current.buckets[0].lower, "high": current.buckets[-1].upper} if current and current.buckets else None,
            "volume_concentration": current.poc.volume_percent if current and current.poc else 0,
            "confluences": [x.model_dump(mode="json") for x in snapshot.confluences],
            "confidence": snapshot.confidence_summary.get("overall", 0),
            "volume_source_quality": snapshot.volume_data_quality.model_dump(mode="json"),
            "analytical_timestamp": snapshot.analysis_timestamp.isoformat(),
            "source_traceability": snapshot.market_data_boundary,
            "data_quality": snapshot.quality_summary,
            "configuration_version": snapshot.configuration_version,
            "engine_version": snapshot.engine_version,
            "snapshot_id": str(snapshot.id),
        }
