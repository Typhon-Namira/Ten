"""Async SMC service consuming candles exclusively through MarketDataService."""

from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

from backend.app.core.bounded import BoundedSet
from backend.app.engines.market_data_engine import Candle, MarketDataService, Timeframe
from backend.app.events import EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .analyzer import BaselineSMCAnalyzer
from .config import SMCConfig
from .liquidity_contract import LiquidityFeatureReader, SMCLiquidityContext, SMCLiquidityLevel
from .events import BOSDetected, BreakerBlockDetected, CHOCHDetected, DealingRangeUpdated, DisplacementDetected, ImbalanceDetected, LiquidityVoidDetected, MSSDetected, MitigationBlockDetected, MultiTimeframeContextUpdated, OrderBlockDetected, SMCAnalysisUpdated, SMCInputDegraded, SMCObjectLifecycleChanged, SMCReplayCompleted, SwingConfirmed
from .models import AnalysisStatus, Evidence, LiquidityReferenceType, MTFConflictState, MultiTimeframeContext, ProcessingMode, SMCAnalysisSnapshot, StructureDirection, StructureEventType, StructureLiquidityReference, ZoneType, stable_id
from .repository import InMemorySMCRepository, SMCRepository


class SMCMetrics:
    def __init__(self) -> None:
        self.analyses = 0
        self.candles_processed = 0
        self.swings_confirmed = 0
        self.bos_count = 0
        self.choch_count = 0
        self.mss_count = 0
        self.displacements = 0
        self.zones_detected = 0
        self.zones_mitigated = 0
        self.zones_invalidated = 0
        self.dealing_ranges = 0
        self.degraded_inputs = 0
        self.processing_latency_ms = 0.0
        self.replay_mismatches = 0

    def snapshot(self) -> dict[str, int | float]:
        return dict(vars(self))


class SMCService:
    def __init__(self, market_data: MarketDataService, event_bus: EventBus, feature_store: FeatureStore, config: SMCConfig | None = None, repository: SMCRepository | None = None, liquidity_reader: LiquidityFeatureReader | None = None) -> None:
        self.market_data = market_data
        self.event_bus = event_bus
        self.feature_store = feature_store
        self.config = config or SMCConfig()
        self.repository = repository or InMemorySMCRepository()
        self.liquidity_reader = liquidity_reader
        self.analyzer = BaselineSMCAnalyzer(self.config)
        self.metrics = SMCMetrics()
        self._published = BoundedSet[UUID](10_000)
        self._recovered: dict[tuple[str, Timeframe], SMCAnalysisSnapshot] = {}

    async def restore(self) -> int:
        checkpoints = await self.repository.checkpoints()
        self._recovered = {(item.symbol.replace("/", "").replace("-", "").upper(), item.timeframe): item for item in checkpoints}
        return len(checkpoints)

    async def analyze(self, symbol: str, timeframe: Timeframe, *, start: datetime | None = None, end: datetime | None = None, limit: int = 500, mode: ProcessingMode = ProcessingMode.HISTORICAL, correlation_id: UUID | None = None) -> SMCAnalysisSnapshot:
        candles = await self.market_data.history(symbol, timeframe, start=start, end=end, limit=limit)
        return await self.analyze_candles(candles, mode=mode, correlation_id=correlation_id)

    async def analyze_candles(self, candles: list[Candle], *, mode: ProcessingMode = ProcessingMode.HISTORICAL, correlation_id: UUID | None = None) -> SMCAnalysisSnapshot:
        normalized = candles
        started = perf_counter()
        snapshot = self.analyzer.analyze_snapshot(normalized, mode)
        if self.liquidity_reader is not None and snapshot.status != AnalysisStatus.INSUFFICIENT_HISTORY:
            external = await self.liquidity_reader.evidence(snapshot.symbol, snapshot.timeframe, snapshot.analysis_timestamp)
            supplied = tuple(StructureLiquidityReference(id=stable_id("external-liquidity", snapshot.symbol, snapshot.timeframe, item.id, item.available_at.isoformat()), symbol=snapshot.symbol, timeframe=snapshot.timeframe, reference_type=LiquidityReferenceType.EXTERNAL_SWEEP, direction=StructureDirection.NEUTRAL, price=item.price, timestamp=item.occurred_at, available_at=item.available_at, external_sweep_id=item.id, confidence_score=item.confidence_score, evidence=(Evidence(code="dedicated_liquidity_engine", description="sweep metadata supplied through the read-only liquidity contract", value=item.event_type),), algorithm_version=snapshot.engine_version) for item in external if item.available_at <= snapshot.analysis_timestamp)
            snapshot = snapshot.model_copy(update={"id": stable_id("snapshot-liquidity", snapshot.symbol, snapshot.timeframe, snapshot.id, *(item.id for item in supplied)), "liquidity_references": (*snapshot.liquidity_references, *supplied)})
        await self.repository.save(snapshot, correlation_id=correlation_id)
        self._record(snapshot, len(normalized), (perf_counter() - started) * 1000)
        await self._publish(snapshot, correlation_id or uuid4())
        return snapshot

    async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, *, limit: int = 500) -> SMCAnalysisSnapshot:
        candles = await self.market_data.replay(symbol, timeframe, timestamp, limit=limit)
        snapshot = await self.analyze_candles(candles, mode=ProcessingMode.REPLAY)
        await self.event_bus.publish(SMCReplayCompleted(correlation_id=uuid4(), source="smc", payload={"snapshot_id": str(snapshot.id), "timestamp": timestamp.isoformat()}))
        return snapshot

    async def state(self, symbol: str, timeframe: Timeframe, timestamp: datetime | None = None) -> SMCAnalysisSnapshot | None:
        persisted = await self.repository.at(symbol, timeframe, timestamp) if timestamp else await self.repository.latest(symbol, timeframe)
        return persisted or (self._recovered.get((symbol.replace("/", "").replace("-", "").upper(), timeframe)) if timestamp is None else None)

    async def liquidity_context(self, symbol: str, timeframe: Timeframe, at: datetime) -> SMCLiquidityContext:
        snapshot = await self.state(symbol, timeframe, at)
        if snapshot is None:
            snapshot = await self.replay(symbol, timeframe, at)
        levels = tuple(SMCLiquidityLevel(id=str(item.id), symbol=item.symbol, timeframe=item.timeframe, kind=item.swing_type.value, scope=item.scope.value, price=item.price, occurred_at=item.timestamp, available_at=item.confirmed_at or item.detected_at, confidence_score=item.confidence_score, quality_score=item.quality_score) for item in snapshot.swings if (item.confirmed_at or item.detected_at) <= at)
        protected = tuple(str(item) for item in (snapshot.structure_state.protected_high_id, snapshot.structure_state.protected_low_id) if item)
        return SMCLiquidityContext(symbol=snapshot.symbol, timeframe=snapshot.timeframe, analyzed_through=snapshot.analysis_timestamp, structure_direction=snapshot.structure_state.current_direction.value, levels=levels, protected_level_ids=protected, structural_event_ids=tuple(str(item.id) for item in snapshot.structure_events), configuration_version=snapshot.configuration_version, engine_version=snapshot.engine_version)

    async def bounded_recalculate(self, symbol: str, timeframe: Timeframe, corrected_at: datetime) -> SMCAnalysisSnapshot:
        start = corrected_at - timeframe.duration * self.config.processing.recalculation_window
        return await self.analyze(symbol, timeframe, start=start, limit=self.config.processing.recalculation_window, mode=ProcessingMode.REBUILD)

    async def multi_timeframe(self, symbol: str, timeframe: Timeframe, timestamp: datetime | None = None, *, limit: int = 500) -> MultiTimeframeContext:
        directions: dict[str, StructureDirection] = {}
        through = timestamp
        for item in Timeframe:
            try:
                snapshot = await self.replay(symbol, item, timestamp, limit=limit) if timestamp else await self.analyze(symbol, item, limit=limit)
            except Exception:
                continue
            if snapshot.status != AnalysisStatus.INSUFFICIENT_HISTORY:
                directions[item.value] = snapshot.structure_state.current_direction
                through = snapshot.analysis_timestamp if through is None else min(through, snapshot.analysis_timestamp)
        daily = await self.market_data.history(symbol, Timeframe.D1, end=timestamp, limit=limit)
        directions.update(self._calendar_directions(daily))
        non_neutral = [value for value in directions.values() if value not in (StructureDirection.NEUTRAL, StructureDirection.TRANSITIONAL)]
        if not non_neutral:
            conflict = MTFConflictState.INSUFFICIENT
            alignment = 0.0
        else:
            bullish = sum(value == StructureDirection.BULLISH for value in non_neutral)
            bearish = len(non_neutral) - bullish
            alignment = max(bullish, bearish) / len(non_neutral) * 100.0
            conflict = MTFConflictState.ALIGNED if alignment == 100 else MTFConflictState.PARTIAL if alignment >= 60 else MTFConflictState.CONFLICTED
        context = MultiTimeframeContext(symbol=symbol, requested_timeframe=timeframe.value, directions=directions, conflict_state=conflict, alignment_score=alignment, confidence_score=min(100.0, alignment * len(non_neutral) / max(len(self.config.multi_timeframe.hierarchy), 1)), analyzed_through=through or datetime(1970, 1, 1, tzinfo=UTC), reasoning_metadata={"hierarchy": self.config.multi_timeframe.hierarchy, "no_future_candles": True})
        await self.event_bus.publish(MultiTimeframeContextUpdated(correlation_id=uuid4(), source="smc", payload=context.model_dump(mode="json")))
        return context

    @staticmethod
    def _calendar_directions(candles: list[Candle]) -> dict[str, StructureDirection]:
        if not candles:
            return {}
        weekly: dict[tuple[int, int], list[Candle]] = {}
        monthly: dict[tuple[int, int], list[Candle]] = {}
        for candle in candles:
            iso = candle.timestamp.isocalendar()
            weekly.setdefault((iso.year, iso.week), []).append(candle)
            monthly.setdefault((candle.timestamp.year, candle.timestamp.month), []).append(candle)
        def direction(groups: dict[tuple[int, int], list[Candle]]) -> StructureDirection:
            ordered = [groups[key] for key in sorted(groups)]
            if len(ordered) < 2:
                return StructureDirection.NEUTRAL
            prior, current = ordered[-2][-1].close, ordered[-1][-1].close
            return StructureDirection.BULLISH if current > prior else StructureDirection.BEARISH if current < prior else StructureDirection.NEUTRAL
        return {"W1": direction(weekly), "MN1": direction(monthly)}

    def health(self) -> dict[str, object]:
        persistence_metrics = getattr(self.repository, "metrics", None)
        return {
            "status": "healthy",
            "engine_version": self.analyzer.version,
            "configuration_version": self.config.version,
            "persistence": persistence_metrics.snapshot() if persistence_metrics is not None else None,
            "checked_at": datetime.now(UTC),
        }

    def _record(self, snapshot: SMCAnalysisSnapshot, candle_count: int, latency: float) -> None:
        self.metrics.analyses += 1
        self.metrics.candles_processed += candle_count
        self.metrics.swings_confirmed += len(snapshot.swings)
        self.metrics.bos_count += sum(item.event_type == StructureEventType.BOS for item in snapshot.structure_events)
        self.metrics.choch_count += sum(item.event_type == StructureEventType.CHOCH for item in snapshot.structure_events)
        self.metrics.mss_count += sum(item.event_type == StructureEventType.MSS for item in snapshot.structure_events)
        self.metrics.displacements += len(snapshot.displacements)
        self.metrics.zones_detected += len(snapshot.zones)
        self.metrics.zones_mitigated += sum(item.lifecycle_state.value == "mitigated" for item in snapshot.zones)
        self.metrics.zones_invalidated += sum(item.lifecycle_state.value == "invalidated" for item in snapshot.zones)
        self.metrics.dealing_ranges += len(snapshot.dealing_ranges)
        self.metrics.degraded_inputs += snapshot.status == AnalysisStatus.DEGRADED_INPUT
        self.metrics.processing_latency_ms = latency

    async def _publish(self, snapshot: SMCAnalysisSnapshot, correlation_id: UUID) -> None:
        if snapshot.id in self._published:
            return
        for swing in snapshot.swings:
            await self.event_bus.publish(SwingConfirmed(correlation_id=correlation_id, source="smc", payload={"object_id": str(swing.id), "symbol": swing.symbol, "timeframe": swing.timeframe.value, "analytical_timestamp": swing.confirmed_at.isoformat() if swing.confirmed_at else None, "confidence": swing.confidence_score, "processing_mode": snapshot.processing_mode.value}))
        event_types = {StructureEventType.BOS: BOSDetected, StructureEventType.CHOCH: CHOCHDetected, StructureEventType.MSS: MSSDetected}
        for event in snapshot.structure_events:
            event_class = event_types.get(event.event_type)
            if event_class:
                await self.event_bus.publish(event_class(correlation_id=correlation_id, source="smc", payload=event.model_dump(mode="json")))
        for displacement in snapshot.displacements:
            await self.event_bus.publish(DisplacementDetected(correlation_id=correlation_id, source="smc", payload=displacement.model_dump(mode="json")))
        zone_events = {
            ZoneType.BULLISH_FVG: ImbalanceDetected, ZoneType.BEARISH_FVG: ImbalanceDetected,
            ZoneType.BULLISH_INVERSION_FVG: ImbalanceDetected, ZoneType.BEARISH_INVERSION_FVG: ImbalanceDetected,
            ZoneType.LIQUIDITY_VOID: LiquidityVoidDetected, ZoneType.BULLISH_ORDER_BLOCK: OrderBlockDetected,
            ZoneType.BEARISH_ORDER_BLOCK: OrderBlockDetected, ZoneType.BULLISH_BREAKER: BreakerBlockDetected,
            ZoneType.BEARISH_BREAKER: BreakerBlockDetected, ZoneType.BULLISH_MITIGATION_BLOCK: MitigationBlockDetected,
            ZoneType.BEARISH_MITIGATION_BLOCK: MitigationBlockDetected,
        }
        for zone in snapshot.zones:
            await self.event_bus.publish(zone_events[zone.zone_type](correlation_id=correlation_id, source="smc", payload=zone.model_dump(mode="json")))
            if zone.version > 1:
                await self.event_bus.publish(SMCObjectLifecycleChanged(correlation_id=correlation_id, source="smc", payload={"object_id": str(zone.id), "state": zone.lifecycle_state.value, "version": zone.version}))
        for dealing_range in snapshot.dealing_ranges:
            await self.event_bus.publish(DealingRangeUpdated(correlation_id=correlation_id, source="smc", payload=dealing_range.model_dump(mode="json")))
        if snapshot.status == AnalysisStatus.DEGRADED_INPUT:
            await self.event_bus.publish(SMCInputDegraded(correlation_id=correlation_id, source="smc", payload={"snapshot_id": str(snapshot.id), "quality": snapshot.quality_summary}))
        await self.feature_store.write(FeatureRecord(correlation_id=correlation_id, namespace="smc", engine_name="smc", engine_version=self.analyzer.version, compatibility_version="2.0", values=self.features(snapshot)))
        await self.event_bus.publish(SMCAnalysisUpdated(correlation_id=correlation_id, source="smc", payload={"snapshot_id": str(snapshot.id), "status": snapshot.status.value}))
        self._published.add(snapshot.id)

    @staticmethod
    def features(snapshot: SMCAnalysisSnapshot) -> dict[str, object]:
        state = snapshot.structure_state
        return {"current_structure_direction": state.current_direction.value, "internal_structure_direction": state.internal_direction.value, "external_structure_direction": state.external_direction.value, "active_swing_high": str(state.active_swing_high_id) if state.active_swing_high_id else None, "active_swing_low": str(state.active_swing_low_id) if state.active_swing_low_id else None, "protected_high": str(state.protected_high_id) if state.protected_high_id else None, "protected_low": str(state.protected_low_id) if state.protected_low_id else None, "last_bos": str(state.last_bos_id) if state.last_bos_id else None, "last_choch": str(state.last_choch_id) if state.last_choch_id else None, "last_mss": str(state.last_mss_id) if state.last_mss_id else None, "smc_confidence": snapshot.confidence_summary.get("overall", 0), "smc_input_quality": snapshot.quality_summary.get("average", 0), "displacements": [item.model_dump(mode="json") for item in snapshot.displacements], "zones": [item.model_dump(mode="json") for item in snapshot.zones], "liquidity_references": [item.model_dump(mode="json") for item in snapshot.liquidity_references], "dealing_ranges": [item.model_dump(mode="json") for item in snapshot.dealing_ranges], "replay_mode": snapshot.processing_mode.value, "analysis_timestamp": snapshot.analysis_timestamp.isoformat(), "engine_version": snapshot.engine_version, "configuration_version": snapshot.configuration_version, "snapshot_id": str(snapshot.id)}
