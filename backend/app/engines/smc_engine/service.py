"""Async SMC service consuming candles exclusively through MarketDataService."""

from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

from backend.app.engines.market_data_engine import Candle, MarketDataService, Timeframe
from backend.app.events import EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .analyzer import BaselineSMCAnalyzer
from .config import SMCConfig
from .events import BOSDetected, CHOCHDetected, MSSDetected, SMCAnalysisUpdated, SMCInputDegraded, SMCReplayCompleted, SwingConfirmed
from .models import AnalysisStatus, ProcessingMode, SMCAnalysisSnapshot, StructureEventType
from .repository import InMemorySMCRepository, SMCRepository


class SMCMetrics:
    def __init__(self) -> None:
        self.analyses = 0
        self.candles_processed = 0
        self.swings_confirmed = 0
        self.bos_count = 0
        self.choch_count = 0
        self.mss_count = 0
        self.degraded_inputs = 0
        self.processing_latency_ms = 0.0
        self.replay_mismatches = 0

    def snapshot(self) -> dict[str, int | float]:
        return dict(vars(self))


class SMCService:
    def __init__(self, market_data: MarketDataService, event_bus: EventBus, feature_store: FeatureStore, config: SMCConfig | None = None, repository: SMCRepository | None = None) -> None:
        self.market_data = market_data
        self.event_bus = event_bus
        self.feature_store = feature_store
        self.config = config or SMCConfig()
        self.repository = repository or InMemorySMCRepository()
        self.analyzer = BaselineSMCAnalyzer(self.config)
        self.metrics = SMCMetrics()
        self._published: set[UUID] = set()

    async def analyze(self, symbol: str, timeframe: Timeframe, *, start: datetime | None = None, end: datetime | None = None, limit: int = 500, mode: ProcessingMode = ProcessingMode.HISTORICAL, correlation_id: UUID | None = None) -> SMCAnalysisSnapshot:
        candles = await self.market_data.history(symbol, timeframe, start=start, end=end, limit=limit)
        return await self.analyze_candles(candles, mode=mode, correlation_id=correlation_id)

    async def analyze_candles(self, candles: list[Candle], *, mode: ProcessingMode = ProcessingMode.HISTORICAL, correlation_id: UUID | None = None) -> SMCAnalysisSnapshot:
        normalized = candles
        started = perf_counter()
        snapshot = self.analyzer.analyze_snapshot(normalized, mode)
        await self.repository.save(snapshot)
        self._record(snapshot, len(normalized), (perf_counter() - started) * 1000)
        await self._publish(snapshot, correlation_id or uuid4())
        return snapshot

    async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, *, limit: int = 500) -> SMCAnalysisSnapshot:
        candles = await self.market_data.replay(symbol, timeframe, timestamp, limit=limit)
        snapshot = await self.analyze_candles(candles, mode=ProcessingMode.REPLAY)
        await self.event_bus.publish(SMCReplayCompleted(correlation_id=uuid4(), source="smc", payload={"snapshot_id": str(snapshot.id), "timestamp": timestamp.isoformat()}))
        return snapshot

    async def state(self, symbol: str, timeframe: Timeframe, timestamp: datetime | None = None) -> SMCAnalysisSnapshot | None:
        return await self.repository.at(symbol, timeframe, timestamp) if timestamp else await self.repository.latest(symbol, timeframe)

    async def bounded_recalculate(self, symbol: str, timeframe: Timeframe, corrected_at: datetime) -> SMCAnalysisSnapshot:
        start = corrected_at - timeframe.duration * self.config.processing.recalculation_window
        return await self.analyze(symbol, timeframe, start=start, limit=self.config.processing.recalculation_window, mode=ProcessingMode.REBUILD)

    def health(self) -> dict[str, object]:
        return {"status": "healthy", "engine_version": self.analyzer.version, "configuration_version": self.config.version, "checked_at": datetime.now(UTC)}

    def _record(self, snapshot: SMCAnalysisSnapshot, candle_count: int, latency: float) -> None:
        self.metrics.analyses += 1
        self.metrics.candles_processed += candle_count
        self.metrics.swings_confirmed += len(snapshot.swings)
        self.metrics.bos_count += sum(item.event_type == StructureEventType.BOS for item in snapshot.structure_events)
        self.metrics.choch_count += sum(item.event_type == StructureEventType.CHOCH for item in snapshot.structure_events)
        self.metrics.mss_count += sum(item.event_type == StructureEventType.MSS for item in snapshot.structure_events)
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
        if snapshot.status == AnalysisStatus.DEGRADED_INPUT:
            await self.event_bus.publish(SMCInputDegraded(correlation_id=correlation_id, source="smc", payload={"snapshot_id": str(snapshot.id), "quality": snapshot.quality_summary}))
        await self.feature_store.write(FeatureRecord(correlation_id=correlation_id, namespace="smc", engine_name="smc", engine_version=self.analyzer.version, compatibility_version="2.0", values=self.features(snapshot)))
        await self.event_bus.publish(SMCAnalysisUpdated(correlation_id=correlation_id, source="smc", payload={"snapshot_id": str(snapshot.id), "status": snapshot.status.value}))
        self._published.add(snapshot.id)

    @staticmethod
    def features(snapshot: SMCAnalysisSnapshot) -> dict[str, object]:
        state = snapshot.structure_state
        return {"current_structure_direction": state.current_direction.value, "internal_structure_direction": state.internal_direction.value, "external_structure_direction": state.external_direction.value, "active_swing_high": str(state.active_swing_high_id) if state.active_swing_high_id else None, "active_swing_low": str(state.active_swing_low_id) if state.active_swing_low_id else None, "protected_high": str(state.protected_high_id) if state.protected_high_id else None, "protected_low": str(state.protected_low_id) if state.protected_low_id else None, "last_bos": str(state.last_bos_id) if state.last_bos_id else None, "last_choch": str(state.last_choch_id) if state.last_choch_id else None, "last_mss": str(state.last_mss_id) if state.last_mss_id else None, "smc_confidence": snapshot.confidence_summary.get("overall", 0), "smc_input_quality": snapshot.quality_summary.get("average", 0), "analysis_timestamp": snapshot.analysis_timestamp.isoformat(), "engine_version": snapshot.engine_version, "configuration_version": snapshot.configuration_version, "snapshot_id": str(snapshot.id)}
