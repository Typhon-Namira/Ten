from datetime import UTC, datetime
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

from backend.app.engines.market_data_engine import MarketDataService, Timeframe
from backend.app.events import Event, EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .analyzer import BaselineInstitutionalFlowAnalyzer
from .config import InstitutionalFlowConfig
from .contracts import InstitutionalFlowContext
from .events import (
    AbsorptionLikeBehaviorInferred,
    CampaignPhaseChanged,
    CrossSessionFlowAnalyzed,
    DirectionalPressureChanged,
    ExhaustionLikeBehaviorInferred,
    InitiativeActivityInferred,
    InstitutionalFlowCheckpointRecovered,
    InstitutionalFlowDegraded,
    InstitutionalFlowReplayCompleted,
    InstitutionalFlowUpdated,
    InventoryBehaviorInferred,
    ParticipationChanged,
    ResponsiveActivityInferred,
)
from .models import (
    AnalysisStatus,
    CorrelationGroup,
    EvidenceSourceEngine,
    EvidenceType,
    FlowDirection,
    InstitutionalFlowAnalysisSnapshot,
    InstitutionalFlowEvidence,
    MultiTimeframeInstitutionalFlow,
    ProcessingMode,
    SessionType,
    stable_id,
)
from .repository import InMemoryInstitutionalFlowRepository, InstitutionalFlowRepository


class InstitutionalFlowMetrics:
    def __init__(self) -> None:
        self.analyses_completed = 0
        self.analyses_degraded = 0
        self.analyses_failed = 0
        self.candles_considered = 0
        self.upstream_evidence_consumed = 0
        self.evidence_rejected_as_future = 0
        self.evidence_rejected_as_invalid = 0
        self.evidence_deduplicated = 0
        self.evidence_correlation_discounted = 0
        self.evidence_conflicts = 0
        self.initiative_events = 0
        self.responsive_events = 0
        self.absorption_inferences = 0
        self.exhaustion_inferences = 0
        self.campaign_phase_transitions = 0
        self.directional_pressure_transitions = 0
        self.cross_session_analyses = 0
        self.multi_timeframe_analyses = 0
        self.checkpoint_recoveries = 0
        self.replay_runs = 0
        self.persistence_failures = 0
        self.event_publication_failures = 0
        self.feature_publication_failures = 0
        self.average_latency_ms = 0.0
        self.p95_latency_ms = 0.0
        self.latest_successful_analysis_timestamp: str | None = None
        self._latencies: list[float] = []

    def observe_latency(self, latency: float) -> None:
        self._latencies.append(latency)
        self._latencies = self._latencies[-100:]
        self.average_latency_ms = sum(self._latencies) / len(self._latencies)
        self.p95_latency_ms = sorted(self._latencies)[max(0, round(0.95 * len(self._latencies)) - 1)]

    def snapshot(self) -> dict[str, int | float | str | None]:
        return {key: value for key, value in vars(self).items() if key != "_latencies"}


class InstitutionalFlowService:
    def __init__(
        self,
        market_data: MarketDataService,
        smc: Any,
        liquidity: Any,
        volume_profile: Any,
        event_bus: EventBus,
        feature_store: FeatureStore,
        config: InstitutionalFlowConfig | None = None,
        repository: InstitutionalFlowRepository | None = None,
        repository_mode: str = "memory",
    ) -> None:
        self.market_data, self.smc, self.liquidity, self.volume_profile = market_data, smc, liquidity, volume_profile
        self.event_bus, self.feature_store = event_bus, feature_store
        self.config = config or InstitutionalFlowConfig()
        self.repository = repository or InMemoryInstitutionalFlowRepository()
        self.repository_mode = repository_mode
        self.analyzer = BaselineInstitutionalFlowAnalyzer(self.config)
        self.metrics = InstitutionalFlowMetrics()
        self._published: set[UUID] = set()
        self._recovered: dict[tuple[str, Timeframe], InstitutionalFlowAnalysisSnapshot] = {}
        self.recovery_status = "not_attempted"

    async def restore(self) -> int:
        items = tuple(item for item in await self.repository.checkpoints() if item.configuration_version == self.config.version and item.engine_version == self.analyzer.version)
        self._recovered = {(item.symbol, item.timeframe): item for item in items}
        self.metrics.checkpoint_recoveries += len(items)
        self.recovery_status = "recovered" if items else "clean_start"
        if items:
            await self.event_bus.publish(InstitutionalFlowCheckpointRecovered(correlation_id=uuid4(), source="institutional_flow", payload={"count": len(items)}))
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
    ) -> InstitutionalFlowAnalysisSnapshot:
        limit = min(limit, self.config.processing.maximum_candles)
        candles = await self.market_data.history(symbol, timeframe, start=start, end=end, limit=limit)
        if not candles:
            raise ValueError("Institutional Flow source data is unavailable")
        boundary = candles[-1].timestamp
        evidence = await self._upstream_evidence(symbol, timeframe, boundary)
        session_name = self.market_data.sessions.session_at(boundary).value
        try:
            session = SessionType(session_name)
        except ValueError:
            session = SessionType.UNKNOWN
        context = InstitutionalFlowContext(tuple(candles), evidence, session, boundary, (("market_data", "1.0.0"), ("smc", "1.0.0"), ("liquidity", "1.0.0"), ("volume_profile", "1.0.0")))
        return await self.analyze_context(context, mode, correlation_id)

    async def _upstream_evidence(self, symbol: str, timeframe: Timeframe, boundary: datetime) -> tuple[InstitutionalFlowEvidence, ...]:
        result: list[InstitutionalFlowEvidence] = []
        for provider in (self.smc, self.liquidity, self.volume_profile):
            reader = getattr(provider, "institutional_flow_evidence", None)
            if reader:
                result.extend(await reader(symbol, timeframe, boundary))
                continue
            state_reader = getattr(provider, "state", None)
            if state_reader:
                state = await state_reader(symbol, timeframe, boundary)
                if provider is self.smc:
                    result.extend(self._smc_evidence(state, boundary))
                elif provider is self.liquidity:
                    result.extend(self._liquidity_evidence(state, boundary))
                elif provider is self.volume_profile:
                    result.extend(self._volume_profile_evidence(state, boundary))
        return tuple(result)

    def _evidence(
        self,
        *,
        source: EvidenceSourceEngine,
        kind: EvidenceType,
        source_id: object,
        timestamp: datetime,
        timeframe: Timeframe,
        direction: FlowDirection,
        strength: float,
        confidence: float,
        quality: float,
        group: CorrelationGroup,
        explanation: str,
    ) -> InstitutionalFlowEvidence:
        return InstitutionalFlowEvidence(
            id=stable_id("upstream", source, kind, source_id, timestamp),
            source_engine=source,
            evidence_type=kind,
            source_object_id=str(source_id),
            source_timestamp=timestamp,
            availability_timestamp=timestamp,
            timeframe=timeframe,
            session=SessionType.UNKNOWN,
            direction=direction,
            strength=max(0.0, min(1.0, strength)),
            confidence=max(0.0, min(1.0, confidence)),
            quality=max(0.0, min(1.0, quality)),
            correlation_group=group,
            explanation=explanation,
            configuration_version=self.config.version,
            engine_version=self.analyzer.version,
        )

    def _smc_evidence(self, state: object | None, boundary: datetime) -> tuple[InstitutionalFlowEvidence, ...]:
        result = []
        for item in getattr(state, "structure_events", ()) if state else ():
            timestamp = item.timestamp
            if timestamp > boundary:
                continue
            direction = FlowDirection.BULLISH if item.direction.value == "bullish" else FlowDirection.BEARISH if item.direction.value == "bearish" else FlowDirection.INDETERMINATE
            kind = EvidenceType.DISPLACEMENT if item.displacement_score >= 50 else EvidenceType.STRUCTURAL_BREAK
            result.append(self._evidence(source=EvidenceSourceEngine.SMC, kind=kind, source_id=item.id, timestamp=timestamp, timeframe=item.timeframe, direction=direction, strength=max(item.displacement_score, item.confidence_score) / 100, confidence=item.confidence_score / 100, quality=item.quality_score / 100, group=CorrelationGroup.STRUCTURE, explanation="Time-valid public SMC structure evidence; structure is not redetected by Institutional Flow."))
        return tuple(result)

    def _liquidity_evidence(self, state: object | None, boundary: datetime) -> tuple[InstitutionalFlowEvidence, ...]:
        result = []
        for item in getattr(state, "events", ()) if state else ():
            timestamp = item.available_at
            if timestamp > boundary:
                continue
            direction = FlowDirection.BEARISH if item.side.value in {"buy_side", "buy"} else FlowDirection.BULLISH if item.side.value in {"sell_side", "sell"} else FlowDirection.INDETERMINATE
            result.append(self._evidence(source=EvidenceSourceEngine.LIQUIDITY, kind=EvidenceType.LIQUIDITY_EVENT, source_id=item.id, timestamp=timestamp, timeframe=item.timeframe, direction=direction, strength=item.confidence_score / 100, confidence=item.confidence_score / 100, quality=item.quality_score / 100, group=CorrelationGroup.LIQUIDITY, explanation="Time-valid public Liquidity event evidence; sweep and pool logic remain owned upstream."))
        return tuple(result)

    def _volume_profile_evidence(self, state: object | None, boundary: datetime) -> tuple[InstitutionalFlowEvidence, ...]:
        if state is None:
            return ()
        result = []
        for item in getattr(state, "migrations", ()):
            timestamp = item.available_at
            if timestamp > boundary:
                continue
            value = item.migration_type.value
            direction = FlowDirection.BULLISH if value == "upward" else FlowDirection.BEARISH if value == "downward" else FlowDirection.NEUTRAL
            timeframe = cast(Any, state).timeframe
            result.append(self._evidence(source=EvidenceSourceEngine.VOLUME_PROFILE, kind=EvidenceType.PROFILE_MIGRATION, source_id=item.id, timestamp=timestamp, timeframe=timeframe, direction=direction, strength=min(1.0, abs(item.normalized_change)), confidence=item.confidence_score / 100, quality=item.quality_score / 100, group=CorrelationGroup.PROFILE, explanation="Time-valid public Volume Profile migration evidence; profile calculations remain owned upstream."))
        return tuple(result)

    async def analyze_context(
        self,
        context: InstitutionalFlowContext,
        mode: ProcessingMode = ProcessingMode.HISTORICAL,
        correlation_id: UUID | None = None,
    ) -> InstitutionalFlowAnalysisSnapshot:
        started = perf_counter()
        previous = None
        if context.candles:
            previous = await self.repository.latest(context.candles[-1].symbol, context.candles[-1].timeframe)
        snapshot = self.analyzer.analyze_snapshot(context, mode, previous)
        try:
            await self.repository.save(snapshot)
        except Exception:
            self.metrics.persistence_failures += 1
            self.metrics.analyses_failed += 1
            raise
        self._record(snapshot, len(context.candles), (perf_counter() - started) * 1000)
        await self._publish(snapshot, correlation_id or uuid4())
        return snapshot

    async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, limit: int = 500) -> InstitutionalFlowAnalysisSnapshot:
        candles = await self.market_data.replay(symbol, timeframe, timestamp, limit=min(limit, self.config.processing.maximum_candles))
        snapshot = await self.analyze_context(InstitutionalFlowContext(tuple(candles), analysis_boundary=timestamp), ProcessingMode.REPLAY)
        self.metrics.replay_runs += 1
        await self.event_bus.publish(InstitutionalFlowReplayCompleted(correlation_id=uuid4(), source="institutional_flow", payload={"snapshot_id": str(snapshot.id)}))
        return snapshot

    async def state(self, symbol: str, timeframe: Timeframe, at: datetime | None = None) -> InstitutionalFlowAnalysisSnapshot | None:
        return await self.repository.at(symbol, timeframe, at) if at else await self.repository.latest(symbol, timeframe)

    async def multi_timeframe(self, symbol: str, timeframe: Timeframe, at: datetime | None = None, limit: int = 500) -> MultiTimeframeInstitutionalFlow:
        directions = {}
        through = at
        for name in self.config.multi_timeframe.hierarchy[: self.config.multi_timeframe.maximum_depth]:
            try:
                item = Timeframe(name)
            except ValueError:
                continue
            try:
                snapshot = await self.replay(symbol, item, at, limit) if at else await self.analyze(symbol, item, limit=limit)
            except Exception:
                continue
            directions[name] = snapshot.state.participation.direction
            through = snapshot.analysis_timestamp if through is None else min(through, snapshot.analysis_timestamp)
        non_neutral = [direction for direction in directions.values() if direction.value not in {"neutral", "indeterminate"}]
        aligned = bool(non_neutral) and len(set(non_neutral)) == 1
        conflict = 0.0 if aligned or len(non_neutral) < 2 else 1.0 - max(non_neutral.count(direction) for direction in set(non_neutral)) / len(non_neutral)
        self.metrics.multi_timeframe_analyses += 1
        return MultiTimeframeInstitutionalFlow(requested_timeframe=timeframe, direction_by_timeframe=directions, aligned=aligned, conflict=conflict, confidence=(len(non_neutral) / max(len(directions), 1)) * (1 - conflict), analyzed_through=through or datetime(1970, 1, 1, tzinfo=UTC), maximum_depth=self.config.multi_timeframe.maximum_depth)

    def health(self) -> dict[str, object]:
        reasons = []
        if self.repository_mode == "memory" and self.config.persistence.required_in_production:
            reasons.append("ephemeral_persistence")
        if self.metrics.latest_successful_analysis_timestamp is None:
            reasons.append("no_analysis_completed")
        return {
            "status": "degraded" if reasons else "healthy",
            "engine_version": self.analyzer.version,
            "configuration_version": self.config.version,
            "repository_mode": self.repository_mode,
            "database_status": "available" if self.repository_mode == "sqlalchemy" else "unavailable",
            "checkpoint_status": self.recovery_status,
            "market_data_dependency": "configured",
            "smc_dependency": "configured" if self.smc else "unavailable",
            "liquidity_dependency": "configured" if self.liquidity else "unavailable",
            "volume_profile_dependency": "configured" if self.volume_profile else "unavailable",
            "latest_analysis": self.metrics.latest_successful_analysis_timestamp,
            "upstream_versions": {"market_data": "1.0.0", "smc": "1.0.0", "liquidity": "1.0.0", "volume_profile": "1.0.0"},
            "degradation_reasons": reasons,
            "checked_at": datetime.now(UTC),
        }

    def _record(self, snapshot: InstitutionalFlowAnalysisSnapshot, candle_count: int, latency: float) -> None:
        metrics = self.metrics
        metrics.analyses_completed += 1
        metrics.analyses_degraded += snapshot.status != AnalysisStatus.COMPLETE
        metrics.candles_considered += candle_count
        metrics.upstream_evidence_consumed += len(snapshot.evidence.accepted)
        metrics.evidence_rejected_as_future += len(snapshot.evidence.rejected_future_ids)
        metrics.evidence_rejected_as_invalid += len(snapshot.evidence.rejected_invalid_ids)
        metrics.evidence_deduplicated += len(snapshot.evidence.deduplicated_ids)
        metrics.evidence_correlation_discounted += len(snapshot.evidence.discounted_ids)
        metrics.evidence_conflicts += len(snapshot.state.explanation.contradicting_evidence_ids)
        metrics.initiative_events += snapshot.state.initiative is not None
        metrics.responsive_events += snapshot.state.responsive is not None
        metrics.absorption_inferences += snapshot.state.absorption is not None
        metrics.exhaustion_inferences += snapshot.state.exhaustion is not None
        metrics.campaign_phase_transitions += len(snapshot.transitions)
        metrics.directional_pressure_transitions += len(snapshot.transitions)
        metrics.cross_session_analyses += len(snapshot.state.cross_session)
        metrics.observe_latency(latency)
        metrics.latest_successful_analysis_timestamp = snapshot.analysis_timestamp.isoformat()

    async def _publish(self, snapshot: InstitutionalFlowAnalysisSnapshot, correlation_id: UUID) -> None:
        if snapshot.id in self._published:
            return
        try:
            await self.feature_store.write(FeatureRecord(feature_id=stable_id("feature", snapshot.id), correlation_id=correlation_id, namespace="institutional_flow", engine_name="institutional_flow", engine_version=self.analyzer.version, compatibility_version="1.0", values=self.features(snapshot)))
        except Exception:
            self.metrics.feature_publication_failures += 1
        events: list[tuple[type[Event], object]] = [(ParticipationChanged, snapshot.state.participation)]
        for cls, value in (
            (InitiativeActivityInferred, snapshot.state.initiative),
            (ResponsiveActivityInferred, snapshot.state.responsive),
            (AbsorptionLikeBehaviorInferred, snapshot.state.absorption),
            (ExhaustionLikeBehaviorInferred, snapshot.state.exhaustion),
            (InventoryBehaviorInferred, snapshot.state.inventory),
            (CampaignPhaseChanged, snapshot.state.campaign),
            (DirectionalPressureChanged, snapshot.state.pressure),
        ):
            if value is not None:
                events.append((cls, value))
        events.extend((CrossSessionFlowAnalyzed, value) for value in snapshot.state.cross_session)
        final = InstitutionalFlowDegraded if snapshot.status != AnalysisStatus.COMPLETE else InstitutionalFlowUpdated
        events.append((final, snapshot.state))
        try:
            for event_cls, event_value in events:
                await self.event_bus.publish(event_cls(event_id=stable_id("event", event_cls.__name__, snapshot.id, getattr(event_value, "id", "state")), correlation_id=correlation_id, source="institutional_flow", payload={"snapshot_id": str(snapshot.id), "probabilistic_inference": True, "configuration_version": snapshot.configuration_version, "engine_version": snapshot.engine_version}))
        except Exception:
            self.metrics.event_publication_failures += 1
        self._published.add(snapshot.id)

    @staticmethod
    def features(snapshot: InstitutionalFlowAnalysisSnapshot) -> dict[str, object]:
        state = snapshot.state
        return {
            "participation": state.participation.model_dump(mode="json"),
            "initiative": state.initiative.model_dump(mode="json") if state.initiative else None,
            "responsive": state.responsive.model_dump(mode="json") if state.responsive else None,
            "absorption_like": state.absorption.model_dump(mode="json") if state.absorption else None,
            "exhaustion_like": state.exhaustion.model_dump(mode="json") if state.exhaustion else None,
            "inventory_behavior": state.inventory.model_dump(mode="json"),
            "campaign_phase": state.campaign.model_dump(mode="json"),
            "directional_pressure": state.pressure.model_dump(mode="json"),
            "persistence": state.persistence.model_dump(mode="json"),
            "cross_session": [item.model_dump(mode="json") for item in state.cross_session],
            "confluences": [item.model_dump(mode="json") for item in state.confluences],
            "ambiguity": state.pressure.conflict,
            "quality": snapshot.quality.model_dump(mode="json"),
            "evidence_ids": [str(item.id) for item in snapshot.evidence.accepted],
            "source_traceability": snapshot.market_data_boundary,
            "probabilistic_inference": True,
            "trading_instruction": False,
            "configuration_version": snapshot.configuration_version,
            "engine_version": snapshot.engine_version,
            "snapshot_id": str(snapshot.id),
        }
