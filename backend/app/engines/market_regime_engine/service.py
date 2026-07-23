from datetime import UTC, datetime
from statistics import pstdev
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from backend.app.core.bounded import BoundedSet
from backend.app.engines.market_data_engine import MarketDataService, Timeframe
from backend.app.events import Event, EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .analyzer import BaselineMarketRegimeAnalyzer, clamp
from .config import MarketRegimeConfig
from .contracts import MarketRegimeContext
from .events import (
    AuctionRegimeChanged,
    CompressionDetected,
    CrossSessionHandoffDetected,
    ExpansionDetected,
    MarketRegimeChanged,
    MarketRegimeDegraded,
    MarketRegimeRecoveryCompleted,
    MarketRegimeReplayCompleted,
    MarketRegimeSnapshotCreated,
    MultiTimeframeConflictDetected,
    RegimeReversalRiskDetected,
    RegimeTransitionConfirmed,
    RegimeTransitionFailed,
    RegimeTransitionStarted,
    RegimeWeakeningDetected,
    TrendRegimeChanged,
    VolatilityRegimeChanged,
)
from .models import (
    CrossSessionRegimeState,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceRole,
    ExpansionRegime,
    MarketRegimeEvidence,
    MarketRegimeSnapshot,
    MultiTimeframeRegimeState,
    ProcessingMode,
    RegimeLifecycle,
    RegimeTransition,
    TransitionState,
    TrendMaturity,
    stable_id,
)
from .repository import InMemoryMarketRegimeRepository, MarketRegimeRepository


class MarketRegimeMetrics:
    def __init__(self) -> None:
        self.analysis_count = 0
        self.successful_analysis_count = 0
        self.failed_analysis_count = 0
        self.degraded_analysis_count = 0
        self.accepted_evidence_count = 0
        self.rejected_evidence_count = 0
        self.discounted_evidence_count = 0
        self.contradiction_count = 0
        self.unavailable_evidence_count = 0
        self.missing_dependency_count = 0
        self.regime_transition_count = 0
        self.replay_count = 0
        self.replay_failures = 0
        self.recovery_count = 0
        self.recovery_failures = 0
        self.checkpoint_validation_failures = 0
        self.repository_conflicts = 0
        self.persistence_writes = 0
        self.event_publication_failures = 0
        self.feature_publication_failures = 0
        self.average_latency_ms = 0.0
        self.maximum_latency_ms = 0.0
        self.p95_latency_ms = 0.0
        self.latest_analysis_timestamp: str | None = None
        self.latest_snapshot_id: str | None = None
        self.latest_regime: str | None = None
        self.latest_confidence: float | None = None
        self.latest_ambiguity: float | None = None
        self._latencies: list[float] = []
        self._regimes: dict[str, int] = {}

    def observe(self, snapshot: MarketRegimeSnapshot, latency: float) -> None:
        self.analysis_count += 1
        self.successful_analysis_count += 1
        self.degraded_analysis_count += snapshot.degradation.is_degraded
        self.accepted_evidence_count += sum(item.accepted for item in snapshot.evidence)
        self.rejected_evidence_count += sum(item.rejected for item in snapshot.evidence)
        self.discounted_evidence_count += sum(item.discounted for item in snapshot.evidence)
        self.contradiction_count += sum(item.contradicting for item in snapshot.evidence)
        self.unavailable_evidence_count += sum(item.unavailable for item in snapshot.evidence)
        self.missing_dependency_count += len(snapshot.degradation.missing_dependencies)
        self.regime_transition_count += snapshot.transition_state == TransitionState.CONFIRMED
        self.persistence_writes += 1
        self._latencies = (self._latencies + [latency])[-100:]
        self.average_latency_ms = sum(self._latencies) / len(self._latencies)
        self.maximum_latency_ms = max(self._latencies)
        self.p95_latency_ms = sorted(self._latencies)[max(0, round(0.95 * len(self._latencies)) - 1)]
        self.latest_analysis_timestamp = snapshot.analysis_timestamp.isoformat()
        self.latest_snapshot_id = str(snapshot.snapshot_id)
        self.latest_regime = snapshot.dominant_regime.value
        self.latest_confidence = snapshot.confidence
        self.latest_ambiguity = snapshot.ambiguity
        self._regimes[snapshot.dominant_regime.value] = self._regimes.get(snapshot.dominant_regime.value, 0) + 1

    def snapshot(self) -> dict[str, object]:
        result = {key: value for key, value in vars(self).items() if key not in {"_latencies", "_regimes"}}
        result["regime_distribution"] = dict(sorted(self._regimes.items()))
        return result


class MarketRegimeService:
    def __init__(
        self,
        market_data: MarketDataService,
        smc: Any,
        liquidity: Any,
        volume_profile: Any,
        institutional_flow: Any,
        event_bus: EventBus,
        feature_store: FeatureStore,
        config: MarketRegimeConfig | None = None,
        repository: MarketRegimeRepository | None = None,
        repository_mode: str = "memory",
    ) -> None:
        self.market_data = market_data
        self.smc = smc
        self.liquidity = liquidity
        self.volume_profile = volume_profile
        self.institutional_flow = institutional_flow
        self.event_bus = event_bus
        self.feature_store = feature_store
        self.config = config or MarketRegimeConfig()
        self.repository = repository or InMemoryMarketRegimeRepository()
        self.repository_mode = repository_mode
        self.analyzer = BaselineMarketRegimeAnalyzer(self.config)
        self.metrics = MarketRegimeMetrics()
        self.recovery_state = "not_attempted"
        self._published = BoundedSet[UUID](10_000)

    async def restore(self) -> int:
        try:
            candidates = await self.repository.checkpoints()
            valid = tuple(
                item
                for item in candidates
                if item.engine_version == self.analyzer.version
                and item.schema_version == self.config.versions.schema_version
                and item.configuration_version == self.config.version
                and item.algorithm_version == self.config.versions.algorithm_version
            )
            self.metrics.recovery_count += len(valid)
            self.recovery_state = "recovered" if valid else "clean_start"
            if valid:
                await self.event_bus.publish(
                    MarketRegimeRecoveryCompleted(
                        event_id=stable_id("recovery-event", *(item.snapshot_id for item in valid)),
                        correlation_id=uuid4(),
                        source="market_regime",
                        payload={"count": len(valid), "probabilistic_inference": True, "trading_instruction": False},
                    )
                )
            return len(valid)
        except Exception:
            self.metrics.recovery_failures += 1
            self.metrics.checkpoint_validation_failures += 1
            self.recovery_state = "failed"
            raise

    async def analyze_snapshot(
        self, symbol: str, timeframe: Timeframe, *, timestamp: datetime | None = None, limit: int = 500, mode: ProcessingMode = ProcessingMode.SNAPSHOT
    ) -> MarketRegimeSnapshot:
        limit = min(limit, self.config.processing.maximum_candles)
        candles = await (
            self.market_data.replay(symbol, timeframe, timestamp, limit=limit)
            if timestamp
            else self.market_data.history(symbol, timeframe, end=timestamp, limit=limit)
        )
        if not candles:
            raise ValueError("Market Data is unavailable; Market Regime cannot safely proceed")
        boundary = timestamp or candles[-1].timestamp
        evidence, missing, failed, session = await self._upstream(symbol, timeframe, boundary)
        context = MarketRegimeContext(tuple(candles), evidence, boundary, missing, failed, cross_session=session)
        return await self.analyze_context(context, mode)

    async def update_incremental(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> MarketRegimeSnapshot:
        return await self.analyze_snapshot(symbol, timeframe, limit=limit, mode=ProcessingMode.INCREMENTAL)

    async def replay(self, symbol: str, timeframe: Timeframe, timestamp: datetime, limit: int = 500) -> MarketRegimeSnapshot:
        try:
            snapshot = await self.analyze_snapshot(symbol, timeframe, timestamp=timestamp, limit=limit, mode=ProcessingMode.REPLAY)
            self.metrics.replay_count += 1
            await self.event_bus.publish(
                MarketRegimeReplayCompleted(
                    event_id=stable_id("replay-event", snapshot.snapshot_id),
                    correlation_id=uuid4(),
                    source="market_regime",
                    payload=self._event_payload(snapshot),
                )
            )
            return snapshot
        except Exception:
            self.metrics.replay_failures += 1
            raise

    async def recover(self, symbol: str, timeframe: Timeframe) -> MarketRegimeSnapshot | None:
        snapshot = await self.repository.load_checkpoint(symbol, timeframe)
        if snapshot and (
            snapshot.engine_version != self.analyzer.version
            or snapshot.schema_version != self.config.versions.schema_version
            or snapshot.configuration_version != self.config.version
            or snapshot.algorithm_version != self.config.versions.algorithm_version
            or snapshot.symbol != symbol.replace("/", "")
            or snapshot.timeframe != timeframe
        ):
            raise ValueError("checkpoint is incompatible with the requested engine, versions, symbol, or timeframe")
        return snapshot

    async def analyze_context(
        self, context: MarketRegimeContext, mode: ProcessingMode = ProcessingMode.SNAPSHOT, correlation_id: UUID | None = None
    ) -> MarketRegimeSnapshot:
        started = perf_counter()
        previous = None
        if context.candles:
            previous = await self.repository.get_latest_snapshot(context.candles[-1].symbol, context.candles[-1].timeframe)
        try:
            snapshot = self.analyzer.analyze_snapshot(context, mode, previous, self.repository_mode, self.recovery_state)
            await self.repository.save_snapshot(snapshot)
            await self.repository.save_evidence(snapshot)
            await self.repository.save_checkpoint(snapshot)
            transition = self._transition(snapshot)
            if transition:
                await self.repository.save_transition(transition)
            await self.repository.prune_history(snapshot.symbol, snapshot.timeframe, self.config.processing.retention_snapshots)
        except Exception:
            self.metrics.failed_analysis_count += 1
            raise
        self.metrics.observe(snapshot, (perf_counter() - started) * 1000)
        await self._publish(snapshot, previous, correlation_id or uuid4())
        return snapshot

    async def state(self, symbol: str, timeframe: Timeframe, at: datetime | None = None) -> MarketRegimeSnapshot | None:
        if at is None:
            return await self.repository.get_latest_snapshot(symbol, timeframe)
        values = await self.repository.list_snapshots(symbol, timeframe, 0, self.config.processing.retention_snapshots)
        return next((item for item in values if item.analysis_timestamp <= at), None)

    async def history(self, symbol: str, timeframe: Timeframe, offset: int = 0, limit: int = 100) -> tuple[MarketRegimeSnapshot, ...]:
        return await self.repository.list_snapshots(symbol, timeframe, offset, min(limit, self.config.processing.maximum_page_size))

    async def multi_timeframe(self, symbol: str, timeframe: Timeframe, at: datetime | None = None, limit: int = 500) -> MultiTimeframeRegimeState:
        included: list[str] = []
        unavailable: list[str] = []
        regimes: list[MarketRegimeSnapshot] = []
        supported = {item.value: item for item in Timeframe}
        for name in self.config.multi_timeframe.hierarchy[: self.config.multi_timeframe.maximum_depth]:
            item = supported.get(name)
            if item is None:
                unavailable.append(name)
                continue
            try:
                snapshot = await (self.replay(symbol, item, at, limit) if at else self.analyze_snapshot(symbol, item, limit=limit))
            except Exception:
                unavailable.append(name)
                continue
            if at and snapshot.analysis_timestamp > at:
                unavailable.append(name)
                continue
            included.append(name)
            regimes.append(snapshot)
        directions = [item.directional_bias for item in regimes if item.directional_bias != EvidenceDirection.NEUTRAL]
        aligned = bool(directions) and len(set(directions)) == 1
        conflict = 0.0 if aligned or len(directions) < 2 else 1 - max(directions.count(value) for value in set(directions)) / len(directions)
        alignment = 1 - conflict if regimes else 0.0
        return MultiTimeframeRegimeState(
            requested_timeframe=timeframe.value,
            included_timeframes=tuple(included),
            excluded_timeframes=tuple(self.config.multi_timeframe.hierarchy[self.config.multi_timeframe.maximum_depth :]),
            unavailable_timeframes=tuple(unavailable),
            dominant_timeframe=included[-1] if included else None,
            higher_timeframe_regime=regimes[-1].dominant_regime if regimes else None,
            lower_timeframe_regime=regimes[0].dominant_regime if regimes else None,
            alignment_score=alignment,
            conflict_score=conflict,
            directional_alignment=alignment,
            volatility_alignment=clamp(1 - (pstdev([item.volatility_score for item in regimes]) if len(regimes) > 1 else 0)),
            auction_alignment=clamp(1 - len({item.auction_regime for item in regimes}) / max(len(regimes), 1)),
            confidence=clamp(len(regimes) / self.config.multi_timeframe.maximum_depth * alignment),
            ambiguity=clamp(conflict + len(unavailable) / max(len(included) + len(unavailable), 1)),
            explanation="Time-valid regimes are aligned."
            if aligned
            else "Time-valid regimes are mixed or unavailable; lower-timeframe state is not overridden.",
        )

    async def _upstream(
        self, symbol: str, timeframe: Timeframe, boundary: datetime
    ) -> tuple[tuple[MarketRegimeEvidence, ...], tuple[str, ...], tuple[str, ...], CrossSessionRegimeState]:
        missing = []
        failed = []
        values: list[MarketRegimeEvidence] = []
        flow_snapshot = None
        if self.institutional_flow is None:
            missing.append("institutional_flow")
        else:
            try:
                flow_snapshot = await self.institutional_flow.state(symbol, timeframe, boundary)
                if flow_snapshot is None:
                    flow_snapshot = await self.institutional_flow.replay(symbol, timeframe, boundary, self.config.processing.default_candles)
                values.extend(self._flow_evidence(flow_snapshot, boundary))
            except Exception:
                failed.append("institutional_flow")
        for name, provider in (("smc", self.smc), ("liquidity", self.liquidity), ("volume_profile", self.volume_profile)):
            if provider is None:
                missing.append(name)
            elif not any(item.source_engine == name for item in values):
                try:
                    state = await provider.state(symbol, timeframe, boundary)
                    if state is None:
                        missing.append(name)
                except Exception:
                    failed.append(name)
        return tuple(values), tuple(missing), tuple(failed), self._session(flow_snapshot)

    def _flow_evidence(self, snapshot: Any, boundary: datetime) -> tuple[MarketRegimeEvidence, ...]:
        mapping = {
            "market_data": EvidenceFamily.MARKET_DATA,
            "smc": EvidenceFamily.STRUCTURE,
            "liquidity": EvidenceFamily.LIQUIDITY,
            "volume_profile": EvidenceFamily.VOLUME_PROFILE,
        }
        result = []
        for item in snapshot.evidence.accepted:
            family = mapping.get(item.source_engine.value, EvidenceFamily.INSTITUTIONAL_FLOW)
            direction = EvidenceDirection(item.direction.value) if item.direction.value in {"bullish", "bearish", "neutral"} else EvidenceDirection.UNKNOWN
            result.append(
                MarketRegimeEvidence(
                    evidence_id=stable_id("upstream", item.id),
                    source_engine=item.source_engine.value,
                    source_engine_version=item.engine_version,
                    source_object_type=item.evidence_type.value,
                    source_object_id=item.source_object_id,
                    source_snapshot_id=str(snapshot.id),
                    symbol=snapshot.symbol,
                    timeframe=item.timeframe,
                    session=item.session.value,
                    event_timestamp=item.source_timestamp,
                    available_at=item.availability_timestamp,
                    analysis_boundary=boundary,
                    direction=direction,
                    role=EvidenceRole.CONTRADICTING if item.role.value == "contradicting" else EvidenceRole.SUPPORTING,
                    family=family,
                    subfamily=item.evidence_type.value,
                    raw_strength=item.strength,
                    normalized_strength=item.strength,
                    source_confidence=item.confidence,
                    source_quality=item.quality,
                    effective_weight=0,
                    correlation_group=f"{item.source_engine.value}:{item.correlation_group.value}:{item.source_object_id}",
                    correlation_discount=1,
                    decay_factor=1,
                    contradicting=item.role.value == "contradicting",
                    payload_summary=item.explanation,
                    metadata={"upstream_evidence_id": str(item.id)},
                )
            )
        pressure = snapshot.state.pressure
        direction = (
            EvidenceDirection.BULLISH
            if pressure.net_pressure > 0.1
            else EvidenceDirection.BEARISH
            if pressure.net_pressure < -0.1
            else EvidenceDirection.NEUTRAL
        )
        result.append(
            MarketRegimeEvidence(
                evidence_id=stable_id("flow", snapshot.id),
                source_engine="institutional_flow",
                source_engine_version=snapshot.engine_version,
                source_object_type="institutional_flow_snapshot",
                source_object_id=str(snapshot.state.id),
                source_snapshot_id=str(snapshot.id),
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                session=snapshot.session.value,
                event_timestamp=snapshot.analysis_timestamp,
                available_at=snapshot.availability_timestamp,
                analysis_boundary=boundary,
                direction=direction,
                family=EvidenceFamily.INSTITUTIONAL_FLOW,
                subfamily="directional_pressure",
                raw_strength=abs(pressure.net_pressure),
                normalized_strength=abs(pressure.net_pressure),
                source_confidence=pressure.confidence,
                source_quality=pressure.quality,
                effective_weight=0,
                correlation_group=f"institutional-flow:{snapshot.id}",
                correlation_discount=1,
                decay_factor=1,
                contradicting=pressure.conflict > self.config.thresholds.trend,
                payload_summary="Public probabilistic Institutional Flow pressure and campaign context; no participant identity is inferred.",
                metadata={"campaign": snapshot.state.campaign.phase.value, "participation": snapshot.state.participation.level.value},
            )
        )
        return tuple(result)

    @staticmethod
    def _session(snapshot: Any | None) -> CrossSessionRegimeState:
        sessions = snapshot.state.cross_session if snapshot else ()
        if not sessions:
            return CrossSessionRegimeState(
                current_session=snapshot.session.value if snapshot else "unknown",
                continuation_score=0,
                handoff_score=0,
                reversal_score=0,
                session_alignment="no_prior_session_context",
                confidence=0,
                ambiguity=1,
                explanation="No completed prior-session context is time-valid.",
            )
        item = sessions[-1]
        return CrossSessionRegimeState(
            current_session=item.current_session.value,
            previous_session=item.previous_session.value,
            continuation_score=item.strength if item.relationship == "continuation" else 0,
            handoff_score=item.strength if item.relationship == "handoff" else 0,
            reversal_score=item.strength if item.relationship == "reversal" else 0,
            session_alignment=item.relationship,
            dominant_session=item.current_session.value,
            confidence=item.confidence,
            ambiguity=1 - item.confidence,
            explanation=f"Time-valid session relationship is {item.relationship}.",
        )

    def _transition(self, snapshot: MarketRegimeSnapshot) -> RegimeTransition | None:
        if snapshot.previous_dominant_regime is None or snapshot.transition_state == TransitionState.NONE:
            return None
        return RegimeTransition(
            transition_id=stable_id(
                "transition", snapshot.symbol, snapshot.timeframe, snapshot.previous_dominant_regime, snapshot.dominant_regime, snapshot.transition_started_at
            ),
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            from_regime=snapshot.previous_dominant_regime,
            to_regime=snapshot.dominant_regime,
            started_at=snapshot.transition_started_at or snapshot.analysis_timestamp,
            confirmed_at=snapshot.transition_confirmed_at,
            state=snapshot.transition_state,
            confidence=snapshot.confidence,
            ambiguity=snapshot.ambiguity,
            supporting_evidence_ids=tuple(item.evidence_id for item in snapshot.evidence if item.accepted and not item.contradicting),
            contradicting_evidence_ids=tuple(item.evidence_id for item in snapshot.evidence if item.contradicting),
            reasoning_summary=snapshot.reasoning_summary,
        )

    async def _publish(self, snapshot: MarketRegimeSnapshot, previous: MarketRegimeSnapshot | None, correlation_id: UUID) -> None:
        if snapshot.snapshot_id in self._published:
            return
        try:
            await self.feature_store.write(
                FeatureRecord(
                    feature_id=stable_id("feature", snapshot.snapshot_id),
                    correlation_id=correlation_id,
                    namespace="market_regime",
                    engine_name="market_regime",
                    engine_version=self.analyzer.version,
                    compatibility_version=self.config.compatibility_version,
                    values=self.features(snapshot),
                )
            )
        except Exception:
            self.metrics.feature_publication_failures += 1
        events: list[type[Event]] = [MarketRegimeSnapshotCreated]
        if previous and previous.dominant_regime != snapshot.dominant_regime:
            events.append(MarketRegimeChanged)
        if previous and previous.trend_regime != snapshot.trend_regime:
            events.append(TrendRegimeChanged)
        if previous and previous.volatility_regime != snapshot.volatility_regime:
            events.append(VolatilityRegimeChanged)
        if previous and previous.auction_regime != snapshot.auction_regime:
            events.append(AuctionRegimeChanged)
        if snapshot.expansion_regime == ExpansionRegime.COMPRESSION:
            events.append(CompressionDetected)
        if snapshot.expansion_regime in {ExpansionRegime.EARLY_EXPANSION, ExpansionRegime.EXPANSION, ExpansionRegime.LATE_EXPANSION}:
            events.append(ExpansionDetected)
        if snapshot.transition_state in {TransitionState.WATCH, TransitionState.DEVELOPING}:
            events.append(RegimeTransitionStarted)
        if snapshot.transition_state == TransitionState.CONFIRMED:
            events.append(RegimeTransitionConfirmed)
        if snapshot.transition_state == TransitionState.FAILED:
            events.append(RegimeTransitionFailed)
        if snapshot.lifecycle == RegimeLifecycle.WEAKENING:
            events.append(RegimeWeakeningDetected)
        if snapshot.trend_maturity == TrendMaturity.EXHAUSTION_RISK:
            events.append(RegimeReversalRiskDetected)
        if snapshot.multi_timeframe.conflict_score > 0.4:
            events.append(MultiTimeframeConflictDetected)
        if snapshot.cross_session.handoff_score > 0.4:
            events.append(CrossSessionHandoffDetected)
        if snapshot.degradation.is_degraded:
            events.append(MarketRegimeDegraded)
        try:
            for cls in events:
                await self.event_bus.publish(
                    cls(
                        event_id=stable_id("event", cls.__name__, snapshot.snapshot_id),
                        correlation_id=correlation_id,
                        source="market_regime",
                        payload=self._event_payload(snapshot),
                    )
                )
        except Exception:
            self.metrics.event_publication_failures += 1
        self._published.add(snapshot.snapshot_id)

    @staticmethod
    def _event_payload(snapshot: MarketRegimeSnapshot) -> dict[str, object]:
        return {
            "snapshot_id": str(snapshot.snapshot_id),
            "symbol": snapshot.symbol,
            "timeframe": snapshot.timeframe.value,
            "analysis_timestamp": snapshot.analysis_timestamp.isoformat(),
            "current_state": snapshot.dominant_regime.value,
            "confidence": snapshot.confidence,
            "ambiguity": snapshot.ambiguity,
            "configuration_version": snapshot.configuration_version,
            "probabilistic_inference": True,
            "trading_instruction": False,
            "replay_mode": snapshot.processing_mode == ProcessingMode.REPLAY,
        }

    @staticmethod
    def features(snapshot: MarketRegimeSnapshot) -> dict[str, object]:
        values = {
            "dominant": snapshot.dominant_regime.value,
            "trend": snapshot.trend_regime.value,
            "volatility": snapshot.volatility_regime.value,
            "auction": snapshot.auction_regime.value,
            "expansion": snapshot.expansion_regime.value,
            "structure": snapshot.structural_regime.value,
            "participation": snapshot.participation_regime.value,
            "inventory": snapshot.inventory_regime.value,
            "lifecycle": snapshot.lifecycle.value,
            "persistence": snapshot.persistence.value,
            "trend_maturity": snapshot.trend_maturity.value,
            "directional_bias": snapshot.directional_bias.value,
            "bullish_score": snapshot.bullish_score,
            "bearish_score": snapshot.bearish_score,
            "net_directional_score": snapshot.net_directional_score,
            "balance_score": snapshot.balance_score,
            "imbalance_score": snapshot.imbalance_score,
            "compression_score": snapshot.compression_score,
            "expansion_score": snapshot.expansion_score,
            "trend_strength": snapshot.trend_strength,
            "confidence": snapshot.confidence,
            "quality": snapshot.quality,
            "ambiguity": snapshot.ambiguity,
            "conflict": snapshot.conflict_score,
            "transition": snapshot.transition_state.value,
            "mtf_alignment": snapshot.multi_timeframe.alignment_score,
            "session_state": snapshot.cross_session.session_alignment,
        }
        return {
            **values,
            "feature_version": "1.0",
            "schema_version": snapshot.schema_version,
            "configuration_version": snapshot.configuration_version,
            "analysis_timestamp": snapshot.analysis_timestamp.isoformat(),
            "snapshot_id": str(snapshot.snapshot_id),
            "probabilistic_inference": True,
            "trading_instruction": False,
        }

    def health(self) -> dict[str, object]:
        reasons = []
        if self.repository_mode == "memory" and self.config.persistence.required_in_production:
            reasons.append("ephemeral_persistence")
        if self.metrics.latest_analysis_timestamp is None:
            reasons.append("no_analysis_completed")
        dependencies = {
            name: "configured" if value is not None else "unavailable"
            for name, value in (
                ("market_data", self.market_data),
                ("smc", self.smc),
                ("liquidity", self.liquidity),
                ("volume_profile", self.volume_profile),
                ("institutional_flow", self.institutional_flow),
            )
        }
        return {
            "status": "degraded" if reasons or "unavailable" in dependencies.values() else "healthy",
            "engine_name": "market_regime",
            "engine_version": self.analyzer.version,
            "schema_version": self.config.versions.schema_version,
            "configuration_version": self.config.version,
            "algorithm_version": self.config.versions.algorithm_version,
            "enabled": self.config.enabled,
            "initialized": True,
            "repository_mode": self.repository_mode,
            "recovery_state": self.recovery_state,
            "is_degraded": bool(reasons),
            "dependency_status": dependencies,
            "latest_analysis_timestamp": self.metrics.latest_analysis_timestamp,
            "latest_snapshot_id": self.metrics.latest_snapshot_id,
            "latest_regime": self.metrics.latest_regime,
            "latest_confidence": self.metrics.latest_confidence,
            "latest_ambiguity": self.metrics.latest_ambiguity,
            "evidence_statistics": {
                "accepted": self.metrics.accepted_evidence_count,
                "rejected": self.metrics.rejected_evidence_count,
                "discounted": self.metrics.discounted_evidence_count,
            },
            "transition_statistics": {"confirmed": self.metrics.regime_transition_count},
            "latency_statistics": {
                "average_ms": self.metrics.average_latency_ms,
                "maximum_ms": self.metrics.maximum_latency_ms,
                "p95_ms": self.metrics.p95_latency_ms,
            },
            "failure_statistics": {
                "analysis": self.metrics.failed_analysis_count,
                "event": self.metrics.event_publication_failures,
                "feature": self.metrics.feature_publication_failures,
            },
            "checkpoint_statistics": {"recoveries": self.metrics.recovery_count, "validation_failures": self.metrics.checkpoint_validation_failures},
            "replay_statistics": {"runs": self.metrics.replay_count, "failures": self.metrics.replay_failures},
            "degradation_reasons": reasons,
            "checked_at": datetime.now(UTC),
        }
