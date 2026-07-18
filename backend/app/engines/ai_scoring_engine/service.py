from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from time import perf_counter
from typing import Any

from backend.app.engines.market_data_engine import Timeframe
from backend.app.events import Event, EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .clock import Clock, SystemClock
from .config import AIScoringConfig
from .engine import DeterministicAIScoringEngine
from .events import AIScoringCompleted, AIScoringConflictDetected, AIScoringDegraded, AIScoringInsufficientEvidence
from .models import AIScoreSnapshot, ScoreMode, ScoreRequest, ScoreStatus, ScoringInput, SourceEvidence, SourceHealth, SourceState, stable_id
from .normalization import normalized_source
from .repository import AIScoringRepository

logger = logging.getLogger(__name__)


@dataclass
class AIScoringMetrics:
    requests_total: int = 0
    completed_total: int = 0
    failed_total: int = 0
    degraded_total: int = 0
    insufficient_total: int = 0
    conflicts_total: int = 0
    persistence_failures_total: int = 0
    feature_publish_failures_total: int = 0
    event_publish_failures_total: int = 0
    durations_ms: list[float] = field(default_factory=list)
    latest_snapshot_id: str | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "requests_total": self.requests_total,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
            "degraded_total": self.degraded_total,
            "insufficient_evidence_total": self.insufficient_total,
            "conflicts_total": self.conflicts_total,
            "persistence_failures_total": self.persistence_failures_total,
            "feature_publish_failures_total": self.feature_publish_failures_total,
            "event_publish_failures_total": self.event_publish_failures_total,
            "average_duration_ms": sum(self.durations_ms) / len(self.durations_ms) if self.durations_ms else 0,
            "maximum_duration_ms": max(self.durations_ms, default=0),
            "latest_snapshot_id": self.latest_snapshot_id,
        }


class AIScoringService:
    def __init__(
        self,
        repository: AIScoringRepository,
        event_bus: EventBus,
        feature_store: FeatureStore,
        config: AIScoringConfig,
        *,
        market_data: Any = None,
        smc: Any = None,
        liquidity: Any = None,
        volume_profile: Any = None,
        institutional_flow: Any = None,
        market_regime: Any = None,
        economic_calendar: Any = None,
        repository_mode: str = "memory",
        clock: Clock | None = None,
    ) -> None:
        self.repository = repository
        self.event_bus = event_bus
        self.feature_store = feature_store
        self.config = config
        self.clock = clock or SystemClock()
        self.engine = DeterministicAIScoringEngine(config, self.clock)
        self.repository_mode = repository_mode
        self.dependencies = {
            "market_data": market_data,
            "smc": smc,
            "liquidity": liquidity,
            "volume_profile": volume_profile,
            "institutional_flow": institutional_flow,
            "market_regime": market_regime,
            "economic_calendar": economic_calendar,
        }
        self.metrics = AIScoringMetrics()
        self.initialized = False
        self.closed = False

    async def start(self) -> None:
        self.initialized = True
        self.closed = False
        logger.info("AI Scoring Engine initialized", extra={"engine": "ai_scoring", "policy_version": self.config.policy_version})

    async def stop(self) -> None:
        self.closed = True
        logger.info("AI Scoring Engine stopped", extra={"engine": "ai_scoring"})

    async def calculate(self, request: ScoreRequest) -> AIScoreSnapshot:
        started = perf_counter()
        self.metrics.requests_total += 1
        try:
            scoring_input = await self.collect_input(request)
            candidate = self.engine.score(scoring_input)
            existing = await self.repository.find_by_fingerprint(candidate.metadata.input_fingerprint, candidate.mode)
            if existing:
                return existing
            snapshot = await self.repository.save_snapshot(candidate) if request.persist else candidate
            await self._publish(snapshot, request.publish_events)
            self.metrics.completed_total += 1
            self.metrics.degraded_total += snapshot.status in {ScoreStatus.DEGRADED, ScoreStatus.STALE}
            self.metrics.insufficient_total += snapshot.status == ScoreStatus.INSUFFICIENT_EVIDENCE
            self.metrics.conflicts_total += len(snapshot.conflicts)
            self.metrics.latest_snapshot_id = str(snapshot.snapshot_id)
            logger.info(
                "AI score completed",
                extra={"engine": "ai_scoring", "instrument": snapshot.instrument, "timeframe": snapshot.timeframe, "status": snapshot.status.value, "component_count": len(snapshot.components), "conflict_count": len(snapshot.conflicts)},
            )
            return snapshot
        except Exception:
            self.metrics.failed_total += 1
            logger.exception("AI score failed", extra={"engine": "ai_scoring", "instrument": request.instrument, "timeframe": request.timeframe})
            raise
        finally:
            self.metrics.durations_ms.append((perf_counter() - started) * 1000)

    async def calculate_input(self, scoring_input: ScoringInput, *, persist: bool = True, publish_events: bool = True) -> AIScoreSnapshot:
        candidate = self.engine.score(scoring_input)
        snapshot = await self.repository.save_snapshot(candidate) if persist else candidate
        await self._publish(snapshot, publish_events)
        return snapshot

    async def collect_input(self, request: ScoreRequest) -> ScoringInput:
        boundary = request.as_of or self.clock.now()
        if boundary.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        boundary = boundary.astimezone(UTC)
        if request.mode == ScoreMode.LIVE and boundary > self.clock.now() + timedelta(seconds=self.config.future_clock_skew_seconds):
            raise ValueError("future scoring boundary is prohibited")
        timeframe = Timeframe(request.timeframe)
        results = await asyncio.gather(
            *(self._collect_source(name, service, request.instrument, timeframe, boundary) for name, service in sorted(self.dependencies.items())),
            return_exceptions=True,
        )
        evidence: dict[str, SourceEvidence] = {}
        health = []
        for name, result in zip(sorted(self.dependencies), results, strict=True):
            if isinstance(result, BaseException):
                health.append(SourceHealth(source=name, state=SourceState.UNAVAILABLE, checked_at=boundary, reason_codes=("dependency_collection_failed",)))
            elif result is None:
                health.append(SourceHealth(source=name, state=SourceState.UNAVAILABLE, checked_at=boundary, reason_codes=("no_point_in_time_snapshot",)))
            else:
                evidence[name] = result
                health.append(SourceHealth(source=name, state=SourceState.DEGRADED if result.degraded else SourceState.AVAILABLE, checked_at=boundary, reason_codes=result.reason_codes))
        return ScoringInput(
            instrument=request.instrument,
            timeframe=request.timeframe,
            as_of=boundary,
            requested_at=self.clock.now(),
            mode=request.mode,
            market_data=evidence.get("market_data"),
            market_regime=evidence.get("market_regime"),
            smc=evidence.get("smc"),
            liquidity=evidence.get("liquidity"),
            volume_profile=evidence.get("volume_profile"),
            institutional_flow=evidence.get("institutional_flow"),
            economic_calendar=evidence.get("economic_calendar"),
            source_health=tuple(health),
        )

    async def _collect_source(self, name: str, service: Any, instrument: str, timeframe: Timeframe, boundary: datetime) -> SourceEvidence | None:
        if service is None:
            return None
        cfg = self.config.components[name]
        if name == "economic_calendar":
            context = await service.context(instrument, as_of=boundary, publish=False)
            values = context.model_dump(mode="json")
            timestamp = context.analysis_timestamp
            return normalized_source(name, cfg.source_group, values, timestamp, "1.0.0", str(context.context_id))
        if name == "market_data":
            state = await service.state(instrument, timeframe, at=boundary)
            quality = 0.0 if state.provider_health in {"offline", "unavailable"} else 1.0
            return SourceEvidence(
                source=name,
                source_group=cfg.source_group,
                source_version="1.0.0",
                evidence_id=f"{instrument}:{timeframe.value}:{state.as_of.isoformat()}",
                source_timestamp=state.as_of,
                observation_timestamp=state.as_of,
                publication_timestamp=state.as_of,
                direction=0,
                confidence=quality,
                quality=quality,
                risk=0.5 if state.provider_health != "healthy" else 0,
                degraded=quality < 1,
                reason_codes=(f"provider_{state.provider_health}",),
            )
        argument_name = "timestamp" if name == "smc" else "at"
        snapshot = await service.state(instrument, timeframe, **{argument_name: boundary})
        if snapshot is None:
            return None
        values = service.features(snapshot)
        timestamp = snapshot.analysis_timestamp
        identifier = str(snapshot.snapshot_id if name == "market_regime" else snapshot.id)
        version = str(snapshot.engine_version)
        return normalized_source(name, cfg.source_group, values, timestamp, version, identifier)

    async def _publish(self, snapshot: AIScoreSnapshot, publish_events: bool) -> None:
        correlation_id = stable_id("ai-score-correlation", snapshot.metadata.input_fingerprint)
        try:
            await self.feature_store.write(
                FeatureRecord(
                    feature_id=stable_id("ai-score-feature", snapshot.snapshot_id),
                    correlation_id=correlation_id,
                    namespace="ai_score",
                    engine_name="ai_scoring",
                    engine_version=self.config.engine_version,
                    compatibility_version=self.config.schema_version,
                    values={
                        "directional": snapshot.directional_score,
                        "confidence": snapshot.confidence_score,
                        "risk": snapshot.market_risk_score,
                        "alignment": snapshot.evidence_alignment_score,
                        "quality": snapshot.data_quality_score,
                        "composite": snapshot.composite_score,
                        "status": snapshot.status.value,
                        "policy_version": snapshot.policy_version,
                        "as_of": snapshot.as_of.isoformat(),
                        "trading_instruction": False,
                    },
                    created_at=snapshot.calculated_at,
                )
            )
        except Exception:
            self.metrics.feature_publish_failures_total += 1
        if not publish_events or (snapshot.mode == ScoreMode.REPLAY and not self.config.publish_replay_events):
            return
        event_type: type[Event] = AIScoringCompleted
        if snapshot.status in {ScoreStatus.DEGRADED, ScoreStatus.STALE}:
            event_type = AIScoringDegraded
        elif snapshot.status == ScoreStatus.INSUFFICIENT_EVIDENCE:
            event_type = AIScoringInsufficientEvidence
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "snapshot_id": str(snapshot.snapshot_id),
            "instrument": snapshot.instrument,
            "timeframe": snapshot.timeframe,
            "as_of": snapshot.as_of.isoformat(),
            "status": snapshot.status.value,
            "directional_score": snapshot.directional_score,
            "confidence_score": snapshot.confidence_score,
            "risk_score": snapshot.market_risk_score,
            "policy_version": snapshot.policy_version,
            "input_fingerprint": snapshot.metadata.input_fingerprint,
            "missing_source_count": len(snapshot.missing_sources),
            "degraded_source_count": len(snapshot.degraded_sources),
            "conflict_count": len(snapshot.conflicts),
            "trading_instruction": False,
        }
        try:
            await self.event_bus.publish(event_type(event_id=stable_id("ai-score-event", event_type.__name__, snapshot.snapshot_id), correlation_id=correlation_id, occurred_at=snapshot.calculated_at, source="ai_scoring", payload=payload))
            if snapshot.conflicts:
                await self.event_bus.publish(AIScoringConflictDetected(event_id=stable_id("ai-score-conflicts", snapshot.snapshot_id), correlation_id=correlation_id, occurred_at=snapshot.calculated_at, source="ai_scoring", payload=payload))
        except Exception:
            self.metrics.event_publish_failures_total += 1

    async def cleanup(self) -> int:
        now = self.clock.now()
        live = await self.repository.prune(now - timedelta(days=self.config.retention.live_days), ScoreMode.LIVE, self.config.retention.cleanup_batch_size)
        replay = await self.repository.prune(now - timedelta(days=self.config.retention.replay_days), ScoreMode.REPLAY, self.config.retention.cleanup_batch_size)
        return live + replay

    def health(self) -> dict[str, object]:
        sources = {name: "configured" if service is not None else "unavailable" for name, service in self.dependencies.items()}
        reasons = []
        if not self.initialized or self.closed:
            reasons.append("service_not_running")
        if self.repository_mode == "memory" and self.config.persistence_required_in_production:
            reasons.append("ephemeral_persistence")
        if sum(value == "configured" for value in sources.values()) < self.config.evidence.minimum_components:
            reasons.append("insufficient_configured_sources")
        status = "unavailable" if not self.initialized or self.closed else "degraded" if reasons or "unavailable" in sources.values() else "healthy"
        return {
            "status": status,
            "engine": "ai_scoring",
            "version": self.config.engine_version,
            "policy": {"name": self.config.policy_name, "version": self.config.policy_version},
            "persistence": {"status": "healthy" if self.repository_mode != "memory" else "degraded", "mode": self.repository_mode},
            "feature_store": {"status": "healthy" if self.metrics.feature_publish_failures_total == 0 else "degraded"},
            "event_bus": {"status": "healthy" if self.metrics.event_publish_failures_total == 0 else "degraded"},
            "sources": sources,
            "degradation_reasons": reasons,
            "probabilistic_context": True,
            "trading_instruction": False,
            "timestamp": self.clock.now(),
        }
