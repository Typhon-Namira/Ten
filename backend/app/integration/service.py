from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from backend.app.engines.ai_scoring_engine import ScoreMode, ScoreRequest
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.events import NewCandle
from backend.app.engines.signal_decision_engine import DecisionMode, DecisionRequest
from backend.app.events import Event, EventBus

from .config import IntegrationConfig
from .models import CanonicalEventEnvelope, DataQualityIssue, DataQualityStatus, EvidenceReference, IntegrationMode, IntegrationSnapshot, IntegrationTraceRecord, OperationalSignal, SnapshotStatus, TraceStatus, canonical_hash, semantic_uuid
from .repository import IntegrationRepository


class FullSystemIntegrationService:
    """Coordinates existing engines at a final-candle boundary; contains no analytics."""

    def __init__(self, *, event_bus: EventBus, repository: IntegrationRepository, config: IntegrationConfig, market_data: Any, smc: Any, liquidity: Any, volume_profile: Any, institutional_flow: Any, market_regime: Any, economic_calendar: Any, ai_scoring: Any, signal_decision: Any, repository_mode: str = "memory", clock: Callable[[], datetime] | None = None) -> None:
        self.event_bus, self.repository, self.config = event_bus, repository, config
        self.market_data, self.smc, self.liquidity = market_data, smc, liquidity
        self.volume_profile, self.institutional_flow = volume_profile, institutional_flow
        self.market_regime, self.economic_calendar = market_regime, economic_calendar
        self.ai_scoring, self.signal_decision = ai_scoring, signal_decision
        self.repository_mode = repository_mode
        self.clock = clock or (lambda: datetime.now(UTC))
        self._unsubscribe: Callable[[], None] | None = None
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.started = False
        self.failures = 0

    async def start(self) -> None:
        if self._unsubscribe is None:
            self._unsubscribe = self.event_bus.subscribe(NewCandle, self._on_candle)
        self.started = True

    async def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        self.started = False

    async def _on_candle(self, event: Event) -> None:
        candle = Candle.model_validate(event.payload)
        now = self.clock()
        close = candle.timestamp + candle.timeframe.duration
        envelope = CanonicalEventEnvelope.final_candle(candle, event.correlation_id, max(now, close, candle.ingestion_timestamp))
        self.config.instrument(candle.symbol)
        await self.repository.enqueue(envelope)
        if self.config.worker.embedded_api_worker:
            await self.process_outbox_once()

    async def process_outbox_once(self) -> int:
        items = await self.repository.pending(self.clock(), self.config.limits.outbox_batch_size)
        for item in items:
            try:
                await self.process(item.envelope)
                await self.repository.complete(item.outbox_id, self.clock())
            except Exception as exc:
                self.failures += 1
                await self.repository.fail(item.outbox_id, type(exc).__name__)
        return len(items)

    async def process(self, envelope: CanonicalEventEnvelope) -> OperationalSignal | None:
        if envelope.mode != IntegrationMode.LIVE:
            raise ValueError("live integration rejects replay envelopes")
        if envelope.schema_version != self.config.policy.schema_version or envelope.event_type != "market.candle.closed":
            raise ValueError("unsupported integration envelope")
        if await self.repository.processed(envelope.event_id):
            await self._trace(envelope, TraceStatus.DUPLICATE, "integration", (), ())
            return await self.repository.latest_signal(envelope.instrument_id, envelope.timeframe)
        assert envelope.instrument_id and envelope.timeframe
        lock = self._locks.setdefault((envelope.instrument_id, envelope.timeframe), asyncio.Lock())
        async with lock:
            if await self.repository.processed(envelope.event_id):
                return await self.repository.latest_signal(envelope.instrument_id, envelope.timeframe)
            if envelope.data_quality_status == DataQualityStatus.REJECTED:
                await self._quality_issue(envelope, DataQualityStatus.REJECTED, ("quality_threshold_failed",))
                await self.repository.mark_processed(envelope.event_id)
                await self._trace(envelope, TraceStatus.BLOCKED, "quality", (), ())
                return None
            age = (self.clock() - envelope.available_at).total_seconds()
            if age > self.config.policy.stale_after_seconds:
                await self._quality_issue(envelope, DataQualityStatus.STALE, ("final_candle_stale",))
                await self.repository.mark_processed(envelope.event_id)
                await self._trace(envelope, TraceStatus.BLOCKED, "freshness", (), ())
                return None
            return await self._run(envelope)

    async def _run(self, envelope: CanonicalEventEnvelope) -> OperationalSignal | None:
        started = perf_counter()
        symbol, timeframe_name = envelope.instrument_id or "", envelope.timeframe
        assert timeframe_name is not None
        timeframe = Timeframe(timeframe_name)
        boundary = envelope.available_at
        candles = await self.market_data.history(symbol, timeframe, end=boundary, limit=self.config.limits.maximum_candles)
        outputs: list[tuple[str, object]] = []
        correlation = envelope.correlation_id
        if candles:
            outputs.append(("smc", await self.smc.analyze_candles(candles, correlation_id=correlation)))
            outputs.append(("liquidity", await self.liquidity.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)))
            outputs.append(("volume_profile", await self.volume_profile.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)))
            outputs.append(("institutional_flow", await self.institutional_flow.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)))
            outputs.append(("market_regime", await self.market_regime.analyze_snapshot(symbol, timeframe, timestamp=boundary)))
        outputs.append(("economic_calendar", await self.economic_calendar.context(symbol, as_of=boundary)))
        evidence = [EvidenceReference(engine="market_data", evidence_id=envelope.event_id, engine_version="1.0.0", effective_at=boundary)]
        for name, value in outputs:
            identifier = next((getattr(value, key) for key in ("snapshot_id", "id", "context_id") if getattr(value, key, None) is not None), semantic_uuid(name, envelope.event_id))
            timestamp = next((getattr(value, key) for key in ("analysis_timestamp", "as_of", "created_at") if getattr(value, key, None) is not None), boundary)
            evidence.append(EvidenceReference(engine=name, evidence_id=str(identifier), engine_version=str(getattr(value, "engine_version", "1.0.0")), effective_at=timestamp))
        missing = tuple(name for name in self.config.policy.required_evidence if name not in {item.engine for item in evidence})
        evidence_payload = [item.model_dump(mode="json") for item in evidence]
        snapshot_hash = canonical_hash({"event": envelope.event_id, "policy": self.config.policy.version, "evidence": evidence_payload, "missing": missing})
        snapshot = IntegrationSnapshot(snapshot_id=semantic_uuid("snapshot", snapshot_hash), semantic_hash=snapshot_hash, mode=IntegrationMode.LIVE, instrument=symbol, timeframe=timeframe.value, analytical_boundary=boundary, market_event_id=envelope.event_id, evidence=tuple(evidence), missing_required=missing, data_quality_status=envelope.data_quality_status, status=SnapshotStatus.READY if not missing else SnapshotStatus.INSUFFICIENT_DATA, created_at=self.clock())
        await self.repository.save_snapshot(snapshot)
        if missing:
            await self.repository.mark_processed(envelope.event_id)
            await self._trace(envelope, TraceStatus.BLOCKED, "snapshot_barrier", (envelope.event_id,), (str(snapshot.snapshot_id),), started)
            return None
        score = await self.ai_scoring.calculate(ScoreRequest(instrument=symbol, timeframe=timeframe.value, as_of=boundary, mode=ScoreMode.LIVE))
        decision = await self.signal_decision.evaluate(DecisionRequest(instrument=symbol, timeframe=timeframe.value, ai_score_snapshot_id=score.snapshot_id, as_of=boundary, mode=DecisionMode.LIVE))
        blocker_codes = tuple(item.reason_code for item in decision.blockers)
        warning_codes = tuple(item.reason_code for item in decision.warnings)
        semantic = canonical_hash({"snapshot": snapshot.semantic_hash, "score": score.metadata.input_fingerprint, "decision": decision.input_fingerprint, "policies": (score.policy_version, decision.decision_policy_version)})
        signal = OperationalSignal(operational_signal_id=semantic_uuid("signal", semantic), semantic_hash=semantic, decision_id=decision.decision_id, ai_score_id=score.snapshot_id, snapshot_id=snapshot.snapshot_id, trace_id=envelope.trace_id, market_event_id=envelope.event_id, instrument=symbol, timeframe=timeframe.value, mode=IntegrationMode.LIVE, direction=decision.direction.value, state=decision.state.value, confidence=decision.confidence_score, effective_at=decision.as_of, expires_at=decision.valid_until, data_quality_status=envelope.data_quality_status, provider_provenance=(envelope.source_name,), evidence=tuple(evidence), blockers=blocker_codes, warnings=warning_codes, ai_scoring_policy_version=score.policy_version, signal_decision_policy_version=decision.decision_policy_version, created_at=self.clock())
        signal = await self.repository.save_signal(signal)
        await self.repository.mark_processed(envelope.event_id)
        await self._trace(envelope, TraceStatus.COMPLETED, "full_system", (envelope.event_id,), (str(snapshot.snapshot_id), str(score.snapshot_id), str(decision.decision_id), str(signal.operational_signal_id)), started)
        return signal

    async def _trace(self, envelope: CanonicalEventEnvelope, status: TraceStatus, consumer: str, inputs: tuple[str, ...], outputs: tuple[str, ...], started: float | None = None) -> None:
        duration = (perf_counter() - started) * 1000 if started is not None else 0.0
        now = self.clock()
        record = IntegrationTraceRecord(trace_record_id=semantic_uuid("trace-record", envelope.event_id, consumer, status.value), trace_id=envelope.trace_id, event_id=envelope.event_id, event_type=envelope.event_type, producer=envelope.producer_engine, consumer=consumer, instrument=envelope.instrument_id, timeframe=envelope.timeframe, mode=envelope.mode, status=status, started_at=now, completed_at=now, duration_ms=duration, input_references=inputs, output_references=outputs, correlation_id=envelope.correlation_id, causation_id=envelope.causation_id)
        await self.repository.save_trace(record)

    async def _quality_issue(self, envelope: CanonicalEventEnvelope, status: DataQualityStatus, reasons: tuple[str, ...]) -> None:
        assert envelope.instrument_id and envelope.timeframe
        await self.repository.save_issue(DataQualityIssue(issue_id=semantic_uuid("quality", envelope.event_id, status.value), event_id=envelope.event_id, instrument=envelope.instrument_id, timeframe=envelope.timeframe, status=status, reason_codes=reasons, observed_at=self.clock(), provider=envelope.source_name))

    def health(self) -> dict[str, object]:
        state = "healthy" if self.started and self.repository_mode != "memory" else "degraded" if self.started else "unavailable"
        return {"status": state, "ready": self.started and self.repository_mode != "memory", "repository_mode": self.repository_mode, "worker": "embedded" if self.config.worker.embedded_api_worker else "external", "failures": self.failures, **self.repository.metrics()}
