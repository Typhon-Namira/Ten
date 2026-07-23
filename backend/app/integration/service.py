from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import logging
from time import perf_counter
from typing import Any

from backend.app.engines.ai_scoring_engine import ScoreMode, ScoreRequest
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.events import NewCandle
from backend.app.engines.signal_decision_engine import DecisionMode, DecisionRequest, DecisionState
from backend.app.events import Event, EventBus

from .config import IntegrationConfig
from .models import CanonicalEventEnvelope, DataQualityIssue, DataQualityStatus, EvidenceReference, IntegrationMode, IntegrationSnapshot, IntegrationTraceRecord, MarketCandlePayload, OperationalSignal, SnapshotStatus, TraceStatus, canonical_hash, semantic_uuid
from .repository import IntegrationRepository
from .stage_tracker import PipelineStageTracker

logger = logging.getLogger(__name__)


def _stage_status(result: object) -> str:
    """Success unless the engine's own returned status reports degraded input/quality.

    Engines never raise for a degraded-but-completed analysis (only for genuine failures, e.g.
    persistence errors) — the distinction lives in the returned snapshot's `status` field. Each
    engine defines its own local status enum, so this matches on the serialized value rather than
    a specific enum type, which keeps this helper decoupled from any one engine's model classes.
    """
    value = getattr(getattr(result, "status", None), "value", None)
    if isinstance(value, str) and "degrad" in value.lower():
        return "degraded"
    return "success"


class FullSystemIntegrationService:
    """Coordinates existing engines at a final-candle boundary; contains no analytics."""

    def __init__(self, *, event_bus: EventBus, repository: IntegrationRepository, config: IntegrationConfig, market_data: Any, smc: Any, liquidity: Any, volume_profile: Any, institutional_flow: Any, market_regime: Any, economic_calendar: Any, ai_scoring: Any, signal_decision: Any, repository_mode: str = "memory", clock: Callable[[], datetime] | None = None, stage_tracker: PipelineStageTracker | None = None, unified_market_state: Any | None = None, quantitative_forecasting: Any | None = None, ai_reasoning: Any | None = None, ai_centric_shadow_mode: bool = False) -> None:
        self.event_bus, self.repository, self.config = event_bus, repository, config
        self.market_data, self.smc, self.liquidity = market_data, smc, liquidity
        self.volume_profile, self.institutional_flow = volume_profile, institutional_flow
        self.market_regime, self.economic_calendar = market_regime, economic_calendar
        self.ai_scoring, self.signal_decision = ai_scoring, signal_decision
        self.repository_mode = repository_mode
        self.clock = clock or (lambda: datetime.now(UTC))
        self.stage_tracker = stage_tracker
        self.unified_market_state = unified_market_state
        self.quantitative_forecasting = quantitative_forecasting
        self.ai_reasoning = ai_reasoning
        self.ai_centric_shadow_mode = ai_centric_shadow_mode
        self._unsubscribe: Callable[[], None] | None = None
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.started = False
        self.failures = 0
        self.last_batch_failures = 0
        self.last_cycle_started_at: datetime | None = None
        self.last_cycle_completed_at: datetime | None = None
        self.last_cycle_failed_at: datetime | None = None
        self.last_decision_persisted_at: datetime | None = None
        self.last_signal_published_at: datetime | None = None

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
        self.last_batch_failures = 0
        for item in items:
            try:
                await self.process(item.envelope)
                await self.repository.complete(item.outbox_id, self.clock())
            except Exception as exc:
                self.failures += 1
                self.last_batch_failures += 1
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
            assert isinstance(envelope.payload, MarketCandlePayload)
            age = (self.clock() - envelope.payload.close_time).total_seconds()
            if age > self.config.policy.stale_after_seconds:
                await self._quality_issue(envelope, DataQualityStatus.STALE, ("final_candle_stale",))
                await self.repository.mark_processed(envelope.event_id)
                await self._trace(envelope, TraceStatus.BLOCKED, "freshness", (), ())
                return None
            return await self._run(envelope)

    async def process_historical_candle(self, candle: Candle) -> None:
        """Build persisted replay-mode evidence during bootstrap without a live signal."""
        envelope = CanonicalEventEnvelope.historical_candle(candle, semantic_uuid("bootstrap", candle.symbol, candle.timeframe.value), self.clock())
        if await self.repository.processed(envelope.event_id):
            return
        assert envelope.instrument_id and envelope.timeframe
        self.config.instrument(candle.symbol)
        lock = self._locks.setdefault((envelope.instrument_id, envelope.timeframe), asyncio.Lock())
        async with lock:
            if not await self.repository.processed(envelope.event_id):
                await self._run(envelope, publish_signal=False)

    async def _run(self, envelope: CanonicalEventEnvelope, *, publish_signal: bool = True) -> OperationalSignal | None:
        started = perf_counter()
        symbol, timeframe_name = envelope.instrument_id or "", envelope.timeframe
        assert timeframe_name is not None
        timeframe = Timeframe(timeframe_name)
        boundary = envelope.available_at
        self.last_cycle_started_at = self.clock()
        log_context = {
            "correlation_id": str(envelope.correlation_id),
            "cycle_id": str(envelope.trace_id),
            "candle_id": envelope.event_id,
            "symbol": symbol,
            "canonical_instrument": symbol,
            "timeframe": timeframe.value,
            "canonical_timeframe": timeframe.value,
            "mode": envelope.mode.value,
            "candle_timestamp": boundary.isoformat(),
        }
        logger.info("snapshot.started", extra=log_context)
        tracker = self.stage_tracker
        correlation = envelope.correlation_id
        if tracker is not None:
            tracker.begin(symbol, timeframe.value, boundary, correlation_id=str(correlation))
            tracker.mark(symbol, timeframe.value, boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")
        outputs: list[tuple[str, object]] = []
        # Everything from here through the final `tracker.complete(...)` below is one failure
        # domain: ANY exception in this span — including `market_data.history()` and
        # `repository.save_snapshot()`, which used to sit outside this block — must finalize the
        # cycle via `fail_in_flight()`. Root cause this closes: a provider failure (e.g. a 429/
        # rate-limit circuit-breaker trip) or a snapshot-persistence error raised from one of
        # those two unguarded calls used to propagate straight out of `_run()` without ever
        # touching the tracker, permanently freezing that candle's cycle at "running" on whichever
        # stage was next — while later, unrelated candles kept completing normally and publishing
        # their own events, which is exactly what made the frozen cycle look contradictory next to
        # a live log that had already moved on.
        try:
            candles = await self.market_data.history(symbol, timeframe, end=boundary, limit=self.config.limits.maximum_candles)
            if candles:
                smc_snapshot = await self.smc.analyze_candles(candles, correlation_id=correlation)
                outputs.append(("smc", smc_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("smc_analysis",), _stage_status(smc_snapshot))
                liquidity_snapshot = await self.liquidity.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)
                outputs.append(("liquidity", liquidity_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("liquidity_analysis",), _stage_status(liquidity_snapshot))
                volume_profile_snapshot = await self.volume_profile.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)
                outputs.append(("volume_profile", volume_profile_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("volume_profile",), _stage_status(volume_profile_snapshot))
                institutional_flow_snapshot = await self.institutional_flow.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)
                outputs.append(("institutional_flow", institutional_flow_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("institutional_flow",), _stage_status(institutional_flow_snapshot))
                market_regime_snapshot = await self.market_regime.analyze_snapshot(symbol, timeframe, timestamp=boundary)
                outputs.append(("market_regime", market_regime_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("market_regime",), _stage_status(market_regime_snapshot))
            elif tracker is not None:
                tracker.mark(symbol, timeframe.value, boundary, ("smc_analysis", "liquidity_analysis", "volume_profile", "institutional_flow", "market_regime"), "skipped")
            outputs.append(("economic_calendar", await self.economic_calendar.context(symbol, as_of=boundary)))
            if self.ai_centric_shadow_mode and self.unified_market_state is not None:
                # Phase 1 is observational only.  A shadow-state capture failure is logged but can
                # never change legacy scoring, decision gates, or signal publication.
                try:
                    market_state = await self.unified_market_state.capture_cycle(envelope, dict(outputs))
                    if market_state is not None and self.quantitative_forecasting is not None:
                        quantitative_forecast = await self.quantitative_forecasting.forecast(market_state)
                        if quantitative_forecast is not None and self.ai_reasoning is not None:
                            await self.ai_reasoning.process(market_state, quantitative_forecast)
                except Exception:
                    logger.exception("ai_centric_shadow_pipeline_failed", extra=log_context)
            evidence = [EvidenceReference(engine="market_data", evidence_id=envelope.event_id, engine_version="1.0.0", effective_at=boundary)]
            for name, value in outputs:
                identifier = next((getattr(value, key) for key in ("snapshot_id", "id", "context_id") if getattr(value, key, None) is not None), semantic_uuid(name, envelope.event_id))
                timestamp = next((getattr(value, key) for key in ("analysis_timestamp", "as_of", "created_at") if getattr(value, key, None) is not None), boundary)
                evidence.append(EvidenceReference(engine=name, evidence_id=str(identifier), engine_version=str(getattr(value, "engine_version", "1.0.0")), effective_at=timestamp))
            missing = tuple(name for name in self.config.policy.required_evidence if name not in {item.engine for item in evidence})
            evidence_payload = [item.model_dump(mode="json") for item in evidence]
            snapshot_hash = canonical_hash({"event": envelope.event_id, "policy": self.config.policy.version, "evidence": evidence_payload, "missing": missing})
            snapshot = IntegrationSnapshot(snapshot_id=semantic_uuid("snapshot", snapshot_hash), semantic_hash=snapshot_hash, mode=envelope.mode, instrument=symbol, timeframe=timeframe.value, analytical_boundary=boundary, market_event_id=envelope.event_id, evidence=tuple(evidence), missing_required=missing, data_quality_status=envelope.data_quality_status, status=SnapshotStatus.READY if not missing else SnapshotStatus.INSUFFICIENT_DATA, created_at=self.clock())
            await self.repository.save_snapshot(snapshot)
            if missing:
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("ai_scoring", "confidence_calculation", "scenario_decision"), "skipped")
                    tracker.complete(symbol, timeframe.value, boundary)
            else:
                score_mode = ScoreMode.LIVE if envelope.mode == IntegrationMode.LIVE else ScoreMode.REPLAY
                decision_mode = DecisionMode.LIVE if envelope.mode == IntegrationMode.LIVE else DecisionMode.REPLAY
                score = await self.ai_scoring.calculate(ScoreRequest(instrument=symbol, timeframe=timeframe.value, as_of=boundary, mode=score_mode))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("ai_scoring", "confidence_calculation"), "success")
                decision = await self.signal_decision.evaluate(DecisionRequest(instrument=symbol, timeframe=timeframe.value, ai_score_snapshot_id=score.snapshot_id, as_of=boundary, mode=decision_mode))
                self.last_decision_persisted_at = self.clock()
                logger.info(
                    "decision.persist.completed",
                    extra={**log_context, "snapshot_id": str(snapshot.snapshot_id), "decision_id": str(decision.decision_id), "decision_status": decision.state.value},
                )
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("scenario_decision",), "success")
                    tracker.complete(symbol, timeframe.value, boundary)
        except Exception as exc:
            self.last_cycle_failed_at = self.clock()
            if tracker is not None:
                tracker.fail_in_flight(symbol, timeframe.value, boundary, exc=exc)
            logger.exception(
                "snapshot.failed",
                extra={**log_context, "failure_class": type(exc).__name__, "duration_ms": (perf_counter() - started) * 1000},
            )
            try:
                await self._trace(envelope, TraceStatus.FAILED, "full_system", (envelope.event_id,), (), started)
            except Exception:
                logger.exception("snapshot.failure_trace_failed", extra=log_context)
            raise
        try:
            if missing:
                await self.repository.mark_processed(envelope.event_id)
                await self._trace(envelope, TraceStatus.BLOCKED, "snapshot_barrier", (envelope.event_id,), (str(snapshot.snapshot_id),), started)
                self.last_cycle_completed_at = self.clock()
                logger.warning("snapshot.completed_degraded", extra={**log_context, "snapshot_id": str(snapshot.snapshot_id), "missing": missing, "duration_ms": (perf_counter() - started) * 1000})
                return None
            logger.info("signal_decision.completed", extra={"snapshot_id": str(snapshot.snapshot_id), "decision_id": str(decision.decision_id), "state": decision.state.value, "mode": envelope.mode.value})
            blocker_codes = tuple(item.reason_code for item in decision.blockers)
            warning_codes = tuple(item.reason_code for item in decision.warnings)
            semantic = canonical_hash({"snapshot": snapshot.semantic_hash, "score": score.metadata.input_fingerprint, "decision": decision.input_fingerprint, "policies": (score.policy_version, decision.decision_policy_version)})
            if not publish_signal or decision.state.value != DecisionState.ELIGIBLE.value:
                await self.repository.mark_processed(envelope.event_id)
                await self._trace(envelope, TraceStatus.COMPLETED, "full_system", (envelope.event_id,), (str(snapshot.snapshot_id), str(score.snapshot_id), str(decision.decision_id)), started)
                self.last_cycle_completed_at = self.clock()
                logger.info("snapshot.completed", extra={**log_context, "snapshot_id": str(snapshot.snapshot_id), "decision_id": str(decision.decision_id), "decision_status": decision.state.value, "scenario_published": False, "duration_ms": (perf_counter() - started) * 1000})
                return None
            signal = OperationalSignal(operational_signal_id=semantic_uuid("signal", semantic), semantic_hash=semantic, decision_id=decision.decision_id, ai_score_id=score.snapshot_id, snapshot_id=snapshot.snapshot_id, trace_id=envelope.trace_id, market_event_id=envelope.event_id, instrument=symbol, timeframe=timeframe.value, mode=envelope.mode, direction=decision.direction.value, state=decision.state.value, confidence=decision.confidence_score, effective_at=decision.as_of, expires_at=decision.valid_until, data_quality_status=envelope.data_quality_status, provider_provenance=(envelope.source_name,), evidence=tuple(evidence), blockers=blocker_codes, warnings=warning_codes, ai_scoring_policy_version=score.policy_version, signal_decision_policy_version=decision.decision_policy_version, created_at=self.clock())
            signal = await self.repository.save_signal(signal)
            self.last_signal_published_at = self.clock()
            logger.info("scenario.published", extra={"snapshot_id": str(snapshot.snapshot_id), "signal_id": str(signal.operational_signal_id), "symbol": symbol, "timeframe": timeframe.value})
            await self.repository.mark_processed(envelope.event_id)
            await self._trace(envelope, TraceStatus.COMPLETED, "full_system", (envelope.event_id,), (str(snapshot.snapshot_id), str(score.snapshot_id), str(decision.decision_id), str(signal.operational_signal_id)), started)
            self.last_cycle_completed_at = self.clock()
            logger.info("snapshot.completed", extra={**log_context, "snapshot_id": str(snapshot.snapshot_id), "decision_id": str(decision.decision_id), "signal_id": str(signal.operational_signal_id), "decision_status": decision.state.value, "scenario_published": True, "duration_ms": (perf_counter() - started) * 1000})
            return signal
        except Exception as exc:
            self.last_cycle_failed_at = self.clock()
            logger.exception("snapshot.failed", extra={**log_context, "failure_class": type(exc).__name__, "duration_ms": (perf_counter() - started) * 1000})
            try:
                await self._trace(envelope, TraceStatus.FAILED, "full_system", (envelope.event_id,), (), started)
            except Exception:
                logger.exception("snapshot.failure_trace_failed", extra=log_context)
            raise

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
        if state == "healthy" and self.last_batch_failures:
            state = "degraded"
        return {
            "status": state,
            "ready": self.started and self.repository_mode != "memory" and self.last_batch_failures == 0,
            "repository_mode": self.repository_mode,
            "worker": "embedded" if self.config.worker.embedded_api_worker else "external",
            "failures": self.failures,
            "last_batch_failures": self.last_batch_failures,
            "last_cycle_started": self.last_cycle_started_at,
            "last_cycle_completed": self.last_cycle_completed_at,
            "last_cycle_failed": self.last_cycle_failed_at,
            "last_decision_persisted": self.last_decision_persisted_at,
            "last_signal_published": self.last_signal_published_at,
            **self.repository.metrics(),
        }
