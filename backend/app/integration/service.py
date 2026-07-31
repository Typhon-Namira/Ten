from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
import logging
import os
from time import perf_counter
from typing import Any

from backend.app.engines.ai_scoring_engine import ScoreMode, ScoreRequest
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.events import NewCandle
from backend.app.engines.signal_decision_engine import DecisionMode, DecisionRequest, DecisionState
from backend.app.events import Event, EventBus
from backend.app.ai_reasoning.cadence import (
    synchronized_cycle_eligibility,
)

from .config import IntegrationConfig
from .models import CanonicalEventEnvelope, DataQualityIssue, DataQualityStatus, EvidenceReference, IntegrationMode, IntegrationSnapshot, IntegrationTraceRecord, MarketCandlePayload, OperationalSignal, SnapshotStatus, TraceStatus, canonical_hash, semantic_uuid
from .repository import IntegrationRepository
from .stage_tracker import PipelineStageTracker

logger = logging.getLogger(__name__)


def _is_storage_exhausted(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        name = type(current).__name__.lower()
        message = str(current).lower()
        if (
            "diskfull" in name
            or "no space left on device" in message
            or "could not extend file" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


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

    def __init__(self, *, event_bus: EventBus, repository: IntegrationRepository, config: IntegrationConfig, market_data: Any, smc: Any, liquidity: Any, volume_profile: Any, institutional_flow: Any, market_regime: Any, economic_calendar: Any, ai_scoring: Any, signal_decision: Any, repository_mode: str = "memory", clock: Callable[[], datetime] | None = None, stage_tracker: PipelineStageTracker | None = None, unified_market_state: Any | None = None, quantitative_forecasting: Any | None = None, ai_reasoning: Any | None = None, signal_synthesizer: Any | None = None, signal_synthesis_repository: Any | None = None, scenario_forecasting: Any | None = None, market_simulation: Any | None = None, ai_centric_shadow_mode: bool = False) -> None:
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
        self.signal_synthesizer = signal_synthesizer
        self.signal_synthesis_repository = signal_synthesis_repository
        self.scenario_forecasting = scenario_forecasting
        self.market_simulation = market_simulation
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
        self.storage_exhausted_until: datetime | None = None

    async def recover_authoritative_ai_analysis(self, instrument: str) -> bool:
        """Retrieve or regenerate the latest eligible M15 analysis before simulation recovery."""

        if (
            self.unified_market_state is None
            or self.quantitative_forecasting is None
            or self.ai_reasoning is None
            or self.signal_synthesizer is None
            or self.signal_synthesis_repository is None
        ):
            return False
        state = await self.unified_market_state.repository.latest_state(
            instrument,
            trigger_timeframe="M15",
        )
        if state is None:
            return False
        recovery_max_age = getattr(
            self.market_simulation,
            "recovery_max_age_seconds",
            7200,
        )
        if (
            self.clock().astimezone(UTC) - state.market_data_boundary
        ).total_seconds() > recovery_max_age:
            await self.ai_reasoning.record_gate_decision(
                state=state,
                attempted_cutoff=state.market_data_boundary,
                gate_decision="SKIPPED",
                gate_skip_reason="recovery_lookback_exceeded",
                details={"recovery": True, "maximum_age_seconds": recovery_max_age},
            )
            return False
        quant = await self.quantitative_forecasting.repository.result_for_state(
            state.state_id
        )
        if quant is None:
            quant = await self.quantitative_forecasting.forecast(state)
        if quant is None:
            await self.ai_reasoning.record_gate_decision(
                state=state,
                attempted_cutoff=state.market_data_boundary,
                gate_decision="SKIPPED",
                gate_skip_reason="quantitative_forecast_not_ready",
                details={"recovery": True},
            )
            return False
        analysis = await self.ai_reasoning.repository.analysis_for_state(
            state.state_id
        )
        exact_match = bool(
            analysis is not None
            and analysis.market_snapshot_id == state.state_id
            and analysis.analysis_timestamp == state.market_data_boundary
            and analysis.validation_passed
        )
        logger.info(
            "ai_reasoning.recovery.lookup",
            extra={
                "instrument": instrument,
                "attempted_cutoff": state.market_data_boundary.isoformat(),
                "analysis_lookup_cutoff": state.market_data_boundary.isoformat(),
                "market_state_id": str(state.state_id),
                "snapshot_id": str(state.state_id),
                "gate_decision": "REUSED" if exact_match else "REGENERATE",
                "gate_skip_reason": None,
                "existing_analysis_id": (
                    str(analysis.analysis_id) if analysis is not None else None
                ),
                "analysis_created_at": (
                    analysis.created_at.isoformat() if analysis is not None else None
                ),
                "analysis_market_cutoff": (
                    analysis.analysis_timestamp.isoformat()
                    if analysis is not None
                    else None
                ),
            },
        )
        if not exact_match:
            validated = await self.ai_reasoning.process(state, quant)
            analysis = validated.analysis if validated is not None else (
                await self.ai_reasoning.repository.analysis_for_state(
                    state.state_id
                )
            )
        if (
            analysis is None
            or analysis.market_snapshot_id != state.state_id
            or analysis.analysis_timestamp != state.market_data_boundary
            or not analysis.validation_passed
        ):
            return False
        synthesis = await self.signal_synthesis_repository.for_state(state.state_id)
        if synthesis is None:
            synthesis = self.signal_synthesizer.synthesize(state, quant, analysis)
            await self.signal_synthesis_repository.save(synthesis)
        return True

    async def _skip_ai_reasoning_gate(
        self,
        *,
        context: Mapping[str, object],
        instrument: str,
        attempted_cutoff: datetime,
        skip_reason: str,
        reason_code: str,
        state: Any | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        existing_analysis = (
            await self.ai_reasoning.repository.analysis_for_state(state.state_id)
            if self.ai_reasoning is not None and state is not None
            else None
        )
        payload = {
            **context,
            "attempted_cutoff": attempted_cutoff.isoformat(),
            "analysis_lookup_cutoff": attempted_cutoff.isoformat(),
            "market_state_id": (
                str(state.state_id) if state is not None else None
            ),
            "snapshot_id": str(state.state_id) if state is not None else None,
            "gate_decision": "SKIPPED",
            "skip_reason": skip_reason,
            "gate_skip_reason": skip_reason,
            "reason_code": reason_code,
            "existing_analysis_id": (
                str(existing_analysis.analysis_id)
                if existing_analysis is not None
                else None
            ),
            "analysis_created_at": (
                existing_analysis.created_at.isoformat()
                if existing_analysis is not None
                else None
            ),
            "analysis_market_cutoff": (
                existing_analysis.analysis_timestamp.isoformat()
                if existing_analysis is not None
                else None
            ),
            "provider_call_made": False,
            "details": details or {},
        }
        logger.info("ai_reasoning.gate.skipped", extra=payload)
        if self.ai_reasoning is not None:
            await self.ai_reasoning.record_gate_decision(
                state=state,
                instrument=instrument,
                attempted_cutoff=attempted_cutoff,
                gate_decision="SKIPPED",
                gate_skip_reason=skip_reason,
                existing_analysis=existing_analysis,
                details=details,
                trigger_timeframe=context.get("timeframe")
                if isinstance(context.get("timeframe"), str)
                else None,
            )
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
        now = self.clock()
        if self.storage_exhausted_until is not None and now < self.storage_exhausted_until:
            logger.warning(
                "integration.storage_circuit_open",
                extra={
                    "reason": "storage_exhausted",
                    "retry_at": self.storage_exhausted_until.isoformat(),
                },
            )
            return 0
        items = await self.repository.pending(self.clock(), self.config.limits.outbox_batch_size)
        self.last_batch_failures = 0
        attempted = 0
        for item in items:
            attempted += 1
            try:
                await self.process(item.envelope)
                await self.repository.complete(item.outbox_id, self.clock())
            except Exception as exc:
                self.failures += 1
                self.last_batch_failures += 1
                storage_exhausted = _is_storage_exhausted(exc)
                if storage_exhausted:
                    self.storage_exhausted_until = self.clock() + timedelta(minutes=5)
                    logger.critical(
                        "integration.storage_exhausted",
                        extra={
                            "reason": "storage_exhausted",
                            "circuit_open_seconds": 300,
                            "exception_class": type(exc).__name__,
                        },
                    )
                try:
                    await self.repository.fail(item.outbox_id, type(exc).__name__)
                except Exception as failure_record_error:
                    if not storage_exhausted:
                        raise
                    logger.warning(
                        "integration.storage_failure_record.skipped",
                        extra={
                            "reason": "storage_exhausted",
                            "exception_class": type(failure_record_error).__name__,
                        },
                    )
                if storage_exhausted:
                    break
            else:
                self.storage_exhausted_until = None
        return attempted

    async def process(self, envelope: CanonicalEventEnvelope) -> OperationalSignal | None:
        if envelope.mode != IntegrationMode.LIVE:
            raise ValueError("live integration rejects replay envelopes")
        if envelope.schema_version != self.config.policy.schema_version or envelope.event_type != "market.candle.closed":
            raise ValueError("unsupported integration envelope")
        # Outbox delivery normally guarantees this already. Keeping the invariant at the public
        # processing boundary also protects direct callers and makes a retry self-healing if an
        # earlier ingestion transaction committed the envelope but delivery was interrupted.
        await self.repository.persist_event(envelope)
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
        # Historical bootstrap intentionally does not create an outbox item, but it still owns a
        # canonical integration event. Previously it skipped persistence entirely and later
        # violated integration_processed_events_event_id_fkey in mark_processed().
        await self.repository.persist_event(envelope)
        logger.info(
            "integration.event.persisted",
            extra={
                "event_id": envelope.event_id,
                "stage": "historical_bootstrap_parent",
                "symbol": envelope.instrument_id,
                "timeframe": envelope.timeframe,
                "mode": envelope.mode.value,
            },
        )
        if await self.repository.processed(envelope.event_id):
            logger.info(
                "integration.bootstrap.already_persisted",
                extra={
                    "event_id": envelope.event_id,
                    "symbol": envelope.instrument_id,
                    "timeframe": envelope.timeframe,
                    "mode": envelope.mode.value,
                },
            )
            return
        assert envelope.instrument_id and envelope.timeframe
        self.config.instrument(candle.symbol)
        lock = self._locks.setdefault((envelope.instrument_id, envelope.timeframe), asyncio.Lock())
        async with lock:
            if not await self.repository.processed(envelope.event_id):
                await self._run(envelope, publish_signal=False)

    async def _run(self, envelope: CanonicalEventEnvelope, *, publish_signal: bool = True) -> OperationalSignal | None:
        started = perf_counter()
        if not isinstance(envelope.payload, MarketCandlePayload):
            raise ValueError("integration analytical cycles require a market candle payload")
        symbol, timeframe_name = envelope.instrument_id or "", envelope.timeframe
        assert timeframe_name is not None
        timeframe = Timeframe(timeframe_name)
        boundary = envelope.available_at
        self.last_cycle_started_at = self.clock()
        log_context = {
            "event_id": envelope.event_id,
            "correlation_id": str(envelope.correlation_id),
            "cycle_id": str(envelope.trace_id),
            "candle_id": envelope.event_id,
            "symbol": symbol,
            "canonical_instrument": symbol,
            "timeframe": timeframe.value,
            "canonical_timeframe": timeframe.value,
            "mode": envelope.mode.value,
            "candle_timestamp": boundary.isoformat(),
            "boundary_timestamp": boundary.isoformat(),
            "logical_identity": f"{symbol}:{timeframe.value}:{boundary.isoformat()}:{envelope.mode.value}",
            "worker_id": f"{os.getenv('RAILWAY_REPLICA_ID') or os.getenv('HOSTNAME') or 'local'}:{os.getpid()}",
            "git_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_SHA"),
        }
        logger.info("snapshot.started", extra=log_context)
        tracker = self.stage_tracker
        correlation = envelope.correlation_id
        if tracker is not None:
            tracker.begin(symbol, timeframe.value, boundary, correlation_id=str(correlation))
            tracker.mark(symbol, timeframe.value, boundary, ("candle_received", "candle_normalized", "stored_in_database"), "success")
        outputs: list[tuple[str, object]] = []
        validated_analysis = None
        primary_scenario = None
        market_state = None
        quantitative_forecast = None
        synthesis = None
        market_simulation_invoked = False
        failure_stage = "market_data_history"
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
                failure_stage = "smc_analysis"
                smc_snapshot = await self.smc.analyze_candles(candles, correlation_id=correlation)
                outputs.append(("smc", smc_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("smc_analysis",), _stage_status(smc_snapshot))
                failure_stage = "liquidity_analysis"
                liquidity_snapshot = await self.liquidity.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)
                outputs.append(("liquidity", liquidity_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("liquidity_analysis",), _stage_status(liquidity_snapshot))
                failure_stage = "volume_profile"
                volume_profile_snapshot = await self.volume_profile.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)
                outputs.append(("volume_profile", volume_profile_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("volume_profile",), _stage_status(volume_profile_snapshot))
                failure_stage = "institutional_flow"
                institutional_flow_snapshot = await self.institutional_flow.analyze(symbol, timeframe, end=boundary, correlation_id=correlation)
                outputs.append(("institutional_flow", institutional_flow_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("institutional_flow",), _stage_status(institutional_flow_snapshot))
                failure_stage = "market_regime"
                market_regime_snapshot = await self.market_regime.analyze_snapshot(symbol, timeframe, timestamp=boundary)
                outputs.append(("market_regime", market_regime_snapshot))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("market_regime",), _stage_status(market_regime_snapshot))
            elif tracker is not None:
                tracker.mark(symbol, timeframe.value, boundary, ("smc_analysis", "liquidity_analysis", "volume_profile", "institutional_flow", "market_regime"), "skipped")
            failure_stage = "economic_calendar"
            outputs.append(("economic_calendar", await self.economic_calendar.context(symbol, as_of=boundary)))
            gate_context = {
                **log_context,
                "ai_centric_shadow_mode": self.ai_centric_shadow_mode,
                "unified_market_state_configured": self.unified_market_state is not None,
                "quantitative_forecasting_configured": self.quantitative_forecasting is not None,
                "ai_reasoning_configured": self.ai_reasoning is not None,
            }
            logger.info("ai_reasoning.gate.entered", extra=gate_context)
            if not self.ai_centric_shadow_mode:
                await self._skip_ai_reasoning_gate(
                    context=gate_context,
                    instrument=symbol,
                    attempted_cutoff=boundary,
                    skip_reason="ai_centric_shadow_mode_disabled",
                    reason_code="disabled",
                )
            elif self.unified_market_state is None:
                await self._skip_ai_reasoning_gate(
                    context=gate_context,
                    instrument=symbol,
                    attempted_cutoff=boundary,
                    skip_reason="unified_market_state_service_unavailable",
                    reason_code="missing_prerequisite",
                    details={"prerequisite": "unified_market_state"},
                )
            else:
                # Phase 1 is observational only.  A shadow-state capture failure is logged but can
                # never change legacy scoring, decision gates, or signal publication.
                try:
                    failure_stage = "unified_market_state"
                    market_state = await self.unified_market_state.capture_cycle(envelope, dict(outputs))
                    if market_state is None:
                        await self._skip_ai_reasoning_gate(
                            context=gate_context,
                            instrument=symbol,
                            attempted_cutoff=boundary,
                            skip_reason="synchronized_market_state_not_ready",
                            reason_code="missing_prerequisite",
                            details={"prerequisite": "synchronized_market_state"},
                        )
                    elif self.quantitative_forecasting is None:
                        await self._skip_ai_reasoning_gate(
                            context=gate_context,
                            instrument=symbol,
                            attempted_cutoff=boundary,
                            skip_reason="quantitative_forecasting_service_unavailable",
                            reason_code="missing_prerequisite",
                            state=market_state,
                            details={"prerequisite": "quantitative_forecasting"},
                        )
                    else:
                        failure_stage = "quantitative_forecast"
                        quantitative_forecast = await self.quantitative_forecasting.forecast(market_state)
                        if quantitative_forecast is None:
                            await self._skip_ai_reasoning_gate(
                                context=gate_context,
                                instrument=symbol,
                                attempted_cutoff=boundary,
                                skip_reason="quantitative_forecast_not_ready",
                                reason_code="missing_prerequisite",
                                state=market_state,
                                details={"prerequisite": "quantitative_forecast"},
                            )
                        elif (
                            hasattr(market_state, "timeframes")
                            and (
                                eligibility_reason
                                := synchronized_cycle_eligibility(market_state)
                            )
                            is not None
                        ):
                            await self._skip_ai_reasoning_gate(
                                context=gate_context,
                                instrument=symbol,
                                attempted_cutoff=boundary,
                                skip_reason=eligibility_reason.value,
                                reason_code=eligibility_reason.value,
                                state=market_state,
                            )
                        elif self.ai_reasoning is None:
                            await self._skip_ai_reasoning_gate(
                                context=gate_context,
                                instrument=symbol,
                                attempted_cutoff=boundary,
                                skip_reason="ai_reasoning_service_unavailable",
                                reason_code="missing_prerequisite",
                                state=market_state,
                                details={"prerequisite": "ai_reasoning_service"},
                            )
                        else:
                            failure_stage = "ai_reasoning"
                            logger.info(
                                "ai_reasoning.job.enqueued",
                                extra={
                                    **gate_context,
                                    "market_state_id": str(getattr(market_state, "state_id", "unknown")),
                                    "quantitative_forecast_id": str(
                                        getattr(quantitative_forecast, "result_id", "unknown")
                                    ),
                                    "dispatch_mode": "inline_integration_worker",
                                },
                            )
                            validated_analysis = await self.ai_reasoning.process(
                                market_state,
                                quantitative_forecast,
                            )
                            if (
                                validated_analysis is not None
                                and self.signal_synthesizer is not None
                                and self.signal_synthesis_repository is not None
                            ):
                                failure_stage = "multi_timeframe_signal_synthesis"
                                synthesis = self.signal_synthesizer.synthesize(
                                    market_state,
                                    quantitative_forecast,
                                    validated_analysis.analysis,
                                )
                                synthesis = await self.signal_synthesis_repository.save(
                                    synthesis
                                )
                                logger.info(
                                    "multi_timeframe_signal.persist.completed",
                                    extra={
                                        **log_context,
                                        "synthesis_id": str(synthesis.synthesis_id),
                                        "market_state_id": str(synthesis.market_state_id),
                                        "analysis_id": str(synthesis.analysis_id),
                                        "M5": synthesis.timeframe_signals[0].analytical_direction.value,
                                        "M15": synthesis.timeframe_signals[1].analytical_direction.value,
                                        "combined": synthesis.combined_signal.analytical_direction.value,
                                        "execution_status": synthesis.combined_signal.execution_status.value,
                                    },
                                )
                                if self.scenario_forecasting is not None:
                                    failure_stage = "scenario_forecasting"
                                    await self.scenario_forecasting.process(
                                        market_state,
                                        quantitative_forecast,
                                        synthesis,
                                        trigger_timeframe=timeframe.value,
                                        candles=tuple(candles),
                                        evaluated_at=self.clock(),
                                    )
                                if self.market_simulation is not None:
                                    failure_stage = "market_simulation"
                                    market_simulation_invoked = True
                                    primary_scenario = await self.market_simulation.process(
                                        market_state,
                                        quantitative_forecast,
                                        synthesis,
                                        trigger_timeframe=timeframe.value,
                                        candles=tuple(candles),
                                        evaluated_at=self.clock(),
                                    )
                except Exception:
                    logger.exception("ai_centric_shadow_pipeline_failed", extra=log_context)
            if (
                timeframe == Timeframe.M15
                and self.market_simulation is not None
                and not market_simulation_invoked
            ):
                payload = envelope.payload
                reason = (
                    "MARKET_STATE_MISSING"
                    if market_state is None
                    else "QUANT_FORECAST_MISSING"
                    if quantitative_forecast is None
                    else "AI_ANALYSIS_MISSING"
                    if validated_analysis is None
                    else "SYNTHESIS_MISSING"
                    if synthesis is None
                    else "SIMULATION_NOT_INVOKED"
                )
                logger.info(
                    "market_simulation.m15_eligibility",
                    extra={
                        **log_context,
                        "provider_timestamp": payload.ingestion_time.isoformat(),
                        "candle_open_time": payload.open_time.isoformat(),
                        "candle_close_time": payload.close_time.isoformat(),
                        "resolved_market_cutoff": boundary.isoformat(),
                        "server_time": self.clock().isoformat(),
                        "timezone": "UTC",
                        "eligibility_result": True,
                        "eligibility_reason": "completed_final_m15_candle",
                        "simulation_result": "BLOCKED",
                        "simulation_reason": reason,
                    },
                )
                await self.market_simulation.record_blocked_cutoff(
                    instrument=symbol,
                    market_cutoff=boundary,
                    server_time=self.clock(),
                    reason=reason,
                    provider_timestamp=payload.ingestion_time,
                    candle_open_time=payload.open_time,
                    candle_close_time=payload.close_time,
                    failure_stage=failure_stage,
                )
            failure_stage = "evidence_assembly"
            evidence = [EvidenceReference(engine="market_data", evidence_id=envelope.event_id, engine_version="1.0.0", effective_at=boundary)]
            for name, value in outputs:
                identifier = next((getattr(value, key) for key in ("snapshot_id", "id", "context_id") if getattr(value, key, None) is not None), semantic_uuid(name, envelope.event_id))
                timestamp = next((getattr(value, key) for key in ("analysis_timestamp", "as_of", "created_at") if getattr(value, key, None) is not None), boundary)
                evidence.append(EvidenceReference(engine=name, evidence_id=str(identifier), engine_version=str(getattr(value, "engine_version", "1.0.0")), effective_at=timestamp))
            missing = tuple(name for name in self.config.policy.required_evidence if name not in {item.engine for item in evidence})
            evidence_payload = [item.model_dump(mode="json") for item in evidence]
            snapshot_hash = canonical_hash({"event": envelope.event_id, "policy": self.config.policy.version, "evidence": evidence_payload, "missing": missing})
            snapshot = IntegrationSnapshot(snapshot_id=semantic_uuid("snapshot", snapshot_hash), semantic_hash=snapshot_hash, mode=envelope.mode, instrument=symbol, timeframe=timeframe.value, analytical_boundary=boundary, market_event_id=envelope.event_id, evidence=tuple(evidence), missing_required=missing, data_quality_status=envelope.data_quality_status, status=SnapshotStatus.READY if not missing else SnapshotStatus.INSUFFICIENT_DATA, created_at=self.clock())
            failure_stage = "snapshot_persistence"
            await self.repository.save_snapshot(snapshot)
            if missing:
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("ai_scoring", "confidence_calculation", "scenario_decision"), "skipped")
                    tracker.complete(symbol, timeframe.value, boundary)
            else:
                score_mode = ScoreMode.LIVE if envelope.mode == IntegrationMode.LIVE else ScoreMode.REPLAY
                decision_mode = DecisionMode.LIVE if envelope.mode == IntegrationMode.LIVE else DecisionMode.REPLAY
                failure_stage = "ai_scoring"
                score = await self.ai_scoring.calculate(ScoreRequest(instrument=symbol, timeframe=timeframe.value, as_of=boundary, mode=score_mode))
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("ai_scoring", "confidence_calculation"), "success")
                failure_stage = "signal_decision"
                forecast_predictions = (
                    quantitative_forecast.predictions
                    if quantitative_forecast is not None
                    and hasattr(quantitative_forecast, "predictions")
                    else ()
                )
                prediction = forecast_predictions[0] if forecast_predictions else None
                decision = await self.signal_decision.evaluate(
                    DecisionRequest(
                        instrument=symbol,
                        timeframe=timeframe.value,
                        ai_score_snapshot_id=score.snapshot_id,
                        as_of=boundary,
                        mode=decision_mode,
                        current_ai_analysis=(
                            validated_analysis.analysis
                            if validated_analysis is not None
                            else None
                        ),
                        current_ai_signal=(
                            validated_analysis.signal
                            if validated_analysis is not None
                            else None
                        ),
                        temporal_context=(
                            validated_analysis.temporal_context
                            if validated_analysis is not None
                            else None
                        ),
                        temporal_metrics=(
                            validated_analysis.temporal_metrics
                            if validated_analysis is not None
                            else None
                        ),
                        market_snapshot_id=(
                            getattr(market_state, "state_id", None)
                            if market_state is not None
                            else None
                        ),
                        quantitative_forecast_id=(
                            getattr(quantitative_forecast, "result_id", None)
                            if quantitative_forecast is not None
                            else None
                        ),
                        current_price=(
                            getattr(prediction, "reference_price", None)
                            if prediction
                            else None
                        ),
                        expected_move=(
                            primary_scenario.primary.expected_move
                            if primary_scenario is not None
                            and primary_scenario.primary is not None
                            else None
                        ),
                        current_primary_scenario=primary_scenario,
                    )
                )
                self.last_decision_persisted_at = self.clock()
                logger.info(
                    "decision.persist.completed",
                    extra={
                        **log_context,
                        "snapshot_id": str(snapshot.snapshot_id),
                        "analysis_id": (
                            str(validated_analysis.analysis.analysis_id)
                            if validated_analysis is not None
                            else None
                        ),
                        "analysis_signal_id": (
                            str(validated_analysis.signal.signal_id)
                            if validated_analysis is not None
                            and validated_analysis.signal is not None
                            else None
                        ),
                        "decision_id": str(decision.decision_id),
                        "decision_status": decision.state.value,
                    },
                )
                logger.info(
                    "final_decision.completed",
                    extra={
                        **log_context,
                        "snapshot_id": str(snapshot.snapshot_id),
                        "analysis_id": (
                            str(validated_analysis.analysis.analysis_id)
                            if validated_analysis is not None
                            else None
                        ),
                        "signal_id": (
                            str(validated_analysis.signal.signal_id)
                            if validated_analysis is not None
                            and validated_analysis.signal is not None
                            else None
                        ),
                        "decision_id": str(decision.decision_id),
                        "final_action": getattr(
                            getattr(decision, "final_action", None),
                            "value",
                            None,
                        ),
                        "decision_status": decision.state.value,
                        "publication_eligible": getattr(
                            decision,
                            "publication_eligible",
                            None,
                        ),
                    },
                )
                if tracker is not None:
                    tracker.mark(symbol, timeframe.value, boundary, ("scenario_decision",), "success")
                    tracker.complete(symbol, timeframe.value, boundary)
        except Exception as exc:
            exc.__dict__["integration_event_id"] = envelope.event_id
            exc.__dict__["integration_stage"] = failure_stage
            self.last_cycle_failed_at = self.clock()
            if tracker is not None:
                tracker.fail_in_flight(symbol, timeframe.value, boundary, exc=exc)
            logger.exception(
                "snapshot.failed",
                extra={
                    **log_context,
                    "engine": failure_stage,
                    "pipeline_stage": failure_stage,
                    "repository_method": failure_stage,
                    "constraint_name": None,
                    "attempt": 1,
                    "exception_class": type(exc).__name__,
                    "failure_class": type(exc).__name__,
                    "failure_stage": failure_stage,
                    "duration_ms": (perf_counter() - started) * 1000,
                },
            )
            try:
                await self._trace(envelope, TraceStatus.FAILED, "full_system", (envelope.event_id,), (), started)
            except Exception:
                logger.exception("snapshot.failure_trace_failed", extra=log_context)
            raise
        try:
            if missing:
                failure_stage = "mark_processed"
                await self.repository.mark_processed(envelope.event_id)
                failure_stage = "trace_blocked"
                await self._trace(envelope, TraceStatus.BLOCKED, "snapshot_barrier", (envelope.event_id,), (str(snapshot.snapshot_id),), started)
                self.last_cycle_completed_at = self.clock()
                logger.warning("snapshot.completed_degraded", extra={**log_context, "snapshot_id": str(snapshot.snapshot_id), "missing": missing, "duration_ms": (perf_counter() - started) * 1000})
                return None
            logger.info("signal_decision.completed", extra={"snapshot_id": str(snapshot.snapshot_id), "decision_id": str(decision.decision_id), "state": decision.state.value, "mode": envelope.mode.value})
            blocker_codes = tuple(item.reason_code for item in decision.blockers)
            warning_codes = tuple(item.reason_code for item in decision.warnings)
            semantic = canonical_hash({"snapshot": snapshot.semantic_hash, "score": score.metadata.input_fingerprint, "decision": decision.input_fingerprint, "policies": (score.policy_version, decision.decision_policy_version)})
            if (
                not publish_signal
                or decision.state.value != DecisionState.ELIGIBLE.value
                or not getattr(
                    decision,
                    "publication_eligible",
                    decision.state.value == DecisionState.ELIGIBLE.value,
                )
            ):
                failure_stage = "mark_processed"
                await self.repository.mark_processed(envelope.event_id)
                failure_stage = "trace_completed"
                await self._trace(envelope, TraceStatus.COMPLETED, "full_system", (envelope.event_id,), (str(snapshot.snapshot_id), str(score.snapshot_id), str(decision.decision_id)), started)
                self.last_cycle_completed_at = self.clock()
                logger.info("snapshot.completed", extra={**log_context, "snapshot_id": str(snapshot.snapshot_id), "decision_id": str(decision.decision_id), "decision_status": decision.state.value, "scenario_published": False, "duration_ms": (perf_counter() - started) * 1000})
                return None
            signal = OperationalSignal(operational_signal_id=semantic_uuid("signal", semantic), semantic_hash=semantic, decision_id=decision.decision_id, ai_score_id=score.snapshot_id, snapshot_id=snapshot.snapshot_id, trace_id=envelope.trace_id, market_event_id=envelope.event_id, instrument=symbol, timeframe=timeframe.value, mode=envelope.mode, direction=decision.direction.value, state=decision.state.value, confidence=decision.confidence_score, effective_at=decision.as_of, expires_at=decision.valid_until, data_quality_status=envelope.data_quality_status, provider_provenance=(envelope.source_name,), evidence=tuple(evidence), blockers=blocker_codes, warnings=warning_codes, ai_scoring_policy_version=score.policy_version, signal_decision_policy_version=decision.decision_policy_version, created_at=self.clock())
            failure_stage = "signal_persistence"
            signal = await self.repository.save_signal(signal)
            self.last_signal_published_at = self.clock()
            logger.info("scenario.published", extra={"snapshot_id": str(snapshot.snapshot_id), "signal_id": str(signal.operational_signal_id), "symbol": symbol, "timeframe": timeframe.value})
            failure_stage = "mark_processed"
            await self.repository.mark_processed(envelope.event_id)
            failure_stage = "trace_completed"
            await self._trace(envelope, TraceStatus.COMPLETED, "full_system", (envelope.event_id,), (str(snapshot.snapshot_id), str(score.snapshot_id), str(decision.decision_id), str(signal.operational_signal_id)), started)
            self.last_cycle_completed_at = self.clock()
            logger.info("snapshot.completed", extra={**log_context, "snapshot_id": str(snapshot.snapshot_id), "decision_id": str(decision.decision_id), "signal_id": str(signal.operational_signal_id), "decision_status": decision.state.value, "scenario_published": True, "duration_ms": (perf_counter() - started) * 1000})
            return signal
        except Exception as exc:
            exc.__dict__["integration_event_id"] = envelope.event_id
            exc.__dict__["integration_stage"] = failure_stage
            self.last_cycle_failed_at = self.clock()
            logger.exception(
                "snapshot.failed",
                extra={
                    **log_context,
                    "engine": failure_stage,
                    "pipeline_stage": failure_stage,
                    "repository_method": failure_stage,
                    "constraint_name": None,
                    "attempt": 1,
                    "exception_class": type(exc).__name__,
                    "failure_class": type(exc).__name__,
                    "failure_stage": failure_stage,
                    "duration_ms": (perf_counter() - started) * 1000,
                },
            )
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
