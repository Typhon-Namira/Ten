"""Feature-gated, failure-isolated analysis-only AI orchestration."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
import logging
import os
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.core.exceptions import AIProviderRequestError
from backend.app.final_decision.models import LLMUsageMetric
from backend.app.final_decision.service import FinalDecisionService
from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult

from .analysis import (
    AIMarketAnalysis,
    AIAnalysisTemporalContext,
    AIProviderMetadata,
    AnalysisStatus,
    DEFAULT_TEMPORAL_ANCHORS,
    TemporalContextAnalyzer,
    TemporalDataQuality,
    ValidatedAIAnalysis,
    analysis_reference,
)
from .cadence import (
    AI_ANALYSIS_INTERVAL_MINUTES,
    AI_ANALYSIS_TIMEFRAME,
    AIEligibilityReason,
    five_minute_window_start,
    synchronized_cycle_eligibility,
)
from .gate import AIReasoningGateDecision
from .config import AIReasoningConfig
from .llm_context import LLM_ANALYSIS_CONTEXT_SCHEMA_VERSION, build_llm_analysis_context
from .memory import MarketMemory
from .models import LLMStructuredOutputFailure
from .provider import AIReasoningProvider
from .repository import AIReasoningClaim, AIReasoningRepository
from .request_builder import AIReasoningRequestBuilder
from .signal import DeterministicAnalysisSignalGenerator
from .signal_outcomes import AnalysisSignalOutcomeEvaluator
from .validation import StructuredAIOutputError, StructuredAIOutputValidator

logger = logging.getLogger(__name__)

class AIAnalysisSkipReason(StrEnum):
    NOT_FIVE_MINUTE_BOUNDARY = "not_five_minute_boundary"
    CYCLE_NOT_COMPLETE = "cycle_not_complete"
    ANALYSIS_ALREADY_EXISTS = "analysis_already_exists"
    ACTIVE_CLAIM = "active_claim"
    DUPLICATE_MARKET_STATE = "duplicate_market_state"
    MARKET_DATA_INCOMPLETE = "market_data_incomplete"
    MARKET_DATA_STALE = "market_data_stale"
    AI_DISABLED = "ai_disabled"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    REQUEST_PREFLIGHT_FAILED = "request_preflight_failed"


def reasoning_cycle_idempotency_key(
    instrument: str,
    analysis_timeframe: str,
    five_minute_window: datetime,
    market_state_hash: str,
    analysis_contract_version: str,
) -> str:
    normalized_instrument = "".join(
        character
        for character in instrument.strip().upper()
        if character.isalnum()
    )
    material = "|".join(
        (
            normalized_instrument,
            analysis_timeframe.upper(),
            five_minute_window.astimezone(UTC).isoformat(),
            market_state_hash,
            analysis_contract_version,
        )
    )
    return sha256(material.encode()).hexdigest()


class AIReasoningService:
    def __init__(
        self,
        repository: AIReasoningRepository,
        provider: AIReasoningProvider,
        builder: AIReasoningRequestBuilder,
        validator: StructuredAIOutputValidator,
        config: AIReasoningConfig,
        *,
        shadow_enabled: bool = False,
        proposals_enabled: bool,
        monitoring_enabled: bool,
        final_decision: FinalDecisionService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.builder = builder
        self.validator = validator
        self.config = config
        self.shadow_enabled = shadow_enabled
        self.proposals_enabled = proposals_enabled
        self.monitoring_enabled = monitoring_enabled
        self.final_decision = final_decision
        self.clock = clock or (lambda: datetime.now(UTC))
        self._llm_semaphore = asyncio.Semaphore(config.llm_concurrency_limit)
        host = os.getenv("RAILWAY_REPLICA_ID") or os.getenv("HOSTNAME") or os.getenv("COMPUTERNAME") or "worker"
        self.worker_id = f"{host}:{os.getpid()}"
        self._claim_heartbeats: dict[str, asyncio.Task[None]] = {}
        self._active_claims: dict[str, AIReasoningClaim] = {}
        self.memory = MarketMemory(config.maximum_memory_entries)
        self.signal_generator = DeterministicAnalysisSignalGenerator(config)
        self.signal_outcomes = AnalysisSignalOutcomeEvaluator()
        self.requests = 0
        self.failed_requests = 0
        self.last_latency_ms: float | None = None
        self.last_validation_passed: bool | None = None
        self.last_retry_count = 0
        self.last_failure_state: str | None = None
        self.last_cycle_outcome: str | None = None
        self.last_eligible_cycle_at: datetime | None = None
        self.metrics: Counter[str] = Counter()
        self.skip_reasons: Counter[str] = Counter()
        self.deployment_id = (
            os.getenv("RAILWAY_DEPLOYMENT_ID")
            or os.getenv("RAILWAY_GIT_COMMIT_SHA")
            or os.getenv("GIT_SHA")
            or "local"
        )
        self.boot_session_id = str(
            uuid5(
                NAMESPACE_URL,
                f"ten:ai-boot:{self.deployment_id}:{id(self)}",
            )
        )

    @property
    def enabled(self) -> bool:
        return self.shadow_enabled

    def _claim_log_context(self, claim: AIReasoningClaim, **extra: object) -> dict[str, object]:
        return {
            "instrument": claim.instrument,
            "cutoff": claim.market_cutoff.isoformat(),
            "claim_id": str(claim.claim_id),
            "worker_id": self.worker_id,
            "market_state_id": str(claim.market_state_id) if claim.market_state_id else None,
            "snapshot_id": str(claim.snapshot_id) if claim.snapshot_id else None,
            "analysis_id": str(claim.analysis_id) if claim.analysis_id else None,
            "claimed_at": claim.claimed_at.isoformat(),
            "heartbeat_at": claim.heartbeat_at.isoformat(),
            "lease_expires_at": claim.lease_expires_at.isoformat(),
            **extra,
        }

    def _start_claim_heartbeat(self, claim: AIReasoningClaim) -> None:
        self._active_claims[claim.idempotency_key] = claim
        async def heartbeat() -> None:
            deadline = claim.claimed_at + timedelta(seconds=self.config.claim_max_runtime_seconds)
            try:
                while True:
                    await asyncio.sleep(self.config.claim_heartbeat_seconds)
                    now = self.clock().astimezone(UTC)
                    if now >= deadline:
                        released = await self.repository.release_reasoning_cycle(
                            claim.idempotency_key,
                            claim.claim_id,
                            now,
                            status="FAILED",
                            failure_reason="claim_max_runtime_exceeded",
                        )
                        if released:
                            logger.error(
                                "ai_reasoning.claim.failed",
                                extra=self._claim_log_context(
                                    claim,
                                    failure_reason="claim_max_runtime_exceeded",
                                    released_at=now.isoformat(),
                                ),
                            )
                        return
                    renewed = await self.repository.heartbeat_reasoning_cycle(
                        claim.idempotency_key,
                        claim.claim_id,
                        self.worker_id,
                        now,
                        self.config.claim_lease_seconds,
                    )
                    if not renewed:
                        return
                    logger.info(
                        "ai_reasoning.claim.heartbeat",
                        extra=self._claim_log_context(
                            claim,
                            heartbeat_at=now.isoformat(),
                            lease_expires_at=(
                                now + timedelta(seconds=self.config.claim_lease_seconds)
                            ).isoformat(),
                        ),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "ai_reasoning.claim.heartbeat_failed",
                    extra=self._claim_log_context(claim),
                )
            finally:
                self._claim_heartbeats.pop(claim.idempotency_key, None)
                self._active_claims.pop(claim.idempotency_key, None)

        self._claim_heartbeats[claim.idempotency_key] = asyncio.create_task(
            heartbeat(), name=f"ai-claim-heartbeat:{claim.idempotency_key[:12]}"
        )

    async def shutdown(self) -> None:
        """Release locally owned claims before worker/process shutdown."""

        tasks = tuple(self._claim_heartbeats.items())
        claims = tuple(self._active_claims.values())
        for _, task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
        now = self.clock().astimezone(UTC)
        for claim in claims:
            released = await self.repository.release_reasoning_cycle(
                claim.idempotency_key,
                claim.claim_id,
                now,
                failure_reason="worker_shutdown",
            )
            if released:
                logger.info(
                    "ai_reasoning.claim.released",
                    extra=self._claim_log_context(
                        claim,
                        failure_reason="worker_shutdown",
                        released_at=now.isoformat(),
                    ),
                )

    async def _complete_claim(
        self,
        claim: AIReasoningClaim,
        *,
        request_id: object | None,
        analysis_id: object | None,
        status: str,
        failure_reason: str | None = None,
    ) -> None:
        now = self.clock().astimezone(UTC)
        await self.repository.complete_analysis_cycle(
            claim.idempotency_key,
            request_id,
            analysis_id,
            status,
            now,
            claim_id=claim.claim_id,
            failure_reason=failure_reason,
        )
        task = self._claim_heartbeats.pop(claim.idempotency_key, None)
        self._active_claims.pop(claim.idempotency_key, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        event = {
            "COMMITTED": "ai_reasoning.analysis.committed",
            "RECOVERED": "ai_reasoning.analysis.committed",
            "FAILED": "ai_reasoning.claim.failed",
            "RELEASED": "ai_reasoning.claim.released",
            "EXPIRED": "ai_reasoning.claim.expired",
        }.get(status, "ai_reasoning.claim.released")
        logger.info(
            event,
            extra=self._claim_log_context(
                claim,
                analysis_id=str(analysis_id) if analysis_id else None,
                status=status,
                failure_reason=failure_reason,
                released_at=now.isoformat(),
                claim_duration_seconds=max(0.0, (now - claim.claimed_at).total_seconds()),
            ),
        )

    async def record_gate_decision(
        self,
        *,
        state: UnifiedMarketState | None,
        attempted_cutoff: datetime,
        gate_decision: str,
        gate_skip_reason: str | None = None,
        existing_analysis: AIMarketAnalysis | None = None,
        details: Mapping[str, object] | None = None,
        trigger_timeframe: str | None = None,
        instrument: str | None = None,
    ) -> AIReasoningGateDecision:
        """Persist one exact-cutoff AI gate decision for operational diagnosis."""

        created_at = self.clock().astimezone(UTC)
        cutoff = attempted_cutoff.astimezone(UTC)
        market_state_id = state.state_id if state is not None else None
        trigger = trigger_timeframe or (
            state.trigger_timeframe if state is not None else None
        )
        decision = AIReasoningGateDecision(
            decision_id=uuid5(
                NAMESPACE_URL,
                "|".join(
                    (
                        "ten:ai-reasoning-gate",
                        state.instrument if state is not None else instrument or "unknown",
                        cutoff.isoformat(),
                        str(market_state_id),
                        gate_decision,
                        gate_skip_reason or "none",
                    )
                ),
            ),
            instrument=state.instrument if state is not None else instrument or "unknown",
            trigger_timeframe=trigger,
            attempted_cutoff=cutoff,
            analysis_lookup_cutoff=cutoff,
            market_state_id=market_state_id,
            snapshot_id=market_state_id,
            gate_decision=gate_decision,
            gate_skip_reason=gate_skip_reason,
            existing_analysis_id=(
                existing_analysis.analysis_id if existing_analysis is not None else None
            ),
            analysis_created_at=(
                existing_analysis.created_at if existing_analysis is not None else None
            ),
            analysis_market_cutoff=(
                existing_analysis.analysis_timestamp
                if existing_analysis is not None
                else None
            ),
            details=dict(details or {}),
            created_at=created_at,
        )
        return await self.repository.save_gate_decision(decision)

    async def process(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
    ) -> ValidatedAIAnalysis | None:
        """Persist exactly one analysis-only result for a completed market cycle."""

        worker_context = {
            "cycle_id": str(state.cycle_id),
            "market_state_id": str(state.state_id),
            "quantitative_forecast_id": str(quant.result_id),
            "instrument": state.instrument,
            "trigger": "integration_worker",
            "artifact_type": "ai_market_analysis",
            "attempted_cutoff": state.market_data_boundary.isoformat(),
            "analysis_lookup_cutoff": state.market_data_boundary.isoformat(),
            "snapshot_id": str(state.state_id),
            "trigger_timeframe": state.trigger_timeframe,
        }
        self.metrics["scheduler_ticks"] += 1
        logger.info("ai_reasoning.worker.received", extra=worker_context)
        if not self.enabled:
            await self._skip(
                AIAnalysisSkipReason.AI_DISABLED,
                worker_context,
                state=state,
            )
            return None
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        quant = QuantForecastResult.model_validate(quant.model_dump(mode="python"))
        boundary = state.market_data_boundary.astimezone(UTC)
        window_start = five_minute_window_start(boundary)
        eligibility_failure = self._eligibility_failure(state, boundary, window_start)
        if eligibility_failure is not None:
            await self._skip(
                eligibility_failure,
                worker_context,
                state=state,
                window_start=window_start,
            )
            return None
        self.metrics["eligible_five_minute_cycles"] += 1
        self.last_eligible_cycle_at = self.clock().astimezone(UTC)
        existing_for_state = await self.repository.analysis_for_state(state.state_id)
        if (
            existing_for_state is not None
            and existing_for_state.analysis_timestamp == boundary
            and existing_for_state.market_snapshot_id == state.state_id
        ):
            self.metrics["analyses_reused"] += 1
            await self._skip(
                AIAnalysisSkipReason.ANALYSIS_ALREADY_EXISTS,
                worker_context,
                state=state,
                existing_analysis=existing_for_state,
                window_start=window_start,
            )
            return await self._validated_analysis(existing_for_state, state, quant)
        contract_version = ":".join(
            (
                LLM_ANALYSIS_CONTEXT_SCHEMA_VERSION,
                "ai-market-analysis-2.0",
                self.config.reasoning_policy_version,
            )
        )
        claim_timeframe = (
            f"{AI_ANALYSIS_TIMEFRAME}:{state.trigger_timeframe.upper()}"
        )
        idempotency_key = reasoning_cycle_idempotency_key(
            state.instrument,
            claim_timeframe,
            window_start,
            state.state_hash,
            contract_version,
        )
        claim = await self.repository.claim_reasoning_cycle(
            idempotency_key,
            state.instrument,
            window_start,
            state.schema_version,
            contract_version,
            self.clock(),
            analysis_timeframe=claim_timeframe,
            five_minute_window_start=window_start,
            market_state_hash=state.state_hash,
            analysis_contract_version=contract_version,
            market_state_id=state.state_id,
            snapshot_id=state.state_id,
            claimed_by=self.worker_id,
            lease_seconds=self.config.claim_lease_seconds,
        )
        if not claim.acquired:
            cached = await self.repository.analysis_for_reasoning_cycle(idempotency_key)
            if claim.outcome == "DUPLICATE_COMMITTED_ANALYSIS" and cached is not None:
                self.metrics["analyses_reused"] += 1
                await self._skip(
                    AIAnalysisSkipReason.ANALYSIS_ALREADY_EXISTS,
                    worker_context,
                    state=state,
                    existing_analysis=cached,
                    window_start=window_start,
                    idempotency_key=idempotency_key,
                )
                return await self._validated_analysis(cached, state, quant)
            duplicate = await self.repository.analysis_for_market_state_hash(
                state.instrument,
                state.state_hash,
                contract_version,
            )
            reason = (
                AIAnalysisSkipReason.DUPLICATE_MARKET_STATE
                if duplicate is not None
                else AIAnalysisSkipReason.ACTIVE_CLAIM
            )
            await self._skip(
                reason,
                worker_context,
                state=state,
                existing_analysis=duplicate,
                window_start=window_start,
                idempotency_key=idempotency_key,
                claim=claim,
            )
            return (
                await self._validated_analysis(duplicate, state, quant)
                if duplicate is not None
                else None
            )

        claim_event = (
            "ai_reasoning.claim.takeover"
            if claim.outcome == "CLAIM_TAKEOVER"
            else "ai_reasoning.claim.acquired"
        )
        logger.info(claim_event, extra=self._claim_log_context(claim, outcome=claim.outcome))
        if claim.outcome == "CLAIM_TAKEOVER":
            logger.warning(
                "ai_reasoning.claim.expired",
                extra=self._claim_log_context(
                    claim,
                    expired_claim_count=claim.expired_claim_count,
                ),
            )
        self._start_claim_heartbeat(claim)

        try:
            await self.record_gate_decision(
                state=state,
                attempted_cutoff=boundary,
                gate_decision="PROCEED",
                details={
                    "idempotency_key": idempotency_key,
                    "analysis_timeframe": claim_timeframe,
                    "claim_id": str(claim.claim_id),
                },
            )
        except Exception:
            await self._complete_claim(
                claim,
                request_id=None,
                analysis_id=None,
                status="FAILED",
                failure_reason="gate_decision_persistence",
            )
            raise

        try:
            recent_memory = await self.repository.recent_memory(
                state.instrument,
                self.config.maximum_memory_entries,
            )
            request = self.builder.build(
                state,
                quant,
                self.memory.summarize(recent_memory),
                existing_signal=None,
                previous_forecast=None,
                previous_proposal=None,
            )
            request = request.model_copy(
                update={
                    "idempotency_key": idempotency_key,
                    "analysis_time_bucket": window_start,
                }
            )
            await self.repository.save_request(request)
        except Exception:
            await self._complete_claim(
                claim,
                request_id=None,
                analysis_id=None,
                status="FAILED",
                failure_reason="request_construction_or_persistence",
            )
            logger.exception(
                "ai_reasoning.job.terminal",
                extra={
                    **worker_context,
                    "request_id": (
                        str(request.request_id)
                        if "request" in locals()
                        else None
                    ),
                    "job_state": "FAILED_PERSISTENCE",
                    "failure_phase": "request_construction_or_persistence",
                },
            )
            return None
        response = None
        provider_failure: dict[str, Any] | None = None
        provider_metrics_before = self._provider_metrics()
        provider_delta: dict[str, int] = {}
        provider_attempts: tuple[dict[str, Any], ...] = ()
        failure_state = "llm_unavailable"
        failure_errors: tuple[str, ...] = ()
        terminal_status = "FAILED_PROVIDER"
        try:
            async with self._llm_semaphore:
                logger.info(
                    "ai_reasoning.provider_call.started",
                    extra={
                        **worker_context,
                        "request_id": str(request.request_id),
                        "idempotency_key": idempotency_key,
                    },
                )
                response = await asyncio.wait_for(
                    self.provider.reason(request, prompt_version=request.prompt_version),
                    timeout=self.config.request_timeout_seconds,
                )
            provider_delta = self._metric_delta(
                provider_metrics_before,
                self._provider_metrics(),
            )
            self._consume_provider_metrics(provider_delta)
            self.last_latency_ms = response.latency_ms
            output = self.validator.validate_analysis(response.raw_output)
            analysis = AIMarketAnalysis(
                analysis_id=uuid5(
                    NAMESPACE_URL,
                    f"ten:ai-market-analysis:{request.request_id}:2.0",
                ),
                request_id=request.request_id,
                cycle_id=request.cycle_id,
                symbol=request.instrument,
                timeframe=request.trigger_timeframe,
                market_snapshot_id=request.market_state_id,
                quantitative_forecast_id=request.quantitative_forecast_id,
                analysis_timestamp=request.analysis_timestamp,
                knowledge_cutoff=request.knowledge_cutoff,
                status=AnalysisStatus.AVAILABLE,
                output=output,
                provider_metadata=AIProviderMetadata(
                    provider=response.provider,
                    model=response.model_identifier,
                    prompt_version=request.prompt_version,
                    provider_adapter_version="openai-compatible-analysis-v2",
                    fallback_used=response.fallback_used,
                    fallback_reason=response.fallback_reason,
                    latency_ms=response.latency_ms,
                    token_usage=response.token_usage,
                ),
                validation_passed=True,
                created_at=self.clock(),
            )
            provider_attempts = self._provider_attempts(request.request_id)
            for account_id in ("groq_1", "groq_2", "groq_3", "groq_4"):
                account_attempts = tuple(
                    item
                    for item in provider_attempts
                    if item.get("account_id") == account_id
                )
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    values = [
                        int(value)
                        for item in account_attempts
                        if isinstance((value := item.get(key)), int)
                    ]
                    if values:
                        provider_delta[f"{account_id}_{key}"] = sum(values)
            await self._record_usage(
                request,
                state.state_hash,
                response.token_usage,
                response.latency_ms,
                True,
                None,
                model_identifier=response.model_identifier,
                provider=response.provider,
                provider_metrics=provider_delta,
                provider_attempts=provider_attempts,
            )
            logger.info(
                "structured_validation.completed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "validation_status": "valid",
                    "validation_issue_count": 0,
                    "artifact_type": "ai_market_analysis",
                },
            )
        except StructuredAIOutputError as exc:
            terminal_status = "FAILED_SCHEMA"
            if not provider_delta:
                provider_delta = self._metric_delta(
                    provider_metrics_before,
                    self._provider_metrics(),
                )
                self._consume_provider_metrics(provider_delta)
            self.metrics["validation_failures"] += 1
            failure_state = "structured_output_invalid"
            failure_errors = exc.errors
            logger.error(
                "ai_reasoning.request.failed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "failure_phase": "structured_output_validation",
                    "field_path": exc.first_issue.field_path if exc.first_issue else None,
                    "expected_type": exc.first_issue.expected_type if exc.first_issue else None,
                    "actual_value": exc.first_issue.actual_value if exc.first_issue else None,
                    "validator_name": exc.first_issue.validator_name if exc.first_issue else None,
                    "offending_json_fragment": (
                        exc.first_issue.offending_json_fragment if exc.first_issue else None
                    ),
                },
            )
            analysis = self._failed_analysis(request, failure_state, failure_errors)
        except TimeoutError as exc:
            provider_delta = self._metric_delta(
                provider_metrics_before,
                self._provider_metrics(),
            )
            self._consume_provider_metrics(provider_delta)
            failure_state = "hard_terminal_timeout"
            failure_errors = (type(exc).__name__,)
            terminal_status = "TIMED_OUT"
            analysis = self._failed_analysis(request, failure_state, failure_errors)
            logger.error(
                "ai_reasoning.job.terminal",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "job_state": terminal_status,
                    "failure_phase": "provider_timeout",
                },
            )
        except AIProviderRequestError as exc:
            provider_failure = {
                "terminal": asdict(exc.details),
                "providers": self.provider.metadata().get("providers"),
            }
            provider_delta = self._metric_delta(
                provider_metrics_before,
                self._provider_metrics(),
            )
            self._consume_provider_metrics(provider_delta)
            if exc.details.phase == "request_validation":
                await self._skip(
                    AIAnalysisSkipReason.REQUEST_PREFLIGHT_FAILED,
                    worker_context,
                    state=state,
                    window_start=window_start,
                    idempotency_key=idempotency_key,
                )
                await self._complete_claim(
                    claim,
                    request_id=request.request_id,
                    analysis_id=None,
                    status="RELEASED",
                    failure_reason="request_preflight_failed",
                )
                return None
            if exc.details.reason_code not in {
                "output_budget_exceeded",
                "schema_validation_error",
            }:
                self.metrics["provider_failures"] += 1
            self.last_latency_ms = exc.details.elapsed_ms
            failure_state = exc.details.reason_code
            failure_errors = tuple(
                str(item)
                for item in (
                    exc.details.error_code,
                    exc.details.error_message,
                    exc.details.metadata_error_type,
                )
                if item
            )
            analysis = self._failed_analysis(request, failure_state, failure_errors)
            terminal_status = (
                "FAILED_SCHEMA"
                if exc.details.reason_code
                in {"output_budget_exceeded", "schema_validation_error"}
                else "FAILED_PROVIDER"
            )
        except Exception as exc:
            provider_delta = self._metric_delta(
                provider_metrics_before,
                self._provider_metrics(),
            )
            self._consume_provider_metrics(provider_delta)
            failure_errors = (type(exc).__name__, str(exc)[:200])
            analysis = self._failed_analysis(request, failure_state, failure_errors)
            terminal_status = "FAILED_PROVIDER"
            logger.exception(
                "ai_reasoning.request.failed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "failure_phase": "provider_or_domain",
                },
            )

        provider_attempts = self._provider_attempts(request.request_id)
        attempt_usage = self._attempt_usage(provider_attempts)
        for account_id in ("groq_1", "groq_2", "groq_3", "groq_4"):
            account_attempts = tuple(
                item
                for item in provider_attempts
                if item.get("account_id") == account_id
            )
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                values = [
                    int(value)
                    for item in account_attempts
                    if isinstance((value := item.get(key)), int)
                ]
                if values:
                    provider_delta[f"{account_id}_{key}"] = sum(values)

        if provider_failure is not None:
            await self._complete_claim(
                claim,
                request_id=request.request_id,
                analysis_id=None,
                status="FAILED",
                failure_reason=failure_state,
            )
            self.failed_requests += 1
            self.last_validation_passed = False
            self.last_failure_state = failure_state
            self.last_cycle_outcome = "failed"
            await self._record_failure(
                request.request_id,
                0,
                request.model_identifier,
                request.prompt_version,
                None,
                failure_errors,
                failure_state,
                provider_failure,
            )
            await self._record_usage(
                request,
                state.state_hash,
                attempt_usage,
                self.last_latency_ms,
                False,
                failure_state,
                model_identifier=request.model_identifier,
                provider=None,
                provider_metrics=provider_delta,
                provider_attempts=provider_attempts,
            )
            return None

        ownership_confirmed = await self.repository.heartbeat_reasoning_cycle(
            claim.idempotency_key,
            claim.claim_id,
            self.worker_id,
            self.clock().astimezone(UTC),
            self.config.claim_lease_seconds,
        )
        if not ownership_confirmed:
            logger.warning(
                "ai_reasoning.claim.expired",
                extra=self._claim_log_context(
                    claim,
                    failure_reason="lease_lost_before_analysis_persistence",
                ),
            )
            return None
        try:
            analysis = await self.repository.save_analysis(analysis)
        except Exception:
            await self._complete_claim(
                claim,
                request_id=request.request_id,
                analysis_id=None,
                status="FAILED",
                failure_reason="analysis_persistence",
            )
            logger.exception(
                "ai_reasoning.job.terminal",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "job_state": "FAILED_PERSISTENCE",
                    "failure_phase": "analysis_persistence",
                },
            )
            return None
        if analysis.status != AnalysisStatus.AVAILABLE:
            await self._complete_claim(
                claim,
                request_id=request.request_id,
                analysis_id=analysis.analysis_id,
                status="FAILED",
                failure_reason=failure_state,
            )
            self.failed_requests += 1
            self.last_validation_passed = False
            self.last_failure_state = failure_state
            self.last_cycle_outcome = (
                "configuration_error"
                if failure_state
                in {
                    "authentication_failed",
                    "invalid_request",
                    "model_unavailable",
                    "provider_unconfigured",
                }
                else "failed"
            )
            await self._record_failure(
                request.request_id,
                0,
                response.model_identifier if response is not None else request.model_identifier,
                request.prompt_version,
                response.raw_output if response is not None else None,
                failure_errors,
                failure_state,
                provider_failure,
            )
            await self._record_usage(
                request,
                state.state_hash,
                response.token_usage if response is not None else None,
                response.latency_ms if response is not None else None,
                False,
                failure_state,
                model_identifier=(
                    response.model_identifier
                    if response is not None
                    else request.model_identifier
                ),
                provider=response.provider if response is not None else None,
                provider_metrics=provider_delta,
                provider_attempts=provider_attempts,
            )
            return None

        try:
            commit_decision = await self.record_gate_decision(
                state=state,
                attempted_cutoff=analysis.analysis_timestamp,
                gate_decision="COMMITTED",
                existing_analysis=analysis,
                details={
                    "transaction_commit_time": self.clock().isoformat(),
                    "idempotency_key": idempotency_key,
                    "claim_id": str(claim.claim_id),
                    "claim_outcome": claim.outcome,
                },
            )
        except Exception:
            await self._complete_claim(
                claim,
                request_id=request.request_id,
                analysis_id=analysis.analysis_id,
                status="FAILED",
                failure_reason="commit_gate_persistence",
            )
            raise

        self.last_validation_passed = True
        self.last_failure_state = None
        assert response is not None
        mark_persisted = getattr(self.provider, "mark_analysis_persisted", None)
        if callable(mark_persisted):
            mark_persisted(response.provider, analysis.created_at)
        logger.info(
            "ai_reasoning.persist.completed",
            extra={
                **worker_context,
                "request_id": str(request.request_id),
                "analysis_id": str(analysis.analysis_id),
                "ai_analysis_cutoff": analysis.analysis_timestamp.isoformat(),
                "transaction_commit_time": commit_decision.created_at.isoformat(),
                "status": analysis.status.value,
            },
        )
        try:
            validated = await self._validated_analysis(
                analysis,
                state,
                quant,
                persist_signal=True,
            )
        except Exception:
            self.failed_requests += 1
            self.last_cycle_outcome = "failed"
            self.last_failure_state = "analysis_signal_persistence_failed"
            await self._complete_claim(
                claim,
                request_id=request.request_id,
                analysis_id=analysis.analysis_id,
                status="FAILED",
                failure_reason="analysis_signal_persistence",
            )
            logger.exception(
                "ai_reasoning.job.terminal",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "analysis_id": str(analysis.analysis_id),
                    "job_state": "FAILED_PERSISTENCE",
                    "failure_phase": "analysis_signal_persistence",
                },
            )
            return None
        self.last_cycle_outcome = "pool_success"
        self.metrics["analyses_successfully_completed"] += 1
        await self._complete_claim(
            claim,
            request_id=request.request_id,
            analysis_id=analysis.analysis_id,
            status="RECOVERED" if claim.outcome == "CLAIM_TAKEOVER" else "COMMITTED",
        )
        logger.info(
            "ai_reasoning.job.terminal",
            extra={
                **worker_context,
                "request_id": str(request.request_id),
                "analysis_id": str(analysis.analysis_id),
                "signal_id": (
                    str(validated.signal.signal_id)
                    if validated is not None and validated.signal is not None
                    else None
                ),
                "job_state": "COMPLETED",
            },
        )
        return validated

    def _failed_analysis(
        self,
        request: Any,
        failure_state: str,
        failure_errors: tuple[str, ...],
    ) -> AIMarketAnalysis:
        metadata = self.provider.metadata()
        return AIMarketAnalysis(
            analysis_id=uuid5(
                NAMESPACE_URL,
                f"ten:ai-market-analysis:{request.request_id}:2.0",
            ),
            request_id=request.request_id,
            cycle_id=request.cycle_id,
            symbol=request.instrument,
            timeframe=request.trigger_timeframe,
            market_snapshot_id=request.market_state_id,
            quantitative_forecast_id=request.quantitative_forecast_id,
            analysis_timestamp=request.analysis_timestamp,
            knowledge_cutoff=request.knowledge_cutoff,
            status=(
                AnalysisStatus.INVALID
                if failure_state == "structured_output_invalid"
                else AnalysisStatus.FAILED
            ),
            output=None,
            provider_metadata=AIProviderMetadata(
                provider=str(metadata.get("provider", "unavailable")),
                model=str(metadata.get("model_identifier", request.model_identifier)),
                prompt_version=request.prompt_version,
                provider_adapter_version="openai-compatible-analysis-v2",
            ),
            validation_passed=False,
            validation_errors=failure_errors,
            created_at=self.clock(),
        )

    async def _validated_analysis(
        self,
        analysis: AIMarketAnalysis,
        state: UnifiedMarketState | None = None,
        quant: QuantForecastResult | None = None,
        *,
        persist_signal: bool = False,
    ) -> ValidatedAIAnalysis | None:
        if analysis.status != AnalysisStatus.AVAILABLE or analysis.output is None:
            return None
        history = await self.repository.analyses_before(
            analysis.symbol,
            analysis.timeframe,
            analysis.analysis_timestamp,
            240,
        )
        references = tuple(
            analysis_reference(item)
            for item in reversed(history)
            if item.output is not None
        )
        tolerances = {
            label: timedelta(minutes=self.config.temporal_tolerance_minutes[label])
            for label in DEFAULT_TEMPORAL_ANCHORS
        }
        lookbacks = {}
        for label, anchor in DEFAULT_TEMPORAL_ANCHORS.items():
            target = analysis.analysis_timestamp - anchor
            candidates = [
                item
                for item in references
                if target - tolerances[label] <= item.analysis_timestamp <= target
            ]
            lookbacks[label] = (
                max(candidates, key=lambda item: item.analysis_timestamp)
                if candidates
                else None
            )
        context = AIAnalysisTemporalContext(
            current_analysis_id=analysis.analysis_id,
            as_of=analysis.analysis_timestamp,
            previous_analysis_id=references[-1].analysis_id if references else None,
            lookbacks=lookbacks,
            rolling_window=references[-self.config.temporal_rolling_window :],
            data_quality=(
                TemporalDataQuality.SUFFICIENT
                if len(references) >= 5
                else TemporalDataQuality.LIMITED
                if references
                else TemporalDataQuality.INSUFFICIENT_HISTORY
            ),
        )
        metrics = TemporalContextAnalyzer().analyze(context, analysis)
        signal = await self.repository.signal_for_analysis(analysis.analysis_id)
        if signal is None and state is not None and quant is not None:
            signal = self.signal_generator.generate(analysis, state, quant)
            logger.info(
                "ai_reasoning.signal.generated",
                extra={
                    "analysis_id": str(analysis.analysis_id),
                    "signal_id": str(signal.signal_id),
                    "cycle_id": str(analysis.cycle_id),
                    "snapshot_id": str(analysis.market_snapshot_id),
                    "signal": signal.signal.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                },
            )
            if persist_signal:
                signal = await self.repository.save_analysis_signal(signal)
                logger.info(
                    "ai_reasoning.signal.persist.completed",
                    extra={
                        "analysis_id": str(analysis.analysis_id),
                        "signal_id": str(signal.signal_id),
                        "cycle_id": str(analysis.cycle_id),
                        "snapshot_id": str(analysis.market_snapshot_id),
                    },
                )
        if signal is not None and state is not None:
            await self._update_analysis_signal_outcomes(signal, state)
        logger.info(
            "ai_analysis.temporal_context.completed",
            extra={
                "analysis_id": str(analysis.analysis_id),
                "instrument": analysis.symbol,
                "timeframe": analysis.timeframe,
                "sample_size": len(references),
                "historical_consistency": metrics.historical_consistency.classification.value,
                "analysis_momentum": metrics.analysis_momentum.direction.value,
            },
        )
        return ValidatedAIAnalysis(
            analysis=analysis,
            temporal_context=context,
            temporal_metrics=metrics,
            signal=signal,
        )

    async def _update_analysis_signal_outcomes(
        self,
        current_signal: Any,
        state: UnifiedMarketState,
    ) -> None:
        current_outcome = await self.repository.analysis_signal_outcome(
            current_signal.signal_id
        )
        if current_outcome is None:
            current_outcome = await self.repository.save_analysis_signal_outcome(
                self.signal_outcomes.initial(current_signal)
            )
        market_items = [
            item
            for item in state.evidence
            if item.source_engine.lower() == "market_data"
            and isinstance(item.raw_value, Mapping)
        ]
        if not market_items:
            return
        latest_market = max(
            market_items,
            key=lambda item: item.source_candle_close_timestamp,
        )
        raw = latest_market.raw_value
        assert isinstance(raw, Mapping)
        high = raw.get("high")
        low = raw.get("low")
        close = raw.get("close")
        numeric_values = tuple(
            float(value)
            for value in (high, low, close)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        if len(numeric_values) != 3:
            return
        candle_high, candle_low, candle_close = numeric_values
        previous_signals = await self.repository.list_analysis_signals(
            current_signal.instrument,
            None,
            None,
            current_signal.generated_at,
            None,
            None,
            None,
            0,
            100,
        )
        for signal in previous_signals:
            if signal.signal_id == current_signal.signal_id:
                continue
            previous = await self.repository.analysis_signal_outcome(signal.signal_id)
            if previous is None:
                previous = self.signal_outcomes.initial(signal)
            evaluated = self.signal_outcomes.evaluate(
                signal,
                previous,
                candle_high=candle_high,
                candle_low=candle_low,
                candle_close=candle_close,
                evaluated_at=state.market_data_boundary,
                superseded=True,
            )
            if evaluated != previous:
                await self.repository.save_analysis_signal_outcome(evaluated)
                logger.info(
                    "ai_reasoning.signal.outcome.updated",
                    extra={
                        "signal_id": str(signal.signal_id),
                        "cycle_id": str(signal.cycle_id),
                        "status": evaluated.status.value,
                        "entry_reached": evaluated.entry_reached,
                        "target_hit": evaluated.target_hit,
                        "stop_hit": evaluated.stop_hit,
                        "actual_risk_reward": evaluated.actual_risk_reward,
                    },
                )

    async def _record_usage(
        self,
        request: Any,
        state_hash: str,
        token_usage: dict[str, int] | None,
        latency_ms: float | None,
        success: bool,
        failure_state: str | None,
        *,
        model_identifier: str | None = None,
        provider: str | None = None,
        provider_metrics: dict[str, int] | None = None,
        provider_attempts: tuple[dict[str, Any], ...] = (),
    ) -> None:
        if self.final_decision is None:
            return
        payload = build_llm_analysis_context(request).model_dump(mode="json")
        request_hash = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        total = token_usage.get("total_tokens") if token_usage else None
        latest_attempt = provider_attempts[-1] if provider_attempts else {}
        analysis_schema_version = latest_attempt.get(
            "analysis_schema_version"
        )
        output_profile = latest_attempt.get("output_profile")
        usage = LLMUsageMetric(
            metric_id=uuid5(NAMESPACE_URL, f"ten:llm-usage:{request.request_id}:{success}:{failure_state}"),
            usage_date=self.clock().date().isoformat(),
            request_hash=request_hash,
            market_state_hash=state_hash,
            model_identifier=model_identifier or request.model_identifier,
            prompt_version=request.prompt_version,
            generation_parameters={
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "provider": provider,
                "telemetry_policy": "five_minute_v1",
                "deployment_id": self.deployment_id,
                "boot_session_id": self.boot_session_id,
                "analysis_schema_version": analysis_schema_version,
                "output_profile": output_profile,
                "provider_attempts": list(provider_attempts),
                **(provider_metrics or {}),
                "provider_failure": int(
                    failure_state is not None
                    and failure_state
                    not in {
                        "structured_output_invalid",
                        "output_budget_exceeded",
                        "schema_validation_error",
                    }
                    and (provider_metrics or {}).get("provider_http_calls", 0) > 0
                ),
                "validation_failure": int(
                    failure_state == "structured_output_invalid"
                ),
            },
            request_count=(provider_metrics or {}).get("provider_http_calls", 0),
            input_tokens=token_usage.get("input_tokens") if token_usage else None,
            output_tokens=token_usage.get("output_tokens") if token_usage else None,
            total_tokens=total,
            latency_ms=latency_ms,
            success=success,
            failure_state=failure_state,
            created_at=self.clock(),
        )
        await self.final_decision.repository.save_usage(usage)

    def _provider_attempts(
        self,
        request_id: object,
    ) -> tuple[dict[str, Any], ...]:
        getter = getattr(self.provider, "attempts_for", None)
        if not callable(getter):
            return ()
        value = getter(request_id)
        return tuple(value) if isinstance(value, (tuple, list)) else ()

    @staticmethod
    def _attempt_usage(
        attempts: tuple[dict[str, Any], ...],
    ) -> dict[str, int] | None:
        usage: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            values = [
                int(value)
                for item in attempts
                if isinstance((value := item.get(key)), int)
            ]
            if values:
                usage[key] = sum(values)
        return usage or None

    async def _record_failure(
        self,
        request_id: object,
        attempt: int,
        model: str,
        prompt: str,
        raw: dict[str, Any] | None,
        errors: tuple[str, ...],
        state: str,
        provider_failure: dict[str, Any] | None,
    ) -> None:
        failure = LLMStructuredOutputFailure(
            failure_id=uuid5(NAMESPACE_URL, f"ten:llm-failure:{request_id}:{attempt}:{state}"),
            request_id=request_id,
            attempt=attempt,
            model_identifier=model,
            prompt_version=prompt,
            # Provider output can contain private reasoning and is never persisted.
            raw_output=None,
            validation_errors=errors,
            failure_state=state,
            provider_failure=provider_failure,
            latency_ms=self.last_latency_ms,
            created_at=self.clock(),
        )
        await self.repository.save_failure(failure)

    def _eligibility_failure(
        self,
        state: UnifiedMarketState,
        boundary: datetime,
        window_start: datetime,
    ) -> AIAnalysisSkipReason | None:
        del window_start
        shared = synchronized_cycle_eligibility(state)
        if shared == AIEligibilityReason.INTERVAL_NOT_DUE:
            return AIAnalysisSkipReason.NOT_FIVE_MINUTE_BOUNDARY
        if shared == AIEligibilityReason.MISSING_PREREQUISITE:
            return AIAnalysisSkipReason.MARKET_DATA_INCOMPLETE
        if shared == AIEligibilityReason.STALE_DATA:
            return AIAnalysisSkipReason.MARKET_DATA_STALE
        if shared == AIEligibilityReason.INVALID_STATE:
            return AIAnalysisSkipReason.CYCLE_NOT_COMPLETE
        if self.clock().astimezone(UTC) < boundary:
            return AIAnalysisSkipReason.CYCLE_NOT_COMPLETE
        return None

    async def _skip(
        self,
        reason: AIAnalysisSkipReason,
        context: Mapping[str, object],
        *,
        state: UnifiedMarketState,
        existing_analysis: AIMarketAnalysis | None = None,
        window_start: datetime | None = None,
        idempotency_key: str | None = None,
        claim: AIReasoningClaim | None = None,
    ) -> None:
        self.skip_reasons[reason.value] += 1
        self.metrics["skipped_before_provider_call"] += 1
        if reason in {
            AIAnalysisSkipReason.ANALYSIS_ALREADY_EXISTS,
            AIAnalysisSkipReason.ACTIVE_CLAIM,
            AIAnalysisSkipReason.DUPLICATE_MARKET_STATE,
        }:
            self.metrics["deduplicated_before_provider_call"] += 1
        gate_decision = (
            "REUSED"
            if reason == AIAnalysisSkipReason.ANALYSIS_ALREADY_EXISTS
            and existing_analysis is not None
            else "SKIPPED"
        )
        persisted = await self.record_gate_decision(
            state=state,
            attempted_cutoff=state.market_data_boundary,
            gate_decision=gate_decision,
            gate_skip_reason=reason.value,
            existing_analysis=existing_analysis,
            details={
                "idempotency_key": idempotency_key,
                "five_minute_window_start": (
                    window_start.isoformat() if window_start else None
                ),
                "claim_id": str(claim.claim_id) if claim is not None else None,
                "claim_status": claim.status if claim is not None else None,
                "claim_outcome": claim.outcome if claim is not None else None,
                "claimed_by": claim.claimed_by if claim is not None else None,
                "claimed_at": claim.claimed_at.isoformat() if claim is not None else None,
                "heartbeat_at": claim.heartbeat_at.isoformat() if claim is not None else None,
                "lease_expires_at": claim.lease_expires_at.isoformat() if claim is not None else None,
            },
        )
        logger.info(
            "ai_reasoning.gate.skipped",
            extra={
                **context,
                "skip_reason": reason.value,
                "reason_code": {
                    AIAnalysisSkipReason.NOT_FIVE_MINUTE_BOUNDARY: "interval_not_due",
                    AIAnalysisSkipReason.CYCLE_NOT_COMPLETE: "invalid_state",
                    AIAnalysisSkipReason.ANALYSIS_ALREADY_EXISTS: "analysis_exists",
                    AIAnalysisSkipReason.ACTIVE_CLAIM: "active_claim",
                    AIAnalysisSkipReason.DUPLICATE_MARKET_STATE: "duplicate_snapshot",
                    AIAnalysisSkipReason.MARKET_DATA_INCOMPLETE: "missing_prerequisite",
                    AIAnalysisSkipReason.MARKET_DATA_STALE: "stale_data",
                    AIAnalysisSkipReason.AI_DISABLED: "disabled",
                    AIAnalysisSkipReason.PROVIDER_UNAVAILABLE: "missing_prerequisite",
                    AIAnalysisSkipReason.REQUEST_PREFLIGHT_FAILED: "invalid_state",
                }[reason],
                "five_minute_window_start": (
                    window_start.isoformat() if window_start else None
                ),
                "idempotency_key": idempotency_key,
                "claim_id": str(claim.claim_id) if claim is not None else None,
                "claim_status": claim.status if claim is not None else None,
                "lease_expires_at": (
                    claim.lease_expires_at.isoformat() if claim is not None else None
                ),
                "provider_call_made": False,
                "snapshot_id": context.get("market_state_id"),
                "attempted_cutoff": persisted.attempted_cutoff.isoformat(),
                "analysis_lookup_cutoff": (
                    persisted.analysis_lookup_cutoff.isoformat()
                ),
                "gate_decision": persisted.gate_decision,
                "existing_analysis_id": (
                    str(persisted.existing_analysis_id)
                    if persisted.existing_analysis_id is not None
                    else None
                ),
                "analysis_created_at": (
                    persisted.analysis_created_at.isoformat()
                    if persisted.analysis_created_at is not None
                    else None
                ),
                "analysis_market_cutoff": (
                    persisted.analysis_market_cutoff.isoformat()
                    if persisted.analysis_market_cutoff is not None
                    else None
                ),
                "details": {
                    "skip_reason": reason.value,
                },
            },
        )

    def _provider_metrics(self) -> dict[str, int]:
        metrics = getattr(self.provider, "metrics", None)
        if not callable(metrics):
            return {}
        value = metrics()
        return {
            str(key): int(item)
            for key, item in value.items()
            if isinstance(item, int)
        }

    @staticmethod
    def _metric_delta(
        before: dict[str, int],
        after: dict[str, int],
    ) -> dict[str, int]:
        return {
            key: max(0, value - before.get(key, 0))
            for key, value in after.items()
        }

    def _consume_provider_metrics(self, delta: dict[str, int]) -> None:
        for key in (
            "provider_http_calls",
            "groq_calls",
            "retry_attempts",
            "schema_corrections",
            "analysis_requests",
            "schema_correction_requests",
            "http_429_responses",
            "initial_parse_failures",
            "initial_schema_validation_failures",
            "schema_corrections_succeeded",
            "schema_corrections_failed",
            "truncated_outputs",
            "compact_retries",
            "request_policy_failures",
            "provider_http_successes",
            "schema_valid_analyses",
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_total_tokens",
        ):
            self.metrics[key] += delta.get(key, 0)
        self.requests = self.metrics["provider_http_calls"]
        self.last_retry_count = delta.get("retry_attempts", 0)

    def health(self) -> dict[str, object]:
        metadata = self.provider.metadata()
        provider_states = metadata.get("providers")
        configured_count = metadata.get("configured_account_count")
        available_count = metadata.get("available_account_count")
        aggregate_reason = metadata.get("aggregate_reason")
        configured_account_count = (
            configured_count if isinstance(configured_count, int) else 0
        )
        available_account_count = (
            available_count if isinstance(available_count, int) else 0
        )
        now = self.clock().astimezone(UTC)
        recent_eligible_cycle = (
            self.last_eligible_cycle_at is not None
            and now - self.last_eligible_cycle_at <= timedelta(minutes=10)
        )
        analysis_requests = self.metrics["analysis_requests"]
        truncation_rate = (
            self.metrics["truncated_outputs"] / analysis_requests
            if analysis_requests
            else 0.0
        )
        consecutive_without_completion = max(
            0,
            self.metrics["eligible_five_minute_cycles"]
            - self.metrics["analyses_successfully_completed"],
        )
        completed_analyses = self.metrics["analyses_successfully_completed"]
        provider_calls_per_analysis = (
            self.metrics["provider_http_calls"] / completed_analyses
            if completed_analyses
            else None
        )
        schema_correction_rate = (
            self.metrics["schema_correction_requests"] / analysis_requests
            if analysis_requests
            else 0.0
        )
        tokens_per_analysis = (
            self.metrics["provider_total_tokens"] / completed_analyses
            if completed_analyses
            else None
        )
        efficiency_degraded = (
            provider_calls_per_analysis is not None
            and provider_calls_per_analysis
            > self.config.provider_calls_per_analysis_degraded_threshold
        ) or (
            schema_correction_rate
            > self.config.schema_correction_degraded_threshold
        ) or (
            tokens_per_analysis is not None
            and tokens_per_analysis
            > self.config.tokens_per_analysis_degraded_threshold
        )
        if not recent_eligible_cycle:
            operations_status = "idle"
        elif (
            truncation_rate >= self.config.truncation_unhealthy_threshold
            or consecutive_without_completion
            >= self.config.zero_completion_cycle_threshold
        ):
            operations_status = "unhealthy"
        elif self.last_cycle_outcome == "pool_success":
            operations_status = (
                "degraded"
                if (
                    truncation_rate >= self.config.truncation_degraded_threshold
                    or efficiency_degraded
                )
                else "healthy"
                if configured_account_count > 0
                and available_account_count == configured_account_count
                else "degraded"
            )
        elif aggregate_reason == "temporarily_rate_limited":
            operations_status = "temporarily_rate_limited"
        elif aggregate_reason == "quota_exhausted":
            operations_status = "quota_exhausted"
        elif available_account_count == 0 and isinstance(provider_states, dict) and all(
            not isinstance(item, dict)
            or item.get("status") in {"CONFIGURATION_ERROR", "DISABLED"}
            for item in provider_states.values()
        ):
            operations_status = "configuration_error"
        else:
            operations_status = "unhealthy"
        active_provider = metadata.get("active_provider")
        return {
            "enabled": self.enabled,
            "shadow_enabled": self.shadow_enabled,
            "proposals_enabled": self.proposals_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "publication_enabled": self.final_decision.publication_enabled if self.final_decision else False,
            "adjustments_enabled": self.final_decision.adjustments_enabled if self.final_decision else False,
            "provider_available": available_account_count > 0,
            "provider": metadata["provider"],
            "primary_provider": metadata.get("primary_provider"),
            "active_provider": active_provider,
            "latest_successful_provider": metadata.get("latest_successful_provider"),
            "latest_successful_analysis_at": metadata.get(
                "latest_successful_analysis_at"
            ),
            "latest_eligible_cycle_at": (
                self.last_eligible_cycle_at.isoformat()
                if self.last_eligible_cycle_at
                else None
            ),
            "configured_account_count": configured_account_count,
            "available_account_count": available_account_count,
            "pool_strategy": metadata.get("pool_strategy"),
            "model_identifier": metadata["model_identifier"],
            "prompt_version": self.config.prompt_version_new_market,
            "reasoning_policy_version": self.config.reasoning_policy_version,
            "latest_latency_ms": self.last_latency_ms,
            "latest_validation_passed": self.last_validation_passed,
            "latest_retry_count": self.last_retry_count,
            "failed_requests": self.failed_requests,
            "call_control": {
                "analysis_timeframe": AI_ANALYSIS_TIMEFRAME,
                "interval_minutes": AI_ANALYSIS_INTERVAL_MINUTES,
                "eligible_five_minute_cycles": self.metrics[
                    "eligible_five_minute_cycles"
                ],
                "analyses_successfully_completed": self.metrics[
                    "analyses_successfully_completed"
                ],
                "provider_http_calls": self.metrics["provider_http_calls"],
                "groq_calls": self.metrics["groq_calls"],
                "retries": self.metrics["retry_attempts"],
                "schema_corrections": self.metrics["schema_corrections"],
                "initial_analysis_requests": self.metrics["analysis_requests"],
                "initial_parse_failures": self.metrics["initial_parse_failures"],
                "initial_schema_validation_failures": self.metrics[
                    "initial_schema_validation_failures"
                ],
                "schema_corrections_attempted": self.metrics[
                    "schema_correction_requests"
                ],
                "schema_corrections_succeeded": self.metrics[
                    "schema_corrections_succeeded"
                ],
                "schema_corrections_failed": self.metrics[
                    "schema_corrections_failed"
                ],
                "http_429_responses": self.metrics["http_429_responses"],
                "provider_http_successes": self.metrics[
                    "provider_http_successes"
                ],
                "schema_valid_analyses": self.metrics[
                    "schema_valid_analyses"
                ],
                "truncated_outputs": self.metrics["truncated_outputs"],
                "compact_retries": self.metrics["compact_retries"],
                "request_policy_failures": self.metrics[
                    "request_policy_failures"
                ],
                "truncation_rate": round(truncation_rate, 4),
                "schema_correction_rate": round(schema_correction_rate, 4),
                "provider_calls_per_completed_analysis": (
                    round(provider_calls_per_analysis, 4)
                    if provider_calls_per_analysis is not None
                    else None
                ),
                "tokens_per_completed_analysis": (
                    round(tokens_per_analysis, 2)
                    if tokens_per_analysis is not None
                    else None
                ),
                "skipped_before_provider_call": self.metrics[
                    "skipped_before_provider_call"
                ],
                "deduplicated_before_provider_call": self.metrics[
                    "deduplicated_before_provider_call"
                ],
                "provider_failures": self.metrics["provider_failures"],
                "validation_failures": self.metrics["validation_failures"],
                "skip_reasons": dict(self.skip_reasons),
            },
            "failure_state": self.last_failure_state,
            "fallback_state": "ai_analysis_unavailable" if self.last_failure_state else None,
            "shadow_only": True,
            "awaiting_guardrail_validation": True,
            "providers": provider_states,
            "circuit_policy": metadata.get("circuit_policy"),
            "provider_readiness": operations_status,
            "operations_status": operations_status,
            "final_decision": self.final_decision.health() if self.final_decision else None,
        }
