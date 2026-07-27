"""Feature-gated, failure-isolated analysis-only AI orchestration."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
import logging
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
    five_minute_window_start,
)
from .config import AIReasoningConfig
from .llm_context import LLM_ANALYSIS_CONTEXT_SCHEMA_VERSION, build_llm_analysis_context
from .memory import MarketMemory
from .models import LLMStructuredOutputFailure
from .provider import AIReasoningProvider
from .repository import AIReasoningRepository
from .request_builder import AIReasoningRequestBuilder
from .validation import StructuredAIOutputError, StructuredAIOutputValidator

logger = logging.getLogger(__name__)

_PROVIDER_REACHABLE_FAILURE_STATES = frozenset({"structured_output_invalid"})


class AIAnalysisSkipReason(StrEnum):
    NOT_FIVE_MINUTE_BOUNDARY = "not_five_minute_boundary"
    CYCLE_NOT_COMPLETE = "cycle_not_complete"
    ANALYSIS_ALREADY_EXISTS = "analysis_already_exists"
    CYCLE_ALREADY_CLAIMED = "cycle_already_claimed"
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
        self.memory = MarketMemory(config.maximum_memory_entries)
        self.requests = 0
        self.failed_requests = 0
        self.last_latency_ms: float | None = None
        self.last_validation_passed: bool | None = None
        self.last_retry_count = 0
        self.last_failure_state: str | None = None
        self.metrics: Counter[str] = Counter()
        self.skip_reasons: Counter[str] = Counter()

    @property
    def enabled(self) -> bool:
        return self.shadow_enabled

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
        }
        self.metrics["scheduler_ticks"] += 1
        logger.info("ai_reasoning.worker.received", extra=worker_context)
        if not self.enabled:
            self._skip(AIAnalysisSkipReason.AI_DISABLED, worker_context)
            return None
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        quant = QuantForecastResult.model_validate(quant.model_dump(mode="python"))
        boundary = state.market_data_boundary.astimezone(UTC)
        window_start = five_minute_window_start(boundary)
        eligibility_failure = self._eligibility_failure(state, boundary, window_start)
        if eligibility_failure is not None:
            self._skip(eligibility_failure, worker_context, window_start=window_start)
            return None
        self.metrics["eligible_five_minute_cycles"] += 1
        contract_version = ":".join(
            (
                LLM_ANALYSIS_CONTEXT_SCHEMA_VERSION,
                "ai-market-analysis-2.0",
                self.config.reasoning_policy_version,
            )
        )
        idempotency_key = reasoning_cycle_idempotency_key(
            state.instrument,
            AI_ANALYSIS_TIMEFRAME,
            window_start,
            state.state_hash,
            contract_version,
        )
        claimed = await self.repository.claim_reasoning_cycle(
            idempotency_key,
            state.instrument,
            window_start,
            state.schema_version,
            contract_version,
            self.clock(),
            analysis_timeframe=AI_ANALYSIS_TIMEFRAME,
            five_minute_window_start=window_start,
            market_state_hash=state.state_hash,
            analysis_contract_version=contract_version,
        )
        if not claimed:
            cached = await self.repository.analysis_for_reasoning_cycle(idempotency_key)
            if cached is not None:
                self.metrics["analyses_reused"] += 1
                self._skip(
                    AIAnalysisSkipReason.ANALYSIS_ALREADY_EXISTS,
                    worker_context,
                    window_start=window_start,
                    idempotency_key=idempotency_key,
                )
                return await self._validated_analysis(cached)
            duplicate = await self.repository.analysis_for_market_state_hash(
                state.instrument,
                state.state_hash,
                contract_version,
            )
            reason = (
                AIAnalysisSkipReason.DUPLICATE_MARKET_STATE
                if duplicate is not None
                else AIAnalysisSkipReason.CYCLE_ALREADY_CLAIMED
            )
            self._skip(
                reason,
                worker_context,
                window_start=window_start,
                idempotency_key=idempotency_key,
            )
            return await self._validated_analysis(duplicate) if duplicate is not None else None

        if not self._primary_provider_eligible():
            self._skip(
                AIAnalysisSkipReason.PROVIDER_UNAVAILABLE,
                worker_context,
                window_start=window_start,
                idempotency_key=idempotency_key,
            )
            await self.repository.complete_analysis_cycle(
                idempotency_key,
                None,
                None,
                "skipped",
                self.clock(),
            )
            return None

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
        response = None
        provider_metrics_before = self._provider_metrics()
        provider_delta: dict[str, int] = {}
        failure_state = "llm_unavailable"
        failure_errors: tuple[str, ...] = ()
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
        except AIProviderRequestError as exc:
            provider_delta = self._metric_delta(
                provider_metrics_before,
                self._provider_metrics(),
            )
            self._consume_provider_metrics(provider_delta)
            if exc.details.phase == "request_validation":
                self._skip(
                    AIAnalysisSkipReason.REQUEST_PREFLIGHT_FAILED,
                    worker_context,
                    window_start=window_start,
                    idempotency_key=idempotency_key,
                )
                await self.repository.complete_analysis_cycle(
                    idempotency_key,
                    request.request_id,
                    None,
                    "skipped",
                    self.clock(),
                )
                return None
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
        except Exception as exc:
            provider_delta = self._metric_delta(
                provider_metrics_before,
                self._provider_metrics(),
            )
            self._consume_provider_metrics(provider_delta)
            failure_errors = (type(exc).__name__, str(exc)[:200])
            analysis = self._failed_analysis(request, failure_state, failure_errors)
            logger.exception(
                "ai_reasoning.request.failed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "failure_phase": "provider_or_domain",
                },
            )

        analysis = await self.repository.save_analysis(analysis)
        completed_status = (
            "completed"
            if analysis.status == AnalysisStatus.AVAILABLE
            else "failed"
        )
        await self.repository.complete_analysis_cycle(
            idempotency_key,
            request.request_id,
            analysis.analysis_id,
            completed_status,
            self.clock(),
        )
        if analysis.status != AnalysisStatus.AVAILABLE:
            self.failed_requests += 1
            self.last_validation_passed = False
            self.last_failure_state = failure_state
            await self._record_failure(
                request.request_id,
                0,
                response.model_identifier if response is not None else request.model_identifier,
                request.prompt_version,
                response.raw_output if response is not None else None,
                failure_errors,
                failure_state,
                None,
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
            )
            return None

        self.last_validation_passed = True
        self.last_failure_state = None
        self.metrics["analyses_successfully_completed"] += 1
        logger.info(
            "ai_reasoning.persist.completed",
            extra={
                **worker_context,
                "request_id": str(request.request_id),
                "analysis_id": str(analysis.analysis_id),
                "status": analysis.status.value,
            },
        )
        return await self._validated_analysis(analysis)

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
    ) -> None:
        if self.final_decision is None:
            return
        payload = build_llm_analysis_context(request).model_dump(mode="json")
        request_hash = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        total = token_usage.get("total_tokens") if token_usage else None
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
                **(provider_metrics or {}),
                "provider_failure": int(
                    failure_state is not None
                    and failure_state != "structured_output_invalid"
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
        if (
            state.trigger_timeframe.upper() != AI_ANALYSIS_TIMEFRAME
            or boundary != window_start
        ):
            return AIAnalysisSkipReason.NOT_FIVE_MINUTE_BOUNDARY
        if self.clock().astimezone(UTC) < boundary:
            return AIAnalysisSkipReason.CYCLE_NOT_COMPLETE
        frames = {item.timeframe.upper(): item for item in state.timeframes}
        if set(frames) != {"M1", "M5", "M15"}:
            return AIAnalysisSkipReason.MARKET_DATA_INCOMPLETE
        if any(item.stale for item in frames.values()):
            return AIAnalysisSkipReason.MARKET_DATA_STALE
        return None

    def _primary_provider_eligible(self) -> bool:
        providers = self.provider.metadata().get("providers")
        if not isinstance(providers, dict):
            return True
        primary = providers.get("cerebras")
        return (
            isinstance(primary, dict)
            and primary.get("status") in {"HEALTHY", "STANDBY"}
        )

    def _skip(
        self,
        reason: AIAnalysisSkipReason,
        context: Mapping[str, object],
        *,
        window_start: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        self.skip_reasons[reason.value] += 1
        self.metrics["skipped_before_provider_call"] += 1
        if reason in {
            AIAnalysisSkipReason.ANALYSIS_ALREADY_EXISTS,
            AIAnalysisSkipReason.CYCLE_ALREADY_CLAIMED,
            AIAnalysisSkipReason.DUPLICATE_MARKET_STATE,
        }:
            self.metrics["deduplicated_before_provider_call"] += 1
        logger.info(
            "ai_reasoning.gate.skipped",
            extra={
                **context,
                "skip_reason": reason.value,
                "five_minute_window_start": (
                    window_start.isoformat() if window_start else None
                ),
                "idempotency_key": idempotency_key,
                "provider_call_made": False,
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
            "cerebras_calls",
            "groq_fallback_calls",
            "retry_attempts",
            "schema_corrections",
        ):
            self.metrics[key] += delta.get(key, 0)
        self.requests = self.metrics["provider_http_calls"]
        self.last_retry_count = delta.get("retry_attempts", 0)

    def health(self) -> dict[str, object]:
        metadata = self.provider.metadata()
        provider_states = metadata.get("providers")
        ready_provider_count = (
            sum(
                1
                for item in provider_states.values()
                if isinstance(item, dict)
                and item.get("status") in {"HEALTHY", "STANDBY"}
            )
            if isinstance(provider_states, dict)
            else 0
        )
        return {
            "enabled": self.enabled,
            "shadow_enabled": self.shadow_enabled,
            "proposals_enabled": self.proposals_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "publication_enabled": self.final_decision.publication_enabled if self.final_decision else False,
            "adjustments_enabled": self.final_decision.adjustments_enabled if self.final_decision else False,
            "provider_available": (
                self.last_failure_state is None
                or self.last_failure_state in _PROVIDER_REACHABLE_FAILURE_STATES
            )
            if self.requests
            else None,
            "provider": metadata["provider"],
            "primary_provider": metadata.get("primary_provider"),
            "active_provider": metadata.get("active_provider"),
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
                "cerebras_calls": self.metrics["cerebras_calls"],
                "groq_fallback_calls": self.metrics["groq_fallback_calls"],
                "retries": self.metrics["retry_attempts"],
                "schema_corrections": self.metrics["schema_corrections"],
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
            "fallback_status": (
                "ACTIVE"
                if metadata.get("active_provider") == "groq"
                else "STANDBY"
            ),
            "shadow_only": True,
            "awaiting_guardrail_validation": True,
            "providers": provider_states,
            "provider_readiness": (
                "failed"
                if ready_provider_count == 0
                else "healthy"
                if ready_provider_count == 2
                else "degraded"
            ),
            "final_decision": self.final_decision.health() if self.final_decision else None,
        }
