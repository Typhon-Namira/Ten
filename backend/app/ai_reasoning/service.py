"""Feature-gated, failure-isolated analysis-only AI orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
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


def reasoning_cycle_idempotency_key(
    instrument: str,
    ums_boundary: datetime,
    cycle_version: str,
    provider_contract_version: str,
) -> str:
    normalized_instrument = "".join(
        character
        for character in instrument.strip().upper()
        if character.isalnum()
    )
    material = "|".join(
        (
            normalized_instrument,
            ums_boundary.astimezone(UTC).isoformat(),
            cycle_version,
            provider_contract_version,
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
        logger.info("ai_reasoning.worker.received", extra=worker_context)
        if not self.enabled:
            logger.info(
                "ai_reasoning.gate.skipped",
                extra={**worker_context, "skip_reason": "ai_analysis_disabled"},
            )
            return None
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        quant = QuantForecastResult.model_validate(quant.model_dump(mode="python"))
        boundary = state.market_data_boundary.astimezone(UTC)
        contract_version = ":".join(
            (
                LLM_ANALYSIS_CONTEXT_SCHEMA_VERSION,
                "ai-market-analysis-2.0",
                self.config.reasoning_policy_version,
            )
        )
        idempotency_key = reasoning_cycle_idempotency_key(
            state.instrument,
            boundary,
            state.schema_version,
            contract_version,
        )
        cached = await self.repository.analysis_for_reasoning_cycle(idempotency_key)
        if cached is not None:
            logger.info(
                "ai_analysis.result.reused",
                extra={
                    **worker_context,
                    "analysis_id": str(cached.analysis_id),
                    "idempotency_key": idempotency_key,
                    "provider_call_made": False,
                },
            )
            return await self._validated_analysis(cached)
        claimed = await self.repository.claim_reasoning_cycle(
            idempotency_key,
            state.instrument,
            boundary,
            state.schema_version,
            contract_version,
            self.clock(),
        )
        if not claimed:
            cached = await self.repository.analysis_for_reasoning_cycle(idempotency_key)
            return await self._validated_analysis(cached) if cached is not None else None

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
        await self.repository.save_request(request)
        self.requests += 1
        response = None
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
            return None

        self.last_validation_passed = True
        self.last_failure_state = None
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
            },
            request_count=1,
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
