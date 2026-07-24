"""Feature-gated, failure-isolated AI reasoning and signal monitoring orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import logging
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.core.exceptions import OpenRouterRequestError
from backend.app.final_decision.models import ExecutionContext, LLMUsageMetric, OperationMode
from backend.app.final_decision.service import FinalDecisionService
from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult

from .config import AIReasoningConfig
from .lifecycle import SignalLifecycleService
from .llm_context import build_llm_analysis_context
from .memory import MarketMemory
from .models import (
    AIMarketForecast,
    AIResultStatus,
    AISignalProposal,
    LLMStructuredOutputFailure,
    ManagedSignalState,
    MarketMemoryEntry,
)
from .provider import AIReasoningProvider
from .repository import AIReasoningRepository
from .request_builder import AIReasoningRequestBuilder
from .setup_families import SetupFamilyRegistry
from .validation import StructuredAIOutputError, StructuredAIOutputValidator, ValidatedAIOutput

logger = logging.getLogger(__name__)


class AIReasoningService:
    def __init__(
        self,
        repository: AIReasoningRepository,
        provider: AIReasoningProvider,
        builder: AIReasoningRequestBuilder,
        validator: StructuredAIOutputValidator,
        registry: SetupFamilyRegistry,
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
        self.registry = registry
        self.config = config
        self.shadow_enabled = shadow_enabled
        self.proposals_enabled = proposals_enabled
        self.monitoring_enabled = monitoring_enabled
        self.final_decision = final_decision
        self.clock = clock or (lambda: datetime.now(UTC))
        self._llm_semaphore = asyncio.Semaphore(config.llm_concurrency_limit)
        self._processed_state_hashes: set[str] = set()
        self._provider_failure_streak = 0
        self._provider_backoff_until: datetime | None = None
        self.memory = MarketMemory(config.maximum_memory_entries)
        metadata = provider.metadata()
        self.lifecycle = SignalLifecycleService(
            repository,
            policy_version=config.reasoning_policy_version,
            model_version=str(metadata["model_identifier"]),
            clock=self.clock,
        )
        self.requests = 0
        self.failed_requests = 0
        self.last_latency_ms: float | None = None
        self.last_validation_passed: bool | None = None
        self.last_retry_count = 0
        self.last_failure_state: str | None = None

    @property
    def enabled(self) -> bool:
        return self.shadow_enabled or self.proposals_enabled or self.monitoring_enabled

    async def process(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
    ) -> ValidatedAIOutput | None:
        worker_context = {
            "cycle_id": str(state.cycle_id),
            "market_state_id": str(state.state_id),
            "quantitative_forecast_id": str(quant.result_id),
            "instrument": state.instrument,
            "shadow_enabled": self.shadow_enabled,
            "proposals_enabled": self.proposals_enabled,
            "monitoring_enabled": self.monitoring_enabled,
        }
        logger.info("ai_reasoning.worker.received", extra=worker_context)
        if not self.enabled:
            logger.info(
                "ai_reasoning.gate.skipped",
                extra={**worker_context, "skip_reason": "all_ai_reasoning_feature_flags_disabled"},
            )
            return None
        # Revalidate immutable Phase 1/2 boundaries before any external call.
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        quant = QuantForecastResult.model_validate(quant.model_dump(mode="python"))
        if state.market_data_boundary > state.knowledge_cutoff:
            raise ValueError("AI reasoning requires a legally closed point-in-time state")
        if state.state_hash in self._processed_state_hashes:
            # One primary reasoning call per immutable closed-cycle state. Monitoring and
            # publication are driven by the persisted first result, never by per-tick polling.
            logger.info(
                "ai_reasoning.gate.skipped",
                extra={**worker_context, "skip_reason": "market_state_already_processed"},
            )
            return None
        if self._provider_backoff_until is not None and self.clock() < self._provider_backoff_until:
            logger.info(
                "ai_reasoning.gate.skipped",
                extra={
                    **worker_context,
                    "skip_reason": "provider_backoff_active",
                    "provider_backoff_until": self._provider_backoff_until.isoformat(),
                },
            )
            return None
        active_signals = await self.repository.active_signals(state.instrument)
        if self.monitoring_enabled and not self.proposals_enabled and not active_signals:
            # Monitoring is independently controllable and cannot create a brand-new opportunity
            # while proposal generation is disabled.
            if not self.shadow_enabled:
                logger.info(
                    "ai_reasoning.gate.skipped",
                    extra={**worker_context, "skip_reason": "monitoring_only_without_active_signal"},
                )
                return None
        existing_signal = active_signals[0] if active_signals else None
        previous_forecast = await self.repository.latest_forecast(state.instrument)
        previous_proposal = await self.repository.latest_proposal()
        recent = await self.repository.recent_memory(state.instrument, self.config.maximum_memory_entries)
        summary = self.memory.summarize(recent)
        request = self.builder.build(
            state,
            quant,
            summary,
            existing_signal=existing_signal,
            previous_forecast=previous_forecast,
            previous_proposal=previous_proposal,
        )
        for family in self.registry.all():
            await self.repository.save_setup_family(family, self.registry.version)
        try:
            await self.repository.save_request(request)
        except Exception as exc:
            logger.exception(
                "ai_reasoning.persist.failed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "failure_phase": "persistence",
                    "exception_class": type(exc).__name__,
                },
            )
            raise
        self.requests += 1
        validated: ValidatedAIOutput | None = None
        last_raw: dict[str, Any] | None = None
        failure_state = "unavailable"
        failure_errors: tuple[str, ...] = ()
        provider_failure: dict[str, Any] | None = None
        # One physical provider request is the hard five-minute-cycle boundary. A failed
        # attempt becomes a typed terminal result; it is never replayed inside the cycle.
        for attempt in range(1):
            try:
                async with self._llm_semaphore:
                    response = await asyncio.wait_for(
                        self.provider.reason(request, prompt_version=request.prompt_version),
                        timeout=self.config.request_timeout_seconds,
                    )
                last_raw = response.raw_output
                self.last_latency_ms = response.latency_ms
                await self._record_usage(request, state.state_hash, response.token_usage, response.latency_ms, True, None)
                candidate = self.validator.validate(response.raw_output, request=request, state=state, quant=quant)
                forecast = candidate.forecast.model_copy(
                    update={
                        "latency_ms": response.latency_ms,
                        "validation_passed": not candidate.degraded_validation,
                        "retry_count": attempt,
                        "token_usage": response.token_usage,
                        "failure_state": (
                            "degraded_structured_output"
                            if candidate.degraded_validation
                            else candidate.forecast.failure_state
                        ),
                        "failure_phase": (
                            "structured_output_validation"
                            if candidate.degraded_validation
                            else candidate.forecast.failure_phase
                        ),
                    }
                )
                validated = ValidatedAIOutput(
                    forecast=forecast,
                    proposal=candidate.proposal,
                    validation_issues=candidate.validation_issues,
                    repaired_fields=candidate.repaired_fields,
                )
                logger.info(
                    "structured_validation.completed",
                    extra={
                        **worker_context,
                        "request_id": str(request.request_id),
                        "validation_status": (
                            "degraded" if candidate.degraded_validation else "valid"
                        ),
                        "validation_issue_count": len(candidate.validation_issues),
                        "repaired_fields": candidate.repaired_fields,
                        "proposal_preserved": candidate.proposal is not None,
                    },
                )
                self.last_validation_passed = not candidate.degraded_validation
                self.last_retry_count = attempt
                self.last_failure_state = None
                self._provider_failure_streak = 0
                self._provider_backoff_until = None
                break
            except StructuredAIOutputError as exc:
                failure_state = "structured_output_invalid"
                failure_errors = exc.errors
                provider_failure = {
                    "reason_code": failure_state,
                    "phase": "structured_output_validation",
                    "request_id": str(request.request_id),
                    "cycle_id": str(request.cycle_id),
                    "model": request.model_identifier,
                    "exception_class": type(exc).__name__,
                }
                logger.error(
                    "ai_reasoning.request.failed",
                    extra={
                        **provider_failure,
                        "failure_phase": "structured_output_validation",
                        "failed_during_http_request": False,
                        "failed_during_response_decoding": False,
                        "failed_during_structured_output_validation": True,
                        "failed_during_domain_parsing": False,
                        "failed_during_persistence": False,
                        "field_path": exc.first_issue.field_path if exc.first_issue else None,
                        "expected_type": exc.first_issue.expected_type if exc.first_issue else None,
                        "actual_value": exc.first_issue.actual_value if exc.first_issue else None,
                        "validator_name": exc.first_issue.validator_name if exc.first_issue else None,
                        "offending_json_fragment": (
                            exc.first_issue.offending_json_fragment if exc.first_issue else None
                        ),
                    },
                )
            except OpenRouterRequestError as exc:
                provider_failure = asdict(exc.details)
                failure_state = exc.details.reason_code
                failure_errors = tuple(
                    value
                    for value in (
                        exc.details.reason_code,
                        exc.details.error_code,
                        exc.details.error_message,
                        exc.details.metadata_error_type,
                        exc.details.metadata_provider_code,
                    )
                    if value
                )
                self.last_latency_ms = exc.details.elapsed_ms
                provider_backoff_failures = {
                    "authentication_failed",
                    "payment_blocked",
                    "key_limit_exhausted",
                    "rate_limited",
                    "provider_unavailable",
                    # Backward-compatible codes from already-deployed adapters.
                    "openrouter_authentication_failed",
                    "openrouter_insufficient_credits",
                    "openrouter_rate_limited",
                    "openrouter_provider_unavailable",
                    "openrouter_transport_error",
                }
                if failure_state in provider_backoff_failures:
                    self._provider_failure_streak += 1
                    backoff = min(
                        self.config.provider_backoff_initial_seconds
                        * (2 ** (self._provider_failure_streak - 1)),
                        self.config.provider_backoff_max_seconds,
                    )
                    self._provider_backoff_until = self.clock() + timedelta(seconds=backoff)
                else:
                    # Local payload/context/cost validation is deterministic and must not be
                    # mislabeled as provider payment backoff.
                    self._provider_failure_streak = 0
                    self._provider_backoff_until = None
                await self._record_usage(request, state.state_hash, None, self.last_latency_ms, False, failure_state)
            except Exception as exc:
                failure_state = "llm_unavailable"
                failure_errors = (type(exc).__name__, str(exc)[:200])
                provider_failure = {
                    "reason_code": failure_state,
                    "phase": "domain_parsing",
                    "request_id": str(request.request_id),
                    "cycle_id": str(request.cycle_id),
                    "model": request.model_identifier,
                    "exception_class": type(exc).__name__,
                }
                self._provider_failure_streak += 1
                backoff = min(
                    self.config.provider_backoff_initial_seconds * (2 ** (self._provider_failure_streak - 1)),
                    self.config.provider_backoff_max_seconds,
                )
                self._provider_backoff_until = self.clock() + timedelta(seconds=backoff)
                await self._record_usage(request, state.state_hash, None, self.last_latency_ms, False, failure_state)
            await self._record_failure(
                request.request_id,
                attempt,
                request.model_identifier,
                request.prompt_version,
                last_raw,
                failure_errors,
                failure_state,
                provider_failure,
            )

        if validated is None:
            self.failed_requests += 1
            self.last_validation_passed = False
            self.last_retry_count = 0
            self.last_failure_state = failure_state
            unavailable = self._unavailable_forecast(
                request,
                failure_state,
                failure_errors,
                provider_failure,
            )
            try:
                await self.repository.save_forecast(unavailable)
            except Exception as exc:
                logger.exception(
                    "ai_reasoning.persist.failed",
                    extra={
                        **worker_context,
                        "request_id": str(request.request_id),
                        "failure_phase": "persistence",
                        "exception_class": type(exc).__name__,
                    },
                )
                raise
            self._processed_state_hashes.add(state.state_hash)
            logger.info(
                "ai_reasoning.persist.completed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "forecast_id": str(unavailable.forecast_id),
                    "status": unavailable.status.value,
                    "failure_reason_code": unavailable.failure_state,
                },
            )
            logger.info(
                "ai_reasoning.completed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "forecast_id": str(unavailable.forecast_id),
                    "status": unavailable.status.value,
                    "failure_reason_code": unavailable.failure_state,
                },
            )
            return ValidatedAIOutput(forecast=unavailable, proposal=None)

        try:
            await self.repository.save_forecast(validated.forecast)
        except Exception as exc:
            logger.exception(
                "ai_reasoning.persist.failed",
                extra={
                    **worker_context,
                    "request_id": str(request.request_id),
                    "failure_phase": "persistence",
                    "exception_class": type(exc).__name__,
                },
            )
            raise
        proposal = validated.proposal
        signal = existing_signal
        if proposal is not None:
            if self.proposals_enabled or existing_signal is not None:
                await self.repository.save_proposal(proposal)
                logger.info(
                    "proposal.generated",
                    extra={
                        **worker_context,
                        "request_id": str(request.request_id),
                        "proposal_id": str(proposal.proposal_id),
                        "recommended_action": proposal.recommended_action.value,
                        "proposal_confidence": proposal.proposal_confidence,
                        "validation_status": (
                            "degraded" if validated.degraded_validation else "valid"
                        ),
                    },
                )
            if self.proposals_enabled or (
                existing_signal is not None
                and proposal.structural_opportunity_key == existing_signal.structural_opportunity_key
            ):
                signal = await self.lifecycle.apply_proposal(
                    validated.forecast,
                    proposal,
                    setup_family=validated.forecast.selected_setup_family or "non_actionable",
                )
        if (
            self.final_decision is not None
            and proposal is not None
            and signal is not None
            and validated.forecast.status == AIResultStatus.AVAILABLE
        ):
            context = self._execution_context(state, quant, request, signal.signal_id, proposal.structural_opportunity_key)
            final_result = await self.final_decision.evaluate(
                state,
                quant,
                validated.forecast,
                proposal,
                signal,
                context,
            )
            if final_result.publication is not None and signal.state == ManagedSignalState.PROPOSED:
                signal = await self.lifecycle.apply_guardrail_approved_transition(
                    signal,
                    ManagedSignalState.CONFIRMED,
                    approval_rule=self.final_decision.config.hard_gate_registry_version,
                    forecast=validated.forecast,
                    proposal=proposal,
                )
        if self.monitoring_enabled:
            for active in active_signals:
                await self.lifecycle.monitor(
                    active,
                    validated.forecast,
                    proposal if proposal and proposal.structural_opportunity_key == active.structural_opportunity_key else None,
                    previous_probability=previous_forecast.dominant_scenario_probability if previous_forecast else None,
                )
        await self._append_memory(state, quant, validated.forecast, proposal, signal)
        self._processed_state_hashes.add(state.state_hash)
        logger.info(
            "ai_reasoning.persist.completed",
            extra={
                **worker_context,
                "request_id": str(request.request_id),
                "forecast_id": str(validated.forecast.forecast_id),
                "status": validated.forecast.status.value,
                "failure_reason_code": None,
            },
        )
        logger.info(
            "ai_reasoning.completed",
            extra={
                **worker_context,
                "request_id": str(request.request_id),
                "forecast_id": str(validated.forecast.forecast_id),
                "status": validated.forecast.status.value,
            },
        )
        return validated

    async def _record_usage(
        self,
        request: Any,
        state_hash: str,
        token_usage: dict[str, int] | None,
        latency_ms: float | None,
        success: bool,
        failure_state: str | None,
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
            model_identifier=request.model_identifier,
            prompt_version=request.prompt_version,
            generation_parameters={"temperature": self.config.temperature, "max_tokens": self.config.max_tokens},
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

    def _execution_context(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        request: Any,
        signal_id: Any,
        opportunity_key: str,
    ) -> ExecutionContext:
        market_open = self._find_boolean(state, ("market_open", "is_market_open"))
        economic_blackout = self._find_boolean(state, ("prohibited_window", "blackout", "event_blackout"))
        session = self._find_string(state, ("session", "session_name")) or "unknown"
        current_price = quant.predictions[0].reference_price if quant.predictions else None
        return ExecutionContext(
            context_id=uuid5(NAMESPACE_URL, f"ten:execution-context:{state.state_id}"),
            instrument=state.instrument,
            evaluated_at=max(self.clock(), state.market_data_boundary),
            operation_mode=OperationMode.ANALYTICAL_LIVE if self.final_decision and self.final_decision.publication_enabled else OperationMode.SHADOW,
            analytical_only=True,
            broker_execution_available=False,
            market_open=market_open,
            current_price=current_price,
            spread=request.spread,
            session=session,
            publication_service_available=True,
            persistence_available=True,
            economic_context_available=bool(request.economic_event_context),
            prohibited_economic_event_window=economic_blackout,
            active_opportunity_keys=(opportunity_key,),
            active_signal_id=signal_id,
        )

    @staticmethod
    def _walk(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            found.append(value)
            for nested in value.values():
                found.extend(AIReasoningService._walk(nested))
        elif isinstance(value, (list, tuple)):
            for nested in value:
                found.extend(AIReasoningService._walk(nested))
        return found

    @classmethod
    def _find_boolean(cls, state: UnifiedMarketState, keys: tuple[str, ...]) -> bool | None:
        for evidence in state.evidence:
            for payload in cls._walk(evidence.raw_value):
                for key in keys:
                    if isinstance(payload.get(key), bool):
                        value = payload[key]
                        assert isinstance(value, bool)
                        return value
        return None

    @classmethod
    def _find_string(cls, state: UnifiedMarketState, keys: tuple[str, ...]) -> str | None:
        for evidence in state.evidence:
            for payload in cls._walk(evidence.raw_value):
                for key in keys:
                    if isinstance(payload.get(key), str):
                        value = payload[key]
                        assert isinstance(value, str)
                        return value
        return None

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

    def _unavailable_forecast(
        self,
        request: Any,
        failure_state: str,
        errors: tuple[str, ...],
        provider_failure: dict[str, Any] | None,
    ) -> AIMarketForecast:
        return AIMarketForecast(
            forecast_id=uuid5(NAMESPACE_URL, f"ten:ai-forecast:{request.request_id}:{failure_state}"),
            request_id=request.request_id,
            market_state_id=request.market_state_id,
            quantitative_forecast_id=request.quantitative_forecast_id,
            cycle_id=request.cycle_id,
            status=AIResultStatus.INVALID if failure_state == "structured_output_invalid" else AIResultStatus.UNAVAILABLE,
            model_provider=str(self.provider.metadata()["provider"]),
            model_identifier=request.model_identifier,
            prompt_version=request.prompt_version,
            reasoning_policy_version=request.reasoning_policy_version,
            setup_family_registry_version=request.setup_family_registry_version,
            quantitative_model_version=request.quantitative_model_version,
            feature_schema_version=request.feature_schema_version,
            market_state_schema_version=request.market_state_schema_version,
            validation_passed=False,
            retry_count=0,
            failure_state=failure_state,
            failure_phase=str(provider_failure.get("phase")) if provider_failure and provider_failure.get("phase") else None,
            provider_http_status=(
                int(provider_failure["http_status"])
                if provider_failure and provider_failure.get("http_status") is not None
                else None
            ),
            provider_error_code=str(provider_failure.get("error_code")) if provider_failure and provider_failure.get("error_code") else None,
            provider_error_message=str(provider_failure.get("error_message")) if provider_failure and provider_failure.get("error_message") else None,
            provider_metadata_error_type=(
                str(provider_failure.get("metadata_error_type"))
                if provider_failure and provider_failure.get("metadata_error_type")
                else None
            ),
            provider_metadata_provider_code=(
                str(provider_failure.get("metadata_provider_code"))
                if provider_failure and provider_failure.get("metadata_provider_code")
                else None
            ),
            fallback_state="no_ai_proposal",
            reasoning_summary="AI result unavailable; no proposal was created.",
            missing_evidence=errors,
            generated_at=max(self.clock(), request.analysis_timestamp),
        )

    async def _append_memory(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        forecast: AIMarketForecast,
        proposal: AISignalProposal | None,
        signal: Any,
    ) -> None:
        await self.repository.append_memory(
            MarketMemoryEntry(
                entry_id=uuid5(NAMESPACE_URL, f"ten:memory:quant-forecast:{quant.result_id}"),
                instrument=state.instrument,
                cycle_id=state.cycle_id,
                market_state_id=state.state_id,
                category="quant_forecast",
                summary=f"Quantitative forecast {quant.status.value} from {quant.model_name} {quant.model_version}",
                structured_payload={
                    "forecast_id": str(quant.result_id),
                    "model_version": quant.model_version,
                    "horizons": [item.horizon.horizon_id for item in quant.predictions],
                },
                occurred_at=quant.generated_at,
            )
        )
        entry = MarketMemoryEntry(
            entry_id=uuid5(NAMESPACE_URL, f"ten:memory:ai-forecast:{forecast.forecast_id}"),
            instrument=state.instrument,
            cycle_id=state.cycle_id,
            market_state_id=state.state_id,
            category="ai_forecast",
            summary=f"Scenario {forecast.dominant_scenario or 'unavailable'}; confidence {forecast.forecast_confidence}",
            structured_payload={
                "direction": forecast.dominant_direction.value if forecast.dominant_direction else None,
                "setup_family": forecast.selected_setup_family,
                "proposal_action": proposal.recommended_action.value if proposal else None,
                "signal_state": signal.state.value if signal else None,
                "levels": {
                    "entry": proposal.entry_zone.model_dump(mode="json") if proposal and proposal.entry_zone else None,
                    "stop_loss": proposal.stop_loss if proposal else None,
                    "take_profit_levels": proposal.take_profit_levels if proposal else (),
                    "invalidation": proposal.invalidation_price if proposal else None,
                },
            },
            evidence_ids=forecast.supporting_evidence_ids + forecast.contradicting_evidence_ids,
            opportunity_key=proposal.structural_opportunity_key if proposal else None,
            signal_id=signal.signal_id if signal else None,
            occurred_at=forecast.generated_at,
        )
        await self.repository.append_memory(entry)

    def health(self) -> dict[str, object]:
        metadata = self.provider.metadata()
        return {
            "enabled": self.enabled,
            "shadow_enabled": self.shadow_enabled,
            "proposals_enabled": self.proposals_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "publication_enabled": self.final_decision.publication_enabled if self.final_decision else False,
            "adjustments_enabled": self.final_decision.adjustments_enabled if self.final_decision else False,
            "provider_available": self.last_failure_state != "llm_unavailable" if self.requests else None,
            "provider": metadata["provider"],
            "model_identifier": metadata["model_identifier"],
            "prompt_version": self.config.prompt_version_new_market,
            "reasoning_policy_version": self.config.reasoning_policy_version,
            "latest_latency_ms": self.last_latency_ms,
            "latest_validation_passed": self.last_validation_passed,
            "latest_retry_count": self.last_retry_count,
            "failed_requests": self.failed_requests,
            "failure_state": self.last_failure_state,
            "fallback_state": "no_ai_proposal" if self.last_failure_state else None,
            "shadow_only": True,
            "awaiting_guardrail_validation": True,
            "provider_backoff_until": self._provider_backoff_until.isoformat() if self._provider_backoff_until else None,
            "deduplicated_market_states": len(self._processed_state_hashes),
            "final_decision": self.final_decision.health() if self.final_decision else None,
        }
