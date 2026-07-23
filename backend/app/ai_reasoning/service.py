"""Feature-gated, failure-isolated AI reasoning and signal monitoring orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult

from .config import AIReasoningConfig
from .lifecycle import SignalLifecycleService
from .memory import MarketMemory
from .models import (
    AIMarketForecast,
    AIResultStatus,
    AISignalProposal,
    LLMStructuredOutputFailure,
    MarketMemoryEntry,
)
from .provider import AIReasoningProvider
from .repository import AIReasoningRepository
from .request_builder import AIReasoningRequestBuilder
from .setup_families import SetupFamilyRegistry
from .validation import StructuredAIOutputError, StructuredAIOutputValidator, ValidatedAIOutput


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
        proposals_enabled: bool,
        monitoring_enabled: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.builder = builder
        self.validator = validator
        self.registry = registry
        self.config = config
        self.proposals_enabled = proposals_enabled
        self.monitoring_enabled = monitoring_enabled
        self.clock = clock or (lambda: datetime.now(UTC))
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
        return self.proposals_enabled or self.monitoring_enabled

    async def process(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
    ) -> ValidatedAIOutput | None:
        if not self.enabled:
            return None
        # Revalidate immutable Phase 1/2 boundaries before any external call.
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        quant = QuantForecastResult.model_validate(quant.model_dump(mode="python"))
        if state.market_data_boundary > state.knowledge_cutoff:
            raise ValueError("AI reasoning requires a legally closed point-in-time state")
        active_signals = await self.repository.active_signals(state.instrument)
        if self.monitoring_enabled and not self.proposals_enabled and not active_signals:
            # Monitoring is independently controllable and cannot create a brand-new opportunity
            # while proposal generation is disabled.
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
        await self.repository.save_request(request)
        self.requests += 1
        validated: ValidatedAIOutput | None = None
        last_raw: dict[str, Any] | None = None
        failure_state = "unavailable"
        failure_errors: tuple[str, ...] = ()
        for attempt in range(self.config.maximum_retries + 1):
            try:
                response = await self.provider.reason(request, prompt_version=request.prompt_version)
                last_raw = response.raw_output
                self.last_latency_ms = response.latency_ms
                candidate = self.validator.validate(response.raw_output, request=request, state=state, quant=quant)
                forecast = candidate.forecast.model_copy(
                    update={
                        "latency_ms": response.latency_ms,
                        "validation_passed": True,
                        "retry_count": attempt,
                        "token_usage": response.token_usage,
                    }
                )
                validated = ValidatedAIOutput(forecast=forecast, proposal=candidate.proposal)
                self.last_validation_passed = True
                self.last_retry_count = attempt
                self.last_failure_state = None
                break
            except StructuredAIOutputError as exc:
                failure_state = "structured_output_invalid"
                failure_errors = exc.errors
            except Exception as exc:
                failure_state = "llm_unavailable"
                failure_errors = (type(exc).__name__,)
            await self._record_failure(request.request_id, attempt, request.model_identifier, request.prompt_version, last_raw, failure_errors, failure_state)

        if validated is None:
            self.failed_requests += 1
            self.last_validation_passed = False
            self.last_retry_count = self.config.maximum_retries
            self.last_failure_state = failure_state
            unavailable = self._unavailable_forecast(request, failure_state, failure_errors)
            await self.repository.save_forecast(unavailable)
            return ValidatedAIOutput(forecast=unavailable, proposal=None)

        await self.repository.save_forecast(validated.forecast)
        proposal = validated.proposal
        signal = existing_signal
        if proposal is not None:
            if self.proposals_enabled or existing_signal is not None:
                await self.repository.save_proposal(proposal)
            if self.proposals_enabled or (
                existing_signal is not None
                and proposal.structural_opportunity_key == existing_signal.structural_opportunity_key
            ):
                signal = await self.lifecycle.apply_proposal(
                    validated.forecast,
                    proposal,
                    setup_family=validated.forecast.selected_setup_family or "non_actionable",
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
        return validated

    async def _record_failure(
        self,
        request_id: object,
        attempt: int,
        model: str,
        prompt: str,
        raw: dict[str, Any] | None,
        errors: tuple[str, ...],
        state: str,
    ) -> None:
        failure = LLMStructuredOutputFailure(
            failure_id=uuid5(NAMESPACE_URL, f"ten:llm-failure:{request_id}:{attempt}:{state}"),
            request_id=request_id,
            attempt=attempt,
            model_identifier=model,
            prompt_version=prompt,
            raw_output=raw,
            validation_errors=errors,
            failure_state=state,
            latency_ms=self.last_latency_ms,
            created_at=self.clock(),
        )
        await self.repository.save_failure(failure)

    def _unavailable_forecast(self, request: Any, failure_state: str, errors: tuple[str, ...]) -> AIMarketForecast:
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
            retry_count=self.config.maximum_retries,
            failure_state=failure_state,
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
            "proposals_enabled": self.proposals_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "publication_enabled": False,
            "adjustments_enabled": False,
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
        }
