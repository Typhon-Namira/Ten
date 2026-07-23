"""Existing-OpenRouter-only provider boundary for AI market reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from backend.app.ai.openrouter_client.client import OpenRouterClient
from backend.app.ai.prompts.loader import PromptLoader

from .models import AIReasoningRequest


@dataclass(frozen=True)
class AIProviderResponse:
    raw_output: dict[str, Any]
    provider: str
    model_identifier: str
    latency_ms: float
    token_usage: dict[str, int] | None


class AIReasoningProvider(Protocol):
    async def reason(self, request: AIReasoningRequest, *, prompt_version: str) -> AIProviderResponse: ...
    def metadata(self) -> dict[str, object]: ...


class ExistingOpenRouterReasoningProvider:
    """Uses the exact OpenRouter client and model already configured by TEN."""

    provider_name = "openrouter"

    def __init__(
        self,
        client: OpenRouterClient,
        prompts: PromptLoader,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def reason(self, request: AIReasoningRequest, *, prompt_version: str) -> AIProviderResponse:
        started = perf_counter()
        raw = await self.client.complete_json(
            system_prompt=self.prompts.load(prompt_version),
            payload={
                "analysis_request": request.model_dump(mode="json"),
                "response_contract": self._response_contract(request),
            },
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_id=str(request.request_id),
            cycle_id=str(request.cycle_id),
        )
        return AIProviderResponse(
            raw_output=raw,
            provider=self.provider_name,
            model_identifier=self.model,
            latency_ms=(perf_counter() - started) * 1000,
            # The existing narrow client does not expose OpenRouter usage metadata. Unknown stays
            # unknown; the reasoning layer never invents token counts.
            token_usage=None,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider_name,
            "model_identifier": self.model,
            "external_ai_apis": ("openrouter",),
            "token_usage_available": False,
        }

    @staticmethod
    def _response_contract(request: AIReasoningRequest) -> dict[str, Any]:
        """Give the model an explicit compact contract without duplicating full JSON Schema."""

        return {
            "top_level": {"forecast": "object (required)", "proposal": "object or null"},
            "immutable_values": {
                "request_id": str(request.request_id),
                "cycle_id": str(request.cycle_id),
                "market_state_id": str(request.market_state_id),
                "quantitative_forecast_id": str(request.quantitative_forecast_id),
                "model_identifier": request.model_identifier,
                "prompt_version": request.prompt_version,
                "reasoning_policy_version": request.reasoning_policy_version,
                "setup_family_registry_version": request.setup_family_registry_version,
                "quantitative_model_version": request.quantitative_model_version,
                "feature_schema_version": request.feature_schema_version,
                "market_state_schema_version": request.market_state_schema_version,
            },
            "forecast_required": {
                "forecast_id": "UUID",
                "request_id": "UUID copied from immutable_values",
                "market_state_id": "UUID copied from immutable_values",
                "quantitative_forecast_id": "UUID copied from immutable_values",
                "cycle_id": "UUID copied from immutable_values",
                "status": "available | non_actionable",
                "dominant_direction": "BUY | SELL | NEUTRAL",
                "buy_probability": "number 0..1",
                "sell_probability": "number 0..1",
                "neutral_probability": "number 0..1; all three sum to 1",
                "dominant_scenario": "string",
                "dominant_scenario_probability": "number 0..1",
                "model_provider": "openrouter",
                "model_identifier": "copied from immutable_values",
                "prompt_version": "copied from immutable_values",
                "reasoning_policy_version": "copied from immutable_values",
                "setup_family_registry_version": "copied from immutable_values",
                "quantitative_model_version": "copied from immutable_values",
                "feature_schema_version": "copied from immutable_values",
                "market_state_schema_version": "copied from immutable_values",
                "validation_passed": True,
                "retry_count": 0,
                "shadow_only": True,
                "awaiting_guardrail_validation": True,
                "approved_for_publication": False,
                "generated_at": "timezone-aware ISO-8601 timestamp",
            },
            "proposal_required_when_object": {
                "proposal_id": "UUID",
                "forecast_id": "same UUID as forecast.forecast_id",
                "market_state_id": "UUID copied from immutable_values",
                "recommended_action": "BUY | SELL | WAIT",
                "direction": "BUY | SELL | NEUTRAL",
                "setup_readiness": "not_ready | developing | ready | active",
                "proposal_confidence": "number 0..1",
                "model_identifier": "copied from immutable_values",
                "policy_version": "reasoning_policy_version copied from immutable_values",
                "shadow_only": True,
                "awaiting_guardrail_validation": True,
                "approved_for_publication": False,
                "created_at": "timezone-aware ISO-8601 timestamp",
            },
            "rules": [
                "forecast must never be a string",
                "WAIT must use direction NEUTRAL and null/empty execution levels",
                "BUY/SELL requires ordered entry_zone, stop_loss, and at least one take_profit_level",
                "use only evidence IDs present in analysis_request",
                "unknown fields are unnecessary",
            ],
        }
