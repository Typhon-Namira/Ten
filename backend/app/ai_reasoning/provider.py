"""Existing-OpenRouter-only provider boundary for AI market reasoning."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from time import perf_counter
from typing import Any, Protocol

from backend.app.ai.openrouter_client.client import (
    OpenRouterClient,
    build_request_body,
    measure_request_body,
)
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.exceptions import OpenRouterFailureDetails, OpenRouterRequestError

from .llm_context import build_llm_analysis_context
from .models import AIReasoningRequest

logger = logging.getLogger(__name__)
AI_REASONING_RESPONSE_SCHEMA_TYPE = "ten_ai_reasoning_response"
AI_REASONING_RESPONSE_SCHEMA_VERSION = "1.0"


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
        target_input_tokens: int = 4_000,
        warning_input_tokens: int = 8_000,
        hard_input_tokens: int = 16_000,
        absolute_max_output_tokens: int = 2_000,
        maximum_request_cost_usd: float = 0.05,
        input_cost_per_million_usd: float = 1.04,
        output_cost_per_million_usd: float = 2.25,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.target_input_tokens = target_input_tokens
        self.warning_input_tokens = warning_input_tokens
        self.hard_input_tokens = hard_input_tokens
        self.absolute_max_output_tokens = absolute_max_output_tokens
        self.maximum_request_cost_usd = maximum_request_cost_usd
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd

    async def reason(self, request: AIReasoningRequest, *, prompt_version: str) -> AIProviderResponse:
        started = perf_counter()
        context = build_llm_analysis_context(request)
        payload = {
            "analysis_context": context.model_dump(mode="json"),
            "response_contract": self._response_contract(),
        }
        system_prompt = self.prompts.load(prompt_version)
        request_body = build_request_body(
            system_prompt=system_prompt,
            payload=payload,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        metrics = measure_request_body(
            request_body,
            input_cost_per_million_usd=self.input_cost_per_million_usd,
            output_cost_per_million_usd=self.output_cost_per_million_usd,
        )
        section_sizes = {
            key: len(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            )
            for key, value in payload.items()
        }
        largest_sections = tuple(
            f"{key}:{size}"
            for key, size in sorted(
                section_sizes.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        logger.info(
            "ai_reasoning.request.measured",
            extra={
                "request_id": str(request.request_id),
                "cycle_id": str(request.cycle_id),
                "model": self.model,
                "request_schema_version": context.schema_version,
                "response_schema_version": AI_REASONING_RESPONSE_SCHEMA_VERSION,
                "serialized_request_bytes": metrics.serialized_request_bytes,
                "estimated_input_tokens": metrics.estimated_input_tokens,
                "maximum_output_tokens": metrics.maximum_output_tokens,
                "estimated_maximum_cost_usd": metrics.estimated_maximum_cost_usd,
                "message_count": metrics.message_count,
                "system_prompt_characters": metrics.system_prompt_characters,
                "user_prompt_characters": metrics.user_prompt_characters,
                "tool_definition_bytes": metrics.tool_definition_bytes,
                "response_schema_bytes": metrics.response_schema_bytes,
                "largest_payload_sections": largest_sections,
                "over_target": metrics.estimated_input_tokens > self.target_input_tokens,
                "over_warning": metrics.estimated_input_tokens > self.warning_input_tokens,
            },
        )
        rejection_reason: str | None = None
        if metrics.maximum_output_tokens > self.absolute_max_output_tokens:
            rejection_reason = "maximum_output_tokens_exceeded"
        elif metrics.estimated_input_tokens > self.hard_input_tokens:
            rejection_reason = "request_too_large"
        elif metrics.estimated_maximum_cost_usd > self.maximum_request_cost_usd:
            rejection_reason = "maximum_cost_exceeded"
        if rejection_reason is not None:
            details = OpenRouterFailureDetails(
                reason_code=rejection_reason,
                phase="request_validation",
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                model=self.model,
                endpoint=f"{getattr(self.client, 'base_url', 'openrouter')}/chat/completions",
                error_code=rejection_reason,
                error_message=(
                    f"preflight rejected bytes={metrics.serialized_request_bytes} "
                    f"estimated_input_tokens={metrics.estimated_input_tokens} "
                    f"maximum_output_tokens={metrics.maximum_output_tokens} "
                    f"estimated_maximum_cost_usd={metrics.estimated_maximum_cost_usd:.6f} "
                    f"largest_sections={','.join(largest_sections)}"
                )[:500],
                body_length=metrics.serialized_request_bytes,
                exception_class="OpenRouterRequestBudgetError",
            )
            logger.error(
                "openrouter.request.rejected",
                extra={
                    "request_id": details.request_id,
                    "cycle_id": details.cycle_id,
                    "model": details.model,
                    "failure_phase": details.phase,
                    "failure_reason_code": details.reason_code,
                    "serialized_request_bytes": metrics.serialized_request_bytes,
                    "estimated_input_tokens": metrics.estimated_input_tokens,
                    "maximum_output_tokens": metrics.maximum_output_tokens,
                    "estimated_maximum_cost_usd": metrics.estimated_maximum_cost_usd,
                    "largest_payload_sections": largest_sections,
                },
            )
            raise OpenRouterRequestError(details)
        if metrics.estimated_input_tokens > self.warning_input_tokens:
            logger.warning(
                "openrouter.request.budget_warning",
                extra={
                    "request_id": str(request.request_id),
                    "cycle_id": str(request.cycle_id),
                    "estimated_input_tokens": metrics.estimated_input_tokens,
                    "warning_input_tokens": self.warning_input_tokens,
                    "largest_payload_sections": largest_sections,
                },
            )
        raw = await self.client.complete_json(
            system_prompt=system_prompt,
            payload=payload,
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
    def _response_contract() -> dict[str, Any]:
        """Request only the decision fields consumed by deterministic normalization."""

        return {
            "schema_type": AI_REASONING_RESPONSE_SCHEMA_TYPE,
            "schema_version": AI_REASONING_RESPONSE_SCHEMA_VERSION,
            "required": {
                "decision": "LONG | SHORT | WAIT",
                "confidence": "number 0..1",
                "rationale": "concise string, maximum 500 characters",
                "risk_flags": "array of at most 5 concise strings",
                "proposal": "object only for LONG/SHORT; otherwise null",
            },
            "proposal_when_actionable": {
                "setup_family": "string",
                "entry_low": "positive number",
                "entry_high": "positive number",
                "stop_loss": "positive number",
                "take_profit_levels": "array of 1..3 positive numbers",
            },
            "rules": [
                "return exactly one JSON object",
                "do not include chain-of-thought or private reasoning",
                "WAIT requires proposal=null",
                "LONG/SHORT requires valid ordered entry, stop, and target geometry",
                "use only the compact analysis_context",
            ],
        }
