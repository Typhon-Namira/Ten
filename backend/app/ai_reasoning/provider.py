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
            payload=request.model_dump(mode="json"),
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
