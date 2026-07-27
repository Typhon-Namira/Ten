"""One-shot, shadow-only smoke test for the configured AI provider router."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.ai.provider_client import HttpAIProviderClient
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.ai_reasoning.config import AIReasoningConfig
from backend.app.ai_reasoning.provider import AIProviderRouter, CerebrasProvider, GroqProvider
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.ai_reasoning.service import AIReasoningService
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from backend.app.ai_reasoning.validation import StructuredAIOutputValidator
from backend.app.core.config import YamlConfigRepository, get_settings
from tests.ai_reasoning.test_ai_reasoning_lifecycle import NOW, state_and_quant


async def main() -> int:
    settings = get_settings()
    if not settings.cerebras_api_key and not settings.groq_api_key:
        print("No Cerebras or Groq API key is configured; refusing to run.")
        return 2

    configs = YamlConfigRepository()
    config = configs.load_model("ai_reasoning", AIReasoningConfig)
    registry = SetupFamilyRegistry.from_yaml(configs)
    prompts = PromptLoader(
        Path(__file__).resolve().parents[1] / "backend" / "app" / "ai_reasoning" / "prompts"
    )
    common = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "target_input_tokens": config.target_input_tokens,
        "warning_input_tokens": config.warning_input_tokens,
        "hard_input_tokens": config.hard_input_tokens,
        "absolute_max_output_tokens": config.absolute_max_output_tokens,
        "maximum_request_cost_usd": config.maximum_request_cost_usd,
        "input_cost_per_million_usd": config.input_cost_per_million_usd,
        "output_cost_per_million_usd": config.output_cost_per_million_usd,
        "setup_family_ids": tuple(item.setup_family_id for item in registry.all()),
    }
    router = AIProviderRouter(
        CerebrasProvider(
            HttpAIProviderClient(
                "cerebras",
                settings.cerebras_api_key,
                settings.cerebras_base_url,
            ),
            prompts,
            model=settings.cerebras_model,
            **common,
        ),
        GroqProvider(
            HttpAIProviderClient("groq", settings.groq_api_key, settings.groq_base_url),
            prompts,
            model=settings.groq_model,
            **common,
        ),
        maximum_retries=config.maximum_retries,
    )
    service = AIReasoningService(
        InMemoryAIReasoningRepository(),
        router,
        AIReasoningRequestBuilder(
            config,
            model_identifier=settings.cerebras_model,
            clock=lambda: NOW,
        ),
        StructuredAIOutputValidator(registry),
        registry,
        config,
        shadow_enabled=True,
        proposals_enabled=False,
        monitoring_enabled=False,
        clock=lambda: NOW,
    )

    print(
        f"Primary cerebras/{settings.cerebras_model}; "
        f"fallback groq/{settings.groq_model}"
    )
    state, quant = await state_and_quant()
    result = await service.process(state, quant)
    if result is None:
        print("Reasoning was skipped before the provider boundary.")
        return 1
    forecast = result.forecast
    print(f"provider={forecast.model_provider}")
    print(f"model={forecast.model_identifier}")
    print(f"status={forecast.status.value}")
    print(f"failure_state={forecast.failure_state}")
    print(f"fallback_used={forecast.provider_fallback_used}")
    print(f"validation_passed={forecast.validation_passed}")
    return 0 if forecast.status.value in {"available", "non_actionable"} else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
