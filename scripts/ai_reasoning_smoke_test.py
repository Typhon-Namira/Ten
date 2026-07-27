"""One-shot, shadow-only smoke test for the configured Groq account pool."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.ai.provider_client import HttpAIProviderClient
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.ai_reasoning.config import AIReasoningConfig
from backend.app.ai_reasoning.provider import GroqProvider, GroqProviderPool
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.ai_reasoning.service import AIReasoningService
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from backend.app.ai_reasoning.validation import StructuredAIOutputValidator
from backend.app.core.config import YamlConfigRepository, get_settings
from tests.ai_reasoning.test_ai_reasoning_lifecycle import NOW, state_and_quant


async def main() -> int:
    settings = get_settings()
    if not any(settings.groq_pool_api_keys):
        print("No Groq pool API key is configured; refusing to run.")
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
    providers = tuple(
        GroqProvider(
            HttpAIProviderClient(
                f"groq_{index}",
                key,
                settings.groq_base_url,
                settings.groq_request_timeout_seconds,
            ),
            prompts,
            account_id=f"groq_{index}",
            model=settings.groq_model,
            **common,
        )
        for index, key in enumerate(settings.groq_pool_api_keys, start=1)
    )
    router = GroqProviderPool(
        providers,
        maximum_retries=settings.groq_max_retries_per_account,
        rate_limit_cooldown_seconds=settings.groq_rate_limit_cooldown_seconds,
        quota_cooldown_seconds=settings.groq_quota_cooldown_seconds,
    )
    service = AIReasoningService(
        InMemoryAIReasoningRepository(),
        router,
        AIReasoningRequestBuilder(
            config,
            model_identifier=settings.groq_model,
            clock=lambda: NOW,
        ),
        StructuredAIOutputValidator(),
        config,
        shadow_enabled=True,
        proposals_enabled=False,
        monitoring_enabled=False,
        clock=lambda: NOW,
    )

    print(f"Primary Groq pool/{settings.groq_model}; accounts=4")
    state, quant = await state_and_quant()
    result = await service.process(state, quant)
    if result is None:
        print("Reasoning was skipped before the provider boundary.")
        return 1
    analysis = result.analysis
    print(f"provider={analysis.provider_metadata.provider}")
    print(f"model={analysis.provider_metadata.model}")
    print(f"status={analysis.status.value}")
    print(f"fallback_used={analysis.provider_metadata.fallback_used}")
    print(f"validation_passed={analysis.validation_passed}")
    return 0 if analysis.status.value == "available" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
