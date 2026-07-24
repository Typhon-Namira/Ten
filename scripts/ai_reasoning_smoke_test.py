"""One-off, safe smoke test for the AI Reasoning stage against the REAL configured OpenRouter
model and REAL credentials.

NOT executed as part of the automated test suite and NOT run by the agent that wrote this file —
it requires `TEN_OPENROUTER_API_KEY` (and friends) from the actual Railway environment, which was
not available in the sandbox this was written in. Run it yourself wherever those credentials are
available, e.g. `railway run python scripts/ai_reasoning_smoke_test.py`.

Safety, by construction:
  - Uses `InMemoryAIReasoningRepository` — nothing is written to the real database, production or
    otherwise. No production row is created, modified, or read.
  - `proposals_enabled=False`, `monitoring_enabled=False` — `AIReasoningLifecycleService` can
    never create a `ManagedSignal`, so `final_decision.evaluate()` is never reached and no
    publication path is exercised at all, even if the model is willing.
  - Makes exactly one real HTTP call to OpenRouter (or two, if the configured retry count allows
    a second attempt) and exits. No loop, no worker, no polling.
  - The synthetic market state/quant forecast fixtures are imported directly from
    tests/ai_reasoning/test_ai_reasoning_lifecycle.py (`state_and_quant`) rather than
    reimplemented here, so this script exercises the exact same, already-tested construction path
    instead of a second, unverified approximation of it.

What it proves: whether OpenRouter is reachable, whether the configured API key authenticates,
what HTTP status/latency comes back, and whether the response validates against
StructuredAIOutputValidator — the exact chain the required investigation asked to trace end to
end, using the actual production model and actual production credentials, without touching
anything else in the live system.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.ai.openrouter_client.client import HttpOpenRouterClient
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.ai_reasoning.config import AIReasoningConfig
from backend.app.ai_reasoning.provider import ExistingOpenRouterReasoningProvider
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.ai_reasoning.service import AIReasoningService
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from backend.app.ai_reasoning.validation import StructuredAIOutputValidator
from backend.app.core.config import YamlConfigRepository, get_settings

from tests.ai_reasoning.test_ai_reasoning_lifecycle import NOW, state_and_quant


async def main() -> int:
    settings = get_settings()
    if not settings.openrouter_api_key:
        print("TEN_OPENROUTER_API_KEY is not set in this environment — refusing to run (nothing to test).")
        return 2

    configs = YamlConfigRepository()
    ai_config = configs.load_model("ai_reasoning", AIReasoningConfig)
    registry = SetupFamilyRegistry.from_yaml(configs)
    # Mirrors main.py's real construction exactly (backend/app/main.py, around
    # `ai_provider = ExistingOpenRouterReasoningProvider(...)`) so this exercises the same
    # client/provider wiring production actually uses, not an approximation of it.
    client = HttpOpenRouterClient(settings.openrouter_api_key, settings.openrouter_base_url)
    provider = ExistingOpenRouterReasoningProvider(
        client,
        PromptLoader(Path(__file__).resolve().parent.parent / "backend" / "app" / "ai_reasoning" / "prompts"),
        model=settings.openrouter_model,
        temperature=ai_config.temperature,
        max_tokens=ai_config.max_tokens,
    )
    repository = InMemoryAIReasoningRepository()
    service = AIReasoningService(
        repository, provider,
        AIReasoningRequestBuilder(ai_config, model_identifier=settings.openrouter_model, clock=lambda: NOW),
        StructuredAIOutputValidator(registry), registry, ai_config,
        shadow_enabled=True, proposals_enabled=False, monitoring_enabled=False,
        clock=lambda: NOW,
    )

    print(f"Provider: openrouter | Model: {settings.openrouter_model} | Base URL: {settings.openrouter_base_url}")
    print("Calling AIReasoningService.process() once (shadow-only, nothing will be persisted or published)...")
    state, quant = await state_and_quant()
    result = await service.process(state, quant)

    if result is None:
        print("process() returned None — a gate skipped this cycle before any HTTP call was attempted (check shadow_enabled/backoff).")
        return 1

    forecast = result.forecast
    print(f"forecast.status              = {forecast.status.value}")
    print(f"forecast.failure_state       = {forecast.failure_state}")
    print(f"forecast.failure_phase       = {forecast.failure_phase}")
    print(f"forecast.provider_http_status= {forecast.provider_http_status}")
    print(f"forecast.provider_error_code = {forecast.provider_error_code}")
    print(f"forecast.provider_error_msg  = {forecast.provider_error_message}")
    print(f"forecast.latency_ms          = {forecast.latency_ms}")
    print(f"forecast.validation_passed   = {forecast.validation_passed}")
    print(f"proposal produced            = {result.proposal is not None}")
    if result.proposal is not None:
        print(f"proposal.recommended_action  = {result.proposal.recommended_action.value}")
    return 0 if forecast.status.value in {"available", "non_actionable"} else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
