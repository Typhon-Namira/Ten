import asyncio
from typing import Any

from backend.app.ai.openrouter_client.client import OpenRouterClient
from backend.app.ai.prompts import PromptLoader
from backend.app.engines.ai_scoring_engine import OpenRouterScoringEngine, ScoringContext


class FakeOpenRouterClient(OpenRouterClient):
    async def complete_json(self, **_: Any) -> dict[str, Any]:
        return {"confidence": 0.72, "direction": "long", "quality_score": 74, "risk_notes": [], "reasoning": ["Aligned structured evidence"]}


def test_ai_scoring_validates_provider_output() -> None:
    context = ScoringContext(market_structure={}, liquidity={}, flow_score={}, volume_profile={}, news_risk={})
    result = asyncio.run(OpenRouterScoringEngine(FakeOpenRouterClient(), PromptLoader()).score(context))
    assert result.confidence == 0.72
    assert result.model == "meta-llama/llama-3.3-70b-instruct"
