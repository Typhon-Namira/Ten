from datetime import date
from typing import Any
from collections.abc import Mapping

from backend.app.ai.openrouter_client import HttpOpenRouterClient
from backend.app.ai.prompts import PromptLoader
from backend.app.engines.common import EngineMetadata
from backend.app.events import AICompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import AIScoringConfig
from .engine import OpenRouterScoringEngine
from .models import ScoringContext


def _build(context: EngineBuildContext, config: Mapping[str, Any]) -> OpenRouterScoringEngine:
    client = HttpOpenRouterClient(context.settings.openrouter_api_key, context.settings.openrouter_base_url, context.settings.request_timeout_seconds)
    return OpenRouterScoringEngine(client, PromptLoader(), AIScoringConfig.model_validate(config))


async def _execute(engine: OpenRouterScoringEngine, context: PipelineExecutionContext) -> EngineExecutionResult:
    snapshot = await context.feature_store.snapshot(context.correlation_id)
    context.feature_snapshot = snapshot
    result = await engine.score(ScoringContext.from_features(snapshot.features, snapshot.engine_versions))
    features = result.model_dump(mode="json", exclude={"confidence"})
    return EngineExecutionResult(output=result, features=features, namespace="ai", event_type=AICompleted, confidence_factor=result.quality_score / 100)


def register(factory: EngineFactory) -> None:
    factory.register(EngineMetadata(name="ai_scoring", version="1.0.0", compatibility_version="1.0", created_date=date(2026, 7, 16), dependencies=("smc", "liquidity", "institutional_flow", "volume_profile", "economic_calendar"), description="OpenRouter structured-feature quality assessment boundary.", config_key="ai", feature_flag="EnableAI"), _build, _execute)
