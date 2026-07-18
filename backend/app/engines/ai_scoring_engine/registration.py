from collections.abc import Mapping
from datetime import UTC, date
from typing import Any

from backend.app.engines.common import EngineMetadata
from backend.app.events import AICompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import AIScoringConfig
from .engine import DeterministicAIScoringEngine
from .models import ScoreMode, ScoringContext, ScoringInput
from .normalization import normalized_source


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> DeterministicAIScoringEngine:
    return DeterministicAIScoringEngine(AIScoringConfig.model_validate(config))


async def _execute(engine: Any, context: PipelineExecutionContext) -> EngineExecutionResult:
    feature_snapshot = await context.feature_store.snapshot(context.correlation_id)
    context.feature_snapshot = feature_snapshot
    if not isinstance(engine, DeterministicAIScoringEngine):
        legacy = await engine.score(ScoringContext.from_features(feature_snapshot.features, feature_snapshot.engine_versions))
        return EngineExecutionResult(
            output=legacy,
            features=legacy.model_dump(mode="json", exclude={"confidence"}),
            namespace="ai",
            event_type=AICompleted,
            confidence_factor=legacy.quality_score / 100,
        )
    boundary = feature_snapshot.created_at.astimezone(UTC)
    evidence = {}
    for name, values in sorted(feature_snapshot.features.items()):
        if name not in engine.config.components:
            continue
        evidence[name] = normalized_source(
            name,
            engine.config.components[name].source_group,
            values,
            boundary,
            feature_snapshot.engine_versions.get(name, "1.0.0"),
            f"{context.correlation_id}:{name}",
        )
    scoring_input = ScoringInput(
        instrument="XAUUSD",
        timeframe="M15",
        as_of=boundary,
        requested_at=boundary,
        mode=ScoreMode.LIVE,
        market_data=evidence.get("market_data"),
        market_regime=evidence.get("market_regime"),
        smc=evidence.get("smc"),
        liquidity=evidence.get("liquidity"),
        volume_profile=evidence.get("volume_profile"),
        institutional_flow=evidence.get("institutional_flow"),
        economic_calendar=evidence.get("economic_calendar"),
    )
    result = engine.score(scoring_input)
    return EngineExecutionResult(
        output=result,
        features={
            "directional": result.directional_score,
            "confidence": result.confidence_score,
            "risk": result.market_risk_score,
            "quality": result.data_quality_score,
            "status": result.status.value,
            "trading_instruction": False,
        },
        namespace="ai_score",
        event_type=AICompleted,
        confidence_factor=result.confidence_score / 100,
    )


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="ai_scoring",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 19),
            dependencies=("smc", "liquidity", "volume_profile", "institutional_flow", "market_regime", "economic_calendar"),
            description="Deterministic, explainable, point-in-time-safe intelligence aggregation; no trading execution.",
            config_key="ai_scoring",
            feature_flag="EnableAI",
        ),
        _build,
        _execute,
    )
