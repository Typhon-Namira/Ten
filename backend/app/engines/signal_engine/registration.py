from datetime import date
from typing import Any
from collections.abc import Mapping

from backend.app.analytics.statistics import average_true_range
from backend.app.engines.common import EngineMetadata
from backend.app.events import SignalGenerated
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import SignalConfig
from .engine import BaselineSignalEngine
from .models import SignalInputs


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineSignalEngine:
    return BaselineSignalEngine(SignalConfig.model_validate(config))


async def _execute(engine: BaselineSignalEngine, context: PipelineExecutionContext) -> EngineExecutionResult:
    latest = context.candles[-1]
    result = engine.analyze(
        SignalInputs(
            symbol=latest.symbol,
            timeframe=latest.timeframe.value,
            current_price=latest.close,
            average_true_range=average_true_range(context.candles),
            smc=context.results["smc"],
            liquidity=context.results["liquidity"],
            flow=context.results["institutional_flow"],
            volume_profile=context.results["volume_profile"],
            news_risk=context.results["economic_calendar"],
            ai_score=context.results["ai_scoring"],
            calculated_confidence=context.calculated_confidence,
            confidence_breakdown=context.confidence_breakdown,
        )
    )
    return EngineExecutionResult(output=result, features=result.model_dump(mode="json"), namespace="signal", event_type=SignalGenerated)


def register(factory: EngineFactory) -> None:
    factory.register(EngineMetadata(name="signal", version="1.0.0", compatibility_version="1.0", created_date=date(2026, 7, 16), dependencies=("smc", "liquidity", "institutional_flow", "volume_profile", "economic_calendar", "ai_scoring"), description="Central non-executable scenario construction boundary.", config_key="signal"), _build, _execute)
