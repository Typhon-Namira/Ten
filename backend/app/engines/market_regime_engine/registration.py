from collections.abc import Mapping
from datetime import date
from typing import Any

from backend.app.engines.common import EngineMetadata
from backend.app.events import Event
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import MarketRegimeConfig
from .engine import BaselineMarketRegimeEngine


class MarketRegimeCompleted(Event):
    pass


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineMarketRegimeEngine:
    return BaselineMarketRegimeEngine(MarketRegimeConfig.model_validate(config))


async def _execute(engine: BaselineMarketRegimeEngine, context: PipelineExecutionContext) -> EngineExecutionResult:
    result = engine.analyze(context.candles)
    return EngineExecutionResult(
        output=result, features=result.model_dump(mode="json"), namespace="market_regime", event_type=MarketRegimeCompleted, confidence_factor=result.confidence
    )


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="market_regime",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 18),
            dependencies=("market_data", "smc", "liquidity", "volume_profile", "institutional_flow"),
            description="Deterministic, probabilistic, explainable and no-lookahead market-environment synthesis.",
            config_key="market_regime",
            feature_flag="EnableMarketRegime",
        ),
        _build,
        _execute,
    )
