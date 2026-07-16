from datetime import date
from typing import Any
from collections.abc import Mapping

from backend.app.engines.common import EngineMetadata
from backend.app.events import LiquidityCompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .analyzer import BaselineLiquidityAnalyzer
from .config import LiquidityConfig


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineLiquidityAnalyzer:
    return BaselineLiquidityAnalyzer(LiquidityConfig.model_validate(config))


async def _execute(engine: BaselineLiquidityAnalyzer, context: PipelineExecutionContext) -> EngineExecutionResult:
    result = engine.analyze(context.candles)
    factor = min(1.0, len(result.levels) / 2) if result.levels else 0.0
    return EngineExecutionResult(output=result, features=result.model_dump(mode="json"), namespace="liquidity", event_type=LiquidityCompleted, confidence_factor=factor)


def register(factory: EngineFactory) -> None:
    factory.register(EngineMetadata(name="liquidity", version="1.0.0", compatibility_version="1.0", created_date=date(2026, 7, 16), dependencies=("market_data",), description="Liquidity pool and sweep analysis contract.", config_key="liquidity", feature_flag="EnableLiquidity"), _build, _execute)
