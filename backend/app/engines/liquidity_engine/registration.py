from datetime import date
from typing import Any
from collections.abc import Mapping

from backend.app.engines.common import EngineMetadata
from backend.app.engines.smc_engine.liquidity_contract import SMCLiquidityContext, SMCLiquidityLevel
from backend.app.events import LiquidityCompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .analyzer import BaselineLiquidityAnalyzer
from .config import LiquidityConfig
from .contracts import LiquidityContext
from .service import LiquidityService


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineLiquidityAnalyzer:
    return BaselineLiquidityAnalyzer(LiquidityConfig.model_validate(config))


async def _execute(engine: BaselineLiquidityAnalyzer, context: PipelineExecutionContext) -> EngineExecutionResult:
    smc_result = context.results.get("smc")
    snapshot = getattr(smc_result, "snapshot", None)
    smc_context = None
    if snapshot is not None:
        levels = tuple(
            SMCLiquidityLevel(
                id=str(item.id),
                symbol=item.symbol,
                timeframe=item.timeframe,
                kind=item.swing_type.value,
                scope=item.scope.value,
                price=item.price,
                occurred_at=item.timestamp,
                available_at=item.confirmed_at or item.detected_at,
                confidence_score=item.confidence_score,
                quality_score=item.quality_score,
            )
            for item in snapshot.swings
        )
        smc_context = SMCLiquidityContext(
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            analyzed_through=snapshot.analysis_timestamp,
            structure_direction=snapshot.structure_state.current_direction.value,
            levels=levels,
            configuration_version=snapshot.configuration_version,
            engine_version=snapshot.engine_version,
        )
    result = engine.analyze(LiquidityContext(tuple(context.candles), smc_context))
    factor = min(1.0, len(result.levels) / 2) if result.levels else 0.0
    return EngineExecutionResult(
        output=result,
        features=LiquidityService.features(result.snapshot) if result.snapshot else result.model_dump(mode="json"),
        namespace="liquidity",
        event_type=LiquidityCompleted,
        confidence_factor=factor,
    )


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="liquidity",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 17),
            dependencies=("market_data", "smc"),
            description="Deterministic inferred-liquidity analysis engine.",
            config_key="liquidity",
            feature_flag="EnableLiquidity",
        ),
        _build,
        _execute,
    )
