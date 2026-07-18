from datetime import date
from typing import Any
from collections.abc import Mapping

from backend.app.engines.common import EngineMetadata
from backend.app.events import EconomicCalendarCompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import EconomicCalendarConfig
from .engine import BaselineEconomicCalendarEngine


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineEconomicCalendarEngine:
    return BaselineEconomicCalendarEngine(EconomicCalendarConfig.model_validate(config))


async def _execute(engine: BaselineEconomicCalendarEngine, context: PipelineExecutionContext) -> EngineExecutionResult:
    result = engine.analyze((context.now, context.events))
    return EngineExecutionResult(
        output=result,
        features=result.model_dump(mode="json"),
        namespace="economic",
        event_type=EconomicCalendarCompleted,
        confidence_factor=0.0 if result.no_trade else 1.0,
    )


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="economic_calendar",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 16),
            dependencies=(),
            description="Economic-event risk window contract.",
            config_key="economic",
            feature_flag="EnableEconomicFilter",
        ),
        _build,
        _execute,
    )
