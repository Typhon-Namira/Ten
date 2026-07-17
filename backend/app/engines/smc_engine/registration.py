from collections.abc import Mapping
from datetime import date
from typing import Any

from backend.app.engines.common import EngineMetadata
from backend.app.events import SMCCompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .analyzer import BaselineSMCAnalyzer
from .config import SMCConfig
from .models import StructureDirection
from .service import SMCService


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineSMCAnalyzer:
    return BaselineSMCAnalyzer(SMCConfig.model_validate(config))


async def _execute(engine: BaselineSMCAnalyzer, context: PipelineExecutionContext) -> EngineExecutionResult:
    result = engine.analyze(context.candles)
    if result.snapshot is None:
        raise RuntimeError("registered SMC analyzer did not produce a snapshot")
    features = SMCService.features(result.snapshot)
    return EngineExecutionResult(output=result, features=features, namespace="smc", event_type=SMCCompleted, confidence_factor=0.5 if result.bias in (StructureDirection.NEUTRAL, StructureDirection.TRANSITIONAL) else 1.0)


def register(factory: EngineFactory) -> None:
    factory.register(EngineMetadata(name="smc", version="1.0.0", compatibility_version="1.0", created_date=date(2026, 7, 17), dependencies=("market_data",), description="Deterministic no-lookahead swing and market-structure analysis engine.", config_key="smc", feature_flag="EnableSMC"), _build, _execute)
