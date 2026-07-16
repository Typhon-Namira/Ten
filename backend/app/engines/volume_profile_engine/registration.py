from datetime import date
from typing import Any
from collections.abc import Mapping

from backend.app.engines.common import EngineMetadata
from backend.app.events import VolumeProfileCompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .analyzer import BaselineVolumeProfileAnalyzer
from .config import VolumeProfileConfig


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineVolumeProfileAnalyzer:
    return BaselineVolumeProfileAnalyzer(VolumeProfileConfig.model_validate(config))


async def _execute(engine: BaselineVolumeProfileAnalyzer, context: PipelineExecutionContext) -> EngineExecutionResult:
    result = engine.analyze(context.candles)
    return EngineExecutionResult(output=result, features=result.model_dump(mode="json"), namespace="volume_profile", event_type=VolumeProfileCompleted, confidence_factor=1.0 if result.poc is not None else 0.0)


def register(factory: EngineFactory) -> None:
    factory.register(EngineMetadata(name="volume_profile", version="1.0.0", compatibility_version="1.0", created_date=date(2026, 7, 16), dependencies=("market_data",), description="Session and composite volume-profile analysis contract.", config_key="volume_profile", feature_flag="EnableVolumeProfile"), _build, _execute)
