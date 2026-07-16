from datetime import date
from typing import Any, NoReturn
from collections.abc import Mapping

from backend.app.core.exceptions import ConfigurationError
from backend.app.engines.common import EngineLifecycleStatus, EngineMetadata
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext


def _build(_: EngineBuildContext, __: Mapping[str, Any]) -> NoReturn:
    raise ConfigurationError("Replay engine infrastructure is not implemented")


async def _execute(_: Any, __: PipelineExecutionContext) -> EngineExecutionResult:
    raise ConfigurationError("Replay engine infrastructure is not implemented")


def register(factory: EngineFactory) -> None:
    factory.register(EngineMetadata(name="replay", version="1.0.0", compatibility_version="1.0", created_date=date(2026, 7, 16), status=EngineLifecycleStatus.EXPERIMENTAL, dependencies=("market_data",), description="Future historical replay and simulation contract; execution intentionally absent.", enabled=False, config_key="replay", feature_flag="EnableReplay"), _build, _execute)
