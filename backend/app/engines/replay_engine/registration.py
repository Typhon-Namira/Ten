from collections.abc import Mapping
from datetime import date
from typing import Any

from backend.app.engines.common import EngineLifecycleStatus, EngineMetadata
from backend.app.events import Event
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import ReplayConfig


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> ReplayConfig:
    return ReplayConfig.model_validate(config)


async def _execute(config: ReplayConfig, _: PipelineExecutionContext) -> EngineExecutionResult:
    return EngineExecutionResult(
        output={"status": "controlled_by_replay_worker", "trade_execution": False},
        features={"version": config.engine.version, "historical_reconstruction": True},
        namespace="replay",
        event_type=Event,
    )


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="replay",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 19),
            status=EngineLifecycleStatus.STABLE,
            dependencies=("market_data", "ai_scoring", "signal_decision"),
            description="Deterministic point-in-time historical reconstruction with isolated events, durable checkpoints and database leases.",
            enabled=True,
            config_key="replay",
            feature_flag="EnableReplay",
        ),
        _build,
        _execute,
    )
