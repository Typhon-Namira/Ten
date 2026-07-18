from collections.abc import Mapping
from datetime import date
from typing import Any

from backend.app.engines.common import EngineMetadata
from backend.app.events import FlowCompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import InstitutionalFlowConfig
from .engine import BaselineInstitutionalFlowEngine


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineInstitutionalFlowEngine:
    return BaselineInstitutionalFlowEngine(InstitutionalFlowConfig.model_validate(config))


async def _execute(engine: BaselineInstitutionalFlowEngine, context: PipelineExecutionContext) -> EngineExecutionResult:
    result = engine.analyze(context.candles)
    return EngineExecutionResult(output=result, features=result.model_dump(mode="json"), namespace="institutional_flow", event_type=FlowCompleted, confidence_factor=abs(result.score))


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="institutional_flow",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 18),
            dependencies=("market_data", "smc", "liquidity", "volume_profile"),
            description="Probabilistic, evidence-supported institutional-flow inference without participant identity claims.",
            config_key="flow",
            feature_flag="EnableFlow",
        ),
        _build,
        _execute,
    )
