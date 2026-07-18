from collections.abc import Mapping
from datetime import date
from typing import Any

from backend.app.engines.common import EngineMetadata
from backend.app.events import VolumeProfileCompleted
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .analyzer import BaselineVolumeProfileAnalyzer
from .config import VolumeProfileConfig
from .models import VolumeSourceType


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> BaselineVolumeProfileAnalyzer:
    return BaselineVolumeProfileAnalyzer(VolumeProfileConfig.model_validate(config))


async def _execute(engine: BaselineVolumeProfileAnalyzer, context: PipelineExecutionContext) -> EngineExecutionResult:
    smc_result = context.results.get("smc")
    smc = getattr(getattr(smc_result, "snapshot", None), "liquidity_context", None)
    result = engine.analyze(context.candles)
    if smc is not None:
        result = engine.analyze(list(context.candles))
    profile = result.snapshot.profiles[0] if result.snapshot and result.snapshot.profiles else None
    features = {
        "poc": result.poc,
        "vah": result.vah,
        "val": result.val,
        "total_volume": result.total_volume,
        "profile": profile.model_dump(mode="json") if profile else None,
        "volume_source_type": VolumeSourceType.UNKNOWN.value,
    }
    return EngineExecutionResult(
        output=result,
        features=features,
        namespace="volume_profile",
        event_type=VolumeProfileCompleted,
        confidence_factor=(profile.confidence_score / 100) if profile else 0.0,
    )


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="volume_profile",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 18),
            dependencies=("market_data", "smc", "liquidity"),
            description="Deterministic replay-safe analytical volume-at-price profiles.",
            config_key="volume_profile",
            feature_flag="EnableVolumeProfile",
        ),
        _build,
        _execute,
    )
