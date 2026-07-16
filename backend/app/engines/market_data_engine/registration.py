from datetime import date
from typing import Any
from collections.abc import Mapping

from backend.app.engines.common import EngineMetadata
from backend.app.events import MarketDataReady
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext


class MarketDataBoundary:
    """Registry boundary for provider-normalized input supplied to a pipeline run."""


def _build(_: EngineBuildContext, __: Mapping[str, Any]) -> MarketDataBoundary:
    return MarketDataBoundary()


async def _execute(_: MarketDataBoundary, context: PipelineExecutionContext) -> EngineExecutionResult:
    latest = context.candles[-1] if context.candles else None
    features = {
        "count": len(context.candles),
        "symbol": latest.symbol if latest else None,
        "timeframe": latest.timeframe.value if latest else None,
        "open": latest.open if latest else None,
        "high": latest.high if latest else None,
        "low": latest.low if latest else None,
        "close": latest.close if latest else None,
        "volume": latest.volume if latest else None,
        "spread": latest.spread if latest else None,
        "provider": latest.provider if latest else None,
        "quality": latest.quality_score if latest else None,
        "timestamp": latest.timestamp.isoformat() if latest else None,
    }
    return EngineExecutionResult(output=context.candles, features=features, namespace="market_data", event_type=MarketDataReady)


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(name="market_data", version="1.0.0", compatibility_version="1.0", created_date=date(2026, 7, 16), dependencies=(), description="Provider-neutral normalized market-data boundary.", config_key="market_data"),
        _build,
        _execute,
    )
