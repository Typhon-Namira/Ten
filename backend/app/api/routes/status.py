from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from backend.app.api.dependencies import get_engine_registry
from backend.app.api.schemas import MarketStatusResponse
from backend.app.engines.common import EngineStatus
from backend.app.engines.market_data_engine import MarketDataService, Timeframe
from backend.app.api.dependencies import get_market_data_service
from backend.app.services import EngineRegistry

router = APIRouter(tags=["status"])


@router.get("/engines/status", response_model=list[EngineStatus])
async def engine_status(registry: Annotated[EngineRegistry, Depends(get_engine_registry)]) -> list[EngineStatus]:
    """Expose the registered engine implementations and their readiness."""

    return registry.statuses()


@router.get("/market/status", response_model=MarketStatusResponse)
async def market_status(request: Request, service: Annotated[MarketDataService, Depends(get_market_data_service)]) -> MarketStatusResponse:
    """Return timezone-aware trading availability and data freshness for the pipeline's
    configured primary instrument/timeframe — not a hardcoded literal, so this always matches
    whatever `settings.market_data_symbols`/`market_data_timeframes` actually run."""
    settings = request.app.state.settings
    symbol = settings.market_data_symbols[0].upper()
    timeframe = Timeframe(settings.market_data_timeframes[0])
    now = datetime.now(UTC)
    schedule = service.sessions.status_at(now)
    candle = await service.repository.candle_at(symbol, timeframe, now)
    state = await service.state(symbol, timeframe, at=now)
    return MarketStatusResponse(
        symbol=symbol,
        session=schedule.active_session.value if schedule.active_session else None,
        is_open=schedule.market_open,
        checked_at=now,
        note=f"Schedule-aware {symbol} status; provider freshness is reported separately.",
        market_status=schedule.market_status.value,
        closure_reason=schedule.closure_reason,
        next_expected_open_at=schedule.next_expected_open_at,
        server_time_utc=schedule.server_time_utc,
        latest_candle_at=candle.timestamp if candle else None,
        latest_candle_age_seconds=max(0, (now - candle.timestamp).total_seconds()) if candle else None,
        provider_status=state.provider_health,
    )

