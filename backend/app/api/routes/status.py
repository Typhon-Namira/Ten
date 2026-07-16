from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_engine_registry
from backend.app.api.schemas import MarketStatusResponse
from backend.app.engines.common import EngineStatus
from backend.app.services import EngineRegistry

router = APIRouter(tags=["status"])


@router.get("/engines/status", response_model=list[EngineStatus])
async def engine_status(registry: Annotated[EngineRegistry, Depends(get_engine_registry)]) -> list[EngineStatus]:
    """Expose the registered engine implementations and their readiness."""

    return registry.statuses()


@router.get("/market/status", response_model=MarketStatusResponse)
async def market_status() -> MarketStatusResponse:
    """Return a transparent weekday/session approximation, not broker status."""

    now = datetime.now(UTC)
    weekday_open = now.weekday() < 5
    hour = now.hour
    session = "asia" if hour < 7 else "london" if hour < 13 else "new_york" if hour < 21 else "after_hours"
    return MarketStatusResponse(symbol="XAU/USD", session=session, is_open=weekday_open, checked_at=now, note="Indicative weekday/session status; validate against the selected data provider.")

