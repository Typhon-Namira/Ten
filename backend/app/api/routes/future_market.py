"""TEN 2.0 future-market read API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from backend.app.engines.market_data_engine.models import canonical_symbol
from backend.app.future_market import (
    BoundedInMemoryFutureMarketRepository,
    FutureMarketService,
    MAX_FORECASTS_PER_INSTRUMENT,
)

router = APIRouter(prefix="/api/v2/future-market", tags=["future-market-v2"])


def _service(request: Request) -> FutureMarketService:
    service = getattr(request.app.state, "future_market_service", None)
    if service is None:
        repository = BoundedInMemoryFutureMarketRepository(
            limit=MAX_FORECASTS_PER_INSTRUMENT
        )
        service = FutureMarketService(
            repository,
            simulation_repository=getattr(
                request.app.state, "market_simulation_repository", None
            ),
        )
        request.app.state.future_market_repository = repository
        request.app.state.future_market_service = service
    return service


@router.get("/latest")
async def latest_future_market(
    request: Request,
    instrument: str = Query(default="XAUUSD"),
):
    symbol = canonical_symbol(instrument)
    forecast = await _service(request).latest(symbol)
    if forecast is None:
        raise HTTPException(
            status_code=404,
            detail="No completed market scenario is available to bootstrap TEN 2.0 yet",
        )
    return forecast.model_dump(mode="json")


@router.get("/history")
async def future_market_history(
    request: Request,
    instrument: str = Query(default="XAUUSD"),
    limit: int = Query(default=100, ge=1, le=MAX_FORECASTS_PER_INSTRUMENT),
):
    symbol = canonical_symbol(instrument)
    values = await _service(request).history(symbol, limit)
    return [item.model_dump(mode="json") for item in values]


@router.get("/opportunities/current")
async def current_opportunities(
    request: Request,
    instrument: str = Query(default="XAUUSD"),
):
    symbol = canonical_symbol(instrument)
    values = await _service(request).opportunities(symbol)
    return [item.model_dump(mode="json") for item in values]


@router.get("/performance")
async def future_market_performance(
    request: Request,
    instrument: str = Query(default="XAUUSD"),
):
    symbol = canonical_symbol(instrument)
    value = await _service(request).performance(symbol)
    return value.model_dump(mode="json")
