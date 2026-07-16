"""Provider-neutral Market Data Engine REST surface."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.dependencies import get_market_data_service
from backend.app.engines.market_data_engine import Candle, MarketDataService, MarketMetrics, MarketState, Timeframe
from backend.app.engines.market_data_engine.exceptions import MarketDataError

router = APIRouter(prefix="/market", tags=["market-data"])
Service = Annotated[MarketDataService, Depends(get_market_data_service)]


@router.get("/latest", response_model=Candle)
async def latest(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1, refresh: bool = False) -> Candle:
    try:
        candle = await service.latest(symbol, timeframe, refresh=refresh)
    except MarketDataError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    if candle is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No candle available")
    return candle


@router.get("/history", response_model=list[Candle])
async def history(
    service: Service,
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M1,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100_000)] = 500,
    refresh: bool = False,
) -> list[Candle]:
    try:
        return await service.history(symbol, timeframe, start=start, end=end, limit=limit, refresh=refresh)
    except MarketDataError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/replay", response_model=list[Candle])
async def replay(service: Service, at: datetime, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1, limit: Annotated[int, Query(ge=1, le=100_000)] = 500) -> list[Candle]:
    return await service.replay(symbol, timeframe, at, limit=limit)


@router.get("/candle/{timestamp}", response_model=Candle)
async def candle_at(timestamp: datetime, service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1) -> Candle:
    candle = await service.candle_at(symbol, timeframe, timestamp)
    if candle is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No candle existed at the requested timestamp")
    return candle


@router.get("/session/{timestamp}")
async def session_at(timestamp: datetime, service: Service) -> dict[str, object]:
    session = service.sessions.session_at(timestamp)
    return {"timestamp": timestamp, "session": session, "market_open": service.sessions.is_open(timestamp)}


@router.get("/providers")
async def providers(service: Service) -> list[dict[str, object]]:
    return [
        {"name": item.provider_name, "capabilities": item.capabilities}
        for item in service.manager.registry.all()
    ]


@router.get("/provider/status")
async def provider_status(service: Service) -> dict[str, object]:
    return {"current_provider": service.manager.current_provider, "providers": service.manager.statistics}


@router.get("/provider/statistics")
async def provider_statistics(service: Service) -> dict[str, object]:
    return {"rankings": service.manager.rankings("XAUUSD", Timeframe.M1), "statistics": service.manager.statistics}


@router.get("/state", response_model=MarketState)
async def state(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1, at: datetime | None = None) -> MarketState:
    return await service.state(symbol, timeframe, at=at)


@router.get("/metrics", response_model=MarketMetrics)
async def metrics(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1) -> MarketMetrics:
    result = await service.metrics(symbol, timeframe)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insufficient market data")
    return result


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    statistics = service.manager.statistics
    healthy = sum(1 for item in statistics.values() if item.healthy)
    return {
        "status": "healthy" if healthy else "degraded",
        "registered_providers": len(statistics),
        "healthy_providers": healthy,
        "cache": {**service.cache.statistics.model_dump(), "hit_ratio": service.cache.statistics.hit_ratio},
        "historical_sync": service.sync_status,
        "realtime": service.realtime_status,
    }
