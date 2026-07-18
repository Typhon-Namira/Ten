from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import get_market_regime_service
from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_regime_engine import MarketRegimeService, MarketRegimeSnapshot

router = APIRouter(prefix="/market-regime", tags=["market-regime"])
Service = Annotated[MarketRegimeService, Depends(get_market_regime_service)]


async def _state(service: MarketRegimeService, symbol: str, timeframe: Timeframe, timestamp: datetime | None) -> MarketRegimeSnapshot:
    value = await service.state(symbol, timeframe, timestamp)
    if value:
        return value
    try:
        return await service.replay(symbol, timeframe, timestamp) if timestamp else await service.analyze_snapshot(symbol, timeframe)
    except Exception as exc:
        raise HTTPException(503, "Market Regime source data is unavailable") from exc


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/config")
async def config(service: Service) -> dict[str, object]:
    return {"configuration": service.config.model_dump(mode="json"), "configuration_version": service.config.version, "engine_version": service.analyzer.version}


@router.get("/metrics")
async def metrics(service: Service) -> dict[str, object]:
    return service.metrics.snapshot()


@router.get("/state", response_model=MarketRegimeSnapshot)
async def state(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> MarketRegimeSnapshot:
    return await _state(service, symbol, timeframe, timestamp)


@router.get("/history", response_model=list[MarketRegimeSnapshot])
async def history(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[MarketRegimeSnapshot]:
    return list(await service.history(symbol, timeframe, offset, limit))


@router.get("/snapshots/{snapshot_id}", response_model=MarketRegimeSnapshot)
async def snapshot(service: Service, snapshot_id: str) -> MarketRegimeSnapshot:
    value = await service.repository.get_snapshot(snapshot_id)
    if not value:
        from uuid import UUID

        try:
            value = await service.repository.get_snapshot(UUID(snapshot_id))
        except ValueError:
            value = None
    if not value:
        raise HTTPException(404, "Market Regime snapshot not found")
    return value


def derived(path: str, attribute: str) -> None:
    async def endpoint(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
        return getattr(await _state(service, symbol, timeframe, timestamp), attribute)

    endpoint.__name__ = "market_regime_" + attribute
    router.add_api_route(path, endpoint, methods=["GET"])


for _path, _attribute in (
    ("/trend", "trend_regime"),
    ("/volatility", "volatility_regime"),
    ("/auction", "auction_regime"),
    ("/compression", "compression_score"),
    ("/expansion", "expansion_regime"),
    ("/persistence", "persistence"),
    ("/sessions", "cross_session"),
    ("/explanations", "reasoning_summary"),
):
    derived(_path, _attribute)


@router.get("/transitions")
async def transitions(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[object]:
    return list(await service.repository.list_transitions(symbol, timeframe, offset, limit))


@router.get("/mtf")
async def mtf(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return await service.multi_timeframe(symbol, timeframe, timestamp)


@router.get("/evidence")
async def evidence(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=1000)] = 100, source_engine: str | None = None, family: str | None = None) -> list[object]:
    values = list(await service.repository.list_evidence(symbol, timeframe, 0, min(1000, offset + limit)))
    if source_engine:
        values = [item for item in values if item.source_engine == source_engine]
    if family:
        values = [item for item in values if item.family.value == family]
    return list(values[offset : offset + limit])
