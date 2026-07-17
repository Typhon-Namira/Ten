"""Read-only SMC Milestone 2A API."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.dependencies import get_smc_service
from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.smc_engine import (
    ProcessingMode,
    SMCAnalysisSnapshot,
    SMCService,
    StructureDirection,
    StructureEvent,
    StructureEventType,
    StructureScope,
    SwingPoint,
)
from backend.app.engines.smc_engine.exceptions import SMCError

router = APIRouter(prefix="/smc", tags=["smc"])
Service = Annotated[SMCService, Depends(get_smc_service)]


async def _snapshot(service: SMCService, symbol: str, timeframe: Timeframe, at: datetime | None = None, limit: int = 500) -> SMCAnalysisSnapshot:
    existing = await service.state(symbol, timeframe, at)
    if existing is not None:
        return existing
    try:
        return await service.replay(symbol, timeframe, at, limit=limit) if at else await service.analyze(symbol, timeframe, limit=limit)
    except SMCError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="SMC source market data is unavailable") from exc


@router.get("/state")
async def smc_state(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return (await _snapshot(service, symbol, timeframe, timestamp)).structure_state


@router.get("/swings", response_model=list[SwingPoint])
async def smc_swings(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None, scope: StructureScope | None = None, minimum_confidence: Annotated[float, Query(ge=0, le=100)] = 0, limit: Annotated[int, Query(ge=1, le=5000)] = 500) -> list[SwingPoint]:
    items = (await _snapshot(service, symbol, timeframe, timestamp, limit)).swings
    return [item for item in items if (scope is None or item.scope == scope) and item.confidence_score >= minimum_confidence][-limit:]


@router.get("/structure")
async def smc_structure(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> dict[str, object]:
    snapshot = await _snapshot(service, symbol, timeframe, timestamp)
    return {"state": snapshot.structure_state, "legs": snapshot.structure_legs}


@router.get("/events", response_model=list[StructureEvent])
async def smc_events(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None, event_type: StructureEventType | None = None, scope: StructureScope | None = None, direction: StructureDirection | None = None, minimum_confidence: Annotated[float, Query(ge=0, le=100)] = 0, limit: Annotated[int, Query(ge=1, le=5000)] = 500) -> list[StructureEvent]:
    items = (await _snapshot(service, symbol, timeframe, timestamp, limit)).structure_events
    return [item for item in items if (event_type is None or item.event_type == event_type) and (scope is None or item.scope == scope) and (direction is None or item.direction == direction) and item.confidence_score >= minimum_confidence][-limit:]


@router.get("/snapshot", response_model=SMCAnalysisSnapshot)
async def smc_snapshot(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None, limit: Annotated[int, Query(ge=1, le=5000)] = 500) -> SMCAnalysisSnapshot:
    return await _snapshot(service, symbol, timeframe, timestamp, limit)


@router.get("/replay", response_model=SMCAnalysisSnapshot)
async def smc_replay(service: Service, timestamp: datetime, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, limit: Annotated[int, Query(ge=1, le=5000)] = 500) -> SMCAnalysisSnapshot:
    return await service.replay(symbol, timeframe, timestamp, limit=limit)


@router.get("/health")
async def smc_health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/metrics")
async def smc_metrics(service: Service) -> dict[str, int | float]:
    return service.metrics.snapshot()


@router.get("/config")
async def smc_config(service: Service) -> dict[str, object]:
    return {"configuration": service.config.model_dump(mode="json"), "configuration_version": service.config.version, "engine_version": service.analyzer.version, "processing_modes": [item.value for item in ProcessingMode]}
