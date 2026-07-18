from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import get_institutional_flow_service
from backend.app.engines.institutional_flow_engine import InstitutionalFlowAnalysisSnapshot, InstitutionalFlowService
from backend.app.engines.market_data_engine import Timeframe

router = APIRouter(prefix="/institutional-flow", tags=["institutional-flow"])
Service = Annotated[InstitutionalFlowService, Depends(get_institutional_flow_service)]


async def _snapshot(service: InstitutionalFlowService, symbol: str, timeframe: Timeframe, timestamp: datetime | None, limit: int) -> InstitutionalFlowAnalysisSnapshot:
    existing = await service.state(symbol, timeframe, timestamp)
    if existing:
        return existing
    try:
        return await service.replay(symbol, timeframe, timestamp, limit) if timestamp else await service.analyze(symbol, timeframe, limit=limit)
    except Exception as exc:
        raise HTTPException(503, "Institutional Flow source data is unavailable") from exc


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/metrics")
async def metrics(service: Service) -> dict[str, int | float | str | None]:
    return service.metrics.snapshot()


@router.get("/config")
async def config(service: Service) -> dict[str, object]:
    return {"configuration": service.config.model_dump(mode="json"), "configuration_version": service.config.version, "engine_version": service.analyzer.version}


@router.get("/snapshot", response_model=InstitutionalFlowAnalysisSnapshot)
async def snapshot(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None, limit: Annotated[int, Query(ge=1, le=100000)] = 500) -> InstitutionalFlowAnalysisSnapshot:
    return await _snapshot(service, symbol, timeframe, timestamp, limit)


@router.get("/state")
async def state(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return await _snapshot(service, symbol, timeframe, timestamp, 500)


@router.get("/replay", response_model=InstitutionalFlowAnalysisSnapshot)
async def replay(service: Service, timestamp: datetime, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, limit: Annotated[int, Query(ge=1, le=100000)] = 500) -> InstitutionalFlowAnalysisSnapshot:
    return await service.replay(symbol, timeframe, timestamp, limit)


def derived(path: str, attribute: str) -> None:
    async def endpoint(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
        item = await _snapshot(service, symbol, timeframe, timestamp, 500)
        return getattr(item.state, attribute)

    endpoint.__name__ = "institutional_flow_" + attribute
    router.add_api_route(path, endpoint, methods=["GET"])


for _path, _attribute in (
    ("/participation", "participation"),
    ("/initiative", "initiative"),
    ("/responsive", "responsive"),
    ("/absorption", "absorption"),
    ("/exhaustion", "exhaustion"),
    ("/inventory", "inventory"),
    ("/campaign", "campaign"),
    ("/pressure", "pressure"),
    ("/persistence", "persistence"),
    ("/cross-session", "cross_session"),
    ("/confluences", "confluences"),
    ("/explanation", "explanation"),
):
    derived(_path, _attribute)


@router.get("/evidence")
async def evidence(
    service: Service,
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M15,
    timestamp: datetime | None = None,
    source_engine: str | None = None,
    evidence_type: str | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[object]:
    item = await _snapshot(service, symbol, timeframe, timestamp, min(100000, offset + limit))
    values = list(item.evidence.accepted)
    if source_engine:
        values = [value for value in values if value.source_engine.value == source_engine]
    if evidence_type:
        values = [value for value in values if value.evidence_type.value == evidence_type]
    return list(values[offset : offset + limit])


@router.get("/mtf")
async def mtf(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return await service.multi_timeframe(symbol, timeframe, timestamp)
