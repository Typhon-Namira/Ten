from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.dependencies import get_integration_service
from backend.app.core.security import require_admin, require_viewer
from backend.app.integration import FullSystemIntegrationService, IntegrationConfig, IntegrationTraceRecord, OperationalSignal

router = APIRouter(prefix="/integration", tags=["full-system-integration"])


@router.get("/health")
async def health(service: FullSystemIntegrationService = Depends(get_integration_service)) -> dict[str, object]:
    return service.health()


@router.get("/readiness")
async def readiness(service: FullSystemIntegrationService = Depends(get_integration_service)) -> dict[str, object]:
    health = service.health()
    if not health["ready"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health)
    return health


@router.get("/signals", response_model=list[OperationalSignal], dependencies=[Depends(require_viewer)])
async def signals(limit: int = Query(default=100, ge=1, le=200), service: FullSystemIntegrationService = Depends(get_integration_service)) -> tuple[OperationalSignal, ...]:
    return await service.repository.signals(limit)


@router.get("/signals/latest", response_model=OperationalSignal, dependencies=[Depends(require_viewer)])
async def latest_signal(instrument: str | None = None, timeframe: str | None = None, service: FullSystemIntegrationService = Depends(get_integration_service)) -> OperationalSignal:
    value = await service.repository.latest_signal(instrument.upper() if instrument else None, timeframe)
    if value is None:
        raise HTTPException(status_code=404, detail="No integrated operational signal is available")
    return value


@router.get("/traces/{trace_id}", response_model=list[IntegrationTraceRecord], dependencies=[Depends(require_admin)])
async def trace(trace_id: UUID, service: FullSystemIntegrationService = Depends(get_integration_service)) -> tuple[IntegrationTraceRecord, ...]:
    return await service.repository.trace(trace_id)


@router.get("/graph", response_model=IntegrationConfig, dependencies=[Depends(require_admin)])
async def graph(service: FullSystemIntegrationService = Depends(get_integration_service)) -> IntegrationConfig:
    return service.config


@router.post("/outbox/run-once", dependencies=[Depends(require_admin)])
async def run_once(service: FullSystemIntegrationService = Depends(get_integration_service)) -> dict[str, int]:
    return {"processed": await service.process_outbox_once()}
