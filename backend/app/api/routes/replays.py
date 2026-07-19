from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.api.dependencies import get_replay_service
from backend.app.engines.replay_engine import ReplayComparison, ReplayRequest, ReplayService, ReplaySession, ReplayStatus, ReplaySummary
from backend.app.engines.replay_engine.exceptions import ReplayConcurrencyError, ReplayNotFound, ReplayTransitionError, ReplayValidationError

router = APIRouter(prefix="/replays", tags=["replays"])
Service = Annotated[ReplayService, Depends(get_replay_service)]


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    units: int = Field(default=1, ge=1, le=1000)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_replay_id: UUID
    right_replay_id: UUID


async def _domain[T](call: Awaitable[T]) -> T:
    try:
        return await call
    except ReplayNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ReplayTransitionError, ReplayConcurrencyError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except ReplayValidationError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/config")
async def config(service: Service) -> dict[str, object]:
    return {
        "configuration": service.config.model_dump(mode="json"),
        "datasets": [item.model_dump(mode="json") for item in service.coordinator.datasets.datasets()],
        "engines": [item.engine_name for item in service.coordinator.engines.registrations()],
        "sources": service.coordinator.sources.names(),
    }


@router.get("/metrics")
async def metrics(service: Service) -> dict[str, object]:
    return service.metrics.snapshot()


@router.post("", response_model=ReplaySession, status_code=status.HTTP_201_CREATED)
async def create(request: ReplayRequest, service: Service) -> object:
    return await _domain(service.create(request))


@router.get("", response_model=list[ReplaySession])
async def sessions(
    service: Service,
    replay_status: ReplayStatus | None = Query(default=None, alias="status"),
    cursor: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> tuple[ReplaySession, ...]:
    return await service.list(cursor, limit, replay_status)


@router.post("/compare", response_model=ReplayComparison)
async def compare(request: CompareRequest, service: Service) -> object:
    return await _domain(service.compare(request.left_replay_id, request.right_replay_id))


@router.post("/{replay_id}/start", response_model=ReplaySession, status_code=status.HTTP_202_ACCEPTED)
async def start(replay_id: UUID, service: Service) -> object:
    return await _domain(service.command_start(replay_id))


@router.post("/{replay_id}/pause", response_model=ReplaySession, status_code=status.HTTP_202_ACCEPTED)
async def pause(replay_id: UUID, service: Service) -> object:
    return await _domain(service.pause(replay_id))


@router.post("/{replay_id}/resume", response_model=ReplaySession, status_code=status.HTTP_202_ACCEPTED)
async def resume(replay_id: UUID, service: Service) -> object:
    return await _domain(service.resume(replay_id))


@router.post("/{replay_id}/cancel", response_model=ReplaySession, status_code=status.HTTP_202_ACCEPTED)
async def cancel(replay_id: UUID, service: Service) -> object:
    return await _domain(service.cancel(replay_id))


@router.post("/{replay_id}/step", response_model=ReplaySession)
async def step(replay_id: UUID, request: StepRequest, service: Service) -> object:
    return await _domain(service.step(replay_id, request.units))


@router.get("/{replay_id}", response_model=ReplaySession)
async def session(replay_id: UUID, service: Service) -> object:
    return await _domain(service.get(replay_id))


@router.get("/{replay_id}/checkpoints")
async def checkpoints(replay_id: UUID, service: Service, cursor: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 100) -> object:
    await _domain(service.get(replay_id))
    return await service.repository.list_checkpoints(replay_id, cursor, limit)


@router.get("/{replay_id}/transitions")
async def transitions(replay_id: UUID, service: Service, cursor: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 100) -> object:
    await _domain(service.get(replay_id))
    return await service.repository.list_transitions(replay_id, cursor, limit)


@router.get("/{replay_id}/summary", response_model=ReplaySummary)
async def summary(replay_id: UUID, service: Service) -> object:
    return await _domain(service.summary(replay_id))


@router.get("/{replay_id}/outputs")
async def outputs(replay_id: UUID, service: Service, output_type: str | None = None, cursor: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 100) -> object:
    await _domain(service.get(replay_id))
    return await service.repository.list_outputs(replay_id, output_type, cursor, limit)


@router.get("/{replay_id}/ai-scores")
async def ai_scores(replay_id: UUID, service: Service, cursor: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 100) -> object:
    await _domain(service.get(replay_id))
    return await service.repository.list_outputs(replay_id, "ai_score", cursor, limit)


@router.get("/{replay_id}/signal-decisions")
async def signal_decisions(replay_id: UUID, service: Service, cursor: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 100) -> object:
    await _domain(service.get(replay_id))
    return await service.repository.list_outputs(replay_id, "signal_decision", cursor, limit)


@router.get("/{replay_id}/trace")
async def trace(replay_id: UUID, service: Service, cursor: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 100) -> object:
    if not service.config.trace.enabled:
        raise HTTPException(404, "replay trace is disabled")
    await _domain(service.get(replay_id))
    return await service.repository.list_trace(replay_id, cursor, limit)
