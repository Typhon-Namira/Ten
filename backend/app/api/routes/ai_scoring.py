from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import get_ai_scoring_service
from backend.app.engines.ai_scoring_engine import AIScoreSnapshot, AIScoringService, ScoreMode, ScoreRequest, ScoreStatus

router = APIRouter(prefix="/ai-scoring", tags=["ai-scoring"])
Service = Annotated[AIScoringService, Depends(get_ai_scoring_service)]


def _aware(value: datetime | None, name: str) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(422, f"{name} must include a timezone")
    return value.astimezone(UTC) if value else None


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/config")
async def config(service: Service) -> dict[str, object]:
    return {"configuration": service.config.model_dump(mode="json"), "configuration_hash": service.engine.configuration_hash}


@router.get("/metrics")
async def metrics(service: Service) -> dict[str, object]:
    return service.metrics.snapshot()


@router.post("/score", response_model=AIScoreSnapshot)
async def score(request: ScoreRequest, service: Service) -> AIScoreSnapshot:
    try:
        return await service.calculate(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/replay", response_model=AIScoreSnapshot)
async def replay(request: ScoreRequest, service: Service) -> AIScoreSnapshot:
    if request.as_of is None:
        raise HTTPException(422, "replay requires as_of")
    replay_request = request.model_copy(update={"mode": ScoreMode.REPLAY, "publish_events": False})
    try:
        return await service.calculate(replay_request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/latest", response_model=AIScoreSnapshot)
async def latest(service: Service, instrument: str = "XAUUSD", timeframe: str = "M15", policy_version: str | None = None) -> AIScoreSnapshot:
    value = await service.repository.get_latest_snapshot(instrument.upper(), timeframe, policy_version)
    if value is None:
        raise HTTPException(404, "AI Scoring snapshot not found")
    return value


@router.get("/history", response_model=list[AIScoreSnapshot])
async def history(
    service: Service,
    instrument: str = "XAUUSD",
    timeframe: str = "M15",
    start: datetime | None = None,
    end: datetime | None = None,
    status: ScoreStatus | None = None,
    policy_version: str | None = None,
    mode: ScoreMode | None = None,
    cursor: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[AIScoreSnapshot]:
    start = _aware(start, "start")
    end = _aware(end, "end")
    if start and end and start > end:
        raise HTTPException(422, "start must precede end")
    if start and end and end - start > timedelta(days=service.config.api.maximum_history_days):
        raise HTTPException(422, "history range exceeds configured maximum")
    bounded_limit = min(limit, service.config.api.maximum_page_size)
    return list(await service.repository.list_snapshots(instrument.upper(), timeframe, start, end, status, policy_version, mode, cursor, bounded_limit))


@router.get("/snapshots/{snapshot_id}", response_model=AIScoreSnapshot)
async def snapshot(snapshot_id: UUID, service: Service) -> AIScoreSnapshot:
    value = await service.repository.get_snapshot(snapshot_id)
    if value is None:
        raise HTTPException(404, "AI Scoring snapshot not found")
    return value


@router.get("/snapshots/{snapshot_id}/explanation")
async def explanation(snapshot_id: UUID, service: Service) -> object:
    return (await snapshot(snapshot_id, service)).explanation
