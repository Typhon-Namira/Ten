from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.app.api.dependencies import get_signal_decision_service
from backend.app.api.rejection_diagnostics import build_rejection_diagnostics
from backend.app.api.safe import safe_call
from backend.app.engines.signal_decision_engine import (
    DecisionDirection,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    SignalDecision,
    SignalDecisionConfigurationError,
    SignalDecisionInputError,
    SignalDecisionService,
    SignalDecisionSnapshotNotFound,
)

router = APIRouter(prefix="/signal-decisions", tags=["signal-decisions"])
Service = Annotated[SignalDecisionService, Depends(get_signal_decision_service)]


def _aware(value: datetime | None, name: str) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(422, f"{name} must include a timezone")
    return value.astimezone(UTC) if value else None


async def _evaluate(request: DecisionRequest, service: SignalDecisionService) -> SignalDecision:
    try:
        return await service.evaluate(request)
    except SignalDecisionSnapshotNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (SignalDecisionInputError, SignalDecisionConfigurationError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/config")
async def config(service: Service) -> dict[str, object]:
    return {"configuration": service.config.model_dump(mode="json"), "configuration_hash": service.policy.configuration_hash}


@router.get("/metrics")
async def metrics(service: Service) -> dict[str, object]:
    return service.metrics.snapshot()


@router.post("/evaluate", response_model=SignalDecision)
async def evaluate(request: DecisionRequest, service: Service) -> SignalDecision:
    return await _evaluate(request, service)


@router.post("/replay", response_model=SignalDecision)
async def replay(request: DecisionRequest, service: Service) -> SignalDecision:
    if request.as_of is None:
        raise HTTPException(422, "replay requires as_of")
    replay_request = request.model_copy(update={"mode": DecisionMode.REPLAY, "publish_events": False})
    return await _evaluate(replay_request, service)


@router.get("/latest", response_model=SignalDecision)
async def latest(
    service: Service,
    instrument: str = "XAUUSD",
    timeframe: str = "M15",
    direction: DecisionDirection | None = None,
    state: DecisionState | None = None,
) -> SignalDecision:
    value = await service.repository.get_active_decision(instrument.upper(), timeframe, service.clock.now(), direction, state)
    if value is None:
        raise HTTPException(404, "active Signal Decision not found")
    return value


@router.get("/history", response_model=list[SignalDecision])
async def history(
    service: Service,
    instrument: str = "XAUUSD",
    timeframe: str = "M15",
    direction: DecisionDirection | None = None,
    state: DecisionState | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    decision_policy_version: str | None = None,
    ai_score_policy_version: str | None = None,
    mode: DecisionMode | None = None,
    cursor: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[SignalDecision]:
    start = _aware(start, "start")
    end = _aware(end, "end")
    if start and end and start > end:
        raise HTTPException(422, "start must precede end")
    if start and end and end - start > timedelta(days=service.config.api.maximum_history_days):
        raise HTTPException(422, "history range exceeds configured maximum")
    bounded = min(limit, service.config.api.maximum_page_size)
    return list(await service.repository.list_decisions(instrument.upper(), timeframe, start, end, direction, state, decision_policy_version, ai_score_policy_version, mode, cursor, bounded))


@router.get("/rejections/recent")
async def rejections_recent(
    request: Request,
    service: Service,
    instrument: str = "XAUUSD",
    timeframe: str = "M15",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> dict[str, object]:
    """Recent non-eligible decisions with a full, human-readable breakdown of every failed condition."""
    now = service.clock.now()
    decisions = await service.repository.find_recent_decisions(instrument.upper(), timeframe, now, limit)
    rejected = [item for item in decisions if item.state.value != "eligible"]
    app = request.app
    market = app.state.market_data_service
    try:
        session_status = market.sessions.status_at(now)
        session_error = None
    except Exception as exc:
        session_status, session_error = None, type(exc).__name__
    active_session = session_status.active_session if session_status else None
    items = []
    for item in rejected:
        score, score_error = await safe_call(lambda item=item: app.state.ai_scoring_service.repository.get_snapshot(item.ai_score_snapshot_id))
        diagnostics = build_rejection_diagnostics(item, score, active_session)
        items.append(
            {
                "decision_id": str(item.decision_id),
                "instrument": item.instrument,
                "timeframe": item.timeframe,
                "state": item.state.value,
                "direction": item.direction.value,
                "as_of": item.as_of,
                "confidence_score": item.confidence_score,
                "blockers": [reason.reason_code for reason in item.blockers],
                "warnings": [reason.reason_code for reason in item.warnings],
                "ai_score_unavailable": score_error,
                "diagnostics": diagnostics,
            }
        )
    return {"instrument": instrument.upper(), "timeframe": timeframe, "count": len(items), "session_status_error": session_error, "rejections": items}


async def _decision(decision_id: UUID, service: SignalDecisionService) -> SignalDecision:
    value = await service.get_decision(decision_id)
    if value is None:
        raise HTTPException(404, "Signal Decision not found")
    return value


@router.get("/{decision_id}", response_model=SignalDecision)
async def decision(decision_id: UUID, service: Service) -> SignalDecision:
    return await _decision(decision_id, service)


@router.get("/{decision_id}/rules")
async def rules(decision_id: UUID, service: Service) -> object:
    return (await _decision(decision_id, service)).rules


@router.get("/{decision_id}/explanation")
async def explanation(decision_id: UUID, service: Service) -> object:
    return (await _decision(decision_id, service)).explanation
