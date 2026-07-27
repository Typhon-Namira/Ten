"""Read-only explainability projections over already-persisted TEN evidence."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.explainability import ChatTurn
from backend.app.explainability.models import EngineInfluence, Explanation
from backend.app.explainability.context import build_context

from .system import _default_selection

router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])


def _persisted_explanation(context: object) -> Explanation:
    engines = list(getattr(context, "engines", ()))
    available = [item for item in engines if item.available]
    unavailable = [item for item in engines if not item.available]
    decision = getattr(context, "decision", None)
    decision_state = (
        str(decision.get("state", "not available"))
        if isinstance(decision, dict)
        else "not available"
    )
    return Explanation(
        summary=(
            f"Persisted TEN evidence reports {len(available)} of {len(engines)} "
            f"analytical engines available; decision state is {decision_state}."
        ),
        primary_reasons=[
            f"{item.engine} has persisted evidence for this cycle."
            for item in available[:5]
        ],
        opposing_factors=[
            f"{item.engine} is unavailable: {item.error or 'no persisted evidence'}."
            for item in unavailable[:5]
        ],
        engine_breakdown=[
            EngineInfluence(
                engine=item.engine,
                influence="available" if item.available else "unavailable",
                note=(
                    "Persisted evidence is available."
                    if item.available
                    else item.error or "No persisted evidence is available."
                ),
            )
            for item in engines
        ],
        required_for_change=[],
        caveats=[
            "This explanation is a deterministic read of persisted evidence; "
            "dashboard requests never call an AI provider."
        ],
    )


async def _explain_response(request: Request, *, instrument: str, timeframe: str, question: str | None = None, decision: object | None = None) -> dict[str, object]:
    app = request.app
    context = await build_context(app, instrument=instrument, timeframe=timeframe, question=question, decision=decision)
    explanation = _persisted_explanation(context)
    return {
        "instrument": context.instrument,
        "timeframe": context.timeframe,
        "generated_at": context.generated_at,
        "explanation": explanation.model_dump(mode="json"),
        "error": None,
        "explainability_score": context.explainability_score(),
        "evidence": [item.model_dump(mode="json") for item in context.evidence_list()],
        "engines": [item.model_dump(mode="json") for item in context.engines],
    }


@router.get("/current")
async def explain_current(request: Request, instrument: str | None = None, timeframe: str | None = None) -> dict[str, object]:
    """Explains the current pipeline/engine state for one instrument+timeframe."""
    default_instrument, default_timeframe = _default_selection(request)
    return await _explain_response(request, instrument=(instrument or default_instrument).upper(), timeframe=timeframe or default_timeframe)


@router.get("/decision/{decision_id}")
async def explain_decision(request: Request, decision_id: UUID) -> dict[str, object]:
    """Explains one specific decision — grounded on the engine state as of that decision's own
    timestamp, not whatever is currently live, so the explanation matches what the pipeline
    actually saw when it decided."""
    service = request.app.state.signal_decision_service
    decision = await service.get_decision(decision_id)
    if decision is None:
        raise HTTPException(404, "Signal Decision not found")
    return await _explain_response(request, instrument=decision.instrument, timeframe=decision.timeframe, decision=decision)


@router.get("/rejection/{decision_id}")
async def explain_rejection(request: Request, decision_id: UUID) -> dict[str, object]:
    """Same as `/decision/{id}` but only for a decision that was actually rejected — returns 422
    for an eligible decision instead of silently explaining "why" something that wasn't blocked."""
    service = request.app.state.signal_decision_service
    decision = await service.get_decision(decision_id)
    if decision is None:
        raise HTTPException(404, "Signal Decision not found")
    if decision.state.value == "eligible":
        raise HTTPException(422, "this decision was not rejected")
    question = "Why was this scenario rejected, and what would need to change for it to become valid?"
    return await _explain_response(request, instrument=decision.instrument, timeframe=decision.timeframe, question=question, decision=decision)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)
    instrument: str | None = None
    timeframe: str | None = None


@router.post("/chat")
async def explain_chat(request: Request, body: ChatRequest) -> dict[str, object]:
    """Free-form Q&A grounded in the same context every other `/explain/*` route uses for the same
    instrument/timeframe — so a chat answer can never disagree with the dashboard's own panels."""
    app = request.app
    default_instrument, default_timeframe = _default_selection(request)
    instrument = (body.instrument or default_instrument).upper()
    timeframe = body.timeframe or default_timeframe
    context = await build_context(app, instrument=instrument, timeframe=timeframe, question=body.message)
    explanation = _persisted_explanation(context)
    return {
        "instrument": context.instrument,
        "timeframe": context.timeframe,
        "generated_at": context.generated_at,
        "explanation": explanation.model_dump(mode="json"),
        "error": None,
        "explainability_score": context.explainability_score(),
        "evidence": [item.model_dump(mode="json") for item in context.evidence_list()],
    }
