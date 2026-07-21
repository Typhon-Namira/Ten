"""AI Explainability API — natural-language explanations grounded entirely in TEN's own,
already-persisted engine outputs. See `backend.app.explainability` for the grounding/prose
separation this router relies on: every route below assembles a context with plain Python, then
asks the LLM only to write prose about it. A failing OpenRouter call degrades to
`explanation: null` + a reported `error`, never a 500 and never a fabricated explanation.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.core.exceptions import ConfigurationError, ExternalServiceError
from backend.app.explainability import ChatTurn
from backend.app.explainability.context import build_context

from .system import _default_selection

router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])


async def _explain_response(request: Request, *, instrument: str, timeframe: str, question: str | None = None, decision: object | None = None) -> dict[str, object]:
    app = request.app
    context = await build_context(app, instrument=instrument, timeframe=timeframe, question=question, decision=decision)
    explanation = None
    error = None
    try:
        explanation = await app.state.explainability_service.explain(context)
    except (ConfigurationError, ExternalServiceError) as exc:
        error = str(exc)
    except Exception as exc:  # the model returned a shape Explanation can't validate — degrade, don't 500
        error = f"{type(exc).__name__}: could not parse the model's response"
    return {
        "instrument": context.instrument,
        "timeframe": context.timeframe,
        "generated_at": context.generated_at,
        "explanation": explanation.model_dump(mode="json") if explanation else None,
        "error": error,
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
    explanation = None
    error = None
    try:
        explanation = await app.state.explainability_service.chat(context, tuple(body.history))
    except (ConfigurationError, ExternalServiceError) as exc:
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: could not parse the model's response"
    return {
        "instrument": context.instrument,
        "timeframe": context.timeframe,
        "generated_at": context.generated_at,
        "explanation": explanation.model_dump(mode="json") if explanation else None,
        "error": error,
        "explainability_score": context.explainability_score(),
        "evidence": [item.model_dump(mode="json") for item in context.evidence_list()],
    }
