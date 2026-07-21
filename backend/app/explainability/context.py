"""Assembles the grounded `ExplainabilityContext` from already-persisted TEN state.

This is the entire "grounding" boundary: every fact the LLM ever sees is produced here, by plain
Python reading real repositories/services — never by the model itself. A source that fails or has
no snapshot yet becomes `EngineFact(available=False, error=...)`, never a silently-omitted field
and never a fabricated placeholder.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.api.safe import safe_call
from backend.app.engines.market_data_engine import Timeframe

from .models import EngineFact, Evidence, ExplainabilityContext

MAX_LIST_ITEMS = 6
MAX_DEPTH = 3


def _trim(value: Any, depth: int = 0) -> Any:
    """Bounds the size of a snapshot dump so the grounding bundle stays legible to both the model
    and a human reading the raw context — truncates deep/huge structures with an explicit
    "...and N more" marker rather than silently dropping them, so trimming is never mistaken for
    "this data doesn't exist"."""
    if depth >= MAX_DEPTH:
        return "…" if isinstance(value, (dict, list, tuple)) else value
    if isinstance(value, dict):
        return {key: _trim(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_trim(item, depth + 1) for item in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            items.append(f"…and {len(value) - MAX_LIST_ITEMS} more")
        return items
    return value


def _summarize(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    dumped = model.model_dump(mode="json") if hasattr(model, "model_dump") else model
    trimmed = _trim(dumped)
    return trimmed if isinstance(trimmed, dict) else {"value": trimmed}


async def _engine_fact(engine: str, fetch: Any, id_field: str = "id", timestamp_field: str = "analysis_timestamp") -> EngineFact:
    value, error = await safe_call(fetch)
    if error is not None:
        return EngineFact(engine=engine, available=False, error=error)
    if value is None:
        return EngineFact(engine=engine, available=False, error="no snapshot persisted yet")
    reference = next((getattr(value, name, None) for name in (id_field, "snapshot_id", "decision_id", "context_id", "id")), None)
    timestamp = next((getattr(value, name, None) for name in (timestamp_field, "as_of", "analysis_timestamp", "created_at") if getattr(value, name, None) is not None), None)
    evidence = Evidence(source=f"{engine}_snapshot", reference_id=str(reference) if reference is not None else "unknown", timestamp=timestamp)
    return EngineFact(engine=engine, available=True, summary=_summarize(value) or {}, evidence=evidence)


async def build_context(
    app: Any,
    *,
    instrument: str,
    timeframe: str,
    question: str | None = None,
    decision: Any = None,
    as_of: datetime | None = None,
) -> ExplainabilityContext:
    """The one place every `/explain/*` endpoint gets its grounding from — identical for a
    dashboard "explain current state" call and a chat question, so an answer in the chat can never
    disagree with what the explanation panels show for the same instrument/timeframe.

    `decision`, when given (by `/explain/decision/{id}` and `/explain/rejection/{id}`), replaces
    the "most recent decision" lookup with that exact, already-fetched `SignalDecision` — grounding
    an explanation of a specific past decision, not whatever is currently latest. `as_of` then
    defaults to that decision's own timestamp so every engine snapshot reflects the state the
    pipeline actually saw at decision time, not the state right now.
    """
    symbol = instrument.upper()
    tf = Timeframe(timeframe)
    now = datetime.now(UTC)
    snapshot_at = as_of or (decision.as_of if decision is not None else None)

    smc, liquidity, volume_profile, institutional_flow, market_regime = [
        await _engine_fact(name, fetch)
        for name, fetch in (
            ("smc", lambda: app.state.smc_service.state(symbol, tf, snapshot_at)),
            ("liquidity", lambda: app.state.liquidity_service.state(symbol, tf, snapshot_at)),
            ("volume_profile", lambda: app.state.volume_profile_service.state(symbol, tf, snapshot_at)),
            ("institutional_flow", lambda: app.state.institutional_flow_service.state(symbol, tf, snapshot_at)),
            ("market_regime", lambda: app.state.market_regime_service.state(symbol, tf, snapshot_at)),
        )
    ]
    economic = await _engine_fact("economic_calendar", lambda: app.state.economic_calendar_service.context(symbol, as_of=snapshot_at or now, publish=False), timestamp_field="analysis_timestamp")
    # The context alone only says "unavailable" — the model needs the actual provider telemetry
    # (which provider, connection state, HTTP status, failure reason) to explain WHY, instead of
    # repeating the bare word "unavailable" the way it used to.
    provider_statuses, _ = await safe_call(lambda: app.state.economic_calendar_service.provider_status())
    if provider_statuses:
        economic = economic.model_copy(update={"summary": {**economic.summary, "provider_status": [_summarize(item) for item in provider_statuses]}})

    if decision is None:
        decisions, decision_error = await safe_call(lambda: app.state.signal_decision_service.repository.find_recent_decisions(symbol, timeframe, now, 1))
        decision = decisions[0] if decisions else None
    else:
        decision_error = None

    if decision is not None and getattr(decision, "ai_score_snapshot_id", None) is not None:
        ai_score, ai_error = await safe_call(lambda: app.state.ai_scoring_service.repository.get_snapshot(decision.ai_score_snapshot_id))
    else:
        ai_score, ai_error = await safe_call(lambda: app.state.ai_scoring_service.repository.get_latest_snapshot(symbol, timeframe))
    # `list_snapshots` sorts newest-first; offset=0/limit=2 gets [current, previous] in one call —
    # index 1 (not 0) is genuinely the *previous* snapshot, not the current one repeated.
    recent_scores, _ = await safe_call(lambda: app.state.ai_scoring_service.repository.list_snapshots(symbol, timeframe, None, None, None, None, None, 0, 2))
    previous_score = recent_scores[1] if recent_scores and len(recent_scores) > 1 else None

    ai_score_fact = EngineFact(
        engine="ai_scoring",
        available=ai_score is not None,
        summary=_summarize(ai_score) or {},
        evidence=Evidence(source="ai_score", reference_id=str(ai_score.snapshot_id), timestamp=ai_score.as_of) if ai_score else None,
        error=ai_error or ("no AI score persisted yet" if ai_score is None else None),
    )
    decision_fact = EngineFact(
        engine="signal_decision",
        available=decision is not None,
        summary=_summarize(decision) or {},
        evidence=Evidence(source="decision", reference_id=str(decision.decision_id), timestamp=decision.as_of) if decision else None,
        error=decision_error or ("no decision persisted yet" if decision is None else None),
    )

    stage = app.state.pipeline_stage_tracker.latest(symbol, timeframe)

    return ExplainabilityContext(
        instrument=symbol,
        timeframe=timeframe,
        generated_at=now,
        engines=[smc, liquidity, volume_profile, institutional_flow, market_regime, economic, ai_score_fact, decision_fact],
        ai_score=_summarize(ai_score),
        decision=_summarize(decision),
        pipeline_stage=_trim(stage) if stage else None,
        previous_ai_score=_summarize(previous_score),
        question=question,
    )
