"""Per-candle pipeline stage visibility (Part 3): every named stage, including ones that
produce no signal, with an explicit Running/Waiting/Success/Failed/Skipped status."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.api.routes.system import _default_selection

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.get("/stages/latest")
async def latest_stages(request: Request, instrument: str | None = None, timeframe: str | None = None) -> dict[str, object]:
    default_instrument, default_timeframe = _default_selection(request)
    symbol = (instrument or default_instrument).upper()
    resolved_timeframe = timeframe or default_timeframe
    tracker = request.app.state.pipeline_stage_tracker
    cycle = tracker.latest(symbol, resolved_timeframe)
    if cycle is None:
        return {"symbol": symbol, "timeframe": resolved_timeframe, "available": False, "reason": "no_candle_processed_yet", "stages": []}
    return {"available": True, **cycle}


@router.get("/stages/recent")
async def recent_stages(request: Request, instrument: str | None = None, timeframe: str | None = None, limit: int = 5) -> dict[str, object]:
    default_instrument, default_timeframe = _default_selection(request)
    symbol = (instrument or default_instrument).upper()
    resolved_timeframe = timeframe or default_timeframe
    tracker = request.app.state.pipeline_stage_tracker
    cycles = tracker.recent(symbol, resolved_timeframe, max(1, min(limit, 20)))
    return {"symbol": symbol, "timeframe": resolved_timeframe, "cycles": cycles}


@router.get("/activity")
async def activity(request: Request, limit: int = 200) -> dict[str, object]:
    """Recent raw pipeline events, newest last — backs the live log panel's initial load."""
    activity_log = request.app.state.pipeline_activity_log
    entries = activity_log.snapshot(max(1, min(limit, 500)))
    return {"count": len(entries), "events": [entry.as_dict() for entry in entries]}
