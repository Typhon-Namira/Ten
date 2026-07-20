"""Per-candle pipeline stage visibility (Part 3): every named stage, including ones that
produce no signal, with an explicit Running/Waiting/Success/Failed/Skipped status."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.get("/stages/latest")
async def latest_stages(request: Request, instrument: str = "XAUUSD", timeframe: str = "M15") -> dict[str, object]:
    tracker = request.app.state.pipeline_stage_tracker
    cycle = tracker.latest(instrument.upper(), timeframe)
    if cycle is None:
        return {"symbol": instrument.upper(), "timeframe": timeframe, "available": False, "reason": "no_candle_processed_yet", "stages": []}
    return {"available": True, **cycle}


@router.get("/stages/recent")
async def recent_stages(request: Request, instrument: str = "XAUUSD", timeframe: str = "M15", limit: int = 5) -> dict[str, object]:
    tracker = request.app.state.pipeline_stage_tracker
    cycles = tracker.recent(instrument.upper(), timeframe, max(1, min(limit, 20)))
    return {"symbol": instrument.upper(), "timeframe": timeframe, "cycles": cycles}


@router.get("/activity")
async def activity(request: Request, limit: int = 200) -> dict[str, object]:
    """Recent raw pipeline events, newest last — backs the live log panel's initial load."""
    activity_log = request.app.state.pipeline_activity_log
    entries = activity_log.snapshot(max(1, min(limit, 500)))
    return {"count": len(entries), "events": [entry.as_dict() for entry in entries]}
