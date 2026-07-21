"""Chart-plottable aggregation of candles + every engine's overlay objects.

Pure presentation layer, same policy as `market_intelligence.py`: reads only already-persisted
snapshots (never triggers analysis), and a failing source becomes an empty overlay list rather
than a 500 — the chart must always render whatever data IS available.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Request

from backend.app.api.routes.system import _default_selection
from backend.app.api.safe import safe_call
from backend.app.engines.market_data_engine import Candle, Timeframe

router = APIRouter(prefix="/api/v1/chart", tags=["chart"])


def _epoch(value: datetime) -> int:
    return int(value.timestamp())


def _candle_point(candle: Candle) -> dict[str, Any]:
    return {"time": _epoch(candle.timestamp), "open": candle.open, "high": candle.high, "low": candle.low, "close": candle.close, "volume": candle.volume}


def _structure_events(smc: Any) -> list[dict[str, Any]]:
    if smc is None:
        return []
    return [
        {
            "id": str(item.id),
            "kind": item.event_type.value,
            "direction": item.direction.value,
            "time": _epoch(item.timestamp),
            "price": item.broken_level,
            "confidence": item.confidence_score,
        }
        for item in smc.structure_events
    ]


def _zones(smc: Any) -> list[dict[str, Any]]:
    if smc is None:
        return []
    return [
        {
            "id": str(item.id),
            "kind": item.zone_type.value,
            "direction": item.direction.value,
            "upper": item.upper_price,
            "lower": item.lower_price,
            "start_time": _epoch(item.origin_timestamp),
            "lifecycle_state": item.lifecycle_state.value,
            "mitigation_percentage": item.mitigation_percentage,
        }
        for item in smc.zones
        if item.lifecycle_state.value not in {"invalidated", "archived", "expired", "superseded"}
    ]


def _dealing_range(smc: Any) -> dict[str, Any] | None:
    if smc is None or not smc.dealing_ranges:
        return None
    item = max(smc.dealing_ranges, key=lambda entry: entry.start_timestamp)
    return {
        "range_high": item.range_high,
        "range_low": item.range_low,
        "equilibrium": item.equilibrium,
        "premium_boundary": item.premium_boundary,
        "discount_boundary": item.discount_boundary,
        "golden_zone_low": item.golden_zone_low,
        "golden_zone_high": item.golden_zone_high,
        "start_time": _epoch(item.start_timestamp),
        "end_time": _epoch(item.end_timestamp),
        "direction": item.direction.value,
    }


def _liquidity_pools(liquidity: Any) -> list[dict[str, Any]]:
    if liquidity is None:
        return []
    return [
        {
            "id": str(item.id),
            "side": item.side.value,
            "upper": item.upper_bound,
            "lower": item.lower_bound,
            "start_time": _epoch(item.available_at),
            "lifecycle_state": item.lifecycle_state.value,
            "strength": item.strength_score,
            "target_rank": item.target_rank,
        }
        for item in liquidity.pools
        if item.lifecycle_state.value not in {"invalidated", "archived", "expired"}
    ]


def _liquidity_sweeps(liquidity: Any) -> list[dict[str, Any]]:
    if liquidity is None:
        return []
    return [{"id": str(item.id), "kind": type(item).__name__, "time": _epoch(item.occurred_at), "price": item.price, "side": item.side.value} for item in liquidity.sweeps]


def _equal_levels(liquidity: Any) -> list[dict[str, Any]]:
    if liquidity is None:
        return []
    return [
        {"id": str(item.id), "side": item.side.value, "price": item.price, "time": _epoch(item.available_at), "member_count": len(item.member_prices)}
        for item in liquidity.equal_levels
        if item.lifecycle_state.value not in {"invalidated", "archived", "expired"}
    ]


def _economic_events(events: Any) -> list[dict[str, Any]]:
    if not events:
        return []
    return [
        {"id": str(item.event_id), "name": item.display_name, "importance": item.importance.value, "time": _epoch(item.scheduled_at_utc)}
        for item in events
        if item.scheduled_at_utc is not None
    ]


def _decision_annotation(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "direction": decision.direction.value,
        "state": decision.state.value,
        "confidence": decision.confidence_score,
        "time": _epoch(decision.as_of),
    }


def _sessions(liquidity: Any) -> list[dict[str, Any]]:
    if liquidity is None:
        return []
    return [
        {"session": item.session.value, "high": item.high, "low": item.low, "opened_at": _epoch(item.opened_at), "completed": item.completed}
        for item in liquidity.sessions
    ]


def _volume_profile(volume_profile: Any) -> dict[str, Any] | None:
    if volume_profile is None or not volume_profile.profiles:
        return None
    item = max(volume_profile.profiles, key=lambda entry: entry.end_timestamp)
    return {
        "poc": item.poc.price if item.poc else None,
        "vah": item.value_area.vah if item.value_area else None,
        "val": item.value_area.val if item.value_area else None,
        "start_time": _epoch(item.start_timestamp),
        "end_time": _epoch(item.end_timestamp),
    }


@router.get("/overlays")
async def overlays(request: Request, instrument: str | None = None, timeframe: str | None = None, limit: int = 300) -> dict[str, object]:
    """Everything needed to render the live chart: OHLCV candles plus every analysis engine's
    plottable objects (structure events, zones, dealing range, liquidity pools/sweeps, sessions,
    volume profile POC/VAH/VAL) — one call, one instrument/timeframe pair, so the chart and every
    other dashboard widget are always looking at the same analysis."""
    app = request.app
    default_instrument, default_timeframe = _default_selection(request)
    symbol = (instrument or default_instrument).upper()
    resolved_timeframe = timeframe or default_timeframe
    tf = Timeframe(resolved_timeframe)
    now = datetime.now(UTC)
    errors: dict[str, str | None] = {}

    candles, errors["candles"] = await safe_call(lambda: app.state.market_data_service.history(symbol, tf, limit=max(50, min(limit, 5000))))
    smc, errors["smc"] = await safe_call(lambda: app.state.smc_service.state(symbol, tf))
    liquidity, errors["liquidity"] = await safe_call(lambda: app.state.liquidity_service.state(symbol, tf))
    volume_profile, errors["volume_profile"] = await safe_call(lambda: app.state.volume_profile_service.state(symbol, tf))
    economic_events, errors["economic_calendar"] = await safe_call(
        lambda: app.state.economic_calendar_service.events(now - timedelta(days=2), now + timedelta(days=5), now, 100)
    )
    decisions, errors["signal_decision"] = await safe_call(lambda: app.state.signal_decision_service.repository.find_recent_decisions(symbol, resolved_timeframe, now, 1))

    return {
        "instrument": symbol,
        "timeframe": resolved_timeframe,
        "generated_at": now.isoformat(),
        "candles": [_candle_point(item) for item in (candles or [])],
        "structure_events": _structure_events(smc),
        "zones": _zones(smc),
        "dealing_range": _dealing_range(smc),
        "liquidity_pools": _liquidity_pools(liquidity),
        "liquidity_sweeps": _liquidity_sweeps(liquidity),
        "equal_levels": _equal_levels(liquidity),
        "sessions": _sessions(liquidity),
        "volume_profile": _volume_profile(volume_profile),
        "economic_events": _economic_events(economic_events),
        "decision": _decision_annotation(decisions[0] if decisions else None),
        "source_errors": {key: value for key, value in errors.items() if value is not None},
    }
