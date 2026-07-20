"""Flattens the latest snapshot from every analysis engine into one live "current state" view.

Pure presentation layer: every field is read from already-computed, already-persisted engine
snapshots. It calls no engine analysis methods and cannot change any engine's output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

FVG_ZONE_TYPES = {"bullish_fvg", "bearish_fvg", "bullish_inversion_fvg", "bearish_inversion_fvg"}
OB_ZONE_TYPES = {"bullish_order_block", "bearish_order_block", "bullish_breaker", "bearish_breaker", "bullish_mitigation_block", "bearish_mitigation_block"}
INACTIVE_LIFECYCLE = {"invalidated", "archived", "expired", "superseded"}


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if value is not None else None


def _is_active(item: Any) -> bool:
    lifecycle = getattr(item, "lifecycle_state", None)
    return lifecycle is None or lifecycle.value not in INACTIVE_LIFECYCLE


def _most_recent(items: tuple, timestamp_attr: str, *, where: Any = None) -> Any:
    candidates = [item for item in items if where is None or where(item)]
    active = [item for item in candidates if _is_active(item)]
    pool = active or candidates
    if not pool:
        return None
    return max(pool, key=lambda item: getattr(item, timestamp_attr))


def _smc_summary(snapshot: Any) -> dict[str, Any]:
    if snapshot is None:
        return {"available": False}
    structure = snapshot.structure_state
    bos = _most_recent(tuple(e for e in snapshot.structure_events if e.event_type.value == "bos"), "timestamp")
    choch = _most_recent(tuple(e for e in snapshot.structure_events if e.event_type.value == "choch"), "timestamp")
    fvg = _most_recent(tuple(z for z in snapshot.zones if z.zone_type.value in FVG_ZONE_TYPES), "origin_timestamp")
    order_block = _most_recent(tuple(z for z in snapshot.zones if z.zone_type.value in OB_ZONE_TYPES), "origin_timestamp")
    active_range = next((r for r in snapshot.dealing_ranges if r.id == structure.active_dealing_range_id), None)
    active_range = active_range or _most_recent(snapshot.dealing_ranges, "end_timestamp")
    return {
        "available": True,
        "status": snapshot.status.value,
        "current_bias": structure.current_direction.value,
        "htf_bias": snapshot.multi_timeframe_context.directions if snapshot.multi_timeframe_context else None,
        "current_bos": {"direction": bos.direction.value, "at": bos.timestamp, "price": bos.broken_level} if bos else None,
        "current_choch": {"direction": choch.direction.value, "at": choch.timestamp, "price": choch.broken_level} if choch else None,
        "current_fvg": {"type": fvg.zone_type.value, "upper": fvg.upper_price, "lower": fvg.lower_price, "lifecycle": fvg.lifecycle_state.value} if fvg else None,
        "current_order_block": {"type": order_block.zone_type.value, "upper": order_block.upper_price, "lower": order_block.lower_price, "lifecycle": order_block.lifecycle_state.value} if order_block else None,
        "dealing_range": {"range_high": active_range.range_high, "range_low": active_range.range_low, "equilibrium": active_range.equilibrium, "premium_boundary": active_range.premium_boundary, "discount_boundary": active_range.discount_boundary} if active_range else None,
        "analysis_timestamp": snapshot.analysis_timestamp,
    }


def _premium_discount(smc_summary: dict[str, Any], close: float | None) -> str:
    range_data = smc_summary.get("dealing_range")
    if not range_data or close is None:
        return "unknown"
    if close >= range_data["premium_boundary"]:
        return "premium"
    if close <= range_data["discount_boundary"]:
        return "discount"
    return "equilibrium"


def build_market_intelligence(
    *,
    instrument: str,
    timeframe: str,
    now: datetime,
    session,
    candle: Any,
    smc: Any,
    liquidity: Any,
    volume_profile: Any,
    institutional_flow: Any,
    market_regime: Any,
    economic_context: Any,
    ai_score: Any,
    decision: Any,
    errors: dict[str, str | None],
) -> dict[str, Any]:
    smc_summary = _smc_summary(smc)
    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "as_of": now,
        "current_session": session.active_session.value if session and session.active_session else "closed",
        "market_open": session.market_open if session else None,
        "current_candle": {"timestamp": candle.timestamp, "open": candle.open, "high": candle.high, "low": candle.low, "close": candle.close, "volume": candle.volume, "spread": candle.spread} if candle else None,
        "spread": candle.spread if candle else None,
        "smc": smc_summary,
        "current_bias": smc_summary.get("current_bias"),
        "htf_bias": smc_summary.get("htf_bias"),
        "current_bos": smc_summary.get("current_bos"),
        "current_choch": smc_summary.get("current_choch"),
        "current_fvg": smc_summary.get("current_fvg"),
        "current_order_block": smc_summary.get("current_order_block"),
        "premium_discount": _premium_discount(smc_summary, candle.close if candle else None),
        "liquidity": {"available": liquidity is not None, "state": _dump(getattr(liquidity, "state", None)), "analysis_timestamp": getattr(liquidity, "analysis_timestamp", None)},
        "volume_profile": {"available": volume_profile is not None, "quality": _dump(getattr(volume_profile, "volume_data_quality", None)), "analysis_timestamp": getattr(volume_profile, "analysis_timestamp", None)},
        "institutional_flow": {"available": institutional_flow is not None, "state": _dump(getattr(institutional_flow, "state", None)), "quality": _dump(getattr(institutional_flow, "quality", None)), "analysis_timestamp": getattr(institutional_flow, "analysis_timestamp", None)},
        "market_regime": {
            "available": market_regime is not None,
            "dominant_regime": getattr(market_regime, "dominant_regime", None) and market_regime.dominant_regime.value,
            "trend_regime": getattr(market_regime, "trend_regime", None) and market_regime.trend_regime.value,
            "directional_bias": getattr(market_regime, "directional_bias", None) and market_regime.directional_bias.value,
            "trend_strength": getattr(market_regime, "trend_strength", None),
            "volatility_score": getattr(market_regime, "volatility_score", None),
            "confidence": getattr(market_regime, "confidence", None),
            "analysis_timestamp": getattr(market_regime, "analysis_timestamp", None),
        },
        "economic_status": {
            "available": economic_context is not None,
            "degraded": bool(getattr(economic_context, "unavailable_context", None)) if economic_context is not None else True,
            "risk_window_phase": getattr(economic_context, "risk_window_phase", None) and economic_context.risk_window_phase.value,
            "risk_score": getattr(economic_context, "risk_score", None),
            "next_relevant_event": getattr(economic_context, "next_relevant_event", None) and economic_context.next_relevant_event.display_name,
        },
        "confidence_percent": getattr(ai_score, "confidence_score", None),
        "ai_directional_label": getattr(ai_score, "directional_label", None) and ai_score.directional_label.value,
        "ai_composite_score": getattr(ai_score, "composite_score", None),
        "ai_missing_sources": list(getattr(ai_score, "missing_sources", ()) or ()),
        "ai_degraded_sources": list(getattr(ai_score, "degraded_sources", ()) or ()),
        "scenario_readiness_percent": getattr(decision, "eligibility_score", None),
        "decision_status": getattr(decision, "state", None) and decision.state.value,
        "decision_direction": getattr(decision, "direction", None) and decision.direction.value,
        "last_update_time": now,
        "source_errors": errors,
    }
