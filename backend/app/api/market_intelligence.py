"""Flattens the latest snapshot from every analysis engine into one live "current state" view.

Pure presentation layer: every field is read from already-computed, already-persisted engine
snapshots. It calls no engine analysis methods and cannot change any engine's output.

Retrieval policy (see docs/MARKET_INTELLIGENCE_TRACE.md-equivalent reasoning inline): each source
is read independently via that engine's own "latest snapshot ever persisted" accessor — never a
validity-windowed or exact-boundary-matched query — because engines legitimately complete a few
seconds apart and a stale-but-present snapshot is far more useful to an operator than a false
"unavailable". Every field's own timestamp and freshness state travels with it in `diagnostics`,
so staleness is always explicit rather than silently hidden or silently trusted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

FVG_ZONE_TYPES = {"bullish_fvg", "bearish_fvg", "bullish_inversion_fvg", "bearish_inversion_fvg"}
OB_ZONE_TYPES = {"bullish_order_block", "bearish_order_block", "bullish_breaker", "bearish_breaker", "bullish_mitigation_block", "bearish_mitigation_block"}
INACTIVE_LIFECYCLE = {"invalidated", "archived", "expired", "superseded"}

# Freshness policy is purely an observability label — it never gates or filters retrieval (a
# snapshot older than STALE_SECONDS is still returned, just reported as "stale" rather than
# silently hidden or silently presented as current).
FRESH_SECONDS = 20 * 60
AGING_SECONDS = 60 * 60


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if value is not None else None


def _is_active(item: Any) -> bool:
    lifecycle = getattr(item, "lifecycle_state", None)
    return lifecycle is None or lifecycle.value not in INACTIVE_LIFECYCLE


def _most_recent(items: tuple[Any, ...], timestamp_attr: str, *, where: Any = None) -> Any:
    candidates = [item for item in items if where is None or where(item)]
    active = [item for item in candidates if _is_active(item)]
    pool = active or candidates
    if not pool:
        return None
    return max(pool, key=lambda item: getattr(item, timestamp_attr))


def _freshness(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "unknown"
    if age_seconds <= FRESH_SECONDS:
        return "fresh"
    if age_seconds <= AGING_SECONDS:
        return "aging"
    return "stale"


def _snapshot_timestamp(snapshot: Any, timestamp_attrs: tuple[str, ...]) -> datetime | None:
    for attr in timestamp_attrs:
        value = getattr(snapshot, attr, None)
        if isinstance(value, datetime):
            return value
    return None


def source_diagnostic(
    name: str,
    instrument: str,
    timeframe: str,
    snapshot: Any,
    error: str | None,
    now: datetime,
    timestamp_attrs: tuple[str, ...] = ("analysis_timestamp", "as_of", "created_at"),
    *,
    degraded: bool = False,
    latest_attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-source retrieval diagnostics: exactly what was queried, what came back, and how old it is.

    `last_successful_snapshot_at` describes the persisted snapshot this endpoint retrieved — it
    can be stale. `latest_attempt_*` describes the pipeline's most recent *execution* of that
    stage (from `PipelineStageTracker`), independent of whether it ever succeeded. The two are
    reported separately on purpose: a source is never "ok" just because an old snapshot exists —
    if the latest attempt failed, `latest_attempt_status` says so even while `status` still
    reflects a usable (if stale) snapshot.
    """
    timestamp = _snapshot_timestamp(snapshot, timestamp_attrs) if snapshot is not None else None
    age_seconds = (now - timestamp).total_seconds() if timestamp is not None else None
    if error:
        status = "error"
    elif degraded:
        status = "degraded"
    elif snapshot is not None:
        status = "ok"
    else:
        status = "unavailable"
    return {
        "source": name,
        "instrument": instrument,
        "timeframe": timeframe,
        "status": status,
        "snapshot_found": snapshot is not None,
        "last_successful_snapshot_at": timestamp,
        "age_seconds": age_seconds,
        "freshness": _freshness(age_seconds),
        "error": error,
        "latest_attempt_status": latest_attempt["status"] if latest_attempt else None,
        "latest_attempt_at": latest_attempt["updated_at"] if latest_attempt else None,
        "latest_attempt_error": latest_attempt["error"] if latest_attempt else None,
    }


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
    session: Any,
    candle: Any,
    spread_supported: bool | None,
    smc: Any,
    liquidity: Any,
    volume_profile: Any,
    institutional_flow: Any,
    market_regime: Any,
    economic_context: Any,
    ai_score: Any,
    decision: Any,
    errors: dict[str, str | None],
    stage_attempts: dict[str, dict[str, Any] | None] | None = None,
    economic_stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    smc_summary = _smc_summary(smc)
    stage_attempts = stage_attempts or {}
    # Real spread requires provider bid/ask capability (see docs/MARKET_DATA_ENGINE.md); providers
    # without it leave Candle.spread at its Pydantic default of 0.0, which is indistinguishable
    # from a genuine zero spread unless the provider's capability is checked explicitly here.
    spread = candle.spread if candle is not None and spread_supported else None
    # `economic_context` is always a freshly-computed object (never a persisted snapshot lookup),
    # so it is structurally always "found" — but a context with no relevant event mapping is a
    # substantively degraded result, not a healthy one. `degraded` here must match the same
    # `unavailable_context` signal the "economic_status" block below reports, so the diagnostics
    # table and the main panel can never contradict each other the way they previously did (the
    # diagnostics table said "ok/fresh" purely because the call succeeded, while the main panel
    # separately said "degraded" based on the data itself).
    economic_degraded = bool(getattr(economic_context, "unavailable_context", None)) if economic_context is not None else False
    diagnostics = [
        source_diagnostic("market_data", instrument, timeframe, candle, errors.get("candle"), now, ("timestamp",)),
        source_diagnostic("smc", instrument, timeframe, smc, errors.get("smc"), now, latest_attempt=stage_attempts.get("smc")),
        source_diagnostic("liquidity", instrument, timeframe, liquidity, errors.get("liquidity"), now, latest_attempt=stage_attempts.get("liquidity")),
        source_diagnostic("volume_profile", instrument, timeframe, volume_profile, errors.get("volume_profile"), now, latest_attempt=stage_attempts.get("volume_profile")),
        source_diagnostic("institutional_flow", instrument, timeframe, institutional_flow, errors.get("institutional_flow"), now, latest_attempt=stage_attempts.get("institutional_flow")),
        source_diagnostic("market_regime", instrument, timeframe, market_regime, errors.get("market_regime"), now, latest_attempt=stage_attempts.get("market_regime")),
        source_diagnostic("economic_calendar", instrument, timeframe, economic_context, errors.get("economic_calendar"), now, ("analysis_timestamp",), degraded=economic_degraded),
        source_diagnostic("ai_scoring", instrument, timeframe, ai_score, errors.get("ai_scoring"), now, ("as_of",), latest_attempt=stage_attempts.get("ai_scoring")),
        source_diagnostic("signal_decision", instrument, timeframe, decision, errors.get("signal_decision"), now, ("as_of",), latest_attempt=stage_attempts.get("signal_decision")),
    ]
    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "generated_at": now,
        "latest_candle_timestamp": candle.timestamp if candle else None,
        "current_session": session.active_session.value if session and session.active_session else "closed",
        "market_open": session.market_open if session else None,
        "current_candle": {"timestamp": candle.timestamp, "open": candle.open, "high": candle.high, "low": candle.low, "close": candle.close, "volume": candle.volume, "spread": spread} if candle else None,
        "spread": spread,
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
            # The exact categorical state (`CalendarContextState`) — the same value
            # signal_decision_engine and the explainability layer read, so the dashboard can
            # render "outside_risk_window" distinctly from "provider_rate_limited" instead of a
            # single collapsed `degraded` boolean.
            "context_state": (getattr(economic_context, "context_state", None) and economic_context.context_state.value) if economic_context is not None else "no_calendar_data",
            "risk_window_phase": getattr(economic_context, "risk_window_phase", None) and economic_context.risk_window_phase.value,
            "risk_score": getattr(economic_context, "risk_score", None),
            "next_relevant_event": getattr(economic_context, "next_relevant_event", None) and economic_context.next_relevant_event.display_name,
            # Provider health -> downloaded events -> mapped events -> relevant events -> trading
            # context, each reported independently (see
            # `economic_calendar_engine.analyzer.staged_diagnostics`). Without this, "degraded"
            # here and "economic_context_unavailable" in a rejected decision's blockers read as
            # two different, contradictory-looking states even when they trace back to the exact
            # same fact (no relevant event mapping right now, which is routine, not a provider
            # outage) — this is what lets the dashboard show *why*, not just *that*.
            "stages": economic_stages,
        },
        "confidence_percent": getattr(ai_score, "confidence_score", None),
        "risk_percent": getattr(ai_score, "market_risk_score", None),
        "ai_directional_label": getattr(ai_score, "directional_label", None) and ai_score.directional_label.value,
        "ai_composite_score": getattr(ai_score, "composite_score", None),
        "ai_missing_sources": list(getattr(ai_score, "missing_sources", ()) or ()),
        "ai_degraded_sources": list(getattr(ai_score, "degraded_sources", ()) or ()),
        "scenario_readiness_percent": getattr(decision, "eligibility_score", None),
        "decision_status": getattr(decision, "state", None) and decision.state.value,
        "decision_direction": getattr(decision, "direction", None) and decision.direction.value,
        "decision_active": bool(decision is not None and getattr(decision, "valid_until", now) > now),
        "diagnostics": diagnostics,
        "source_errors": errors,
    }
