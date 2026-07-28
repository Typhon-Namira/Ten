"""Deterministic scheduling primitives for AI market analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from backend.app.market_state import UnifiedMarketState


AI_ANALYSIS_TIMEFRAME = "M5"
AI_ANALYSIS_INTERVAL_MINUTES = 5


class AIEligibilityReason(StrEnum):
    DUPLICATE_SNAPSHOT = "duplicate_snapshot"
    MISSING_PREREQUISITE = "missing_prerequisite"
    COOLDOWN = "cooldown"
    INTERVAL_NOT_DUE = "interval_not_due"
    ANALYSIS_EXISTS = "analysis_exists"
    CONCURRENCY_LIMIT = "concurrency_limit"
    STALE_DATA = "stale_data"
    INVALID_STATE = "invalid_state"
    DISABLED = "disabled"


def five_minute_window_start(boundary: datetime) -> datetime:
    value = boundary.astimezone(UTC)
    return value.replace(
        minute=value.minute - value.minute % AI_ANALYSIS_INTERVAL_MINUTES,
        second=0,
        microsecond=0,
    )


def synchronized_cycle_eligibility(
    state: UnifiedMarketState,
) -> AIEligibilityReason | None:
    """Shared integration/worker eligibility policy for synchronized cycles."""

    boundary = state.market_data_boundary.astimezone(UTC)
    if (
        state.trigger_timeframe.upper() != AI_ANALYSIS_TIMEFRAME
        or boundary != five_minute_window_start(boundary)
    ):
        return AIEligibilityReason.INTERVAL_NOT_DUE
    frames = {item.timeframe.upper(): item for item in state.timeframes}
    if set(frames) != {"M1", "M5", "M15"}:
        return AIEligibilityReason.MISSING_PREREQUISITE
    if any(item.stale for item in frames.values()) or state.stale_evidence:
        return AIEligibilityReason.STALE_DATA
    if state.market_data_boundary > state.knowledge_cutoff:
        return AIEligibilityReason.INVALID_STATE
    return None
