"""Deterministic scheduling primitives for AI market analysis."""

from __future__ import annotations

from datetime import UTC, datetime


AI_ANALYSIS_TIMEFRAME = "M5"
AI_ANALYSIS_INTERVAL_MINUTES = 5


def five_minute_window_start(boundary: datetime) -> datetime:
    value = boundary.astimezone(UTC)
    return value.replace(
        minute=value.minute - value.minute % AI_ANALYSIS_INTERVAL_MINUTES,
        second=0,
        microsecond=0,
    )
