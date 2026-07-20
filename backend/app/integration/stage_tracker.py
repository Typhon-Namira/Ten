"""Per-candle pipeline stage observability.

`PipelineStageTracker` is reported to directly by `FullSystemIntegrationService._run()` — event
payload shapes vary too much across engines to reliably reconstruct a single candle's run from
the event bus alone, so the pipeline reports its own progress instead. Every call site is pure
instrumentation: it never changes control flow, and any real exception from an engine call is
always re-raised unchanged by the caller. This module makes no engine calls itself.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

STAGE_KEYS: tuple[str, ...] = (
    "candle_received",
    "candle_normalized",
    "stored_in_database",
    "liquidity_analysis",
    "smc_analysis",
    "volume_profile",
    "market_regime",
    "institutional_flow",
    "ai_scoring",
    "confidence_calculation",
    "scenario_decision",
)

STAGE_LABELS: dict[str, str] = {
    "candle_received": "Candle received",
    "candle_normalized": "Candle normalized",
    "stored_in_database": "Stored in database",
    "liquidity_analysis": "Liquidity analysis",
    "smc_analysis": "SMC analysis",
    "volume_profile": "Volume profile",
    "market_regime": "Market regime",
    "institutional_flow": "Institutional flow",
    "ai_scoring": "AI scoring",
    "confidence_calculation": "Confidence calculation",
    "scenario_decision": "Scenario decision",
}


@dataclass
class _CycleState:
    key: tuple[str, str]
    boundary: datetime
    started_at: datetime
    updated_at: datetime
    statuses: dict[str, str] = field(default_factory=lambda: dict.fromkeys(STAGE_KEYS, "waiting"))
    complete: bool = False


class PipelineStageTracker:
    """Bounded, in-memory per-(symbol, timeframe) board of the 10 named pipeline stages."""

    def __init__(self, *, max_cycles_per_series: int = 5) -> None:
        self._max_cycles = max_cycles_per_series
        self._cycles: dict[tuple[str, str], deque[_CycleState]] = {}

    def begin(self, symbol: str, timeframe: str, boundary: datetime) -> None:
        key = (symbol.upper(), timeframe)
        series = self._cycles.setdefault(key, deque(maxlen=self._max_cycles))
        now = datetime.now(UTC)
        series.append(_CycleState(key=key, boundary=boundary, started_at=now, updated_at=now))

    def mark(self, symbol: str, timeframe: str, boundary: datetime, stages: tuple[str, ...], status: str) -> None:
        cycle = self._current(symbol, timeframe, boundary)
        if cycle is None:
            return
        cycle.updated_at = datetime.now(UTC)
        for stage in stages:
            cycle.statuses[stage] = status

    def fail_in_flight(self, symbol: str, timeframe: str, boundary: datetime) -> None:
        """Mark the first not-yet-reached stage as failed and every later stage as skipped."""
        cycle = self._current(symbol, timeframe, boundary)
        if cycle is None:
            return
        cycle.updated_at = datetime.now(UTC)
        cycle.complete = True
        failed_assigned = False
        for stage in STAGE_KEYS:
            if cycle.statuses[stage] == "waiting":
                cycle.statuses[stage] = "failed" if not failed_assigned else "skipped"
                failed_assigned = True

    def complete(self, symbol: str, timeframe: str, boundary: datetime) -> None:
        cycle = self._current(symbol, timeframe, boundary)
        if cycle is None:
            return
        cycle.complete = True
        cycle.updated_at = datetime.now(UTC)

    def _current(self, symbol: str, timeframe: str, boundary: datetime) -> _CycleState | None:
        series = self._cycles.get((symbol.upper(), timeframe))
        if not series:
            return None
        cycle = series[-1]
        return cycle if cycle.boundary == boundary else None

    def latest(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        series = self._cycles.get((symbol.upper(), timeframe))
        if not series:
            return None
        return self._render(series[-1])

    def recent(self, symbol: str, timeframe: str, limit: int = 5) -> list[dict[str, Any]]:
        series = self._cycles.get((symbol.upper(), timeframe), deque())
        return [self._render(cycle) for cycle in list(series)[-limit:][::-1]]

    def known_series(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._cycles.keys())

    @staticmethod
    def _render(cycle: _CycleState) -> dict[str, Any]:
        statuses = dict(cycle.statuses)
        if not cycle.complete:
            for stage in STAGE_KEYS:
                if statuses[stage] == "waiting":
                    statuses[stage] = "running"
                    break
        return {
            "symbol": cycle.key[0],
            "timeframe": cycle.key[1],
            "candle_timestamp": cycle.boundary,
            "started_at": cycle.started_at,
            "updated_at": cycle.updated_at,
            "complete": cycle.complete,
            "stages": [{"key": stage, "label": STAGE_LABELS[stage], "status": statuses[stage]} for stage in STAGE_KEYS],
        }
