"""Per-candle pipeline stage observability.

`PipelineStageTracker` is reported to directly by `FullSystemIntegrationService._run()` — event
payload shapes vary too much across engines to reliably reconstruct a single candle's run from
the event bus alone, so the pipeline reports its own progress instead. Every call site is pure
instrumentation: it never changes control flow, and any real exception from an engine call is
always re-raised unchanged by the caller. This module makes no engine calls itself, listens to
no events, and runs no timers — a stage can only move because `_run()` told it to.

Cycles are identified by the full (symbol, timeframe, candle boundary) tuple, never by "whichever
cycle was inserted most recently". A retried candle (same boundary, e.g. after `process_outbox_once`
re-attempts a failed outbox item) resets its own cycle in place instead of being confused with, or
silently shadowing, any other in-flight cycle for the same symbol/timeframe.

When a stage fails, the triggering exception's class, message, full traceback, and failing
file/line are captured against that specific stage (`fail_in_flight(..., exc=...)`) so a "why did
this fail" question never has to be inferred from a bare status string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
import traceback
from typing import Any

logger = logging.getLogger(__name__)

## Declared order MUST match `FullSystemIntegrationService._run()`'s actual execution order
## (smc -> liquidity -> volume_profile -> institutional_flow -> market_regime -> ai_scoring ->
## signal_decision). `_render()`'s "running" inference and `fail_in_flight()` both walk this
## tuple looking for the first still-"waiting" stage; a declared order that diverges from
## execution order misattributes both "running" and "failed" to the wrong stage (this exact bug
## previously listed liquidity_analysis before smc_analysis, matching the user-facing stage
## *label* order from the product spec rather than the pipeline's real call order).
STAGE_KEYS: tuple[str, ...] = (
    "candle_received",
    "candle_normalized",
    "stored_in_database",
    "smc_analysis",
    "liquidity_analysis",
    "volume_profile",
    "institutional_flow",
    "market_regime",
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

# A stage is "waiting" until told otherwise; "running" is a display-only inference (see _render).
# "degraded" means the engine returned a result without raising, but flagged reduced quality/
# input via its own status field — never conflated with "failed", which requires an exception.
STAGE_STATUSES: frozenset[str] = frozenset({"waiting", "running", "success", "degraded", "failed", "skipped"})


def format_exception(exc: BaseException) -> dict[str, Any]:
    """Full, structured detail for a stage failure — never just `type(exc).__name__`."""
    tb = exc.__traceback__
    frames = traceback.extract_tb(tb)
    last_frame = frames[-1] if frames else None
    return {
        "exception_class": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, tb)),
        "file": last_frame.filename if last_frame else None,
        "line": last_frame.lineno if last_frame else None,
        "function": last_frame.name if last_frame else None,
    }


@dataclass
class _CycleState:
    symbol: str
    timeframe: str
    boundary: datetime
    started_at: datetime
    updated_at: datetime
    correlation_id: str | None = None
    statuses: dict[str, str] = field(default_factory=lambda: dict.fromkeys(STAGE_KEYS, "waiting"))
    stage_updated_at: dict[str, datetime] = field(default_factory=dict)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    complete: bool = False
    attempt: int = 1


def _log(event: str, symbol: str, timeframe: str, boundary: datetime, **extra: object) -> None:
    logger.debug(
        event,
        extra={"symbol": symbol, "timeframe": timeframe, "candle_timestamp": boundary.isoformat(), **extra},
    )


class PipelineStageTracker:
    """Bounded, in-memory per-(symbol, timeframe) board of the 11 named pipeline stages."""

    def __init__(self, *, max_cycles_per_series: int = 5) -> None:
        self._max_cycles = max_cycles_per_series
        # (symbol, timeframe) -> { candle boundary -> cycle }. Cycle identity is always the full
        # (symbol, timeframe, boundary) triple; "latest" is computed by max(boundary), not by
        # insertion order, so an out-of-order completion never gets shadowed by a newer candle
        # that started but hasn't finished yet.
        self._series: dict[tuple[str, str], dict[datetime, _CycleState]] = {}

    def begin(self, symbol: str, timeframe: str, boundary: datetime, *, correlation_id: str | None = None) -> None:
        series_key = (symbol.upper(), timeframe)
        series = self._series.setdefault(series_key, {})
        now = datetime.now(UTC)
        existing = series.get(boundary)
        if existing is not None:
            # Same candle re-entering _run() (outbox retry, or historical-then-live overlap):
            # reset this cycle's own stages in place. Never spawn a second, parallel cycle for
            # the same key — that is exactly what let an old attempt's terminal state (e.g. a
            # transient failure) linger and be reported as "latest" after a later attempt moved on.
            existing.statuses = dict.fromkeys(STAGE_KEYS, "waiting")
            existing.stage_updated_at = {}
            existing.errors = {}
            existing.complete = False
            existing.started_at = now
            existing.updated_at = now
            existing.correlation_id = correlation_id
            existing.attempt += 1
            _log("stage_tracker.begin_retry", *series_key, boundary, attempt=existing.attempt, correlation_id=correlation_id)
            return
        series[boundary] = _CycleState(symbol=series_key[0], timeframe=series_key[1], boundary=boundary, started_at=now, updated_at=now, correlation_id=correlation_id)
        if len(series) > self._max_cycles:
            oldest = min(series)
            if oldest != boundary:
                del series[oldest]
        _log("stage_tracker.begin", *series_key, boundary, correlation_id=correlation_id)

    def mark(self, symbol: str, timeframe: str, boundary: datetime, stages: tuple[str, ...], status: str) -> None:
        cycle = self._lookup(symbol, timeframe, boundary)
        if cycle is None:
            _log("stage_tracker.mark_missed_no_cycle", symbol.upper(), timeframe, boundary, stages=stages, status=status)
            return
        now = datetime.now(UTC)
        cycle.updated_at = now
        for stage in stages:
            cycle.statuses[stage] = status
            cycle.stage_updated_at[stage] = now
            cycle.errors.pop(stage, None)
        _log("stage_tracker.mark", symbol.upper(), timeframe, boundary, stages=stages, status=status)

    def fail_in_flight(self, symbol: str, timeframe: str, boundary: datetime, *, exc: BaseException | None = None) -> None:
        """Mark the first not-yet-reached stage as failed and every later stage as skipped.

        Stages already marked success/degraded are left untouched — a failure further down the
        pipeline must never retroactively overwrite an earlier stage that genuinely completed.
        The full exception (class, message, traceback, failing file/line) is captured against the
        exact stage that raised it, when provided.
        """
        cycle = self._lookup(symbol, timeframe, boundary)
        if cycle is None:
            _log("stage_tracker.fail_missed_no_cycle", symbol.upper(), timeframe, boundary)
            return
        now = datetime.now(UTC)
        cycle.updated_at = now
        cycle.complete = True
        failed_assigned = False
        failed_stage: str | None = None
        for stage in STAGE_KEYS:
            if cycle.statuses[stage] == "waiting":
                cycle.statuses[stage] = "failed" if not failed_assigned else "skipped"
                cycle.stage_updated_at[stage] = now
                if not failed_assigned:
                    failed_stage = stage
                failed_assigned = True
        if failed_stage is not None and exc is not None:
            cycle.errors[failed_stage] = format_exception(exc)
        _log(
            "stage_tracker.fail_in_flight",
            symbol.upper(),
            timeframe,
            boundary,
            failed_stage=failed_stage,
            exception_class=type(exc).__name__ if exc is not None else None,
        )

    def complete(self, symbol: str, timeframe: str, boundary: datetime) -> None:
        cycle = self._lookup(symbol, timeframe, boundary)
        if cycle is None:
            return
        cycle.complete = True
        cycle.updated_at = datetime.now(UTC)
        _log("stage_tracker.complete", symbol.upper(), timeframe, boundary)

    def _lookup(self, symbol: str, timeframe: str, boundary: datetime) -> _CycleState | None:
        return self._series.get((symbol.upper(), timeframe), {}).get(boundary)

    def latest(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        series = self._series.get((symbol.upper(), timeframe))
        if not series:
            return None
        newest_boundary = max(series)
        return self._render(series[newest_boundary])

    def recent(self, symbol: str, timeframe: str, limit: int = 5) -> list[dict[str, Any]]:
        series = self._series.get((symbol.upper(), timeframe), {})
        ordered = sorted(series.values(), key=lambda cycle: cycle.boundary, reverse=True)
        return [self._render(cycle) for cycle in ordered[:limit]]

    def stage_attempt(self, symbol: str, timeframe: str, stage: str) -> dict[str, Any] | None:
        """The latest cycle's status/timestamp/error for one specific stage, independent of
        whether that engine has ever produced a *persisted* snapshot — see
        `market_intelligence.py`'s `latest_attempt_*` fields, which must not be confused with
        `last_successful_snapshot_at`."""
        series = self._series.get((symbol.upper(), timeframe))
        if not series:
            return None
        cycle = series[max(series)]
        rendered = self._render(cycle)
        return next((item for item in rendered["stages"] if item["key"] == stage), None)

    def known_series(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._series.keys())

    @staticmethod
    def _render(cycle: _CycleState) -> dict[str, Any]:
        statuses = dict(cycle.statuses)
        if not cycle.complete:
            for stage in STAGE_KEYS:
                if statuses[stage] == "waiting":
                    statuses[stage] = "running"
                    break
        return {
            "symbol": cycle.symbol,
            "timeframe": cycle.timeframe,
            "candle_timestamp": cycle.boundary,
            "started_at": cycle.started_at,
            "updated_at": cycle.updated_at,
            "correlation_id": cycle.correlation_id,
            "complete": cycle.complete,
            "attempt": cycle.attempt,
            "stages": [
                {
                    "key": stage,
                    "label": STAGE_LABELS[stage],
                    "status": statuses[stage],
                    "updated_at": cycle.stage_updated_at.get(stage, cycle.started_at),
                    "error": cycle.errors.get(stage),
                }
                for stage in STAGE_KEYS
            ],
        }
