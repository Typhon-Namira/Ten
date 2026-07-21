"""Per-source circuit breaker for the public calendar adapters.

Distinct from the retry/backoff inside one HTTP call (see `HttpPublicCalendarSource._get()`) —
this tracks failures ACROSS sync cycles, so a source that has been failing for hours doesn't get
hammered every 6-12 hour schedule-refresh regardless of the reason. Failure categories matter:
a "parser mismatch" (the page layout changed) should back off hard and surface diagnostics rather
than retry rapidly, since retrying won't fix a layout change — only a code update will.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class FailureCategory(StrEnum):
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    HTTP_ERROR = "http_error"
    BLOCKED = "blocked_automated_access"
    PARSER_MISMATCH = "parser_mismatch"
    EMPTY_SCHEDULE = "empty_valid_schedule"
    INVALID_TIMEZONE = "invalid_timezone"
    LAYOUT_CHANGED = "source_layout_changed"


#: Categories where retrying soon is pointless — the fix is a code/config change, not persistence.
#: These get a much longer backoff floor than a transient timeout or 5xx.
_STRUCTURAL_FAILURES = frozenset({FailureCategory.PARSER_MISMATCH, FailureCategory.LAYOUT_CHANGED, FailureCategory.BLOCKED, FailureCategory.INVALID_TIMEZONE})

_BASE_BACKOFF = timedelta(minutes=5)
_STRUCTURAL_BACKOFF_FLOOR = timedelta(hours=6)
_MAX_BACKOFF = timedelta(hours=24)


@dataclass
class CircuitBreakerState:
    consecutive_failures: int = 0
    last_failure_at: datetime | None = None
    last_failure_category: FailureCategory | None = None
    last_success_at: datetime | None = None
    open_until: datetime | None = None

    def record_success(self, now: datetime) -> None:
        self.consecutive_failures = 0
        self.last_failure_category = None
        self.last_success_at = now
        self.open_until = None

    def record_failure(self, now: datetime, category: FailureCategory) -> None:
        self.consecutive_failures += 1
        self.last_failure_at = now
        self.last_failure_category = category
        floor = _STRUCTURAL_BACKOFF_FLOOR if category in _STRUCTURAL_FAILURES else _BASE_BACKOFF
        backoff = min(floor * (2 ** min(self.consecutive_failures - 1, 6)), _MAX_BACKOFF)
        self.open_until = now + backoff

    def should_attempt(self, now: datetime) -> bool:
        return self.open_until is None or now >= self.open_until


class CircuitBreakerRegistry:
    """One `CircuitBreakerState` per source name — a failing BLS fetch must never throttle BEA's
    schedule, and vice versa."""

    def __init__(self) -> None:
        self._states: dict[str, CircuitBreakerState] = {}

    def get(self, source_name: str) -> CircuitBreakerState:
        return self._states.setdefault(source_name, CircuitBreakerState())

    def should_attempt(self, source_name: str, *, now: datetime | None = None) -> bool:
        return self.get(source_name).should_attempt(now or datetime.now(UTC))

    def record_success(self, source_name: str, *, now: datetime | None = None) -> None:
        self.get(source_name).record_success(now or datetime.now(UTC))

    def record_failure(self, source_name: str, category: FailureCategory, *, now: datetime | None = None) -> None:
        self.get(source_name).record_failure(now or datetime.now(UTC), category)


@dataclass
class ParserFailureLog:
    """A small bounded ring buffer of recent parser failures per source — surfaced verbatim in
    diagnostics so "the source layout changed" is visible without reading application logs."""

    entries: list[dict[str, object]] = field(default_factory=list)
    max_entries: int = 20

    def record(self, source_name: str, category: FailureCategory, message: str, *, now: datetime) -> None:
        self.entries.append({"source": source_name, "category": category.value, "message": message[:500], "at": now})
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def for_source(self, source_name: str) -> list[dict[str, object]]:
        return [entry for entry in self.entries if entry["source"] == source_name]
