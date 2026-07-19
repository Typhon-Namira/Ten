from __future__ import annotations

from datetime import UTC, datetime

from .exceptions import ReplayPointInTimeError


class ReplayClock:
    def __init__(self, start_at: datetime, end_at: datetime) -> None:
        if start_at.tzinfo is None or end_at.tzinfo is None:
            raise ValueError("replay clock bounds must be timezone-aware")
        self.start_at = start_at.astimezone(UTC)
        self.end_at = end_at.astimezone(UTC)
        if self.start_at >= self.end_at:
            raise ValueError("replay clock requires a positive interval")
        self._now = self.start_at
        self._frozen = False

    def now(self) -> datetime:
        return self._now

    def advance_to(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            raise ReplayPointInTimeError("virtual time must be timezone-aware")
        target = timestamp.astimezone(UTC)
        if self._frozen and target != self._now:
            raise ReplayPointInTimeError("virtual clock is frozen")
        if target < self._now:
            raise ReplayPointInTimeError("virtual clock cannot move backward")
        if target > self.end_at:
            raise ReplayPointInTimeError("virtual clock cannot exceed replay end")
        self._now = target

    def restore(self, timestamp: datetime) -> None:
        if self._frozen:
            raise ReplayPointInTimeError("virtual clock is frozen")
        if timestamp.tzinfo is None:
            raise ReplayPointInTimeError("checkpoint cursor must be timezone-aware")
        target = timestamp.astimezone(UTC)
        if not self.start_at <= target <= self.end_at:
            raise ReplayPointInTimeError("checkpoint cursor is outside replay bounds")
        self._now = target

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False


class ReplayClockAdapter:
    """Existing TEN engines consume this minimal synchronous clock contract."""

    def __init__(self, clock: ReplayClock) -> None:
        self.clock = clock

    def now(self) -> datetime:
        return self.clock.now()
