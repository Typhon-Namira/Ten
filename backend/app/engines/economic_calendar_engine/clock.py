from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime: ...


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock(Clock):
    def __init__(self, value: datetime) -> None:
        if value.tzinfo is None:
            raise ValueError("clock value must be timezone-aware")
        self.value = value

    def now(self) -> datetime:
        return self.value
