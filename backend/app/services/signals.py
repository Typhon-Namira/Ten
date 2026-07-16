"""Signal repository port and thread-safe development implementation."""

from abc import ABC, abstractmethod

from backend.app.engines.signal_engine import Signal


class SignalRepository(ABC):
    @abstractmethod
    async def list(self, limit: int = 50) -> list[Signal]:
        """Return scenarios newest first."""

    @abstractmethod
    async def latest(self) -> Signal | None:
        """Return the newest scenario, if any."""

    @abstractmethod
    async def add(self, signal: Signal) -> None:
        """Store a scenario."""


class InMemorySignalRepository(SignalRepository):
    """Process-local adapter suitable for tests and zero-infrastructure startup."""

    def __init__(self, initial: list[Signal] | None = None) -> None:
        self._signals = sorted(initial or [], key=lambda item: item.timestamp, reverse=True)

    async def list(self, limit: int = 50) -> list[Signal]:
        return self._signals[:limit]

    async def latest(self) -> Signal | None:
        return self._signals[0] if self._signals else None

    async def add(self, signal: Signal) -> None:
        self._signals.append(signal)
        self._signals.sort(key=lambda item: item.timestamp, reverse=True)

