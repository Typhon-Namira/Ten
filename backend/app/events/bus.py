"""Async in-process publish/subscribe event bus."""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .models import Event

EventT = TypeVar("EventT", bound=Event)
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus(ABC):
    @abstractmethod
    def subscribe(self, event_type: type[EventT], handler: EventHandler) -> Callable[[], None]:
        """Subscribe and return an idempotent unsubscription callback."""

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers in registration order."""

    async def drain(self) -> None:
        """Wait for accepted publications to finish."""
        raise NotImplementedError

    async def close(self) -> None:
        """Stop accepting new publications after draining."""
        raise NotImplementedError


class InMemoryEventBus(EventBus):
    """Process-local event bus with failure isolation and observable history."""

    def __init__(self, capacity: int = 1000, handler_timeout_seconds: float = 30.0) -> None:
        if capacity < 1 or handler_timeout_seconds <= 0:
            raise ValueError("event bus capacity and timeout must be positive")
        self._handlers: dict[type[Event], list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._lock = asyncio.Lock()
        self._capacity = asyncio.Semaphore(capacity)
        self._inflight: set[asyncio.Future[None]] = set()
        self._handler_timeout_seconds = handler_timeout_seconds
        self._closed = False
        self.published_total = 0
        self.failed_total = 0

    def subscribe(self, event_type: type[EventT], handler: EventHandler) -> Callable[[], None]:
        handlers = self._handlers[event_type]
        handlers.append(handler)

        def unsubscribe() -> None:
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: Event) -> None:
        if self._closed:
            raise RuntimeError("event bus is closed")
        async with self._capacity:
            async with self._lock:
                self._history.append(event)
                self.published_total += 1
                handlers = [handler for event_type, registered in self._handlers.items() if isinstance(event, event_type) for handler in registered]
            tasks: set[asyncio.Future[None]] = {asyncio.ensure_future(handler(event)) for handler in handlers}
            self._inflight.update(tasks)
            try:
                if tasks:
                    await asyncio.wait_for(asyncio.gather(*tasks), timeout=self._handler_timeout_seconds)
            except Exception:
                self.failed_total += 1
                raise
            finally:
                self._inflight.difference_update(tasks)

    async def drain(self) -> None:
        pending = tuple(self._inflight)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        await self.drain()

    def metrics(self) -> dict[str, int | bool]:
        return {"published_total": self.published_total, "failed_total": self.failed_total, "inflight": len(self._inflight), "closed": self._closed}

    def history(self) -> tuple[Event, ...]:
        return tuple(self._history)
