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


class InMemoryEventBus(EventBus):
    """Process-local event bus with failure isolation and observable history."""

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: type[EventT], handler: EventHandler) -> Callable[[], None]:
        handlers = self._handlers[event_type]
        handlers.append(handler)

        def unsubscribe() -> None:
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: Event) -> None:
        async with self._lock:
            self._history.append(event)
            handlers = [handler for event_type, registered in self._handlers.items() if isinstance(event, event_type) for handler in registered]
        if handlers:
            await asyncio.gather(*(handler(event) for handler in handlers))

    def history(self) -> tuple[Event, ...]:
        return tuple(self._history)
