import asyncio
from uuid import uuid4

from backend.app.events import Event, InMemoryEventBus, MarketDataReady


def test_event_bus_publishes_typed_events_and_unsubscribes() -> None:
    bus = InMemoryEventBus()
    received: list[str] = []

    async def handler(event: Event) -> None:
        received.append(event.source)

    unsubscribe = bus.subscribe(MarketDataReady, handler)
    asyncio.run(bus.publish(MarketDataReady(correlation_id=uuid4(), source="market_data")))
    unsubscribe()
    asyncio.run(bus.publish(MarketDataReady(correlation_id=uuid4(), source="ignored")))
    assert received == ["market_data"]
    assert len(bus.history()) == 2
