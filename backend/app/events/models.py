"""Typed internal domain events used to decouple pipeline stages."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Event(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)


class MarketDataReady(Event):
    pass


class SMCCompleted(Event):
    pass


class LiquidityCompleted(Event):
    pass


class FlowCompleted(Event):
    pass


class VolumeProfileCompleted(Event):
    pass


class EconomicCalendarCompleted(Event):
    pass


class AICompleted(Event):
    pass


class SignalGenerated(Event):
    pass


class DashboardUpdated(Event):
    pass
