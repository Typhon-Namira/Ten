from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EventImportance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EconomicEvent(BaseModel):
    event_id: str
    name: str
    currency: str = "USD"
    scheduled_at: datetime
    importance: EventImportance
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    source: str = "provider"


class NewsRiskResult(BaseModel):
    risk_level: EventImportance = EventImportance.LOW
    no_trade: bool = False
    active_events: list[EconomicEvent] = Field(default_factory=list)
    minutes_to_nearest: float | None = None
    observations: list[str] = Field(default_factory=list)

