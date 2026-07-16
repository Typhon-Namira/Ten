from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Bias(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureEvent(BaseModel):
    kind: str
    direction: Bias
    price: float
    timestamp: datetime
    confidence: float = Field(ge=0, le=1)


class FairValueGap(BaseModel):
    direction: Bias
    low: float
    high: float
    timestamp: datetime
    mitigated: bool = False


class SMCResult(BaseModel):
    bias: Bias = Bias.NEUTRAL
    structure_events: list[StructureEvent] = Field(default_factory=list)
    fair_value_gaps: list[FairValueGap] = Field(default_factory=list)
    premium_discount_position: str = "equilibrium"
    observations: list[str] = Field(default_factory=list)

