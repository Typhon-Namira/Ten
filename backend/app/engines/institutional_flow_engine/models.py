from enum import StrEnum

from pydantic import BaseModel, Field


class FlowBias(StrEnum):
    BUYING = "buying"
    SELLING = "selling"
    BALANCED = "balanced"


class FlowScore(BaseModel):
    score: float = Field(ge=-1, le=1)
    bias: FlowBias
    volume_pressure: float = Field(ge=-1, le=1)
    price_acceleration: float = Field(ge=-1, le=1)
    delta_estimate: float = Field(ge=-1, le=1)
    absorption_probability: float = Field(ge=0, le=1)
    methodology: str = "OHLCV estimation; not exchange order flow"
    observations: list[str] = Field(default_factory=list)

