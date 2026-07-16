from enum import StrEnum

from pydantic import BaseModel, Field


class MarketRegime(StrEnum):
    TRENDING = "trending"
    RANGING = "ranging"
    EXPANSION = "expansion"
    COMPRESSION = "compression"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class RegimeEvidence(BaseModel):
    feature_name: str
    value: float | str | None = None
    source_engine: str


class MarketRegimeResult(BaseModel):
    regime: MarketRegime = MarketRegime.UNKNOWN
    evidence: list[RegimeEvidence] = Field(default_factory=list)
    implemented: bool = False
