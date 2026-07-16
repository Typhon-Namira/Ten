from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from backend.app.engines.ai_scoring_engine.models import SignalScore
from backend.app.engines.economic_calendar_engine.models import NewsRiskResult
from backend.app.engines.institutional_flow_engine.models import FlowScore
from backend.app.engines.liquidity_engine.models import LiquidityResult
from backend.app.engines.smc_engine.models import SMCResult
from backend.app.engines.volume_profile_engine.models import VolumeProfileResult


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class SignalInputs(BaseModel):
    symbol: str
    timeframe: str
    current_price: float = Field(gt=0)
    average_true_range: float = Field(gt=0)
    smc: SMCResult
    liquidity: LiquidityResult
    flow: FlowScore
    volume_profile: VolumeProfileResult
    news_risk: NewsRiskResult
    ai_score: SignalScore
    calculated_confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)


class SignalExplanation(BaseModel):
    reasons: list[str] = Field(default_factory=list)
    triggered_engines: list[str] = Field(default_factory=list)
    supporting_evidence: dict[str, list[str]] = Field(default_factory=dict)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    risk_notes: list[str] = Field(default_factory=list)
    rejected_conditions: list[str] = Field(default_factory=list)


class Signal(BaseModel):
    symbol: str
    timeframe: str
    direction: Direction
    entry_zone: tuple[float, float]
    stop_loss: float
    take_profit: float
    confidence: float = Field(ge=0, le=1)
    reasoning: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    explanation: SignalExplanation = Field(default_factory=SignalExplanation)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_entry_zone(self) -> "Signal":
        if self.entry_zone[0] > self.entry_zone[1]:
            raise ValueError("entry zone must be ordered low to high")
        return self
