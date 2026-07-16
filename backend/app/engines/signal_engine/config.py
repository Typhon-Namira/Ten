from pydantic import BaseModel, Field


class SignalConfig(BaseModel):
    minimum_confidence: float = Field(default=0.55, ge=0, le=1)
    risk_reward_ratio: float = Field(default=2.0, gt=0)
    atr_stop_multiplier: float = Field(default=1.25, gt=0)

