from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ScoredDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class ScoringContext(BaseModel):
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    market_structure: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    liquidity: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    flow_score: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    volume_profile: dict[str, Any] = Field(default_factory=dict, deprecated=True)
    news_risk: dict[str, Any] = Field(default_factory=dict, deprecated=True)

    @classmethod
    def from_features(cls, features: dict[str, dict[str, Any]], engine_versions: dict[str, str]) -> "ScoringContext":
        """Build the only context used by the production pipeline."""

        return cls(features=features, engine_versions=engine_versions)


class SignalScore(BaseModel):
    confidence: float | None = Field(default=None, ge=0, le=1, deprecated=True)
    direction: ScoredDirection
    quality_score: float = Field(ge=0, le=100)
    risk_notes: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
