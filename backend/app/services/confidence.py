"""Deterministic, configuration-driven confidence calculation."""

from pydantic import BaseModel, Field, model_validator

from backend.app.core.config import YamlConfigRepository


class ConfidenceWeights(BaseModel):
    smc: float = Field(ge=0)
    liquidity: float = Field(ge=0)
    institutional_flow: float = Field(ge=0)
    volume_profile: float = Field(ge=0)
    economic_risk: float = Field(ge=0)
    ai_bonus: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "ConfidenceWeights":
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("At least one confidence weight must be positive")
        return self


class ConfidenceConfig(BaseModel):
    weights: ConfidenceWeights
    maximum_ai_bonus: float = Field(default=0.10, ge=0, le=1)


class ConfidenceFactors(BaseModel):
    smc: float = Field(default=0, ge=0, le=1)
    liquidity: float = Field(default=0, ge=0, le=1)
    institutional_flow: float = Field(default=0, ge=0, le=1)
    volume_profile: float = Field(default=0, ge=0, le=1)
    economic_risk: float = Field(default=0, ge=0, le=1)
    ai_bonus: float = Field(default=0, ge=0, le=1)


class ConfidenceResult(BaseModel):
    confidence: float = Field(ge=0, le=1)
    breakdown: dict[str, float]
    weights: dict[str, float]
    methodology: str = "deterministic_weighted_v1"


class ConfidenceCalculator:
    """Calculates confidence from numeric system evidence, never LLM confidence."""

    def __init__(self, config: ConfidenceConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, configs: YamlConfigRepository) -> "ConfidenceCalculator":
        return cls(configs.load_model("confidence", ConfidenceConfig))

    def calculate(self, factors: ConfidenceFactors) -> ConfidenceResult:
        weights = self.config.weights.model_dump()
        values = factors.model_dump()
        ai_weighted = min(values["ai_bonus"] * weights["ai_bonus"], self.config.maximum_ai_bonus)
        weighted = {name: values[name] * weight for name, weight in weights.items()}
        weighted["ai_bonus"] = ai_weighted
        total_weight = sum(weights.values())
        confidence = max(0.0, min(1.0, sum(weighted.values()) / total_weight))
        return ConfidenceResult(confidence=confidence, breakdown=weighted, weights=weights)
