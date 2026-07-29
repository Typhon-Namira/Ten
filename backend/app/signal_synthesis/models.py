"""Multi-factor, multi-timeframe analytical signal contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SynthesisModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnalyticalDirection(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class AnalyticalStrength(StrEnum):
    VERY_WEAK = "VERY_WEAK"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class ExecutionEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


class ExecutionStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class DirectionalContribution(SynthesisModel):
    evidence_id: UUID
    tool: str
    timeframe: str
    family: str
    correlation_group: str
    directional_contribution: AnalyticalDirection
    raw_value: Any
    normalized_score: float = Field(ge=-1, le=1)
    weight: float = Field(ge=0, le=1)
    effective_weight: float = Field(ge=0, le=1)
    weighted_score: float = Field(ge=-100, le=100)
    quality: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    correlated_discount: float = Field(ge=0, le=1)
    reason: str
    source_fact_identifiers: tuple[str, ...]


class ConfidenceDecomposition(SynthesisModel):
    score_separation: float = Field(ge=0, le=100)
    independent_confluence: float = Field(ge=0, le=100)
    evidence_quality: float = Field(ge=0, le=100)
    evidence_freshness: float = Field(ge=0, le=100)
    evidence_completeness: float = Field(ge=0, le=100)
    timeframe_alignment: float = Field(ge=0, le=100)
    quant_ai_alignment: float = Field(ge=0, le=100)
    contradiction_penalty: float = Field(ge=0, le=100)
    missing_evidence_penalty: float = Field(ge=0, le=100)
    regime_suitability_penalty: float = Field(ge=0, le=100)
    final_confidence: float = Field(ge=0, le=100)


class SignalGeometry(SynthesisModel):
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    risk_reward_ratio: float = Field(gt=0)
    basis_fact_identifiers: tuple[str, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def coherent(self) -> SignalGeometry:
        if len({self.entry, self.stop_loss, self.take_profit}) != 3:
            raise ValueError("entry, stop loss, and take profit must be distinct")
        return self


class TimeframeAnalyticalSignal(SynthesisModel):
    signal_id: UUID
    synthesis_id: UUID
    market_state_id: UUID
    analysis_id: UUID
    quantitative_forecast_id: UUID
    instrument: str
    timeframe: str = Field(pattern=r"^(M5|M15|COMBINED)$")
    analytical_direction: AnalyticalDirection
    confidence: float = Field(ge=0, le=100)
    strength: AnalyticalStrength
    bullish_score: float = Field(ge=0)
    bearish_score: float = Field(ge=0)
    expected_horizon: str
    evidence_breakdown: tuple[DirectionalContribution, ...] = Field(min_length=1)
    confidence_decomposition: ConfidenceDecomposition
    directional_thesis: str
    invalidation_conditions: tuple[str, ...]
    execution_eligibility: ExecutionEligibility
    execution_status: ExecutionStatus
    blocking_reasons: tuple[str, ...]
    geometry: SignalGeometry | None = None
    completed_at: datetime

    @field_validator("completed_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("signal timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def execution_is_separate_from_direction(self) -> TimeframeAnalyticalSignal:
        if self.execution_eligibility == ExecutionEligibility.ELIGIBLE:
            if self.execution_status != ExecutionStatus.READY or self.geometry is None:
                raise ValueError("eligible signal requires ready status and validated geometry")
            if self.blocking_reasons:
                raise ValueError("eligible signal cannot retain execution blockers")
        elif self.execution_status != ExecutionStatus.BLOCKED or not self.blocking_reasons:
            raise ValueError("ineligible signal requires explicit blocking reasons")
        return self


class TimeframeContribution(SynthesisModel):
    timeframe: str = Field(pattern=r"^(M5|M15)$")
    direction: AnalyticalDirection
    confidence: float = Field(ge=0, le=100)
    structural_importance: float = Field(gt=0, le=1)
    evidence_quality: float = Field(ge=0, le=1)
    signed_contribution: float = Field(ge=-100, le=100)
    explanation: str


class MultiTimeframeSignalSet(SynthesisModel):
    schema_version: str = "1.0"
    synthesis_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    analysis_id: UUID
    quantitative_forecast_id: UUID
    instrument: str
    market_timestamp: datetime
    timeframe_signals: tuple[TimeframeAnalyticalSignal, ...]
    combined_signal: TimeframeAnalyticalSignal
    timeframe_contributions: tuple[TimeframeContribution, ...]
    engine_version: str
    configuration_version: str
    created_at: datetime

    @field_validator("market_timestamp", "created_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("synthesis timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def complete_matrix(self) -> MultiTimeframeSignalSet:
        if tuple(item.timeframe for item in self.timeframe_signals) != ("M5", "M15"):
            raise ValueError("synthesis requires independent M5 and M15 signals")
        if self.combined_signal.timeframe != "COMBINED":
            raise ValueError("combined signal must use the COMBINED scope")
        if tuple(item.timeframe for item in self.timeframe_contributions) != ("M5", "M15"):
            raise ValueError("combined synthesis must preserve every timeframe contribution")
        return self


def strength_for(
    confidence: float,
    thresholds: tuple[float, float, float, float] = (40, 55, 70, 85),
) -> AnalyticalStrength:
    very_weak, weak, moderate, strong = thresholds
    if confidence < very_weak:
        return AnalyticalStrength.VERY_WEAK
    if confidence < weak:
        return AnalyticalStrength.WEAK
    if confidence < moderate:
        return AnalyticalStrength.MODERATE
    if confidence < strong:
        return AnalyticalStrength.STRONG
    return AnalyticalStrength.VERY_STRONG
