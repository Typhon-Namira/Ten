"""Immutable contracts for evidence-grounded forward market scenarios."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ScenarioModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScenarioDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGE = "RANGE"
    INCONCLUSIVE = "INCONCLUSIVE"


class ScenarioValidity(StrEnum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class GeometryValidity(StrEnum):
    VALID = "VALID"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_EXECUTABLE = "NOT_EXECUTABLE"


class ScenarioOutcomeStatus(StrEnum):
    PENDING = "PENDING"
    TARGET_REACHED = "TARGET_REACHED"
    DIRECTION_CORRECT = "DIRECTION_CORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    RANGE_CORRECT = "RANGE_CORRECT"
    INVALIDATED = "INVALIDATED"
    ENTRY_NOT_REACHED = "ENTRY_NOT_REACHED"
    EXPIRED = "EXPIRED"
    INCONCLUSIVE = "INCONCLUSIVE"


class PriceZone(ScenarioModel):
    low: float = Field(gt=0)
    high: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> PriceZone:
        if self.low > self.high:
            raise ValueError("price zone must be ordered")
        return self


class ScenarioGeometry(ScenarioModel):
    entry_zone: PriceZone
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    secondary_target: float | None = Field(default=None, gt=0)
    risk_reward_ratio: float = Field(gt=0)
    basis_fact_identifiers: tuple[str, ...] = Field(min_length=1)
    validity: GeometryValidity
    reason: str

    @model_validator(mode="after")
    def coherent(self) -> ScenarioGeometry:
        if self.validity != GeometryValidity.VALID:
            raise ValueError("persisted scenario geometry must be valid")
        buy = self.stop_loss < self.entry < self.take_profit
        sell = self.take_profit < self.entry < self.stop_loss
        if not buy and not sell:
            raise ValueError("scenario geometry ordering is invalid")
        if not self.entry_zone.low <= self.entry <= self.entry_zone.high:
            raise ValueError("entry must belong to its executable zone")
        return self


class AlternativeScenario(ScenarioModel):
    direction: ScenarioDirection
    scenario_type: str
    expected_path: str
    probability: float = Field(ge=0, le=1)
    invalidation: str


class ForwardMarketScenario(ScenarioModel):
    schema_version: str = "1.0"
    scenario_id: UUID
    cycle_id: UUID
    market_state_id: UUID
    synthesis_id: UUID
    analysis_id: UUID
    quantitative_forecast_id: UUID
    instrument: str
    timeframe: str = Field(pattern=r"^(M5|M15)$")
    created_at: datetime
    market_cutoff_time: datetime
    reference_market_price: float = Field(gt=0)
    forecast_horizon_seconds: int = Field(gt=0)
    primary_direction: ScenarioDirection
    scenario_type: str
    expected_price_path: str
    expected_range: PriceZone
    expected_closing_zone: PriceZone
    expected_move: float = Field(ge=0)
    expected_move_unit: str = "price"
    expected_high: float = Field(gt=0)
    expected_low: float = Field(gt=0)
    entry_zone: PriceZone | None = None
    invalidation_level: float | None = Field(default=None, gt=0)
    protective_stop: float | None = Field(default=None, gt=0)
    primary_target: float | None = Field(default=None, gt=0)
    secondary_target: float | None = Field(default=None, gt=0)
    raw_directional_confidence: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=100)
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    calibration_status: str
    evidence_strength: float = Field(ge=0, le=100)
    supporting_fact_ids: tuple[str, ...]
    contradicting_fact_ids: tuple[str, ...]
    narrative: str
    alternative_scenario: AlternativeScenario
    expiry: datetime
    scenario_validity: ScenarioValidity
    scenario_validity_reason: str
    execution_geometry_validity: GeometryValidity
    geometry_rejection_reason: str | None = None
    geometry: ScenarioGeometry | None = None
    outcome_status: ScenarioOutcomeStatus = ScenarioOutcomeStatus.PENDING
    source_timeframe_cycle_id: UUID

    @field_validator("created_at", "market_cutoff_time", "expiry")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("scenario timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def point_in_time_and_geometry(self) -> ForwardMarketScenario:
        if self.created_at < self.market_cutoff_time:
            raise ValueError("scenario cannot be created before its market cutoff")
        if self.expiry <= self.market_cutoff_time:
            raise ValueError("scenario expiry must follow its market cutoff")
        expected = {"M5": 300, "M15": 900}[self.timeframe]
        if self.forecast_horizon_seconds != expected:
            raise ValueError("scenario horizon must match timeframe")
        if self.expected_range.low != self.expected_low or self.expected_range.high != self.expected_high:
            raise ValueError("expected range and high/low are inconsistent")
        if not self.expected_range.low <= self.reference_market_price <= self.expected_range.high:
            raise ValueError("expected range must contain the reference price")
        if self.execution_geometry_validity == GeometryValidity.VALID:
            if self.geometry is None or self.geometry_rejection_reason is not None:
                raise ValueError("valid execution geometry requires geometry without rejection")
        elif self.geometry is not None or not self.geometry_rejection_reason:
            raise ValueError("unavailable geometry requires an explicit rejection reason")
        if self.primary_direction in {ScenarioDirection.RANGE, ScenarioDirection.INCONCLUSIVE}:
            if self.geometry is not None:
                raise ValueError("range/inconclusive scenarios cannot fabricate trade geometry")
        return self


class ScenarioAgreement(StrEnum):
    ALIGNED = "ALIGNED"
    PULLBACK_COMPATIBLE = "PULLBACK_COMPATIBLE"
    CONFLICT = "CONFLICT"
    INCONCLUSIVE = "INCONCLUSIVE"


class CombinedForwardScenario(ScenarioModel):
    schema_version: str = "1.0"
    combined_scenario_id: UUID
    cycle_id: UUID
    instrument: str
    market_state_id: UUID
    m5_scenario_id: UUID
    m15_scenario_id: UUID
    created_at: datetime
    market_cutoff_time: datetime
    agreement: ScenarioAgreement
    combined_direction: ScenarioDirection
    expected_price_path: str
    confidence: float = Field(ge=0, le=100)
    scenario_validity: ScenarioValidity
    scenario_validity_reason: str
    execution_geometry_validity: GeometryValidity
    geometry_rejection_reason: str | None = None
    geometry: ScenarioGeometry | None = None
    expiry: datetime
    publication_status: str = "ANALYTICAL_ONLY"

    @field_validator("created_at", "market_cutoff_time", "expiry")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("combined scenario timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def geometry_contract(self) -> CombinedForwardScenario:
        if self.expiry <= self.market_cutoff_time:
            raise ValueError("combined scenario expiry must follow cutoff")
        if self.execution_geometry_validity == GeometryValidity.VALID:
            if self.geometry is None or self.agreement == ScenarioAgreement.CONFLICT:
                raise ValueError("valid combined geometry requires compatible scenarios")
        elif self.geometry is not None or not self.geometry_rejection_reason:
            raise ValueError("unavailable combined geometry requires a reason")
        return self


class ScenarioOutcome(ScenarioModel):
    outcome_id: UUID
    scenario_id: UUID
    evaluated_at: datetime
    completed_at: datetime
    status: ScenarioOutcomeStatus
    actual_high: float = Field(gt=0)
    actual_low: float = Field(gt=0)
    actual_close: float = Field(gt=0)
    maximum_favorable_excursion: float = Field(ge=0)
    maximum_adverse_excursion: float = Field(ge=0)
    entry_reached: bool
    target_reached: bool
    invalidation_occurred: bool
    directional_accuracy: float = Field(ge=0, le=1)
    target_error: float = Field(ge=0)
    range_error: float = Field(ge=0)
    high_prediction_error: float = Field(ge=0)
    low_prediction_error: float = Field(ge=0)
    close_prediction_error: float = Field(ge=0)
    calibration_bucket: str

    @field_validator("evaluated_at", "completed_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("outcome timestamps must be timezone-aware")
        return value.astimezone(UTC)
