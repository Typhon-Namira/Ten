"""Immutable contracts for quantitative forecasts produced in shadow mode."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ImmutableForecastModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ForecastStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INSUFFICIENT_HISTORY = "insufficient_history"
    INCOMPATIBLE_FEATURES = "incompatible_features"
    FAILED = "failed"


class FeatureAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    STALE = "stale"


class ForecastMode(StrEnum):
    SHADOW = "shadow"
    REPLAY = "replay"


class CalibrationStatus(StrEnum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class ForecastHorizon(ImmutableForecastModel):
    horizon_id: str
    timeframe: str
    candle_count: int = Field(gt=0)
    duration_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def duration_matches_timeframe(self) -> ForecastHorizon:
        seconds = {"M5": 300, "M15": 900}.get(self.timeframe)
        if seconds is None or self.duration_seconds != seconds * self.candle_count:
            raise ValueError("forecast horizon duration must match its candle timeframe")
        return self


class QuantFeatureValue(ImmutableForecastModel):
    name: str
    value: float | str | bool | None = None
    availability: FeatureAvailability
    source_evidence_ids: tuple[UUID, ...] = ()
    source_paths: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unavailable_is_not_zero_filled(self) -> QuantFeatureValue:
        if self.availability == FeatureAvailability.UNAVAILABLE and self.value is not None:
            raise ValueError("unavailable quantitative features cannot carry a value")
        return self


class QuantFeatureVector(ImmutableForecastModel):
    schema_version: str
    vector_id: UUID
    market_state_id: UUID
    instrument: str
    point_in_time: datetime
    features: tuple[QuantFeatureValue, ...]
    created_at: datetime

    @field_validator("point_in_time", "created_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("forecast timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def immutable_point_in_time_vector(self) -> QuantFeatureVector:
        if self.created_at < self.point_in_time:
            raise ValueError("feature vector cannot be created before its point-in-time boundary")
        names = [item.name for item in self.features]
        if len(names) != len(set(names)):
            raise ValueError("feature vector names must be unique")
        return self


class QuantForecastRequest(ImmutableForecastModel):
    schema_version: str = "1.0"
    request_id: UUID
    market_state_id: UUID
    cycle_id: UUID
    instrument: str
    point_in_time: datetime
    requested_horizons: tuple[ForecastHorizon, ...]
    feature_schema_version: str
    model_name: str
    model_version: str
    mode: ForecastMode = ForecastMode.SHADOW
    data_quality_status: str
    created_at: datetime

    @field_validator("point_in_time", "created_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("forecast timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def request_is_complete(self) -> QuantForecastRequest:
        ids = [item.horizon_id for item in self.requested_horizons]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("forecast request horizons must be non-empty and unique")
        return self


class UncertaintyInterval(ImmutableForecastModel):
    low: float
    high: float
    confidence_level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def ordered(self) -> UncertaintyInterval:
        if self.low > self.high:
            raise ValueError("uncertainty interval must be ordered")
        return self


class HorizonPrediction(ImmutableForecastModel):
    horizon: ForecastHorizon
    reference_price: float = Field(gt=0)
    buy_probability: float = Field(ge=0, le=1)
    sell_probability: float = Field(ge=0, le=1)
    neutral_probability: float = Field(ge=0, le=1)
    expected_return: float
    expected_base_movement: float = Field(ge=0)
    expected_minimum_movement: float = Field(ge=0)
    expected_maximum_movement: float = Field(ge=0)
    expected_volatility: float = Field(ge=0)
    expected_mfe: float = Field(ge=0)
    expected_mae: float = Field(ge=0)
    tp1_probability: float = Field(ge=0, le=1)
    tp2_probability: float = Field(ge=0, le=1)
    stop_loss_probability: float = Field(ge=0, le=1)
    sl_before_tp_probability: float = Field(ge=0, le=1)
    uncertainty_interval: UncertaintyInterval
    transition_probabilities: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def probabilities_and_movements_are_coherent(self) -> HorizonPrediction:
        if abs(self.buy_probability + self.sell_probability + self.neutral_probability - 1) > 1e-8:
            raise ValueError("direction probabilities must sum to one")
        if not self.expected_minimum_movement <= self.expected_base_movement <= self.expected_maximum_movement:
            raise ValueError("expected movement bounds are inconsistent")
        if self.transition_probabilities and abs(sum(self.transition_probabilities.values()) - 1) > 1e-8:
            raise ValueError("transition probabilities must sum to one")
        if any(value < 0 or value > 1 for value in self.transition_probabilities.values()):
            raise ValueError("transition probabilities must be probabilities")
        return self


class QuantForecastResult(ImmutableForecastModel):
    schema_version: str = "1.0"
    result_id: UUID
    request_id: UUID
    market_state_id: UUID
    cycle_id: UUID
    instrument: str
    point_in_time: datetime
    status: ForecastStatus
    mode: ForecastMode
    model_name: str
    model_version: str
    training_dataset_version: str
    feature_schema_version: str
    calibration_version: str
    model_kind: str
    calibration_status: CalibrationStatus
    shadow_only: bool = True
    approved_for_publication: bool = False
    predictions: tuple[HorizonPrediction, ...] = ()
    reason_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime

    @field_validator("point_in_time", "generated_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("forecast timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def publication_and_status_invariants(self) -> QuantForecastResult:
        if not self.shadow_only or self.approved_for_publication:
            raise ValueError("Phase 2 forecasts are shadow-only and cannot be published")
        if self.status == ForecastStatus.AVAILABLE and not self.predictions:
            raise ValueError("available forecast must include horizon predictions")
        if self.status != ForecastStatus.AVAILABLE and self.predictions:
            raise ValueError("unavailable/failed forecasts cannot fabricate numeric predictions")
        return self


class ModelMetadata(ImmutableForecastModel):
    model_name: str
    model_version: str
    model_kind: str
    training_dataset_version: str
    feature_schema_version: str
    calibration_version: str
    calibration_status: CalibrationStatus
    shadow_only: bool = True
    approved_for_publication: bool = False
    deterministic: bool


class ModelHealth(ImmutableForecastModel):
    status: str
    ready: bool
    model_name: str
    model_version: str
    calibration_status: CalibrationStatus
    detail: str


class OutcomeStatus(StrEnum):
    PENDING = "pending"
    VALID = "valid"
    MISSING_DATA = "missing_data"
    INCOMPLETE = "incomplete"


class ForecastOutcome(ImmutableForecastModel):
    outcome_id: UUID
    forecast_result_id: UUID
    horizon_id: str
    status: OutcomeStatus
    evaluated_at: datetime
    realized_return: float | None = None
    realized_direction: str | None = None
    maximum_favorable_excursion: float | None = None
    maximum_adverse_excursion: float | None = None
    tp1_hit: bool | None = None
    tp2_hit: bool | None = None
    stop_loss_hit: bool | None = None
    stop_before_tp: bool | None = None
    spread_adjusted_return: float | None = None
    candle_count: int = Field(ge=0)
    reason_codes: tuple[str, ...] = ()

    @field_validator("evaluated_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("outcome timestamp must be timezone-aware")
        return value.astimezone(UTC)


class CalibrationBucket(ImmutableForecastModel):
    horizon_id: str
    dimension: str
    dimension_value: str
    probability_low: float = Field(ge=0, le=1)
    probability_high: float = Field(ge=0, le=1)
    count: int = Field(ge=0)
    mean_confidence: float | None = None
    observed_frequency: float | None = None


class CalibrationObservation(ImmutableForecastModel):
    prediction: HorizonPrediction
    outcome: ForecastOutcome
    session: str
    regime: str
    confidence_band: str
    data_quality_status: str


class CalibrationReport(ImmutableForecastModel):
    report_id: UUID
    model_name: str
    model_version: str
    generated_at: datetime
    sample_count: int = Field(ge=0)
    brier_score: float | None = None
    log_loss: float | None = None
    expected_calibration_error: float | None = None
    buckets: tuple[CalibrationBucket, ...] = ()
    status: CalibrationStatus
    filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("calibration timestamp must be timezone-aware")
        return value.astimezone(UTC)
