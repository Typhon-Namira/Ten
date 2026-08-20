"""TEN 2.0 future-market contracts.

The core product contract is a probabilistic 30-minute market scenario map, not a
BUY/SELL signal.  The models are provider-agnostic so the bootstrap adapter can be
replaced by a Lightning world model without changing the API or dashboard.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FORECAST_HORIZON_SECONDS = 1800
FORECAST_CADENCE_SECONDS = 300
MAX_FORECASTS_PER_INSTRUMENT = 100


class FutureModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScenarioDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGE = "RANGE"
    INCONCLUSIVE = "INCONCLUSIVE"


class OpportunityState(StrEnum):
    WATCHING = "WATCHING"
    FORMING = "FORMING"
    ARMED = "ARMED"
    TRIGGERED = "TRIGGERED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class PriceZone(FutureModel):
    low: float = Field(gt=0)
    high: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> "PriceZone":
        if self.low > self.high:
            raise ValueError("price zone must be ordered")
        return self


class FuturePathStage(FutureModel):
    sequence: int = Field(ge=1)
    minute_from: int = Field(ge=0, le=30)
    minute_to: int = Field(ge=0, le=30)
    event: str
    expected_price_area: PriceZone | None = None
    invalidation_condition: str | None = None

    @model_validator(mode="after")
    def valid_window(self) -> "FuturePathStage":
        if self.minute_to < self.minute_from:
            raise ValueError("stage window must be ordered")
        return self


class ScenarioBranch(FutureModel):
    scenario_id: UUID
    scenario_type: str
    direction: ScenarioDirection
    probability: float = Field(ge=0, le=1)
    expected_range: PriceZone
    likely_close: PriceZone
    path: tuple[FuturePathStage, ...] = Field(min_length=1)
    invalidation: str
    rank: int = Field(ge=1)


class OpportunityWindow(FutureModel):
    opportunity_id: UUID
    scenario_id: UUID
    direction: ScenarioDirection
    state: OpportunityState = OpportunityState.WATCHING
    expected_from_minute: int = Field(ge=0, le=30)
    expected_to_minute: int = Field(ge=0, le=30)
    entry_zone: PriceZone
    trigger_conditions: tuple[str, ...] = Field(min_length=1)
    invalidation_level: float = Field(gt=0)
    targets: tuple[float, ...] = Field(min_length=1)
    probability: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def valid_window(self) -> "OpportunityWindow":
        if self.expected_to_minute < self.expected_from_minute:
            raise ValueError("opportunity window must be ordered")
        return self


class MarketStateSummary(FutureModel):
    regime: str
    uncertainty: float = Field(ge=0, le=1)
    reference_price: float = Field(gt=0)
    context_timeframes: tuple[str, ...] = ("M5", "M15", "H1")


class FutureMarketForecast(FutureModel):
    schema_version: str = "2.0"
    forecast_id: UUID
    instrument: str
    generated_at: datetime
    market_cutoff: datetime
    expires_at: datetime
    forecast_horizon_seconds: int = FORECAST_HORIZON_SECONDS
    forecast_cadence_seconds: int = FORECAST_CADENCE_SECONDS
    provider: str
    model_name: str
    model_version: str
    market_state: MarketStateSummary
    dominant_scenario_id: UUID | None
    scenarios: tuple[ScenarioBranch, ...]
    opportunities: tuple[OpportunityWindow, ...]

    @field_validator("generated_at", "market_cutoff", "expires_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("forecast timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def coherent(self) -> "FutureMarketForecast":
        if self.forecast_horizon_seconds != FORECAST_HORIZON_SECONDS:
            raise ValueError("TEN 2.0 forecast horizon is exactly 30 minutes")
        if self.forecast_cadence_seconds != FORECAST_CADENCE_SECONDS:
            raise ValueError("TEN 2.0 forecast cadence is exactly five minutes")
        if self.expires_at <= self.market_cutoff:
            raise ValueError("forecast must expire after market cutoff")
        if self.dominant_scenario_id is not None and self.dominant_scenario_id not in {
            item.scenario_id for item in self.scenarios
        }:
            raise ValueError("dominant scenario must belong to forecast")
        if self.scenarios:
            total = sum(item.probability for item in self.scenarios)
            if not 0.95 <= total <= 1.05:
                raise ValueError("scenario probabilities must approximately sum to one")
        return self


class ForecastPerformance(FutureModel):
    instrument: str
    forecasts_retained: int = Field(ge=0, le=MAX_FORECASTS_PER_INSTRUMENT)
    retention_limit: int = MAX_FORECASTS_PER_INSTRUMENT
    horizon_minutes: int = 30
    cadence_minutes: int = 5
    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
