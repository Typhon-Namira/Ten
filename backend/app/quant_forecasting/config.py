"""Validated configuration for the Phase 2 shadow forecaster."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import ForecastHorizon


class QuantForecastingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    feature_schema_version: str
    model_name: str
    model_version: str
    minimum_numeric_features: int = Field(ge=1)
    neutral_band: float = Field(gt=0)
    horizons: tuple[ForecastHorizon, ...]

    @model_validator(mode="after")
    def exact_phase_two_horizons(self) -> "QuantForecastingConfig":
        expected = {("3_m1", "M1", 3), ("5_m1", "M1", 5), ("10_m1", "M1", 10), ("3_m5", "M5", 3)}
        actual = {(item.horizon_id, item.timeframe, item.candle_count) for item in self.horizons}
        if actual != expected or len(self.horizons) != 4:
            raise ValueError("Phase 2 requires exactly 3 M1, 5 M1, 10 M1, and 3 M5 horizons")
        return self
