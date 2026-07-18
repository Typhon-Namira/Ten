from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceConfig(FrozenConfig):
    maximum_items: int = Field(512, ge=16, le=10000)
    maximum_age_candles: int = Field(200, ge=1, le=10000)
    minimum_quality: float = Field(0.15, ge=0, le=1)
    decay_per_candle: float = Field(0.985, gt=0, le=1)


class CorrelationConfig(FrozenConfig):
    maximum_group_contribution: float = Field(1.5, gt=0, le=10)
    duplicate_tolerance_seconds: int = Field(60, ge=0, le=86400)
    correlated_discount: float = Field(0.65, gt=0, le=1)


class ThresholdConfig(FrozenConfig):
    moderate_participation: float = Field(0.35, ge=0, le=1)
    high_participation: float = Field(0.68, ge=0, le=1)
    initiative: float = Field(0.48, ge=0, le=1)
    responsive: float = Field(0.42, ge=0, le=1)
    absorption: float = Field(0.5, ge=0, le=1)
    exhaustion: float = Field(0.5, ge=0, le=1)
    inventory: float = Field(0.56, ge=0, le=1)
    conflict: float = Field(0.35, ge=0, le=1)
    strong_pressure: float = Field(0.62, ge=0, le=1)
    moderate_pressure: float = Field(0.2, ge=0, le=1)

    @model_validator(mode="after")
    def participation_order(self) -> "ThresholdConfig":
        if self.moderate_participation >= self.high_participation:
            raise ValueError("moderate participation threshold must be below high")
        return self


class WeightConfig(FrozenConfig):
    market_data: float = Field(0.75, ge=0, le=2)
    smc: float = Field(1.0, ge=0, le=2)
    liquidity: float = Field(0.9, ge=0, le=2)
    volume_profile: float = Field(0.9, ge=0, le=2)
    quality: float = Field(0.3, ge=0, le=1)
    diversity: float = Field(0.2, ge=0, le=1)


class PersistenceConfig(FrozenConfig):
    required_in_production: bool = True
    checkpoint_enabled: bool = True


class ProcessingConfig(FrozenConfig):
    maximum_candles: int = Field(100000, ge=100, le=200000)
    default_candles: int = Field(500, ge=20, le=10000)
    maximum_range_days: int = Field(3650, ge=1, le=36500)
    maximum_page_size: int = Field(1000, ge=1, le=5000)


class MultiTimeframeConfig(FrozenConfig):
    hierarchy: tuple[str, ...] = ("M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
    maximum_depth: int = Field(5, ge=1, le=9)


class InstitutionalFlowConfig(FrozenConfig):
    version: str = "institutional-flow-1.0"
    acceleration_weight: float = Field(0.35, ge=0, le=1)
    volume_weight: float = Field(0.45, ge=0, le=1)
    close_location_weight: float = Field(0.20, ge=0, le=1)
    evidence: EvidenceConfig = EvidenceConfig()
    correlation: CorrelationConfig = CorrelationConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    weights: WeightConfig = WeightConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    processing: ProcessingConfig = ProcessingConfig()
    multi_timeframe: MultiTimeframeConfig = MultiTimeframeConfig()
