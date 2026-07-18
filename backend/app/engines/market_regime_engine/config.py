from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VersionConfig(FrozenConfig):
    schema_version: str = "1.0"
    algorithm_version: str = "1.0.0"


class DependencyConfig(FrozenConfig):
    required: tuple[str, ...] = ("market_data",)
    optional: tuple[str, ...] = ("smc", "liquidity", "volume_profile", "institutional_flow")

    @model_validator(mode="after")
    def required_not_empty(self) -> "DependencyConfig":
        if not self.required:
            raise ValueError("at least one required dependency is required")
        return self


class WeightConfig(FrozenConfig):
    market_data: float = Field(0.8, ge=0, le=2)
    structure: float = Field(1.0, ge=0, le=2)
    liquidity: float = Field(0.75, ge=0, le=2)
    volume_profile: float = Field(1.0, ge=0, le=2)
    institutional_flow: float = Field(1.0, ge=0, le=2)
    volatility: float = Field(0.9, ge=0, le=2)
    session: float = Field(0.5, ge=0, le=2)
    multi_timeframe: float = Field(0.7, ge=0, le=2)
    persistence: float = Field(0.7, ge=0, le=2)
    transition: float = Field(0.6, ge=0, le=2)


class ThresholdConfig(FrozenConfig):
    minimum_candles: int = Field(20, ge=5, le=10000)
    trend: float = Field(0.2, ge=0, le=1)
    strong_trend: float = Field(0.48, ge=0, le=1)
    balance: float = Field(0.58, ge=0, le=1)
    compression: float = Field(0.62, ge=0, le=1)
    expansion: float = Field(0.60, ge=0, le=1)
    transition_watch: float = Field(0.35, ge=0, le=1)
    transition_confirm: float = Field(0.67, ge=0, le=1)
    high_volatility_percentile: float = Field(0.80, ge=0, le=1)
    low_volatility_percentile: float = Field(0.20, ge=0, le=1)

    @model_validator(mode="after")
    def ordering(self) -> "ThresholdConfig":
        if self.trend >= self.strong_trend:
            raise ValueError("trend threshold must be below strong trend")
        if self.transition_watch >= self.transition_confirm:
            raise ValueError("transition watch must be below confirmation")
        if self.low_volatility_percentile >= self.high_volatility_percentile:
            raise ValueError("low volatility percentile must be below high")
        return self


class EvidenceConfig(FrozenConfig):
    maximum_items: int = Field(512, ge=16, le=10000)
    decay_half_life_candles: int = Field(50, ge=1, le=10000)
    minimum_quality: float = Field(0.10, ge=0, le=1)
    correlation_group_cap: float = Field(1.5, gt=0, le=10)


class PersistenceConfig(FrozenConfig):
    window: int = Field(8, ge=2, le=1000)
    checkpoint_enabled: bool = True
    required_in_production: bool = True


class ProcessingConfig(FrozenConfig):
    default_candles: int = Field(500, ge=20, le=10000)
    maximum_candles: int = Field(100000, ge=100, le=200000)
    maximum_page_size: int = Field(1000, ge=1, le=5000)
    retention_snapshots: int = Field(5000, ge=10, le=100000)


class MultiTimeframeConfig(FrozenConfig):
    hierarchy: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
    maximum_depth: int = Field(5, ge=1, le=9)

    @model_validator(mode="after")
    def supported(self) -> "MultiTimeframeConfig":
        allowed = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"}
        if not self.hierarchy or any(item not in allowed for item in self.hierarchy):
            raise ValueError("unsupported timeframe")
        return self


class MarketRegimeConfig(FrozenConfig):
    version: str = "market-regime-1.0"
    compatibility_version: str = "1.0"
    enabled: bool = True
    versions: VersionConfig = Field(default_factory=VersionConfig)
    dependencies: DependencyConfig = Field(default_factory=DependencyConfig)
    weights: WeightConfig = Field(default_factory=WeightConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    multi_timeframe: MultiTimeframeConfig = Field(default_factory=MultiTimeframeConfig)
    repository_mode: str = "auto"

    @model_validator(mode="after")
    def repository(self) -> "MarketRegimeConfig":
        if self.repository_mode not in {"auto", "memory", "sqlalchemy"}:
            raise ValueError("invalid repository mode")
        if not self.version.startswith("market-regime-") or self.compatibility_version != self.versions.schema_version:
            raise ValueError("incompatible versions")
        return self
