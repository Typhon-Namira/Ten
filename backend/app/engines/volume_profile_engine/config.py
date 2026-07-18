from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PriceGridConfig(FrozenConfig):
    method: str = "rows"
    rows: int = Field(24, ge=4, le=5000)
    tick_size: float = Field(0.01, gt=0, le=1000)
    fixed_increment: float = Field(0.25, gt=0, le=1000)
    percentage: float = Field(0.001, gt=0, le=0.1)
    atr_multiplier: float = Field(0.1, gt=0, le=10)
    minimum_bins: int = Field(4, ge=1, le=5000)
    maximum_bins: int = Field(500, ge=4, le=5000)

    @model_validator(mode="after")
    def bounds(self) -> "PriceGridConfig":
        if self.minimum_bins > self.maximum_bins:
            raise ValueError("minimum_bins cannot exceed maximum_bins")
        if self.method not in {"tick", "fixed", "rows", "percentage", "atr", "auto"}:
            raise ValueError("unsupported price grid method")
        return self


class AllocationConfig(FrozenConfig):
    method: str = "uniform_range"
    body_weight: float = Field(0.7, ge=0)
    wick_weight: float = Field(0.3, ge=0)
    directional_approximation: bool = False

    @model_validator(mode="after")
    def method_and_weights(self) -> "AllocationConfig":
        if self.method not in {"close", "typical_price", "uniform_range", "body_wick"}:
            raise ValueError("unsupported allocation method")
        if self.body_weight + self.wick_weight <= 0:
            raise ValueError("allocation weights must have positive sum")
        return self


class NodeConfig(FrozenConfig):
    high_percentile: float = Field(0.75, gt=0, lt=1)
    low_percentile: float = Field(0.25, gt=0, lt=1)
    minimum_prominence: float = Field(0.1, ge=0, le=1)
    merge_distance_bins: int = Field(1, ge=0, le=20)
    shelf_minimum_width: int = Field(3, ge=2, le=100)
    gap_maximum_ratio: float = Field(0.05, ge=0, le=0.5)


class ProcessingConfig(FrozenConfig):
    minimum_candles: int = Field(2, ge=1)
    maximum_candles: int = Field(100000, ge=10, le=100000)
    maximum_range_days: int = Field(366, ge=1, le=3660)
    maximum_composite_profiles: int = Field(32, ge=1, le=1000)


class MultiTimeframeConfig(FrozenConfig):
    hierarchy: tuple[str, ...] = ("M15", "H1", "H4", "D1", "W1", "MN1")
    maximum_depth: int = Field(4, ge=1, le=9)


class PersistenceConfig(FrozenConfig):
    required_in_production: bool = True


class VolumeProfileConfig(FrozenConfig):
    price_grid: PriceGridConfig = PriceGridConfig()
    allocation: AllocationConfig = AllocationConfig()
    nodes: NodeConfig = NodeConfig()
    processing: ProcessingConfig = ProcessingConfig()
    multi_timeframe: MultiTimeframeConfig = MultiTimeframeConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    value_area_percent: float = Field(0.70, gt=0, lt=1)
    default_volume_source: str = "unknown"
    allowed_volume_sources: tuple[str, ...] = ("exchange", "broker", "tick", "synthetic", "unknown")
    profile_types: tuple[str, ...] = ("developing", "fixed_range", "session", "daily", "weekly", "monthly", "composite", "anchored")
    configuration_name: str = "volume-profile-production-1"

    @model_validator(mode="after")
    def volume_source_policy(self) -> "VolumeProfileConfig":
        supported = {"exchange", "broker", "tick", "synthetic", "missing", "unknown"}
        if self.default_volume_source not in self.allowed_volume_sources or not set(self.allowed_volume_sources) <= supported:
            raise ValueError("default and allowed volume sources must use the supported volume-source policy")
        return self

    @property
    def bins(self) -> int:
        return self.price_grid.rows

    @property
    def high_volume_percentile(self) -> float:
        return self.nodes.high_percentile

    @property
    def version(self) -> str:
        payload = self.model_dump_json(exclude={"configuration_name"})
        return sha256(payload.encode()).hexdigest()[:16]
