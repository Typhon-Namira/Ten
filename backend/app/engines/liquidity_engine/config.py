"""Validated and content-versioned Liquidity Engine configuration."""

from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True)


class ToleranceConfig(FrozenConfig):
    absolute: float = Field(default=0.0, ge=0)
    ticks: float = Field(default=2, ge=0)
    tick_size: float = Field(default=0.01, gt=0)
    atr_multiplier: float = Field(default=0.08, ge=0)
    percentage: float = Field(default=0.0005, ge=0, le=0.1)


class EqualLevelConfig(FrozenConfig):
    enabled: bool = True
    allow_micro_candles: bool = True
    minimum_touches: int = Field(default=2, ge=2)
    minimum_separation_candles: int = Field(default=1, ge=1)
    maximum_age_candles: int = Field(default=1000, ge=2)
    merge_multiplier: float = Field(default=1.25, ge=1)
    split_multiplier: float = Field(default=2.5, gt=1)
    outlier_zscore: float = Field(default=2.5, gt=0)


class PoolConfig(FrozenConfig):
    maximum_active: int = Field(default=1000, ge=10)
    approach_atr: float = Field(default=0.25, ge=0)
    expiration_candles: int = Field(default=2000, ge=10)
    structural_weight: float = Field(default=0.35, ge=0, le=1)
    temporal_weight: float = Field(default=0.25, ge=0, le=1)


class SweepConfig(FrozenConfig):
    minimum_penetration_atr: float = Field(default=0.02, ge=0)
    deep_penetration_atr: float = Field(default=0.5, gt=0)
    reclaim_candles: int = Field(default=3, ge=1, le=50)
    raid_minimum_pools: int = Field(default=2, ge=2)
    stop_hunt_minimum_confidence: float = Field(default=65, ge=0, le=100)
    false_break_hold_candles: int = Field(default=1, ge=1, le=20)


class SessionConfig(FrozenConfig):
    enabled: bool = True
    opening_range_candles: int = Field(default=4, ge=1, le=100)
    include_overlap: bool = True


class ReferenceConfig(FrozenConfig):
    previous_day: bool = True
    previous_week: bool = True
    previous_month: bool = True
    current_periods: bool = True


class RoundNumberConfig(FrozenConfig):
    enabled: bool = True
    increments: dict[str, float] = Field(default_factory=lambda: {"XAUUSD": 10.0, "DEFAULT": 1.0})
    minor_divisor: int = Field(default=2, ge=1)
    maximum_distance_atr: float = Field(default=10, gt=0)
    confidence_cap: float = Field(default=55, ge=0, le=100)


class RankingConfig(FrozenConfig):
    distance_weight: float = Field(default=0.25, ge=0, le=1)
    strength_weight: float = Field(default=0.3, ge=0, le=1)
    freshness_weight: float = Field(default=0.2, ge=0, le=1)
    scope_weight: float = Field(default=0.15, ge=0, le=1)
    quality_weight: float = Field(default=0.1, ge=0, le=1)


class MultiTimeframeConfig(FrozenConfig):
    hierarchy: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
    maximum_depth: int = Field(default=5, ge=1, le=9)
    overlap_tolerance_atr: float = Field(default=0.1, ge=0)


class ProcessingConfig(FrozenConfig):
    minimum_history: int = Field(default=5, ge=3)
    maximum_candles: int = Field(default=10000, ge=100)
    candidate_lookback: int = Field(default=2000, ge=10)
    checkpoint_interval: int = Field(default=100, ge=1)
    minimum_input_quality: float = Field(default=60, ge=0, le=100)


class PersistenceConfig(FrozenConfig):
    required_in_production: bool = True
    query_limit: int = Field(default=500, ge=1, le=5000)


class LiquidityConfig(FrozenConfig):
    tolerances: ToleranceConfig = ToleranceConfig()
    equal_levels: EqualLevelConfig = EqualLevelConfig()
    pools: PoolConfig = PoolConfig()
    sweeps: SweepConfig = SweepConfig()
    sessions: SessionConfig = SessionConfig()
    references: ReferenceConfig = ReferenceConfig()
    round_numbers: RoundNumberConfig = RoundNumberConfig()
    ranking: RankingConfig = RankingConfig()
    multi_timeframe: MultiTimeframeConfig = MultiTimeframeConfig()
    processing: ProcessingConfig = ProcessingConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    algorithm_version: str = "1.0.0"

    @model_validator(mode="after")
    def valid_weights(self) -> "LiquidityConfig":
        total = (
            self.ranking.distance_weight
            + self.ranking.strength_weight
            + self.ranking.freshness_weight
            + self.ranking.scope_weight
            + self.ranking.quality_weight
        )
        if abs(total - 1) > 1e-9:
            raise ValueError("target ranking weights must sum to 1")
        if not any((self.tolerances.absolute, self.tolerances.ticks, self.tolerances.atr_multiplier, self.tolerances.percentage)):
            raise ValueError("at least one equality tolerance must be enabled")
        return self

    @property
    def version(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()[:16]

    @property
    def equal_level_tolerance(self) -> float:
        return self.tolerances.percentage

    @property
    def max_levels(self) -> int:
        return self.pools.maximum_active
