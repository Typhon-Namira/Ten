"""Market Data Engine configuration."""

from pydantic import BaseModel, Field

from .models import Timeframe


class ProviderConfig(BaseModel):
    name: str
    enabled: bool = True
    priority: int = Field(default=100, ge=0)
    base_url: str
    # `None` for keyless public-source providers (LBMA, Kraken, OKX, ...) — `build_market_data_service`
    # only requires and reads an env var for providers that actually need one.
    api_key_env: str | None = None
    account_id_env: str | None = None
    request_timeout_seconds: float = Field(default=15, gt=0)
    quota_per_minute: int | None = Field(default=None, ge=1)


class CacheConfig(BaseModel):
    memory_max_entries: int = Field(default=10_000, ge=1)
    realtime_ttl_seconds: int = Field(default=15, ge=1)
    historical_ttl_seconds: int = Field(default=3600, ge=1)
    persistent_directory: str = "data/market_cache"


class ValidationConfig(BaseModel):
    future_tolerance_seconds: int = Field(default=5, ge=0)
    clock_drift_tolerance_seconds: int = Field(default=30, ge=0)
    gap_tolerance_multiplier: float = Field(default=1.5, ge=1)
    volatility_spike_multiplier: float = Field(default=8, gt=1)
    minimum_quality_score: float = Field(default=60, ge=0, le=100)
    # Cross-source close-price deviation thresholds used by `MarketDataValidator.compare()`.
    # Below `cross_source_tolerance`: no anomaly. Between the two: flagged as a
    # `PROVIDER_INCONSISTENCY` anomaly but the candle is still served. Above
    # `cross_source_quarantine_tolerance`: the candle is dropped entirely and treated as a gap
    # (never fabricated/interpolated) — see `MarketDataService._cross_check()`.
    cross_source_tolerance: float = Field(default=0.01, gt=0)
    cross_source_quarantine_tolerance: float = Field(default=0.05, gt=0)


class MarketDataConfig(BaseModel):
    symbols: tuple[str, ...] = ("XAUUSD",)
    timeframes: tuple[Timeframe, ...] = tuple(Timeframe)
    default_lookback: int = Field(default=500, ge=1, le=100_000)
    providers: tuple[ProviderConfig, ...] = ()
    cache: CacheConfig = CacheConfig()
    validation: ValidationConfig = ValidationConfig()
    preferred_provider: str = "twelve_data"
    provider_recovery_successes: int = Field(default=3, ge=1)
    provider_failure_threshold: int = Field(default=3, ge=1)
