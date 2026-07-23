from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import ProviderMode


def _symbol_overrides() -> dict[str, tuple[str, ...]]:
    return {"XAUUSD": ("USD",), "BTCUSD": ("USD",), "SPX": ("USD",), "DAX": ("EUR",)}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionConfig(StrictModel):
    schema_version: str = "1.0"
    normalization_version: str = "1.0.0"


class ProviderConfig(StrictModel):
    name: str
    mode: ProviderMode = ProviderMode.DISABLED
    enabled: bool = False
    version: str = "unknown"
    timezone: str = "UTC"
    priority: int = Field(default=100, ge=0)
    timeout_seconds: float = Field(default=10, gt=0, le=120)
    retry_count: int = Field(default=2, ge=0, le=10)
    requests_per_minute: int = Field(default=30, ge=1, le=10000)
    circuit_failure_threshold: int = Field(default=5, ge=1, le=100)
    file_path: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None

    @field_validator("timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("invalid timezone") from exc
        return value


class RiskWindowConfig(StrictModel):
    pre_minutes: int = Field(ge=0, le=10080)
    imminent_minutes: int = Field(ge=0, le=1440)
    post_minutes: int = Field(ge=0, le=10080)
    cooldown_minutes: int = Field(ge=0, le=10080)

    @model_validator(mode="after")
    def ordering(self) -> RiskWindowConfig:
        if self.imminent_minutes > self.pre_minutes:
            raise ValueError("imminent window cannot exceed pre-event window")
        return self


class ApiConfig(StrictModel):
    default_page_size: int = Field(default=50, ge=1, le=1000)
    maximum_page_size: int = Field(default=500, ge=1, le=5000)
    maximum_query_window_days: int = Field(default=366, ge=1, le=3650)
    maximum_history_window_days: int = Field(default=3650, ge=1, le=36500)
    maximum_observation_records: int = Field(default=500, ge=1, le=10000)
    maximum_revision_records: int = Field(default=500, ge=1, le=10000)


class ProcessingConfig(StrictModel):
    polling_interval_seconds: int = Field(default=3600, ge=300, le=86400)
    adaptive_refresh_seconds: int = Field(default=300, ge=60, le=3600)
    adaptive_lookahead_minutes: int = Field(default=120, ge=30, le=1440)
    failure_backoff_initial_seconds: int = Field(default=60, ge=30, le=3600)
    failure_backoff_max_seconds: int = Field(default=900, ge=60, le=21600)
    retention_events: int = Field(default=100000, ge=100, le=10_000_000)
    retention_observations: int = Field(default=500000, ge=100, le=50_000_000)
    retention_snapshots: int = Field(default=5000, ge=10, le=1_000_000)
    cache_size: int = Field(default=1000, ge=1, le=100000)
    maximum_batch_size: int = Field(default=10000, ge=1, le=100000)

    @model_validator(mode="after")
    def scheduler_ordering(self) -> ProcessingConfig:
        if self.adaptive_refresh_seconds > self.polling_interval_seconds:
            raise ValueError("adaptive calendar refresh cannot exceed the normal interval")
        if self.failure_backoff_initial_seconds > self.failure_backoff_max_seconds:
            raise ValueError("calendar failure backoff bounds are reversed")
        return self


class FreshnessConfig(StrictModel):
    aging_minutes: int = Field(default=30, ge=1)
    stale_minutes: int = Field(default=120, ge=2)
    critical_minutes: int = Field(default=1440, ge=3)

    @model_validator(mode="after")
    def ordered(self) -> FreshnessConfig:
        if not self.aging_minutes < self.stale_minutes < self.critical_minutes:
            raise ValueError("freshness thresholds must be strictly ordered")
        return self


class EconomicCalendarConfig(StrictModel):
    version: str = "1.0.0"
    enabled: bool = True
    versions: VersionConfig = Field(default_factory=VersionConfig)
    providers: tuple[ProviderConfig, ...] = (ProviderConfig(name="disabled", mode=ProviderMode.DISABLED),)
    provider_priority: tuple[str, ...] = ("disabled",)
    windows: dict[str, RiskWindowConfig] = Field(
        default_factory=lambda: {
            "low": RiskWindowConfig(pre_minutes=5, imminent_minutes=2, post_minutes=5, cooldown_minutes=5),
            "medium": RiskWindowConfig(pre_minutes=15, imminent_minutes=5, post_minutes=15, cooldown_minutes=15),
            "high": RiskWindowConfig(pre_minutes=30, imminent_minutes=10, post_minutes=30, cooldown_minutes=30),
            "critical": RiskWindowConfig(pre_minutes=60, imminent_minutes=15, post_minutes=60, cooldown_minutes=60),
            "unknown": RiskWindowConfig(pre_minutes=10, imminent_minutes=5, post_minutes=10, cooldown_minutes=10),
        }
    )
    importance_weights: dict[str, float] = Field(default_factory=lambda: {"low": 0.2, "medium": 0.45, "high": 0.75, "critical": 1.0, "unknown": 0.25})
    relevance_weights: dict[str, float] = Field(
        default_factory=lambda: {"direct": 1.0, "strong": 0.8, "moderate": 0.55, "weak": 0.25, "none": 0.0, "unknown": 0.1}
    )
    cluster_window_minutes: int = Field(default=30, ge=1, le=1440)
    country_currency: dict[str, str] = Field(
        default_factory=lambda: {"US": "USD", "GB": "GBP", "JP": "JPY", "CA": "CAD", "AU": "AUD", "NZ": "NZD", "CH": "CHF", "EU": "EUR"}
    )
    symbol_overrides: dict[str, tuple[str, ...]] = Field(default_factory=_symbol_overrides)
    inverted_surprise_categories: tuple[str, ...] = ("employment",)
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    repository_mode: str = "memory"

    @model_validator(mode="after")
    def consistency(self) -> EconomicCalendarConfig:
        names = [item.name for item in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        if set(self.provider_priority) != set(names):
            raise ValueError("provider priority must list every provider exactly once")
        if any(value < 0 or value > 1 for value in (*self.importance_weights.values(), *self.relevance_weights.values())):
            raise ValueError("weights must be bounded between zero and one")
        if set(self.windows) != {"low", "medium", "high", "critical", "unknown"}:
            raise ValueError("all importance windows are required")
        if self.repository_mode not in {"memory", "postgresql"}:
            raise ValueError("unknown repository mode")
        return self

    @property
    def high_impact_pre_minutes(self) -> int:
        return self.windows["high"].pre_minutes

    @property
    def high_impact_post_minutes(self) -> int:
        return self.windows["high"].post_minutes

    @property
    def medium_impact_pre_minutes(self) -> int:
        return self.windows["medium"].pre_minutes

    @property
    def medium_impact_post_minutes(self) -> int:
        return self.windows["medium"].post_minutes
