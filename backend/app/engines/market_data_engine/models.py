"""Canonical market-data value objects and state contracts."""

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, Field, field_validator, model_validator


class Timeframe(StrEnum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def duration(self) -> timedelta:
        return {
            Timeframe.M1: timedelta(minutes=1),
            Timeframe.M5: timedelta(minutes=5),
            Timeframe.M15: timedelta(minutes=15),
            Timeframe.M30: timedelta(minutes=30),
            Timeframe.H1: timedelta(hours=1),
            Timeframe.H4: timedelta(hours=4),
            Timeframe.D1: timedelta(days=1),
        }[self]


class MarketSession(StrEnum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NEW_YORK_OVERLAP = "london_new_york_overlap"
    CLOSED = "closed"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class MarketStatusCode(StrEnum):
    OPEN = "OPEN"
    MAINTENANCE = "MAINTENANCE"
    CLOSED_WEEKEND = "CLOSED_WEEKEND"
    # Retained for backward-compatible reads of snapshots persisted before the
    # explicit MAINTENANCE status was introduced.
    CLOSED_DAILY_BREAK = "CLOSED_DAILY_BREAK"
    HOLIDAY_OR_PROVIDER_CLOSED = "HOLIDAY_OR_PROVIDER_CLOSED"
    UNKNOWN = "UNKNOWN"


class MarketScheduleStatus(BaseModel):
    market_status: MarketStatusCode
    market_open: bool
    active_session: MarketSession | None
    instrument: str = "XAUUSD"
    timezone: str = "America/New_York"
    status_source: str = "deterministic_xauusd_trading_schedule"
    closure_reason: str | None = None
    next_expected_open_at: datetime | None = None
    server_time_utc: datetime

    @field_validator("server_time_utc", "next_expected_open_at")
    @classmethod
    def schedule_timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("market schedule timestamps must be timezone-aware")
        return value.astimezone(UTC)


class DataQualityLevel(StrEnum):
    NATIVE = "native"
    VERIFIED = "verified"
    RECOVERED = "recovered"
    INTERPOLATED = "interpolated"
    MINOR_ANOMALY = "minor_anomaly"
    MAJOR_ANOMALY = "major_anomaly"
    CORRUPTED = "corrupted"


class SyncStatus(StrEnum):
    IDLE = "idle"
    SYNCING = "syncing"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class RealtimeStatus(StrEnum):
    STOPPED = "stopped"
    POLLING = "polling"
    STREAMING = "streaming"
    DEGRADED = "degraded"


def canonical_symbol(value: str) -> str:
    normalized = value.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if not normalized or not normalized.isalnum():
        raise ValueError("symbol must contain letters or numbers")
    return normalized


class Candle(BaseModel):
    timestamp: datetime
    symbol: str = "XAU/USD"
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(default=0.0, ge=0)
    spread: float = Field(default=0.0, ge=0)
    provider: str = "unknown"
    quality_score: float = Field(default=100.0, ge=0, le=100)
    quality_level: DataQualityLevel = DataQualityLevel.NATIVE
    ingestion_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        canonical_symbol(value)
        return value.upper()

    @field_validator("timestamp", "ingestion_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_values(self) -> "Candle":
        prices = (self.open, self.high, self.low, self.close)
        if not all(isfinite(value) and value > 0 for value in prices):
            raise ValueError("OHLC prices must be finite and positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("high/low must contain open and close")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")
        if not isfinite(self.volume) or not isfinite(self.spread):
            raise ValueError("volume and spread must be finite")
        return self


class Tick(BaseModel):
    symbol: str = "XAU/USD"
    timestamp: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    volume: float = Field(default=0.0, ge=0)
    provider: str = "unknown"
    ingestion_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        canonical_symbol(value)
        return value.upper()

    @field_validator("timestamp", "ingestion_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_spread(self) -> "Tick":
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        return self


class MarketMetrics(BaseModel):
    symbol: str
    timeframe: Timeframe
    atr: float = Field(ge=0)
    current_spread: float = Field(ge=0)
    average_spread: float = Field(ge=0)
    daily_range: float = Field(ge=0)
    session_range: float = Field(ge=0)
    rolling_volatility: float = Field(ge=0)
    price_velocity: float
    tick_frequency: float = Field(ge=0)
    data_freshness_seconds: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MarketState(BaseModel):
    market_open: bool
    session: MarketSession
    current_provider: str | None
    provider_health: str
    current_latency_ms: float | None = Field(default=None, ge=0)
    data_freshness_seconds: float | None = Field(default=None, ge=0)
    symbol: str
    timeframe: Timeframe
    historical_sync_status: SyncStatus = SyncStatus.IDLE
    realtime_status: RealtimeStatus = RealtimeStatus.STOPPED
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
