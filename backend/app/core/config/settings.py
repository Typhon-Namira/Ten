"""Environment-backed application configuration."""

from functools import lru_cache
import json
import os
from typing import Annotated, Any

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from backend.app.core.database.url import normalize_async_database_url


class Settings(BaseSettings):
    """Validated runtime settings. Secrets are loaded only from the environment."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TEN_", extra="ignore")

    app_name: str = "TEN"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    database_url: str = "postgresql+asyncpg://ten:ten@localhost:5432/ten"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    request_timeout_seconds: float = 30.0
    integration_enabled: bool = True
    live_pipeline_enabled: bool = True
    integration_worker_enabled: bool = False
    market_data_worker_enabled: bool = False
    market_data_provider: str = "lbma_gold_price"
    market_data_symbols: Annotated[tuple[str, ...], NoDecode] = ("XAUUSD",)
    market_data_timeframes: Annotated[tuple[str, ...], NoDecode] = ("M15",)
    market_data_bootstrap_enabled: bool = True
    market_data_bootstrap_candles: int = Field(default=2500, ge=50, le=5000)
    market_data_poll_seconds: float = Field(default=60, ge=5, le=3600)
    # Must exceed the largest configured timeframe's own bar duration with real margin: a freshly
    # discovered "latest closed" candle is, by construction (no-lookahead), always somewhere
    # between 0 and just-under-one-bar-duration old (see `market_data_engine.adapters._period_has_closed`).
    # A tolerance equal to (or tighter than) that duration — e.g. 900s for M15 — leaves no room for
    # normal poll-cycle timing/jitter and will intermittently mark genuinely fresh data as stale.
    # Default here is 2x the default M15 timeframe's 900s duration.
    max_candle_staleness_seconds: int = Field(default=1800, ge=60, le=604800)
    replay_worker_enabled: bool = False
    public_read_access: bool = True
    api_keys: dict[str, str] = Field(default_factory=dict)

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: Any) -> Any:
        """Accept Railway's native PostgreSQL URL and select asyncpg."""
        return normalize_async_database_url(value) if isinstance(value, str) else value

    @field_validator("market_data_symbols", "market_data_timeframes", mode="before")
    @classmethod
    def parse_market_data_sequence(cls, value: Any, info: ValidationInfo) -> Any:
        """Accept both JSON arrays and comma-separated Railway variables."""
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if raw.startswith("["):
            items = json.loads(raw)
        else:
            items = tuple(item.strip() for item in raw.split(",") if item.strip())
        if info.field_name != "market_data_timeframes":
            return items
        aliases = {
            "1m": "M1",
            "m1": "M1",
            "5m": "M5",
            "m5": "M5",
            "15m": "M15",
            "m15": "M15",
            "30m": "M30",
            "m30": "M30",
            "1h": "H1",
            "h1": "H1",
            "4h": "H4",
            "h4": "H4",
            "1d": "D1",
            "d1": "D1",
        }
        return tuple(aliases.get(str(item).strip().lower(), str(item).strip()) for item in items)

    @model_validator(mode="after")
    def production_security(self) -> "Settings":
        railway_runtime = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"))
        if self.environment.lower() == "production" or railway_runtime:
            if "market_data_worker_enabled" not in self.model_fields_set:
                self.market_data_worker_enabled = True
            if "integration_worker_enabled" not in self.model_fields_set:
                self.integration_worker_enabled = True
        if self.environment.lower() == "production" and self.integration_enabled:
            if self.public_read_access:
                raise ValueError("production integration requires TEN_PUBLIC_READ_ACCESS=false")
            if not self.api_keys:
                raise ValueError("production integration requires TEN_API_KEYS")
            if "*" in self.cors_origins:
                raise ValueError("production CORS cannot allow every origin")
        if set(self.api_keys.values()) - {"viewer", "operator", "admin"}:
            raise ValueError("TEN_API_KEYS roles must be viewer, operator, or admin")
        # Keyless public sources (active by default) + the disabled-by-default legacy paid/keyed
        # and robots-blocked adapters — see backend/app/engines/market_data_engine/adapters.py.
        supported_providers = {
            "lbma_gold_price",
            "kraken",
            "okx",
            "yahoo_finance",
            "stooq",
            "binance",
            "twelve_data",
            "oanda",
            "alpha_vantage",
            "financial_modeling_prep",
        }
        if self.market_data_provider not in supported_providers:
            raise ValueError("TEN_MARKET_DATA_PROVIDER is unsupported")
        supported_timeframes = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
        if not self.market_data_symbols or any(not item.strip() for item in self.market_data_symbols):
            raise ValueError("TEN_MARKET_DATA_SYMBOLS must contain at least one symbol")
        if not self.market_data_timeframes or set(self.market_data_timeframes) - supported_timeframes:
            raise ValueError("TEN_MARKET_DATA_TIMEFRAMES contains an unsupported timeframe")
        if self.market_data_worker_enabled and not self.database_url.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError("TEN_DATABASE_URL must use an asynchronous SQLAlchemy driver")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()

