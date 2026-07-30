"""Environment-backed application configuration."""

from functools import lru_cache
import json
import logging
import os
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from backend.app.core.database.url import normalize_async_database_url

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Validated runtime settings. Secrets are loaded only from the environment."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TEN_", extra="ignore")

    app_name: str = "TEN"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    database_url: str = "postgresql+asyncpg://ten:ten@localhost:5432/ten"
    db_pool_size: int = Field(default=3, ge=1, le=50)
    db_max_overflow: int = Field(default=2, ge=0, le=50)
    db_pool_timeout_seconds: float = Field(default=30, gt=0, le=300)
    db_pool_recycle_seconds: int = Field(default=1800, ge=60, le=86400)
    db_pool_pre_ping: bool = True
    db_statement_timeout_ms: int = Field(default=30_000, ge=1000, le=300_000)
    db_idle_transaction_timeout_ms: int = Field(default=30_000, ge=1000, le=300_000)
    ai_primary_provider: str = "groq"
    groq_pool_enabled: bool = True
    groq_pool_size: int = Field(default=4, ge=1, le=4)
    groq_api_key_1: str | None = None
    groq_api_key_2: str | None = None
    groq_api_key_3: str | None = None
    groq_api_key_4: str | None = None
    # Temporary migration alias for TEN_GROQ_API_KEY_1. It is never a fifth account.
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "gpt-oss-120b"
    groq_request_timeout_seconds: float = Field(default=60, gt=0, le=300)
    groq_max_retries_per_account: int = Field(default=1, ge=0, le=3)
    groq_rate_limit_cooldown_seconds: float = Field(default=3600, ge=1, le=86400)
    groq_quota_cooldown_seconds: float = Field(default=86400, ge=60, le=604800)
    groq_pool_strategy: str = "ordered_failover"
    ai_output_profile: Literal["compact", "standard", "expanded"] = "compact"
    ai_target_output_tokens: int = Field(default=900, ge=256, le=10_000)
    ai_max_output_tokens: int = Field(default=1400, ge=256, le=10_000)
    ai_input_token_budget: int = Field(default=3500, ge=512, le=100_000)
    ai_token_safety_margin: int = Field(default=256, ge=64, le=4096)
    integration_enabled: bool = True
    live_pipeline_enabled: bool = True
    integration_worker_enabled: bool = False
    market_data_worker_enabled: bool = False
    market_data_provider: str = "lbma_gold_price"
    market_data_symbols: Annotated[tuple[str, ...], NoDecode] = ("XAUUSD",)
    market_data_timeframes: Annotated[tuple[str, ...], NoDecode] = ("M15",)
    market_data_bootstrap_enabled: bool = True
    market_data_bootstrap_candles: int = Field(default=2500, ge=50, le=5000)
    market_data_poll_seconds: float = Field(default=10, ge=5, le=3600)
    market_data_idle_poll_seconds: float = Field(default=30, ge=5, le=3600)
    market_data_provider_backoff_max_seconds: float = Field(default=300, ge=30, le=3600)
    # Must exceed the largest configured timeframe's own bar duration with real margin: a freshly
    # discovered "latest closed" candle is, by construction (no-lookahead), always somewhere
    # between 0 and just-under-one-bar-duration old (see `market_data_engine.adapters._period_has_closed`).
    # A tolerance equal to (or tighter than) that duration — e.g. 900s for M15 — leaves no room for
    # normal poll-cycle timing/jitter and will intermittently mark genuinely fresh data as stale.
    # Default here is 2x the default M15 timeframe's 900s duration.
    max_candle_staleness_seconds: int = Field(default=1800, ge=60, le=604800)
    replay_worker_enabled: bool = False
    max_event_history_size: int = Field(default=1000, ge=100, le=100_000)
    max_feature_store_entries: int = Field(default=10_000, ge=1000, le=1_000_000)
    max_dashboard_event_buffer: int = Field(default=500, ge=100, le=5000)
    max_client_queue_size: int = Field(default=500, ge=10, le=5000)
    log_access_requests: bool = False
    log_market_tick_events: bool = False
    log_health_unchanged: bool = False
    public_read_access: bool = True
    api_keys: dict[str, str] = Field(default_factory=dict)
    # Optional runtime overrides for the AI-centric feature flags. ``None`` deliberately
    # preserves the checked-in YAML default so development, tests, and existing deployments
    # remain unchanged unless an operator explicitly enables a capability.
    ai_centric_shadow_mode: bool | None = None
    ai_signal_proposals: bool | None = None
    ai_signal_monitoring: bool | None = None
    ai_signal_publication: bool | None = None
    ai_signal_adjustments: bool | None = None
    # Cross-cutting retention worker (backend/app/core/database/retention.py) — deletes rows
    # older than these windows from tables that have no engine-specific retention of their own.
    # ai_scoring_engine/signal_decision_engine/replay_engine already have their own configured
    # retention (see each engine's `configs/*.yaml` `retention:` section); this worker only
    # invokes those existing `cleanup()` methods, it does not duplicate their windows.
    retention_worker_enabled: bool = True
    retention_interval_seconds: float = Field(default=3600, ge=60, le=86400)
    retention_batch_size: int = Field(default=500, ge=1, le=10_000)
    # smc_objects / liquidity_objects / volume_profile_objects / institutional_flow_evidence /
    # market_regime_evidence — one row per analytical object per cycle it was re-evaluated in.
    analytical_object_retention_days: int = Field(default=14, ge=1, le=3650)
    # smc_analysis_snapshots / liquidity_snapshots / volume_profile_snapshots /
    # institutional_flow_snapshots — one full-payload row per cycle. market_regime_snapshots is
    # excluded: it already has its own working `prune_history`, called inline after every save.
    analytical_snapshot_retention_days: int = Field(default=14, ge=1, le=3650)
    # integration_events (cascades to integration_outbox/integration_processed_events),
    # integration_event_trace, integration_data_quality_issues, integration_snapshots.
    integration_audit_retention_days: int = Field(default=14, ge=1, le=3650)
    # operational_signals — the actually-published trading signals; kept much longer than the
    # surrounding audit trail since they're the real product output, not diagnostic exhaust.
    operational_signal_retention_days: int = Field(default=180, ge=1, le=3650)
    # provider_metrics / market_quality_history / market_gap_history / market_latency_history /
    # market_synchronization_history / realtime_candles — high-frequency diagnostic exhaust, not
    # the deterministic OHLCV series (historical_candles is never pruned).
    market_data_history_retention_days: int = Field(default=7, ge=1, le=3650)
    signal_email_enabled: bool = False
    signal_email_recipient: str = "tufannamira@gmail.com"
    email_provider: Literal["smtp"] = "smtp"
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    email_from: str | None = None
    signal_email_poll_seconds: float = Field(default=10, ge=1, le=300)
    signal_email_max_attempts: int = Field(default=5, ge=1, le=20)

    @property
    def signal_email_configuration_errors(self) -> tuple[str, ...]:
        if not self.signal_email_enabled:
            return ()
        required = {
            "TEN_SMTP_HOST": self.smtp_host,
            "TEN_EMAIL_FROM": self.email_from,
        }
        if self.smtp_username and not self.smtp_password:
            required["TEN_SMTP_PASSWORD"] = self.smtp_password
        return tuple(name for name, value in required.items() if not value)

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

    @field_validator("groq_base_url")
    @classmethod
    def validate_ai_provider_url(cls, value: str, info: ValidationInfo) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            variable = f"TEN_{(info.field_name or 'AI_PROVIDER_BASE_URL').upper()}"
            raise ValueError(f"{variable} must be an absolute HTTPS URL")
        normalized_path = parsed.path.rstrip("/").lower()
        if normalized_path.endswith("/v1/v1") or normalized_path.endswith(
            "/chat/completions"
        ):
            variable = f"TEN_{(info.field_name or 'AI_PROVIDER_BASE_URL').upper()}"
            raise ValueError(
                f"{variable} must be a base URL without duplicated /v1 "
                "or /chat/completions"
            )
        return value.rstrip("/")

    @field_validator(
        "groq_api_key",
        "groq_api_key_1",
        "groq_api_key_2",
        "groq_api_key_3",
        "groq_api_key_4",
    )
    @classmethod
    def validate_ai_provider_key(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None or value == "":
            return None
        variable = f"TEN_{(info.field_name or 'AI_PROVIDER_API_KEY').upper()}"
        if value != value.strip() or (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            raise ValueError(
                f"{variable} must not contain surrounding whitespace or quotes"
            )
        return value

    @field_validator("groq_model")
    @classmethod
    def validate_ai_provider_model(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip() or any(character.isspace() for character in value):
            variable = f"TEN_{(info.field_name or 'AI_PROVIDER_MODEL').upper()}"
            raise ValueError(f"{variable} must be a non-empty provider model ID")
        return value.strip()

    @model_validator(mode="after")
    def production_security(self) -> "Settings":
        if self.ai_target_output_tokens > self.ai_max_output_tokens:
            raise ValueError(
                "TEN_AI_TARGET_OUTPUT_TOKENS must not exceed "
                "TEN_AI_MAX_OUTPUT_TOKENS"
            )
        if self.ai_primary_provider != "groq":
            raise ValueError("TEN_AI_PRIMARY_PROVIDER must be groq")
        if self.groq_pool_strategy != "ordered_failover":
            raise ValueError("TEN_GROQ_POOL_STRATEGY must be ordered_failover")
        if self.groq_api_key_1 is None and self.groq_api_key is not None:
            self.groq_api_key_1 = self.groq_api_key
            logger.warning(
                "ai_provider.legacy_key_alias.used",
                extra={
                    "legacy_variable": "TEN_GROQ_API_KEY",
                    "replacement_variable": "TEN_GROQ_API_KEY_1",
                    "account_id": "groq_1",
                },
            )
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

    @property
    def groq_pool_api_keys(self) -> tuple[str | None, ...]:
        """Return the four account keys without exposing them through diagnostics."""

        return (
            self.groq_api_key_1,
            self.groq_api_key_2,
            self.groq_api_key_3,
            self.groq_api_key_4,
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()

