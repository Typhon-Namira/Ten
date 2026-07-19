"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    replay_worker_enabled: bool = False
    public_read_access: bool = True
    api_keys: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def production_security(self) -> "Settings":
        if self.environment.lower() == "production" and self.integration_enabled:
            if self.public_read_access:
                raise ValueError("production integration requires TEN_PUBLIC_READ_ACCESS=false")
            if not self.api_keys:
                raise ValueError("production integration requires TEN_API_KEYS")
            if "*" in self.cors_origins:
                raise ValueError("production CORS cannot allow every origin")
        if set(self.api_keys.values()) - {"viewer", "operator", "admin"}:
            raise ValueError("TEN_API_KEYS roles must be viewer, operator, or admin")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()

