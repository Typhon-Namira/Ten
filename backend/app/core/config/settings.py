"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()

