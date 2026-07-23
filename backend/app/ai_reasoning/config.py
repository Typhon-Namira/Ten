"""Validated configuration for AI reasoning and persistent signal management."""

from pydantic import BaseModel, ConfigDict, Field


class AIReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    request_schema_version: str
    reasoning_policy_version: str
    setup_family_registry_version: str
    prompt_version_new_market: str
    prompt_version_existing_signal: str
    maximum_memory_entries: int = Field(ge=1, le=100)
    maximum_retries: int = Field(ge=0, le=3)
    temperature: float = Field(ge=0, le=1)
    max_tokens: int = Field(ge=256, le=10000)
    request_timeout_seconds: float = Field(gt=0)
    llm_concurrency_limit: int = Field(ge=1, le=32)
    provider_backoff_initial_seconds: float = Field(gt=0)
    provider_backoff_max_seconds: float = Field(gt=0)
