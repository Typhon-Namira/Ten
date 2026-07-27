"""Validated configuration for AI reasoning and persistent signal management."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AIReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    request_schema_version: str
    reasoning_policy_version: str
    setup_family_registry_version: str
    analysis_timeframe: Literal["M5"] = "M5"
    analysis_interval_minutes: Literal[5] = 5
    prompt_version_new_market: str
    prompt_version_existing_signal: str
    maximum_memory_entries: int = Field(ge=1, le=100)
    # Each provider may receive one bounded retry for transport/5xx failures.
    maximum_retries: int = Field(ge=0, le=1)
    temperature: float = Field(ge=0, le=1)
    max_tokens: int = Field(ge=256, le=10000)
    request_timeout_seconds: float = Field(gt=0)
    llm_concurrency_limit: int = Field(ge=1, le=32)
    provider_backoff_initial_seconds: float = Field(gt=0)
    provider_backoff_max_seconds: float = Field(gt=0)
    target_input_tokens: int = Field(default=4_000, ge=256, le=100_000)
    warning_input_tokens: int = Field(default=8_000, ge=256, le=100_000)
    hard_input_tokens: int = Field(default=16_000, ge=256, le=100_000)
    absolute_max_output_tokens: int = Field(default=2_000, ge=256, le=10_000)
    maximum_request_cost_usd: float = Field(default=0.05, gt=0)
    input_cost_per_million_usd: float = Field(default=1.04, ge=0)
    output_cost_per_million_usd: float = Field(default=2.25, ge=0)
    temporal_lookback_minutes: tuple[int, ...] = (5, 15, 30, 60, 240)
    temporal_tolerance_minutes: dict[str, int] = {
        "5m": 2,
        "15m": 5,
        "30m": 10,
        "1h": 20,
        "4h": 60,
    }
    temporal_rolling_window: int = Field(default=60, ge=3, le=240)

    @model_validator(mode="after")
    def ordered_request_budgets(self) -> AIReasoningConfig:
        if not self.target_input_tokens <= self.warning_input_tokens <= self.hard_input_tokens:
            raise ValueError("AI input-token thresholds must be ordered")
        if self.max_tokens > self.absolute_max_output_tokens:
            raise ValueError("AI max_tokens exceeds the absolute output-token limit")
        if self.temporal_lookback_minutes != (5, 15, 30, 60, 240):
            raise ValueError("temporal lookback anchors must remain 5m, 15m, 30m, 1h, and 4h")
        return self
