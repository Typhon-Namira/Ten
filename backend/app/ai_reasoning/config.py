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
    analysis_timeframe: Literal["M5_M15"] = "M5_M15"
    analysis_interval_minutes: Literal[5] = 5
    prompt_version_new_market: str
    prompt_version_existing_signal: str
    maximum_memory_entries: int = Field(ge=1, le=100)
    # Each provider may receive one bounded retry for transport/5xx failures.
    maximum_retries: int = Field(ge=0, le=1)
    temperature: float = Field(ge=0, le=1)
    max_tokens: int = Field(ge=256, le=10000)
    output_profile: Literal["compact", "standard", "expanded"] = "compact"
    target_output_tokens: int = Field(default=900, ge=256, le=10_000)
    input_token_budget: int = Field(default=3_500, ge=512, le=100_000)
    token_safety_margin: int = Field(default=256, ge=64, le=4096)
    model_context_limit: int = Field(default=8192, ge=2048, le=1_000_000)
    truncation_degraded_threshold: float = Field(default=0.05, ge=0, le=1)
    truncation_unhealthy_threshold: float = Field(default=0.20, ge=0, le=1)
    zero_completion_cycle_threshold: int = Field(default=3, ge=1, le=100)
    provider_calls_per_analysis_degraded_threshold: float = Field(
        default=1.5,
        ge=1,
        le=10,
    )
    schema_correction_degraded_threshold: float = Field(default=0.10, ge=0, le=1)
    tokens_per_analysis_degraded_threshold: int = Field(
        default=6_000,
        ge=1_000,
        le=1_000_000,
    )
    request_timeout_seconds: float = Field(gt=0)
    llm_concurrency_limit: int = Field(ge=1, le=32)
    claim_lease_seconds: int = Field(default=90, ge=30, le=600)
    claim_heartbeat_seconds: int = Field(default=20, ge=5, le=120)
    claim_max_runtime_seconds: int = Field(default=180, ge=60, le=1800)
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
    signal_minimum_risk_reward: float = Field(default=2.0, ge=1, le=10)
    signal_preferred_risk_reward: float = Field(default=2.5, ge=1, le=10)
    signal_exceptional_risk_reward: float = Field(default=3.0, ge=1, le=20)
    signal_quality_threshold: float = Field(default=65, ge=0, le=100)

    @model_validator(mode="after")
    def ordered_request_budgets(self) -> AIReasoningConfig:
        if not self.target_input_tokens <= self.warning_input_tokens <= self.hard_input_tokens:
            raise ValueError("AI input-token thresholds must be ordered")
        if self.max_tokens > self.absolute_max_output_tokens:
            raise ValueError("AI max_tokens exceeds the absolute output-token limit")
        if self.target_output_tokens > self.max_tokens:
            raise ValueError("AI target output tokens exceed the hard output limit")
        if (
            self.input_token_budget
            + self.max_tokens
            + self.token_safety_margin
            > self.model_context_limit
        ):
            raise ValueError("AI input and output reservations exceed model context")
        if self.truncation_degraded_threshold > self.truncation_unhealthy_threshold:
            raise ValueError("AI truncation health thresholds must be ordered")
        if self.claim_heartbeat_seconds * 2 >= self.claim_lease_seconds:
            raise ValueError("AI claim heartbeat must run at least twice within the lease")
        if self.claim_max_runtime_seconds <= self.claim_lease_seconds:
            raise ValueError("AI claim maximum runtime must exceed the lease")
        if self.temporal_lookback_minutes != (5, 15, 30, 60, 240):
            raise ValueError("temporal lookback anchors must remain 5m, 15m, 30m, 1h, and 4h")
        if not (
            self.signal_minimum_risk_reward
            <= self.signal_preferred_risk_reward
            <= self.signal_exceptional_risk_reward
        ):
            raise ValueError("signal risk/reward thresholds must be ordered")
        return self
