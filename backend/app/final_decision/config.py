"""Versioned deterministic guardrail, publication, replay, and readiness policies."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SessionSpreadPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asia: float = Field(gt=0)
    london: float = Field(gt=0)
    new_york: float = Field(gt=0)
    rollover: float = Field(gt=0)
    unknown: float = Field(gt=0)


class GuardrailPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    hard_gate_registry_version: str
    risk_policy_version: str
    execution_policy_version: str
    publication_schema_version: str
    replay_policy_version: str
    outcome_policy_version: str
    calibration_policy_version: str
    readiness_policy_version: str
    minimum_absolute_risk_to_reward: float = Field(gt=0)
    maximum_spread: SessionSpreadPolicy
    maximum_signal_age_seconds: int = Field(gt=0)
    maximum_setup_expiry_seconds: int = Field(gt=0)
    minimum_stop_distance: float = Field(gt=0)
    maximum_stop_distance: float = Field(gt=0)
    maximum_entry_distance: float = Field(gt=0)
    supported_entry_types: tuple[str, ...]
    price_precision: int = Field(ge=0, le=8)
    economic_event_blackout_required: bool
    minimum_evidence_change_ratio: float = Field(ge=0, le=1)
    llm_request_timeout_seconds: float = Field(gt=0)
    llm_concurrency_limit: int = Field(ge=1, le=32)
    provider_backoff_initial_seconds: float = Field(gt=0)
    provider_backoff_max_seconds: float = Field(gt=0)
    maximum_daily_llm_requests: int = Field(gt=0)
    maximum_daily_llm_tokens: int = Field(gt=0)
    configured_slippage: float = Field(ge=0)
    minimum_readiness_sample_size: int = Field(gt=0)
    maximum_llm_failure_rate: float = Field(ge=0, le=1)
    maximum_publication_failure_rate: float = Field(ge=0, le=1)
    policy_sources: dict[str, str]
