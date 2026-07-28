"""Typed application exceptions."""

from __future__ import annotations

from dataclasses import dataclass


class TenError(Exception):
    """Base exception for expected TEN failures."""


class ConfigurationError(TenError):
    """Raised for invalid or missing runtime configuration."""


class EngineError(TenError):
    """Raised when an analysis engine cannot produce a valid result."""


class ExternalServiceError(TenError):
    """Raised when a provider request fails or returns invalid data."""


@dataclass(frozen=True)
class AIProviderFailureDetails:
    """Sanitized provider failure data safe to persist and expose operationally."""

    provider: str
    reason_code: str
    phase: str
    endpoint: str
    model: str
    http_method: str = "POST"
    endpoint_host: str | None = None
    endpoint_path: str | None = None
    request_id: str | None = None
    cycle_id: str | None = None
    http_status: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata_error_type: str | None = None
    metadata_provider_code: str | None = None
    content_type: str | None = None
    body_length: int | None = None
    sanitized_response_body: str | None = None
    serialized_request_bytes: int | None = None
    estimated_input_tokens: int | None = None
    timeout_category: str | None = None
    network_error_category: str | None = None
    attempt_type: str | None = None
    retry_after: str | None = None
    rate_limit_limit: str | None = None
    rate_limit_remaining: str | None = None
    rate_limit_reset: str | None = None
    rate_limit_request_limit: str | None = None
    rate_limit_request_remaining: str | None = None
    rate_limit_request_reset: str | None = None
    rate_limit_token_limit: str | None = None
    rate_limit_token_remaining: str | None = None
    rate_limit_token_reset: str | None = None
    provider_request_id: str | None = None
    limit_classification: str | None = None
    finish_reason: str | None = None
    response_size_bytes: int | None = None
    response_character_count: int | None = None
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    provider_total_tokens: int | None = None
    target_output_tokens: int | None = None
    hard_output_limit: int | None = None
    output_profile: str | None = None
    analysis_schema_version: str | None = None
    input_budget_utilization_percent: float | None = None
    token_estimator: str | None = None
    context_sections_included: tuple[str, ...] = ()
    context_sections_omitted: tuple[str, ...] = ()
    schema_error_code: str | None = None
    schema_error_path: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    elapsed_ms: float | None = None
    exception_class: str | None = None


class AIProviderRequestError(ExternalServiceError):
    """AI-provider failure carrying only sanitized diagnostic details."""

    def __init__(self, details: AIProviderFailureDetails) -> None:
        self.details = details
        message = details.error_message or details.exception_class or details.reason_code
        super().__init__(f"{details.provider}:{details.reason_code}: {message}")

