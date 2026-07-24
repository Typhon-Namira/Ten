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
class OpenRouterFailureDetails:
    """Sanitized provider failure data safe to persist and expose operationally."""

    reason_code: str
    phase: str
    endpoint: str
    model: str
    request_id: str | None = None
    cycle_id: str | None = None
    http_status: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata_error_type: str | None = None
    metadata_provider_code: str | None = None
    content_type: str | None = None
    body_length: int | None = None
    retry_after: str | None = None
    elapsed_ms: float | None = None
    exception_class: str | None = None


class OpenRouterRequestError(ExternalServiceError):
    """OpenRouter failure carrying only sanitized diagnostic details."""

    def __init__(self, details: OpenRouterFailureDetails) -> None:
        self.details = details
        message = details.error_message or details.exception_class or details.reason_code
        super().__init__(f"{details.reason_code}: {message}")

