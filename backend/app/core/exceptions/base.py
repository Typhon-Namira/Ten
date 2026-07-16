"""Typed application exceptions."""


class TenError(Exception):
    """Base exception for expected TEN failures."""


class ConfigurationError(TenError):
    """Raised for invalid or missing runtime configuration."""


class EngineError(TenError):
    """Raised when an analysis engine cannot produce a valid result."""


class ExternalServiceError(TenError):
    """Raised when a provider request fails or returns invalid data."""

