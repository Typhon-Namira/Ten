"""Structured Market Data Engine exceptions."""

from backend.app.core.exceptions import EngineError, ExternalServiceError


class MarketDataError(EngineError):
    """Base error for deterministic market-data failures."""


class MarketDataValidationError(MarketDataError):
    """Raised when normalized market data violates hard invariants."""


class ProviderUnavailableError(ExternalServiceError):
    """Raised when no capable healthy provider can satisfy a request."""


class ProviderResponseError(ExternalServiceError):
    """Raised when a provider response cannot be normalized safely."""


class ProviderRateLimitedError(ProviderResponseError):
    """Raised when a provider responds HTTP 429 (Too Many Requests)."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class CacheError(MarketDataError):
    """Raised when a cache adapter cannot complete an operation."""


class HistoricalSyncError(MarketDataError):
    """Raised when historical synchronization cannot complete safely."""
