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


class CacheError(MarketDataError):
    """Raised when a cache adapter cannot complete an operation."""


class HistoricalSyncError(MarketDataError):
    """Raised when historical synchronization cannot complete safely."""
