"""Shared helpers so one failing upstream dependency degrades a field instead of 500ing an endpoint."""

from __future__ import annotations

from collections.abc import Awaitable, Callable


async def safe_call[T](call: Callable[[], Awaitable[T]]) -> tuple[T | None, str | None]:
    """Run an awaitable; return (value, None) on success or (None, "ExceptionType") on failure."""
    try:
        return await call(), None
    except Exception as exc:  # deliberately broad: this is a field-level circuit breaker
        return None, type(exc).__name__
