"""Shared helpers so one failing upstream dependency degrades a field instead of 500ing an endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def safe_call[T](
    call: Callable[[], Awaitable[T]],
    *,
    timeout_seconds: float = 5.0,
) -> tuple[T | None, str | None]:
    """Run an awaitable; return (value, None) on success or (None, "ExceptionType") on failure."""
    try:
        return await asyncio.wait_for(call(), timeout=timeout_seconds), None
    except Exception as exc:  # deliberately broad: this is a field-level circuit breaker
        return None, type(exc).__name__
