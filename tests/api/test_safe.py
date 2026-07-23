from __future__ import annotations

import asyncio

import pytest

from backend.app.api.safe import safe_call


@pytest.mark.asyncio
async def test_safe_call_returns_timeout_as_degraded_field() -> None:
    async def blocked_dependency() -> str:
        await asyncio.sleep(1)
        return "unreachable"

    value, error = await safe_call(blocked_dependency, timeout_seconds=0.001)

    assert value is None
    assert error == "TimeoutError"


@pytest.mark.asyncio
async def test_safe_call_preserves_successful_value() -> None:
    async def dependency() -> str:
        return "ready"

    assert await safe_call(dependency) == ("ready", None)
