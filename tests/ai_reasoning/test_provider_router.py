from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.ai_reasoning.provider import (
    AIProviderResponse,
    AIProviderRouter,
    ProviderStatus,
    reasoning_response_schema,
)
from backend.app.core.exceptions import AIProviderFailureDetails, AIProviderRequestError


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def test_reasoning_schema_is_strict_at_every_nested_object() -> None:
    schema = reasoning_response_schema()

    def assert_strict(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or (
                isinstance(value.get("type"), list)
                and "object" in value["type"]
            ):
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            for nested in value.values():
                assert_strict(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_strict(nested)

    assert_strict(schema)


def test_cerebras_wire_schema_stays_within_supported_strict_schema_subset() -> None:
    schema = reasoning_response_schema()
    encoded = json.dumps(schema, separators=(",", ":"))
    unsupported = {
        "description",
        "format",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }

    def schema_keywords(value: object, *, property_map: bool = False) -> set[str]:
        if isinstance(value, dict):
            own = set() if property_map else set(value)
            return own.union(
                *(
                    schema_keywords(item, property_map=key == "properties")
                    for key, item in value.items()
                ),
            )
        if isinstance(value, list):
            return set().union(*(schema_keywords(item) for item in value))
        return set()

    assert len(encoded) <= 5_000
    assert schema_keywords(schema).isdisjoint(unsupported)


class StubProvider:
    def __init__(
        self,
        name: str,
        outcomes: list[AIProviderResponse | AIProviderRequestError],
        *,
        configured: bool = True,
    ) -> None:
        self.provider_name = name
        self.model = f"{name}-model"
        self.configured = configured
        self.client = SimpleNamespace(base_url=f"https://api.{name}.test/v1")
        self.outcomes = outcomes
        self.calls = 0
        self.cycle_ids: list[str] = []

    async def reason(self, request: Any, **kwargs: Any) -> AIProviderResponse:
        self.calls += 1
        self.cycle_ids.append(str(request.cycle_id))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, AIProviderRequestError):
            raise outcome
        return replace(
            outcome,
            fallback_used=bool(kwargs.get("fallback_used")),
            fallback_reason=kwargs.get("fallback_reason"),
        )


def response(provider: str) -> AIProviderResponse:
    return AIProviderResponse(
        raw_output={"decision": "WAIT"},
        provider=provider,
        model_identifier=f"{provider}-model",
        latency_ms=1,
        token_usage=None,
    )


def failure(
    provider: str,
    reason_code: str,
    *,
    status: int,
    phase: str = "http_request",
    rate_limit_reset: str | None = None,
) -> AIProviderRequestError:
    return AIProviderRequestError(
        AIProviderFailureDetails(
            provider=provider,
            reason_code=reason_code,
            phase=phase,
            endpoint=f"https://api.{provider}.test/v1/chat/completions",
            model=f"{provider}-model",
            http_status=status,
            rate_limit_reset=rate_limit_reset,
        )
    )


def request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request-1",
        cycle_id="cycle-1",
        instrument="XAUUSD",
        analysis_timestamp=NOW,
    )


@pytest.mark.asyncio
async def test_primary_success_does_not_contact_fallback() -> None:
    primary = StubProvider("cerebras", [response("cerebras")])
    fallback = StubProvider("groq", [response("groq")])
    router = AIProviderRouter(primary, fallback, clock=lambda: NOW)  # type: ignore[arg-type]

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == "cerebras"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert router.states["cerebras"].circuit_open_until is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason_code", "status"),
    (
        ("authentication_failed", 401),
        ("quota_exhausted", 402),
        ("rate_limited", 429),
        ("model_unavailable", 404),
    ),
)
async def test_typed_primary_failures_fall_back_without_retry(
    reason_code: str,
    status: int,
) -> None:
    primary = StubProvider("cerebras", [failure("cerebras", reason_code, status=status)])
    fallback = StubProvider("groq", [response("groq")])
    router = AIProviderRouter(primary, fallback, clock=lambda: NOW)  # type: ignore[arg-type]

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == "groq"
    assert result.fallback_used is True
    assert result.fallback_reason == f"cerebras_{reason_code}"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert primary.cycle_ids == fallback.cycle_ids == ["cycle-1"]


@pytest.mark.asyncio
async def test_invalid_request_is_not_retried_or_sent_to_fallback() -> None:
    primary = StubProvider(
        "cerebras",
        [failure("cerebras", "invalid_request", status=400)],
    )
    fallback = StubProvider("groq", [response("groq")])
    router = AIProviderRouter(primary, fallback, clock=lambda: NOW)  # type: ignore[arg-type]

    with pytest.raises(AIProviderRequestError):
        await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert primary.calls == 1
    assert fallback.calls == 0
    assert router.states["cerebras"].circuit_open_until is None


@pytest.mark.asyncio
async def test_transient_5xx_gets_one_retry_then_fallback() -> None:
    primary = StubProvider(
        "cerebras",
        [
            failure("cerebras", "provider_unavailable", status=503),
            failure("cerebras", "provider_unavailable", status=503),
        ],
    )
    fallback = StubProvider("groq", [response("groq")])
    router = AIProviderRouter(
        primary,  # type: ignore[arg-type]
        fallback,  # type: ignore[arg-type]
        maximum_retries=1,
        clock=lambda: NOW,
    )

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == "groq"
    assert primary.calls == 2
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_daily_quota_opens_long_primary_circuit_and_next_cycle_uses_fallback() -> None:
    primary = StubProvider(
        "cerebras",
        [failure("cerebras", "quota_exhausted", status=402)],
    )
    fallback = StubProvider("groq", [response("groq"), response("groq")])
    router = AIProviderRouter(
        primary,  # type: ignore[arg-type]
        fallback,  # type: ignore[arg-type]
        circuit_seconds=60,
        clock=lambda: NOW,
    )

    await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]
    await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    state = router.states["cerebras"]
    assert state.status == ProviderStatus.QUOTA_EXHAUSTED
    assert state.circuit_open_until is not None
    assert (state.circuit_open_until - NOW).total_seconds() >= 3600
    assert primary.calls == 1
    assert fallback.calls == 2


@pytest.mark.asyncio
async def test_short_rate_limit_expires_and_cerebras_returns_to_primary() -> None:
    current = [NOW]
    primary = StubProvider(
        "cerebras",
        [
            failure(
                "cerebras",
                "rate_limited",
                status=429,
                rate_limit_reset="60",
            ),
            response("cerebras"),
        ],
    )
    fallback = StubProvider("groq", [response("groq")])
    router = AIProviderRouter(
        primary,  # type: ignore[arg-type]
        fallback,  # type: ignore[arg-type]
        circuit_seconds=300,
        clock=lambda: current[0],
    )

    first = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]
    current[0] = NOW + timedelta(minutes=2)
    second = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert first.provider == "groq"
    assert second.provider == "cerebras"
    assert router.states["cerebras"].status == ProviderStatus.HEALTHY
    assert primary.calls == 2
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_groq_429_never_promotes_groq_ahead_of_recovered_cerebras() -> None:
    current = [NOW]
    primary = StubProvider(
        "cerebras",
        [
            failure(
                "cerebras",
                "rate_limited",
                status=429,
                rate_limit_reset="1",
            ),
            response("cerebras"),
        ],
    )
    fallback = StubProvider(
        "groq",
        [
            failure(
                "groq",
                "rate_limited",
                status=429,
                rate_limit_reset="1",
            )
        ],
    )
    router = AIProviderRouter(
        primary,  # type: ignore[arg-type]
        fallback,  # type: ignore[arg-type]
        clock=lambda: current[0],
    )

    with pytest.raises(AIProviderRequestError):
        await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]
    current[0] = NOW + timedelta(seconds=2)
    recovered = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert recovered.provider == "cerebras"
    assert primary.calls == 2
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_both_providers_fail_once_and_terminal_failure_records_fallback() -> None:
    primary = StubProvider(
        "cerebras",
        [failure("cerebras", "authentication_failed", status=401)],
    )
    fallback = StubProvider(
        "groq",
        [failure("groq", "quota_exhausted", status=402)],
    )
    router = AIProviderRouter(primary, fallback, clock=lambda: NOW)  # type: ignore[arg-type]

    with pytest.raises(AIProviderRequestError) as captured:
        await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert primary.calls == 1
    assert fallback.calls == 1
    assert captured.value.details.provider == "groq"
    assert captured.value.details.fallback_used is True
    assert captured.value.details.fallback_reason == "cerebras_authentication_failed"
