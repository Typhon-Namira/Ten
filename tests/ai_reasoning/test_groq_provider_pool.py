from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.ai_reasoning.provider import (
    AIProviderResponse,
    GroqProviderPool,
    ProviderStatus,
    reasoning_response_schema,
)
from backend.app.core.exceptions import (
    AIProviderFailureDetails,
    AIProviderRequestError,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


class StubProvider:
    def __init__(
        self,
        account_id: str,
        outcomes: list[AIProviderResponse | AIProviderRequestError],
        *,
        configured: bool = True,
    ) -> None:
        self.provider_name = account_id
        self.model = "gpt-oss-120b"
        self.configured = configured
        self.client = SimpleNamespace(base_url="https://api.groq.test/openai/v1")
        self.outcomes = outcomes
        self.calls = 0
        self.correction_attempts = 0
        self.attempt_counters = {
            "analysis_requests": 0,
            "schema_correction_requests": 0,
            "http_429_responses": 0,
            "initial_parse_failures": 0,
            "initial_schema_validation_failures": 0,
            "schema_corrections_succeeded": 0,
            "schema_corrections_failed": 0,
            "truncated_outputs": 0,
            "compact_retries": 0,
            "request_policy_failures": 0,
            "provider_http_successes": 0,
            "schema_valid_analyses": 0,
        }

    @property
    def http_calls(self) -> int:
        return self.calls

    async def reason(self, request: Any, **kwargs: Any) -> AIProviderResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, AIProviderRequestError):
            raise outcome
        return replace(
            outcome,
            fallback_used=bool(kwargs.get("fallback_used")),
            fallback_reason=kwargs.get("fallback_reason"),
        )


def request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request-1",
        cycle_id="cycle-1",
        instrument="XAUUSD",
        analysis_timestamp=NOW,
    )


def response(account_id: str) -> AIProviderResponse:
    return AIProviderResponse(
        raw_output={"market_regime": {}},
        provider=account_id,
        model_identifier="gpt-oss-120b",
        latency_ms=1,
        token_usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        operational_metadata={"status_code": 200},
    )


def failure(
    account_id: str,
    reason_code: str,
    *,
    status: int,
    retry_after: str | None = None,
) -> AIProviderRequestError:
    return AIProviderRequestError(
        AIProviderFailureDetails(
            provider=account_id,
            reason_code=reason_code,
            phase="http_request",
            endpoint="https://api.groq.test/openai/v1/chat/completions",
            model="gpt-oss-120b",
            request_id="request-1",
            cycle_id="cycle-1",
            http_status=status,
            error_code=reason_code,
            retry_after=retry_after,
            exception_class="HTTPStatusError",
        )
    )


def pool(
    providers: list[StubProvider],
    **kwargs: Any,
) -> GroqProviderPool:
    return GroqProviderPool(  # type: ignore[arg-type]
        tuple(providers),
        clock=lambda: NOW,
        **kwargs,
    )


def four(
    outcomes: dict[int, list[AIProviderResponse | AIProviderRequestError]],
) -> list[StubProvider]:
    return [
        StubProvider(
            f"groq_{index}",
            outcomes.get(index, [response(f"groq_{index}")]),
        )
        for index in range(1, 5)
    ]


def test_analysis_schema_remains_strict_at_every_object() -> None:
    schema = reasoning_response_schema()

    def assert_strict(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            for nested in value.values():
                assert_strict(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_strict(nested)

    assert_strict(schema)


@pytest.mark.asyncio
async def test_account_one_success_stops_ordered_failover() -> None:
    providers = four({})
    router = pool(providers)

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == "groq_1"
    assert [provider.calls for provider in providers] == [1, 0, 0, 0]
    assert router.active_provider is None
    router.mark_analysis_persisted("groq_1", NOW)
    assert router.active_provider == "groq_1"
    assert router.states["groq_1"].successful_analyses == 1


@pytest.mark.asyncio
async def test_output_budget_failure_does_not_poison_account_or_fail_over() -> None:
    budget_failure = AIProviderRequestError(
        AIProviderFailureDetails(
            provider="groq_1",
            reason_code="output_budget_exceeded",
            phase="output_budget_policy",
            endpoint="https://api.groq.test/openai/v1/chat/completions",
            model="llama-3.1-8b-instant",
            request_id="request-1",
            cycle_id="cycle-1",
            http_status=200,
            finish_reason="length",
            schema_error_code="OUTPUT_BUDGET_EXCEEDED",
        )
    )
    providers = four({1: [budget_failure]})
    router = pool(providers)

    with pytest.raises(AIProviderRequestError) as captured:
        await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert captured.value.details.reason_code == "output_budget_exceeded"
    assert [provider.calls for provider in providers] == [1, 0, 0, 0]
    state = router.states["groq_1"]
    assert state.status == ProviderStatus.AVAILABLE
    assert state.circuit_open_until is None
    assert state.last_http_status == 200
    assert state.last_request_result == "OUTPUT_BUDGET_EXCEEDED"
    assert state.provider_failures == 0
    assert state.request_policy_failures == 1


@pytest.mark.asyncio
async def test_schema_invalid_http_200_is_latest_attempt_not_provider_outage() -> None:
    schema_failure = AIProviderRequestError(
        AIProviderFailureDetails(
            provider="groq_1",
            reason_code="schema_validation_error",
            phase="structured_output_validation",
            endpoint="https://api.groq.test/openai/v1/chat/completions",
            model="llama-3.1-8b-instant",
            request_id="request-1",
            cycle_id="cycle-1",
            http_status=200,
            schema_error_code="unknown_supply_zone_ref",
            schema_error_path=(
                "provider_response.supply_demand_analysis."
                "nearest_supply_ref"
            ),
        )
    )
    providers = four({1: [schema_failure]})
    router = pool(providers)

    with pytest.raises(AIProviderRequestError):
        await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    state = router.states["groq_1"]
    snapshot = state.snapshot()
    assert [provider.calls for provider in providers] == [1, 0, 0, 0]
    assert state.status == ProviderStatus.AVAILABLE
    assert state.circuit_open_until is None
    assert state.provider_failures == 0
    assert snapshot["latest_attempt_result"] == "SCHEMA_VALIDATION_ERROR"
    assert snapshot["latest_attempt_schema_error"] == "unknown_supply_zone_ref"
    assert snapshot["latest_successful_attempt_at"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_accounts", (1, 2, 3))
async def test_quota_exhaustion_advances_to_next_account(
    failed_accounts: int,
) -> None:
    outcomes = {
        index: [failure(f"groq_{index}", "quota_exhausted", status=429)]
        for index in range(1, failed_accounts + 1)
    }
    providers = four(outcomes)
    router = pool(providers, quota_cooldown_seconds=86400)

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == f"groq_{failed_accounts + 1}"
    assert [provider.calls for provider in providers] == [
        *(1 for _ in range(failed_accounts + 1)),
        *(0 for _ in range(3 - failed_accounts)),
    ]
    for index in range(1, failed_accounts + 1):
        state = router.states[f"groq_{index}"]
        assert state.status == ProviderStatus.QUOTA_EXHAUSTED
        assert state.quota_failures == 1
        assert state.circuit_open_until == NOW + timedelta(days=1)


@pytest.mark.asyncio
async def test_temporary_rate_limit_uses_header_and_moves_immediately() -> None:
    providers = four(
        {1: [failure("groq_1", "rate_limited", status=429, retry_after="90s")]}
    )
    router = pool(providers)

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == "groq_2"
    assert router.states["groq_1"].status == ProviderStatus.RATE_LIMITED
    assert router.states["groq_1"].circuit_open_until == NOW + timedelta(seconds=90)
    assert router.retry_attempts == 0


@pytest.mark.asyncio
async def test_rate_limited_account_becomes_available_only_after_cooldown() -> None:
    current = [NOW]
    providers = four(
        {1: [failure("groq_1", "rate_limited", status=429, retry_after="90s")]}
    )
    router = GroqProviderPool(  # type: ignore[arg-type]
        tuple(providers),
        clock=lambda: current[0],
    )

    await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]
    assert router.metadata()["providers"]["groq_1"]["eligible_now"] is False  # type: ignore[index]

    current[0] = NOW + timedelta(seconds=91)
    assert router.metadata()["providers"]["groq_1"]["eligible_now"] is True  # type: ignore[index]


@pytest.mark.asyncio
async def test_pool_is_temporarily_unavailable_when_all_accounts_are_rate_limited() -> None:
    providers = four(
        {
            index: [failure(f"groq_{index}", "rate_limited", status=429)]
            for index in range(1, 5)
        }
    )
    router = pool(providers)

    with pytest.raises(AIProviderRequestError):
        await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    metadata = router.metadata()
    assert metadata["available_account_count"] == 0
    assert metadata["aggregate_reason"] == "temporarily_rate_limited"


@pytest.mark.asyncio
async def test_authentication_failure_marks_only_one_account() -> None:
    providers = four(
        {1: [failure("groq_1", "authentication_failed", status=401)]}
    )
    router = pool(providers)

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == "groq_2"
    assert router.states["groq_1"].status == ProviderStatus.CONFIGURATION_ERROR
    assert router.states["groq_2"].status == ProviderStatus.AVAILABLE
    assert [provider.calls for provider in providers] == [1, 1, 0, 0]


@pytest.mark.asyncio
async def test_transport_failure_retries_same_account_once_then_fails_over() -> None:
    providers = four(
        {
            1: [
                failure("groq_1", "provider_unavailable", status=503),
                failure("groq_1", "provider_unavailable", status=503),
            ]
        }
    )
    router = pool(providers, maximum_retries=1)

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert result.provider == "groq_2"
    assert [provider.calls for provider in providers] == [2, 1, 0, 0]
    assert router.retry_attempts == 1


@pytest.mark.asyncio
async def test_all_accounts_failing_returns_no_successful_provider() -> None:
    providers = four(
        {
            index: [failure(f"groq_{index}", "quota_exhausted", status=429)]
            for index in range(1, 5)
        }
    )
    router = pool(providers)

    with pytest.raises(AIProviderRequestError) as captured:
        await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    assert captured.value.details.provider == "groq_4"
    assert router.active_provider is None
    assert all(
        state.status == ProviderStatus.QUOTA_EXHAUSTED
        for state in router.states.values()
    )


@pytest.mark.asyncio
async def test_disabled_accounts_are_observable_but_never_called() -> None:
    providers = [
        StubProvider("groq_1", [response("groq_1")], configured=False),
        StubProvider("groq_2", [response("groq_2")]),
        StubProvider("groq_3", [response("groq_3")], configured=False),
        StubProvider("groq_4", [response("groq_4")], configured=False),
    ]
    router = pool(providers)

    result = await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]
    metadata = router.metadata()

    assert result.provider == "groq_2"
    assert [provider.calls for provider in providers] == [0, 1, 0, 0]
    assert metadata["configured_account_count"] == 1
    assert metadata["available_account_count"] == 1
    assert metadata["providers"]["groq_1"]["status"] == "DISABLED"  # type: ignore[index]


@pytest.mark.asyncio
async def test_pool_metrics_include_persistable_per_account_deltas() -> None:
    providers = four({})
    providers[0].correction_attempts = 1
    router = pool(providers)

    await router.reason(request(), prompt_version="v1")  # type: ignore[arg-type]

    metrics = router.metrics()
    assert metrics["provider_http_calls"] == 1
    assert metrics["groq_calls"] == 1
    assert metrics["retry_attempts"] == 0
    assert metrics["schema_corrections"] == 1
    assert metrics["groq_1_calls"] == 1
    assert metrics["groq_1_total_tokens"] == 15
    assert metrics["groq_2_calls"] == 0
    assert metrics["groq_4_schema_corrections_failed"] == 0
    assert "api_key" not in str(router.metadata()).lower()
