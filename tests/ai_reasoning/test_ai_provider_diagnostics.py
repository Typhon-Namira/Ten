from __future__ import annotations

import logging

import httpx
import pytest

from backend.app.ai.provider_client import HttpAIProviderClient
from backend.app.ai.provider_client.client import (
    ProviderJSONDecodeError,
    extract_single_json_object,
)
from backend.app.core.exceptions import AIProviderRequestError


def client_for(handler, provider: str = "groq_1") -> HttpAIProviderClient:
    return HttpAIProviderClient(
        provider,
        "safe-test-key",
        f"https://api.{provider}.test/v1",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.parametrize(
    ("content", "expected_note"),
    (
        ('{"status":"ok"}', None),
        ('  {"status":"ok"}  ', None),
        ('```json\n{"status":"ok"}\n```', "markdown_fence_present"),
        (
            'Result follows: {"status":"ok"} end.',
            "surrounding_prose_removed",
        ),
    ),
)
def test_single_json_object_extraction_accepts_harmless_formatting(
    content: str,
    expected_note: str | None,
) -> None:
    parsed, note = extract_single_json_object(content)

    assert parsed == {"status": "ok"}
    assert note == expected_note


def test_multiple_json_objects_are_rejected_as_ambiguous() -> None:
    with pytest.raises(ProviderJSONDecodeError) as captured:
        extract_single_json_object('{"first":1} {"second":2}')

    assert captured.value.reason_code == "multiple_json_objects"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (500, 502, 503, 504))
async def test_all_5xx_failures_are_typed_provider_unavailable(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": "temporary upstream failure"}},
            request=request,
        )

    with pytest.raises(AIProviderRequestError) as captured:
        await client_for(handler).complete_json(
            system_prompt="system",
            payload={"input": "test"},
            model="gpt-oss-120b",
            temperature=0,
            max_tokens=10,
        )

    assert captured.value.details.provider == "groq_1"
    assert captured.value.details.reason_code == "provider_unavailable"
    assert captured.value.details.http_status == status


@pytest.mark.asyncio
async def test_quota_failure_is_sanitized_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    api_key = "secret-key-that-must-never-be-logged"
    prompt = "sensitive prompt that must never be logged"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            headers={"content-type": "application/json", "retry-after": "60"},
            json={
                "error": {
                    "code": "quota_exhausted",
                    "message": "Daily quota exhausted",
                }
            },
            request=request,
        )

    client = HttpAIProviderClient(
        "groq_1",
        api_key,
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(
        logging.INFO,
        logger="backend.app.ai.provider_client.client",
    ), pytest.raises(AIProviderRequestError) as captured:
        await client.complete_json(
            system_prompt=prompt,
            payload={"private": "payload"},
            model="gpt-oss-120b",
            temperature=0,
            max_tokens=10,
            request_id="request-123",
            cycle_id="cycle-456",
            attempt_type="primary",
        )

    details = captured.value.details
    assert details.reason_code == "quota_exhausted"
    assert details.http_status == 402
    assert details.retry_after == "60"
    assert details.endpoint_host == "api.groq.test"
    assert details.endpoint_path == "/openai/v1/chat/completions"
    assert details.http_method == "POST"
    assert details.sanitized_response_body == (
        '{"error":{"code":"quota_exhausted","message":"Daily quota exhausted",'
        '"error_type":null,"provider_code":null}}'
    )
    assert details.serialized_request_bytes is not None
    assert details.estimated_input_tokens is not None
    assert details.attempt_type == "primary"
    failure = next(
        record for record in caplog.records
        if record.message == "ai_provider.request.failure_diagnostic"
    )
    assert failure.provider == "groq_1"
    assert failure.request_id == "request-123"
    assert failure.endpoint_host == "api.groq.test"
    assert failure.endpoint_path == "/openai/v1/chat/completions"
    assert failure.http_method == "POST"
    assert failure.status_code == 402
    assert failure.response_content_type == "application/json"
    assert failure.response_body_length > 0
    assert failure.attempt_type == "primary"
    assert api_key not in caplog.text
    assert prompt not in caplog.text
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_timeout_failure_has_safe_network_classification() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("safe timeout", request=request)

    with pytest.raises(AIProviderRequestError) as captured:
        await client_for(handler).complete_json(
            system_prompt="system",
            payload={"input": "test"},
            model="gpt-oss-120b",
            temperature=0,
            max_tokens=10,
            attempt_type="retry",
        )

    details = captured.value.details
    assert details.reason_code == "request_timeout"
    assert details.timeout_category == "read_timeout"
    assert details.network_error_category == "timeout"
    assert details.attempt_type == "retry"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("groq_1", "groq_2"))
async def test_http_200_invalid_json_is_typed_response_decoding_failure(
    provider: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "not-json"}}],
                "usage": {
                    "prompt_tokens": 31,
                    "completion_tokens": 7,
                    "total_tokens": 38,
                },
            },
            request=request,
        )

    with pytest.raises(AIProviderRequestError) as captured:
        await client_for(handler, provider).complete_json(
            system_prompt="system",
            payload={"input": "test"},
            model="gpt-oss-120b",
            temperature=0,
            max_tokens=10,
        )

    assert captured.value.details.reason_code == "response_decoding_failed"
    assert captured.value.details.phase == "response_decoding"
    assert captured.value.details.provider == provider
    assert captured.value.details.provider_input_tokens == 31
    assert captured.value.details.provider_output_tokens == 7
    assert captured.value.details.provider_total_tokens == 38


@pytest.mark.asyncio
async def test_success_returns_typed_completion_without_logging_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-request-id": "provider-1"},
            json={
                "choices": [{"message": {"content": '{"decision":"WAIT"}'}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
            request=request,
        )

    with caplog.at_level(logging.INFO, logger="backend.app.ai.provider_client.client"):
        result = await client_for(handler).complete_json(
            system_prompt="secret-system-prompt",
            payload={"private": "payload"},
            model="gpt-oss-120b",
            temperature=0,
            max_tokens=10,
        )

    assert result.content == {"decision": "WAIT"}
    assert result.provider == "groq_1"
    assert result.token_usage == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert "secret-system-prompt" not in caplog.text
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_json_validate_failed_preserves_safe_metadata_without_raw_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failed = '{"output_profile":'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            json={
                "error": {
                    "code": "json_validate_failed",
                    "message": "generated JSON failed",
                    "metadata": {
                        "failed_generation": failed,
                        "reason": "object incomplete",
                        "path": "$.market_structure.short_term",
                    },
                }
            },
            request=request,
        )

    with caplog.at_level(logging.INFO), pytest.raises(
        AIProviderRequestError
    ) as captured:
        await client_for(handler).complete_json(
            system_prompt="system",
            payload={"input": "test"},
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=10,
        )

    details = captured.value.details
    assert details.reason_code == "json_generation_failed"
    assert details.failed_generation_type == "str"
    assert details.failed_generation_length == len(failed)
    assert details.failed_generation_hash is not None
    assert details.failed_generation_reason == "object incomplete"
    assert details.failed_generation_path == "$.market_structure.short_term"
    assert failed not in caplog.text


@pytest.mark.asyncio
async def test_missing_usage_remains_null_and_finish_reason_is_exposed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"status":"ok"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
            request=request,
        )

    result = await client_for(handler).complete_json(
        system_prompt="system",
        payload={"input": "test"},
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=10,
    )

    assert result.token_usage is None
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_groq_tracks_request_and_token_rate_limit_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "14400",
                "x-ratelimit-remaining-requests": "14399",
                "x-ratelimit-reset-requests": "23h59m",
                "x-ratelimit-limit-tokens": "6000",
                "x-ratelimit-remaining-tokens": "5900",
                "x-ratelimit-reset-tokens": "7.5s",
            },
            json={
                "choices": [{"message": {"content": '{"decision":"WAIT"}'}}],
            },
            request=request,
        )

    result = await client_for(handler, "groq").complete_json(
        system_prompt="system",
        payload={"input": "test"},
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=10,
    )

    assert result.rate_limit_request_limit == "14400"
    assert result.rate_limit_request_remaining == "14399"
    assert result.rate_limit_request_reset == "23h59m"
    assert result.rate_limit_token_limit == "6000"
    assert result.rate_limit_token_remaining == "5900"
    assert result.rate_limit_token_reset == "7.5s"


@pytest.mark.asyncio
async def test_groq_temporary_token_rate_limit_is_not_daily_quota() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={
                "x-ratelimit-remaining-requests": "14000",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "8s",
            },
            json={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Token rate limit reached for this model.",
                }
            },
            request=request,
        )

    with pytest.raises(AIProviderRequestError) as captured:
        await client_for(handler, "groq").complete_json(
            system_prompt="system",
            payload={"input": "test"},
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=10,
        )

    details = captured.value.details
    assert details.reason_code == "rate_limited"
    assert details.limit_classification == "RATE_LIMITED_TEMPORARY"
    assert details.rate_limit_request_remaining == "14000"
    assert details.rate_limit_token_remaining == "0"
    assert details.rate_limit_token_reset == "8s"


@pytest.mark.asyncio
async def test_daily_quota_is_distinct_from_temporary_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"x-ratelimit-remaining-requests": "0"},
            json={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Requests per day quota exhausted.",
                }
            },
            request=request,
        )

    with pytest.raises(AIProviderRequestError) as captured:
        await client_for(handler, "groq_1").complete_json(
            system_prompt="system",
            payload={"input": "test"},
            model="gpt-oss-120b",
            temperature=0,
            max_tokens=10,
        )

    assert captured.value.details.reason_code == "quota_exhausted"
