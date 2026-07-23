from __future__ import annotations

import logging

import httpx
import pytest

from backend.app.ai.openrouter_client.client import HttpOpenRouterClient
from backend.app.core.exceptions import OpenRouterRequestError


@pytest.mark.asyncio
async def test_http_402_is_sanitized_logged_and_typed(caplog: pytest.LogCaptureFixture) -> None:
    api_key = "secret-key-that-must-never-be-logged"
    prompt = "sensitive prompt that must never be logged"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            headers={"content-type": "application/json", "retry-after": "60"},
            json={
                "error": {
                    "code": 402,
                    "message": "Insufficient credits",
                    "metadata": {
                        "error_type": "insufficient_credits",
                        "provider_code": "402",
                    },
                }
            },
            request=request,
        )

    client = HttpOpenRouterClient(
        api_key,
        "https://openrouter.ai/api/v1",
        transport=httpx.MockTransport(handler),
    )
    target_logger = logging.getLogger("backend.app.ai.openrouter_client.client")
    target_logger.addHandler(caplog.handler)
    previous_level = target_logger.level
    target_logger.setLevel(logging.INFO)
    try:
        with pytest.raises(OpenRouterRequestError) as captured:
            await client.complete_json(
                system_prompt=prompt,
                payload={"private": "payload"},
                model="meta-llama/llama-3.3-70b-instruct",
                temperature=0,
                max_tokens=10,
                request_id="request-123",
                cycle_id="cycle-456",
            )
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(previous_level)

    details = captured.value.details
    assert details.reason_code == "openrouter_insufficient_credits"
    assert details.phase == "http_request"
    assert details.http_status == 402
    assert details.error_code == "402"
    assert details.error_message == "Insufficient credits"
    assert details.metadata_error_type == "insufficient_credits"
    assert details.metadata_provider_code == "402"
    assert details.content_type == "application/json"
    assert details.body_length and details.body_length > 0
    assert details.retry_after == "60"
    assert details.elapsed_ms is not None

    failure = next(record for record in caplog.records if record.message == "openrouter.request.failed")
    assert failure.request_id == "request-123"
    assert failure.cycle_id == "cycle-456"
    assert failure.failure_phase == "http_request"
    assert failure.failed_during_http_request is True
    assert failure.failed_during_response_decoding is False
    assert failure.failed_during_structured_output_validation is False
    assert failure.failed_during_domain_parsing is False
    assert failure.failed_during_persistence is False
    assert api_key not in caplog.text
    assert prompt not in caplog.text
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_http_200_parse_failure_is_typed_as_response_decoding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"choices": [{"message": {"content": "not-json"}}]},
            request=request,
        )

    client = HttpOpenRouterClient(
        "safe-test-key",
        "https://openrouter.ai/api/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenRouterRequestError) as captured:
        await client.complete_json(
            system_prompt="system",
            payload={"input": "test"},
            model="model",
            temperature=0,
            max_tokens=10,
            request_id="request",
            cycle_id="cycle",
        )

    assert captured.value.details.reason_code == "openrouter_response_decoding_failed"
    assert captured.value.details.phase == "response_decoding"
    assert captured.value.details.http_status == 200
