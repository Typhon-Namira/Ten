from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from backend.app.ai_reasoning.cerebras_probe import (
    run_analysis_probe,
    run_minimal_probe,
)
from backend.app.ai_reasoning.models import AIReasoningRequest, MarketMemorySummary
from backend.app.core.config.settings import Settings
from tests.ai_reasoning.test_ai_reasoning_lifecycle import analysis_payload


def settings() -> Settings:
    return Settings(
        _env_file=None,
        cerebras_api_key="safe-test-key",
        cerebras_base_url="https://api.cerebras.test/v1",
        cerebras_model="gpt-oss-120b",
        market_data_worker_enabled=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "content"),
    (("text", "OK"), ("json", '{"status":"ok"}')),
)
async def test_direct_probe_bypasses_router_and_validates_expected_output(
    mode: str,
    content: str,
) -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "x-request-id": "probe-request-1",
                "x-ratelimit-remaining-requests": "99",
            },
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    result = await run_minimal_probe(
        mode,  # type: ignore[arg-type]
        settings(),
        transport=httpx.MockTransport(handler),
    )

    assert result.success is True
    assert result.http_status == 200
    assert result.endpoint_host == "api.cerebras.test"
    assert result.endpoint_path == "/v1/chat/completions"
    assert result.provider_request_id == "probe-request-1"
    assert result.rate_limit_remaining == "99"
    assert len(captured) == 1
    assert captured[0]["model"] == "gpt-oss-120b"
    if mode == "json":
        assert captured[0]["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in captured[0]


@pytest.mark.asyncio
async def test_direct_probe_rejects_http_200_with_wrong_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-ok"}}]},
            request=request,
        )

    result = await run_minimal_probe(
        "text",
        settings(),
        transport=httpx.MockTransport(handler),
    )

    assert result.success is False
    assert result.http_status == 200
    assert result.sanitized_response.startswith("probe_output_invalid")


@pytest.mark.asyncio
async def test_direct_probe_reports_missing_key_without_network_call() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500, request=request)

    result = await run_minimal_probe(
        "text",
        Settings(
            _env_file=None,
            cerebras_api_key=None,
            market_data_worker_enabled=False,
        ),
        transport=httpx.MockTransport(handler),
    )

    assert result.success is False
    assert result.api_key_present is False
    assert result.http_status is None
    assert result.sanitized_response == "TEN_CEREBRAS_API_KEY is not configured"
    assert called is False


@pytest.mark.asyncio
async def test_full_analysis_probe_uses_production_prompt_and_strict_schema() -> None:
    now = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
    request = AIReasoningRequest(
        request_id=uuid4(),
        cycle_id=uuid4(),
        market_state_id=uuid4(),
        quantitative_forecast_id=uuid4(),
        instrument="XAUUSD",
        analysis_timestamp=now,
        knowledge_cutoff=now,
        trigger_timeframe="M5",
        current_price=3400,
        supported_timeframe_states=(),
        data_quality_summary={},
        quantitative_probabilities={},
        expected_movement={},
        tp_probabilities={},
        market_memory=MarketMemorySummary(entry_count=0),
        prompt_version="new_market_analysis_v1",
        reasoning_policy_version="ai_reasoning_policy_v1",
        setup_family_registry_version="1.0.0",
        model_identifier="gpt-oss-120b",
        quantitative_model_version="1.0.0",
        feature_schema_version="1.0",
        market_state_schema_version="1.0",
        created_at=now,
    )
    captured: list[dict[str, object]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(http_request.content))
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "x-request-id": "analysis-probe-1",
            },
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                analysis_payload(),
                                separators=(",", ":"),
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
            request=http_request,
        )

    result = await run_analysis_probe(
        request,
        settings(),
        transport=httpx.MockTransport(handler),
    )

    assert result.success is True
    assert result.validation_passed is True
    assert result.provider_request_id == "analysis-probe-1"
    assert len(captured) == 1
    assert captured[0]["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert captured[0]["response_format"]["json_schema"]["strict"] is True  # type: ignore[index]
