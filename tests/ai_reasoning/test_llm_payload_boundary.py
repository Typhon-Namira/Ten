from __future__ import annotations

from datetime import timedelta
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from backend.app.ai.provider_client import (
    AIProviderClient,
    AIProviderCompletion,
    HttpAIProviderClient,
    build_request_body,
    measure_request_body,
)
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.ai_reasoning.config import AIReasoningConfig
from backend.app.ai_reasoning.llm_context import LLMAnalysisContext, build_llm_analysis_context
from backend.app.ai_reasoning.memory import MarketMemory
from backend.app.ai_reasoning.models import MarketMemoryEntry, MarketMemorySummary
from backend.app.ai_reasoning.provider import (
    GroqProvider,
    reasoning_response_schema,
)
from backend.app.ai_reasoning.request_persistence import (
    decode_persisted_request,
    persisted_request_payload,
)
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from backend.app.ai_reasoning.validation import (
    StructuredAIOutputError,
    StructuredAIOutputValidator,
)
from backend.app.core.config import YamlConfigRepository
from backend.app.core.exceptions import AIProviderRequestError
from tests.ai_reasoning.test_analysis_architecture_v2 import output as analysis_output
from tests.ai_reasoning.test_ai_reasoning_lifecycle import NOW, state_and_quant


class CapturingClient(AIProviderClient):
    provider = "groq_1"
    base_url = "https://api.groq.test/openai/v1"
    configured = True

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.response = response or analysis_output().model_dump(mode="python")

    async def available_models(self) -> tuple[str, ...]:
        return ("gpt-oss-120b",)

    async def complete_json(self, **kwargs: Any) -> AIProviderCompletion:
        self.calls += 1
        self.payloads.append(kwargs["payload"])
        self.requests.append(kwargs)
        return AIProviderCompletion(
            content=self.response,
            provider="groq_1",
            model=kwargs["model"],
            status_code=200,
            latency_ms=1,
            provider_request_id="test-request",
            token_usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            rate_limit_limit=None,
            rate_limit_remaining=None,
            rate_limit_reset=None,
            retry_after=None,
        )


async def _request(memory: MarketMemorySummary | None = None):
    state, quant = await state_and_quant()
    config = YamlConfigRepository().load_model("ai_reasoning", AIReasoningConfig)
    request = AIReasoningRequestBuilder(
        config,
        model_identifier="gpt-oss-120b",
        clock=lambda: NOW,
    ).build(
        state,
        quant,
        memory or MarketMemorySummary(entry_count=0),
        existing_signal=None,
        previous_forecast=None,
        previous_proposal=None,
    )
    return state, quant, config, request


def _provider(
    client: CapturingClient,
    config: AIReasoningConfig,
    **overrides: Any,
) -> GroqProvider:
    values = {
        "model": "gpt-oss-120b",
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "target_input_tokens": config.target_input_tokens,
        "warning_input_tokens": config.warning_input_tokens,
        "hard_input_tokens": config.hard_input_tokens,
        "absolute_max_output_tokens": config.absolute_max_output_tokens,
        "maximum_request_cost_usd": config.maximum_request_cost_usd,
        "input_cost_per_million_usd": config.input_cost_per_million_usd,
        "output_cost_per_million_usd": config.output_cost_per_million_usd,
        "setup_family_ids": tuple(
            item.setup_family_id
            for item in SetupFamilyRegistry.from_yaml(YamlConfigRepository()).all()
        ),
    }
    values.update(overrides)
    return GroqProvider(
        client,
        PromptLoader(Path("backend/app/ai_reasoning/prompts")),
        account_id="groq_1",
        **values,
    )


def _groq_provider(
    client: AIProviderClient,
    config: AIReasoningConfig,
) -> GroqProvider:
    return GroqProvider(
        client,
        PromptLoader(Path("backend/app/ai_reasoning/prompts")),
        account_id="groq_1",
        model="gpt-oss-120b",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        target_input_tokens=config.target_input_tokens,
        warning_input_tokens=config.warning_input_tokens,
        hard_input_tokens=config.hard_input_tokens,
        absolute_max_output_tokens=config.absolute_max_output_tokens,
        maximum_request_cost_usd=config.maximum_request_cost_usd,
        input_cost_per_million_usd=config.input_cost_per_million_usd,
        output_cost_per_million_usd=config.output_cost_per_million_usd,
        setup_family_ids=tuple(
            item.setup_family_id
            for item in SetupFamilyRegistry.from_yaml(YamlConfigRepository()).all()
        ),
    )


@pytest.mark.asyncio
async def test_provider_serializes_only_typed_compact_context_without_candles_or_engine_objects() -> None:
    _, _, config, request = await _request()
    client = CapturingClient()

    await _provider(client, config).reason(request, prompt_version=request.prompt_version)

    assert client.calls == 1
    payload = client.payloads[0]
    assert set(payload) == {"analysis_context", "response_contract"}
    context = LLMAnalysisContext.model_validate(payload["analysis_context"])
    encoded = json.dumps(payload, sort_keys=True)
    prohibited = (
        '"analysis_request"',
        '"candles"',
        '"raw"',
        '"smc_evidence"',
        '"volume_profile_evidence"',
        '"feature_vector"',
        '"previous_ai_forecast"',
        '"previous_ai_proposal"',
        '"dashboard"',
    )
    assert all(token not in encoded for token in prohibited)
    assert context.current_price > 0


def _request_record(request: Any, payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request.request_id,
        cycle_id=request.cycle_id,
        market_state_id=request.market_state_id,
        quantitative_forecast_id=request.quantitative_forecast_id,
        instrument=request.instrument,
        analysis_timestamp=request.analysis_timestamp,
        prompt_version=request.prompt_version,
        model_identifier=request.model_identifier,
        payload=payload,
        created_at=request.created_at,
    )


@pytest.mark.asyncio
async def test_versioned_compact_request_history_decodes_without_reconstructing_internal_request() -> None:
    _, _, _, request = await _request()
    context = build_llm_analysis_context(request)
    payload = persisted_request_payload(request, context)

    decoded = decode_persisted_request(_request_record(request, payload))

    assert decoded.compatibility_status == "compatible"
    assert decoded.payload_format == "versioned_compact"
    assert decoded.context_schema_version == "2.0"
    assert decoded.request_id == request.request_id
    assert "supported_timeframe_states" not in payload
    assert "smc_evidence" not in payload


@pytest.mark.asyncio
async def test_deployed_legacy_context_envelope_remains_readable() -> None:
    _, _, _, request = await _request()
    context = build_llm_analysis_context(request)
    payload = persisted_request_payload(request, context)
    legacy_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"payload_type", "payload_schema_version", "context_schema_version"}
    }
    legacy_payload["schema_version"] = context.schema_version

    decoded = decode_persisted_request(_request_record(request, legacy_payload))

    assert decoded.compatibility_status == "compatible"
    assert decoded.payload_format == "legacy_compact_context"
    assert decoded.context_schema_version == "2.0"


@pytest.mark.asyncio
async def test_legacy_full_internal_request_remains_readable_as_bounded_snapshot() -> None:
    _, _, _, request = await _request()

    decoded = decode_persisted_request(
        _request_record(request, request.model_dump(mode="json"))
    )

    assert decoded.compatibility_status == "compatible"
    assert decoded.payload_format == "legacy_full_request"
    assert decoded.request_id == request.request_id


@pytest.mark.asyncio
async def test_unknown_request_history_shape_returns_typed_incompatibility_without_raw_payload() -> None:
    _, _, _, request = await _request()

    decoded = decode_persisted_request(
        _request_record(request, {"unexpected": "secret-value"})
    )

    assert decoded.compatibility_status == "incompatible"
    assert decoded.payload_format == "incompatible"
    assert decoded.compatibility_reason == "unrecognized_persisted_request_payload"
    assert "secret-value" not in decoded.model_dump_json()


@pytest.mark.asyncio
async def test_compact_context_collection_cardinalities_are_hard_bounded() -> None:
    _, _, _, request = await _request()
    context = build_llm_analysis_context(request)

    assert len(context.timeframe_trends) <= 3
    assert len(context.nearest_supply_zones) <= 3
    assert len(context.nearest_demand_zones) <= 3
    assert len(context.relevant_order_blocks) <= 3
    assert len(context.relevant_fair_value_gaps) <= 3
    assert len(context.nearest_liquidity_levels) <= 5
    assert len(context.volume_profile.nearest_hvns) <= 3
    assert len(context.volume_profile.nearest_lvns) <= 3
    assert len(context.material_changes) <= 5


@pytest.mark.asyncio
async def test_normal_context_stays_below_target_token_budget() -> None:
    _, _, config, request = await _request()
    context = build_llm_analysis_context(request)
    provider = _provider(CapturingClient(), config)
    prompt = provider.prompts.load(request.prompt_version)
    payload = {
        "analysis_context": context.model_dump(mode="json"),
        "response_contract": provider._response_contract(),
    }
    body = build_request_body(
        system_prompt=prompt,
        payload=payload,
        model=provider.model,
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
    )
    metrics = measure_request_body(
        body,
        input_cost_per_million_usd=config.input_cost_per_million_usd,
        output_cost_per_million_usd=config.output_cost_per_million_usd,
    )

    assert metrics.estimated_input_tokens <= config.target_input_tokens
    assert metrics.maximum_output_tokens == 1_000
    assert metrics.maximum_output_tokens <= config.absolute_max_output_tokens


@pytest.mark.asyncio
async def test_oversized_context_is_rejected_before_provider_and_not_typed_as_credit_failure() -> None:
    _, _, config, request = await _request()
    client = CapturingClient()
    provider = _provider(client, config, hard_input_tokens=100)

    with pytest.raises(AIProviderRequestError) as captured:
        await provider.reason(request, prompt_version=request.prompt_version)

    assert client.calls == 0
    assert captured.value.details.reason_code == "request_too_large"
    assert captured.value.details.phase == "request_validation"
    assert captured.value.details.reason_code != "quota_exhausted"


@pytest.mark.asyncio
async def test_groq_uses_json_object_mode_with_application_validation() -> None:
    _, _, config, request = await _request()
    client = CapturingClient(analysis_output().model_dump(mode="python"))

    response = await _groq_provider(client, config).reason(
        request,
        prompt_version=request.prompt_version,
    )
    validated = StructuredAIOutputValidator().validate_analysis(response.raw_output)

    assert client.calls == 1
    assert client.requests[0]["response_schema"] is None
    assert response.model_identifier == "gpt-oss-120b"
    assert validated.market_regime.classification.value == "bullish"


@pytest.mark.asyncio
async def test_groq_malformed_json_gets_exactly_one_correction_attempt() -> None:
    _, _, config, request = await _request()
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        content = (
            "not-json"
            if len(bodies) == 1
            else json.dumps(analysis_output().model_dump(mode="python"))
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    response = await _groq_provider(client, config).reason(
        request,
        prompt_version=request.prompt_version,
    )

    assert response.raw_output["market_regime"]["classification"] == "bullish"
    assert len(bodies) == 2
    assert all(body["response_format"] == {"type": "json_object"} for body in bodies)


@pytest.mark.asyncio
async def test_groq_missing_executive_summary_gets_one_explicit_schema_correction() -> None:
    _, _, config, request = await _request()
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        output = analysis_output().model_dump(mode="python")
        output.pop("executive_summary", None)
        if len(bodies) == 2:
            output["executive_summary"] = "Validated analysis summary."
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(output)}}],
                "usage": {
                    "prompt_tokens": 10 + len(bodies),
                    "completion_tokens": 2 + len(bodies),
                    "total_tokens": 12 + (2 * len(bodies)),
                },
            },
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = _groq_provider(client, config)
    response = await provider.reason(
        request,
        prompt_version=request.prompt_version,
    )
    validated = StructuredAIOutputValidator().validate_analysis(response.raw_output)

    assert len(bodies) == 2
    correction_payload = json.loads(bodies[1]["messages"][1]["content"])
    assert "provider_response.executive_summary" in correction_payload["validation_error"]
    assert "analysis_context" not in correction_payload
    assert len(bodies[1]["messages"][1]["content"]) < len(
        bodies[0]["messages"][1]["content"]
    )
    assert response.raw_output["executive_summary"] == "Validated analysis summary."
    attempts = provider.attempts_for(request.request_id)
    assert [item["request_kind"] for item in attempts] == [
        "analysis",
        "schema_correction",
    ]
    assert [item["total_tokens"] for item in attempts] == [14, 16]
    assert response.token_usage == {
        "input_tokens": 23,
        "output_tokens": 7,
        "total_tokens": 30,
    }
    assert validated.executive_summary == "Validated analysis summary."


@pytest.mark.asyncio
async def test_finish_reason_length_is_not_sent_as_schema_correction() -> None:
    _, _, config, request = await _request()
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        output = analysis_output().model_dump(mode="python")
        output.pop("executive_summary")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps(output)},
                        "finish_reason": "length",
                    }
                ]
            },
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq_1",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIProviderRequestError) as captured:
        await _groq_provider(client, config).reason(
            request,
            prompt_version=request.prompt_version,
        )

    assert calls == 1
    assert captured.value.details.reason_code == "truncated_response"
    assert captured.value.details.schema_error_code == "finish_reason_length"


@pytest.mark.asyncio
async def test_known_empty_rate_limit_capacity_skips_correction() -> None:
    _, _, config, request = await _request()
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        output = analysis_output().model_dump(mode="python")
        output.pop("executive_summary")
        return httpx.Response(
            200,
            headers={
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "8s",
            },
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq_1",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIProviderRequestError) as captured:
        await _groq_provider(client, config).reason(
            request,
            prompt_version=request.prompt_version,
        )

    assert calls == 1
    assert captured.value.details.reason_code == "rate_limited"
    assert (
        captured.value.details.limit_classification
        == "RATE_LIMITED_TEMPORARY"
    )


@pytest.mark.asyncio
async def test_groq_missing_regime_classification_is_visible_and_corrected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, _, config, request = await _request()
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        output = analysis_output().model_dump(mode="python")
        output["market_regime"].pop("classification", None)
        if len(bodies) == 2:
            output["market_regime"]["classification"] = "bullish"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(logging.INFO):
        response = await _groq_provider(client, config).reason(
            request,
            prompt_version=request.prompt_version,
        )
        validated = StructuredAIOutputValidator().validate_analysis(response.raw_output)

    response_logs = [
        record
        for record in caplog.records
        if record.message == "ai_provider.response.diagnostic"
    ]
    normalized_log = next(
        record
        for record in caplog.records
        if record.message == "ai_provider.response.normalized"
    )

    assert len(bodies) == 2
    assert len(response_logs[0].raw_response_sha256) == 64
    assert response_logs[0].raw_response_character_count > 0
    assert not hasattr(response_logs[0], "raw_provider_json")
    correction_payload = json.loads(bodies[1]["messages"][1]["content"])
    assert "provider_response.market_regime.classification" in correction_payload[
        "validation_error"
    ]
    assert len(normalized_log.normalized_response_sha256) == 64
    assert normalized_log.normalized_response_character_count > 0
    assert not hasattr(normalized_log, "normalized_provider_json")
    assert validated.market_regime.classification.value == "bullish"


@pytest.mark.asyncio
async def test_groq_echoed_contract_metadata_is_removed_locally_without_correction() -> None:
    _, _, config, request = await _request()
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        output = analysis_output().model_dump(mode="python")
        if len(bodies) == 1:
            output["schema_type"] = "ten_ai_reasoning_response"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    response = await _groq_provider(client, config).reason(
        request,
        prompt_version=request.prompt_version,
    )
    validated = StructuredAIOutputValidator().validate_analysis(response.raw_output)

    assert len(bodies) == 1
    assert "schema_type" not in response.raw_output
    assert validated.analysis_confidence > 0


@pytest.mark.asyncio
async def test_groq_second_malformed_json_fails_closed_without_more_attempts() -> None:
    _, _, config, request = await _request()
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "still-not-json"}}]},
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIProviderRequestError) as captured:
        await _groq_provider(client, config).reason(
            request,
            prompt_version=request.prompt_version,
        )

    assert calls == 2
    assert captured.value.details.reason_code == "response_decoding_failed"


@pytest.mark.asyncio
async def test_analysis_response_missing_required_field_is_rejected() -> None:
    raw = analysis_output().model_dump(mode="python")
    raw.pop("analysis_confidence")

    with pytest.raises(StructuredAIOutputError) as captured:
        StructuredAIOutputValidator().validate_analysis(raw)

    assert captured.value.first_issue is not None
    assert captured.value.first_issue.field_path == "provider_response.analysis_confidence"


@pytest.mark.asyncio
async def test_analysis_response_additional_property_is_rejected() -> None:
    raw = analysis_output().model_dump(mode="python")
    raw["unexpected"] = "must-not-be-accepted"

    with pytest.raises(StructuredAIOutputError) as captured:
        StructuredAIOutputValidator().validate_analysis(raw)

    assert captured.value.first_issue is not None
    assert captured.value.first_issue.field_path == "provider_response.unexpected"


@pytest.mark.asyncio
async def test_response_contract_is_analysis_only_and_strict() -> None:
    _, _, config, _ = await _request()

    contract = _provider(CapturingClient(), config)._response_contract()

    assert contract["json_schema"] == reasoning_response_schema()
    encoded = json.dumps(contract)
    for prohibited in ("setup_family", "entry_low", "stop_loss", "take_profit_levels"):
        assert prohibited not in encoded
    assert "market_regime" in contract["json_schema"]["properties"]
    assert "do not recommend BUY, SELL, WAIT" in " ".join(contract["rules"])


def test_prompt_templates_explicitly_require_every_provider_wire_field() -> None:
    directory = Path("backend/app/ai_reasoning/prompts")

    for name in ("deep_market_analysis_v2.txt", "existing_position_market_analysis_v2.txt"):
        prompt = (directory / name).read_text(encoding="utf-8")
        for field in (
            "market_regime",
            "higher_timeframe_context",
            "market_structure",
            "liquidity_analysis",
            "supply_demand_analysis",
            "momentum_analysis",
            "volatility_analysis",
            "analysis_confidence",
            "executive_summary",
        ):
            assert field in prompt
        assert "do not emit response-contract metadata" in prompt.lower()
        assert "Never output BUY, SELL, WAIT" in prompt


def test_setup_family_registry_never_fuzzy_maps_unknown_values() -> None:
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())

    assert registry.canonical_id("breakout") == ("breakout_retest", True)
    assert registry.canonical_id("Trend-Continuation") == ("trend_continuation", True)
    assert registry.canonical_id("breakout_reversal") == (None, False)
    assert registry.canonical_id("best setup") == (None, False)


@pytest.mark.asyncio
async def test_history_is_bounded_to_five_material_changes_and_cannot_accumulate_prompt_turns() -> None:
    common_tail = tuple(f"material-{index}" for index in range(5))
    short_memory = MarketMemorySummary(
        entry_count=25,
        regime_transitions=tuple(f"old-{index}" for index in range(20)) + common_tail,
    )
    long_memory = MarketMemorySummary(
        entry_count=105,
        regime_transitions=tuple(f"older-{index}" for index in range(100)) + common_tail,
    )
    _, _, _, short_request = await _request(short_memory)
    _, _, _, long_request = await _request(long_memory)
    short_context = build_llm_analysis_context(short_request)
    long_context = build_llm_analysis_context(long_request)

    assert short_context.material_changes == long_context.material_changes
    assert len(short_context.material_changes) == 5
    assert short_context.previous_final_decision is None
    assert "messages" not in short_context.model_dump()
    assert len(json.dumps(short_context.model_dump(mode="json"))) == len(
        json.dumps(long_context.model_dump(mode="json"))
    )


def test_market_memory_summary_never_includes_full_structured_payload_history() -> None:
    entries = tuple(
        MarketMemoryEntry(
            entry_id=__import__("uuid").uuid4(),
            instrument="XAUUSD",
            cycle_id=__import__("uuid").uuid4(),
            market_state_id=__import__("uuid").uuid4(),
            category="evidence_change",
            summary=f"summary-{index}",
            structured_payload={"private_full_payload": "x" * 20_000},
            occurred_at=NOW + timedelta(seconds=index),
        )
        for index in range(20)
    )

    summary = MarketMemory(20).summarize(entries)
    encoded = json.dumps(summary.model_dump(mode="json"))
    assert "private_full_payload" not in encoded
    assert len(summary.evidence_changes) == 20
