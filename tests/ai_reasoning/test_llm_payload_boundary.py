from __future__ import annotations

from datetime import timedelta
import json
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
from backend.app.ai_reasoning.provider import CerebrasProvider, GroqProvider
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
from tests.ai_reasoning.test_ai_reasoning_lifecycle import NOW, state_and_quant


class CapturingClient(AIProviderClient):
    provider = "cerebras"
    base_url = "https://api.cerebras.ai/v1"
    configured = True

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.response = response or {
            "decision": "WAIT",
            "confidence": 0.8,
            "rationale": "No actionable setup.",
            "risk_flags": [],
            "proposal": None,
        }

    async def available_models(self) -> tuple[str, ...]:
        return ("gpt-oss-120b",)

    async def complete_json(self, **kwargs: Any) -> AIProviderCompletion:
        self.calls += 1
        self.payloads.append(kwargs["payload"])
        self.requests.append(kwargs)
        return AIProviderCompletion(
            content=self.response,
            provider="cerebras",
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
) -> CerebrasProvider:
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
    return CerebrasProvider(
        client,
        PromptLoader(Path("backend/app/ai_reasoning/prompts")),
        **values,
    )


def _groq_provider(
    client: AIProviderClient,
    config: AIReasoningConfig,
) -> GroqProvider:
    return GroqProvider(
        client,
        PromptLoader(Path("backend/app/ai_reasoning/prompts")),
        model="llama-3.1-8b-instant",
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
async def test_compact_wait_response_and_output_limit() -> None:
    state, quant, config, request = await _request()
    client = CapturingClient()
    response = await _provider(client, config).reason(request, prompt_version=request.prompt_version)
    validated = StructuredAIOutputValidator(
        SetupFamilyRegistry.from_yaml(YamlConfigRepository())
    ).validate(response.raw_output, request=request, state=state, quant=quant)

    assert response.raw_output == {
        "decision": "WAIT",
        "confidence": 0.8,
        "rationale": "No actionable setup.",
        "risk_flags": [],
        "proposal": None,
    }
    assert validated.forecast.status.value == "non_actionable"
    assert validated.forecast.dominant_direction is not None
    assert validated.proposal is None
    assert validated.degraded_validation is False
    assert validated.repaired_fields == ()
    assert validated.validation_issues == ()
    assert client.calls == 1
    assert config.max_tokens == 1_000


@pytest.mark.asyncio
async def test_groq_llama_uses_json_object_mode_with_application_validation() -> None:
    state, quant, config, request = await _request()
    client = CapturingClient()

    response = await _groq_provider(client, config).reason(
        request,
        prompt_version=request.prompt_version,
    )
    validated = StructuredAIOutputValidator(
        SetupFamilyRegistry.from_yaml(YamlConfigRepository())
    ).validate(response.raw_output, request=request, state=state, quant=quant)

    assert client.calls == 1
    assert client.requests[0]["response_schema"] is None
    assert response.model_identifier == "llama-3.1-8b-instant"
    assert validated.forecast.status.value == "non_actionable"


@pytest.mark.asyncio
async def test_groq_malformed_json_gets_exactly_one_correction_attempt() -> None:
    _, _, config, request = await _request()
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        content = (
            "not-json"
            if len(bodies) == 1
            else json.dumps(
                {
                    "decision": "WAIT",
                    "confidence": 0.8,
                    "rationale": "No actionable setup.",
                    "risk_flags": [],
                    "proposal": None,
                }
            )
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

    assert response.raw_output["decision"] == "WAIT"
    assert len(bodies) == 2
    assert all(body["response_format"] == {"type": "json_object"} for body in bodies)


@pytest.mark.asyncio
async def test_groq_missing_rationale_gets_one_explicit_schema_correction() -> None:
    state, quant, config, request = await _request()
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        output: dict[str, Any] = {
            "decision": "WAIT",
            "confidence": 0.8,
            "risk_flags": [],
            "proposal": None,
        }
        if len(bodies) == 2:
            output["rationale"] = "No actionable setup."
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
    validated = StructuredAIOutputValidator(
        SetupFamilyRegistry.from_yaml(YamlConfigRepository())
    ).validate(response.raw_output, request=request, state=state, quant=quant)

    assert len(bodies) == 2
    assert "provider_response.rationale" in bodies[1]["messages"][0]["content"]
    assert response.raw_output["rationale"] == "No actionable setup."
    assert validated.degraded_validation is False


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
async def test_compact_response_missing_required_field_is_rejected() -> None:
    state, quant, _, request = await _request()
    raw = {
        "decision": "WAIT",
        "rationale": "No setup.",
        "risk_flags": [],
        "proposal": None,
    }

    with pytest.raises(StructuredAIOutputError) as captured:
        StructuredAIOutputValidator(
            SetupFamilyRegistry.from_yaml(YamlConfigRepository())
        ).validate(raw, request=request, state=state, quant=quant)

    assert captured.value.first_issue is not None
    assert captured.value.first_issue.field_path == "provider_response.confidence"


@pytest.mark.asyncio
async def test_compact_response_additional_property_is_rejected() -> None:
    state, quant, _, request = await _request()
    raw = {
        "decision": "WAIT",
        "confidence": 0.8,
        "rationale": "No setup.",
        "risk_flags": [],
        "proposal": None,
        "unexpected": "must-not-be-accepted",
    }

    with pytest.raises(StructuredAIOutputError) as captured:
        StructuredAIOutputValidator(
            SetupFamilyRegistry.from_yaml(YamlConfigRepository())
        ).validate(raw, request=request, state=state, quant=quant)

    assert captured.value.first_issue is not None
    assert captured.value.first_issue.field_path == "provider_response.unexpected"


@pytest.mark.asyncio
async def test_response_contract_exposes_only_canonical_setup_family_ids() -> None:
    _, _, config, _ = await _request()
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())

    contract = _provider(CapturingClient(), config)._response_contract()

    assert contract["allowed_setup_families"] == [
        item.setup_family_id for item in registry.all()
    ]
    assert "copy setup_family exactly from allowed_setup_families" in contract["rules"]
    assert len(contract["allowed_setup_families"]) == 9


def _actionable_compact_response(setup_family: str) -> dict[str, Any]:
    return {
        "decision": "LONG",
        "confidence": 0.78,
        "rationale": "Constructive trend continuation.",
        "risk_flags": [],
        "proposal": {
            "setup_family": setup_family,
            "entry_low": 3300,
            "entry_high": 3301,
            "stop_loss": 3295,
            "take_profit_levels": [3311, 3320],
        },
    }


@pytest.mark.asyncio
async def test_known_setup_family_alias_is_repaired_without_weakening_registry_validation() -> None:
    state, quant, _, request = await _request()
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())

    validated = StructuredAIOutputValidator(registry).validate(
        _actionable_compact_response("Trend Following"),
        request=request,
        state=state,
        quant=quant,
    )

    assert validated.forecast.selected_setup_family == "trend_continuation"
    assert validated.proposal is not None
    assert "proposal.setup_family" in validated.repaired_fields
    assert validated.degraded_validation is True


@pytest.mark.asyncio
async def test_canonical_compact_response_is_valid_without_artificial_repair() -> None:
    state, quant, _, request = await _request()
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())

    validated = StructuredAIOutputValidator(registry).validate(
        _actionable_compact_response("trend_continuation"),
        request=request,
        state=state,
        quant=quant,
    )

    assert validated.forecast.selected_setup_family == "trend_continuation"
    assert validated.proposal is not None
    assert validated.repaired_fields == ()
    assert validated.validation_issues == ()
    assert validated.degraded_validation is False


@pytest.mark.asyncio
async def test_unknown_setup_family_preserves_reasoning_but_suppresses_unsafe_proposal() -> None:
    state, quant, _, request = await _request()
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())

    validated = StructuredAIOutputValidator(registry).validate(
        _actionable_compact_response("invented_smart_money_setup"),
        request=request,
        state=state,
        quant=quant,
    )

    assert validated.forecast.dominant_direction.value == "BUY"
    assert validated.forecast.reasoning_summary == "Constructive trend continuation."
    assert validated.forecast.selected_setup_family is None
    assert validated.forecast.setup_readiness.value == "not_ready"
    assert validated.forecast.execution_confidence == 0
    assert validated.proposal is None
    assert validated.degraded_validation is True
    first_issue = json.loads(validated.validation_issues[0])
    assert first_issue["field_path"] == "proposal.setup_family"
    assert first_issue["actual_value"] == "invented_smart_money_setup"
    assert first_issue["validator_name"] == "setup_family_registry"
    assert first_issue["recoverable"] is True


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
