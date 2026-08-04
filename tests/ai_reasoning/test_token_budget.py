from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import httpx
from pydantic import ValidationError

from backend.app.ai.provider_client import AIProviderCompletion, HttpAIProviderClient
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.ai_reasoning.compact_output import (
    CompactAIAnalysisOutput,
    CompactOutputValidationError,
    HIGHER_TIMEFRAME_SUMMARY_LIMIT,
    MARKET_REGIME_EVIDENCE_REF_LIMIT,
    normalize_compact_output_shapes,
    resolve_compact_output,
    truncate_market_regime_evidence_refs,
    validate_evidence_references,
    validate_zone_references,
)
from backend.app.ai_reasoning.llm_context import (
    build_llm_analysis_context,
    provider_context_payload,
)
from backend.app.ai_reasoning.provider import (
    GroqProvider,
    reasoning_response_schema,
    validate_provider_contract,
)
from backend.app.ai_reasoning.token_budget import OutputProfile, TokenBudgetManager
from tests.ai_reasoning.test_llm_payload_boundary import _request
from tests.ai_reasoning.test_ai_reasoning_lifecycle import (
    NOW,
    InMemoryAIReasoningRepository,
    ValidProvider,
    build_service,
)


def compact_output(request: Any, *, retry: bool = False) -> dict[str, Any]:
    context = build_llm_analysis_context(request)
    catalog = context.evidence_catalog
    refs = [catalog[0].evidence_id] if catalog else []
    output: dict[str, Any] = {
        "analysis_schema_version": "compact-1.1",
        "output_profile": "compact",
        "market_regime": {
            "classification": "bullish",
            "strength": 72,
            "confidence": 0.74,
            "evidence_refs": refs,
        },
        "higher_timeframe_context": {
            "bias": "bullish",
            "summary": "Higher-timeframe evidence remains constructive.",
            "evidence_refs": refs,
        },
        "market_structure": {
            "short_term": "Short-term structure is constructive.",
            "medium_term": "Medium-term structure retains higher lows.",
            "recent_change": "No confirmed bearish break.",
            "evidence_refs": refs,
        },
        "liquidity_analysis": {
            "summary": "Nearest liquidity remains unresolved.",
            "events": [],
            "unresolved": ["Nearest mapped pool remains open."],
            "evidence_refs": refs,
        },
        "supply_demand_analysis": {
            "summary": "Price remains between mapped supply and demand.",
            "nearest_supply_ref": (
                context.supply_zone_catalog[0].zone_id
                if context.supply_zone_catalog
                else None
            ),
            "nearest_demand_ref": (
                context.demand_zone_catalog[0].zone_id
                if context.demand_zone_catalog
                else None
            ),
            "evidence_refs": refs,
        },
        "momentum_analysis": {
            "direction": "bullish",
            "strength": 65,
            "trend": "stable",
            "evidence_refs": refs,
        },
        "volatility_analysis": {
            "state": "normal",
            "trend": "stable",
            "evidence_refs": refs,
        },
        "bullish_evidence_refs": refs,
        "bearish_evidence_refs": [],
        "contradiction_refs": [],
        "key_risk_refs": refs,
        "invalidation_conditions": ["A confirmed structural break invalidates the view."],
        "data_quality_warnings": [],
        "analysis_confidence": 0.72,
    }
    output.update(
        {
            "alternative_scenarios": [],
            "executive_summary": (
                "Constructive regime evidence persists with bounded risk and "
                "explicit invalidation."
            ),
        }
    )
    return output


def request_with_zones(request: Any) -> Any:
    evidence = [dict(item) for item in request.smc_evidence]
    evidence[0] = {
        **evidence[0],
        "raw": {
            **dict(evidence[0]["raw"]),
            "zones": [
                {
                    "zone_type": "supply",
                    "lower_price": 3340.0,
                    "upper_price": 3350.0,
                },
                {
                    "zone_type": "demand",
                    "lower_price": 3290.0,
                    "upper_price": 3300.0,
                },
            ],
        },
    }
    return request.model_copy(update={"smc_evidence": tuple(evidence)})


class CompactClient:
    provider = "groq_1"
    base_url = "https://api.groq.test/openai/v1"
    configured = True

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def available_models(self) -> tuple[str, ...]:
        return ("llama-3.1-8b-instant",)

    async def complete_json(self, **kwargs: Any) -> AIProviderCompletion:
        self.calls.append(kwargs)
        return AIProviderCompletion(
            content=self.response,
            provider=self.provider,
            model=kwargs["model"],
            status_code=200,
            latency_ms=5,
            provider_request_id="compact-success",
            token_usage={
                "input_tokens": 2100,
                "output_tokens": 620,
                "total_tokens": 2720,
            },
            rate_limit_limit=None,
            rate_limit_remaining=None,
            rate_limit_reset=None,
            retry_after=None,
            finish_reason="stop",
        )


def provider(client: CompactClient, config: Any) -> GroqProvider:
    return GroqProvider(
        client,  # type: ignore[arg-type]
        PromptLoader(Path("backend/app/ai_reasoning/prompts")),
        account_id="groq_1",
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=config.max_tokens,
        target_input_tokens=config.input_token_budget,
        warning_input_tokens=config.warning_input_tokens,
        hard_input_tokens=config.hard_input_tokens,
        absolute_max_output_tokens=config.absolute_max_output_tokens,
        maximum_request_cost_usd=config.maximum_request_cost_usd,
        input_cost_per_million_usd=config.input_cost_per_million_usd,
        output_cost_per_million_usd=config.output_cost_per_million_usd,
        setup_family_ids=(),
        output_profile=config.output_profile,
        target_output_tokens=config.target_output_tokens,
        token_safety_margin=config.token_safety_margin,
        model_context_limit=config.model_context_limit,
    )


def test_profiles_are_centralized_and_reserve_a_safety_margin() -> None:
    manager = TokenBudgetManager(model="llama-3.1-8b-instant")
    compact = manager.limits(OutputProfile.COMPACT)
    standard = manager.limits(OutputProfile.STANDARD)
    expanded = manager.limits(OutputProfile.EXPANDED)
    retry = manager.limits(OutputProfile.COMPACT_RETRY)

    assert retry.hard_limit < compact.hard_limit < standard.hard_limit < expanded.hard_limit
    plan = manager.plan(
        system_prompt="Return JSON.",
        context={"price": 2400.1},
        schema={"shape": {"regime": "enum"}},
    )
    assert plan.safety_margin_tokens == 256
    assert plan.estimator.startswith("conservative_")
    assert not TokenBudgetManager.estimate("test").exact


@pytest.mark.asyncio
async def test_compact_contract_reduces_schema_and_input_cost() -> None:
    _, _, config, request = await _request()
    selected = provider_context_payload(
        build_llm_analysis_context(request),
        OutputProfile.COMPACT,
    )
    manager = TokenBudgetManager(
        model="llama-3.1-8b-instant",
        maximum_input_tokens=config.input_token_budget,
    )
    prompt = Path(
        "backend/app/ai_reasoning/prompts/deep_market_analysis_v2.txt"
    ).read_text(encoding="utf-8")
    compact_schema = reasoning_response_schema(OutputProfile.COMPACT)
    standard_schema = reasoning_response_schema(OutputProfile.STANDARD)
    compact_plan = manager.plan(
        system_prompt=prompt,
        context=selected[0],
        schema=compact_schema,
        included_sections=selected[1],
        omitted_sections=selected[2],
    )
    standard_plan = manager.plan(
        system_prompt=prompt,
        context=build_llm_analysis_context(request).model_dump(mode="json"),
        schema=standard_schema,
        profile=OutputProfile.STANDARD,
    )

    assert compact_plan.schema_token_cost < standard_plan.schema_token_cost
    assert compact_plan.context_token_cost < standard_plan.context_token_cost
    assert compact_plan.estimated_input_tokens < standard_plan.estimated_input_tokens
    assert set(selected[1]) >= {
        "current_price",
        "timeframe_trends",
        "quant",
        "risk",
        "evidence_catalog",
    }
    assert "previous_final_decision" in selected[2]


@pytest.mark.asyncio
async def test_compact_schema_preserves_critical_analysis_and_resolves_evidence() -> None:
    _, _, _, request = await _request()
    context = build_llm_analysis_context(request)
    wire = CompactAIAnalysisOutput.model_validate(compact_output(request))
    validate_evidence_references(wire, context.evidence_catalog)
    resolved = resolve_compact_output(wire, context.evidence_catalog)

    assert resolved.market_regime.classification.value == "bullish"
    assert resolved.market_regime.evidence
    assert resolved.contradictions == ()
    assert resolved.key_risks
    assert resolved.invalidation_conditions
    assert resolved.analysis_confidence == 0.72
    assert TokenBudgetManager.estimate(
        json.dumps(wire.model_dump(mode="json"), separators=(",", ":"))
    ).tokens < 900


@pytest.mark.asyncio
async def test_unknown_evidence_reference_fails_closed() -> None:
    _, _, _, request = await _request()
    context = build_llm_analysis_context(request)
    raw = compact_output(request)
    raw["market_regime"]["evidence_refs"] = ["E99"]
    wire = CompactAIAnalysisOutput.model_validate(raw)

    with pytest.raises(CompactOutputValidationError) as captured:
        validate_evidence_references(wire, context.evidence_catalog)

    assert captured.value.code == "unknown_evidence_reference"


@pytest.mark.asyncio
async def test_zone_references_resolve_only_from_deterministic_catalog() -> None:
    _, _, _, request = await _request()
    request = request_with_zones(request)
    context = build_llm_analysis_context(request)
    raw = compact_output(request)
    wire = CompactAIAnalysisOutput.model_validate(raw)

    validate_zone_references(
        wire,
        context.supply_zone_catalog,
        context.demand_zone_catalog,
    )
    resolved = resolve_compact_output(
        wire,
        context.evidence_catalog,
        context.supply_zone_catalog,
        context.demand_zone_catalog,
    )

    assert raw["supply_demand_analysis"]["nearest_supply_ref"] == "SZ1"
    assert raw["supply_demand_analysis"]["nearest_demand_ref"] == "DZ1"
    assert resolved.supply_demand_analysis.nearest_supply == 3345.0
    assert resolved.supply_demand_analysis.nearest_demand == 3295.0


@pytest.mark.asyncio
async def test_unknown_and_non_reference_zone_values_fail_closed() -> None:
    _, _, _, request = await _request()
    request = request_with_zones(request)
    context = build_llm_analysis_context(request)

    raw = compact_output(request)
    raw["supply_demand_analysis"]["nearest_supply_ref"] = "SZ3"
    wire = CompactAIAnalysisOutput.model_validate(raw)
    with pytest.raises(CompactOutputValidationError) as unknown:
        validate_zone_references(
            wire,
            context.supply_zone_catalog,
            context.demand_zone_catalog,
        )
    assert unknown.value.code == "unknown_supply_zone_ref"

    for invalid in (3345.0, {"price": 3345.0}, 0, ""):
        raw = compact_output(request)
        raw["supply_demand_analysis"]["nearest_supply_ref"] = invalid
        with pytest.raises(ValidationError):
            CompactAIAnalysisOutput.model_validate(raw)


@pytest.mark.asyncio
async def test_empty_zone_catalog_requires_null_and_only_safe_syntax_normalizes() -> None:
    _, _, _, request = await _request()
    context = build_llm_analysis_context(request)
    raw = compact_output(request)
    raw["supply_demand_analysis"]["nearest_supply_ref"] = "SZ1"
    wire = CompactAIAnalysisOutput.model_validate(raw)
    with pytest.raises(CompactOutputValidationError) as empty:
        validate_zone_references(
            wire,
            context.supply_zone_catalog,
            context.demand_zone_catalog,
        )
    assert empty.value.code == "reference_must_be_null_when_catalog_empty"

@pytest.mark.asyncio
async def test_higher_timeframe_string_list_is_joined_locally_without_correction() -> None:
    """Reproduce production wrong_type at the exact provider field path."""

    _, _, config, request = await _request()
    raw = compact_output(request)
    raw["higher_timeframe_context"]["summary"] = [
        "M5 structure remains constructive.",
        "M15 context retains a bullish bias.",
    ]
    client = CompactClient(raw)
    selected_provider = provider(client, config)

    response = await selected_provider.reason(
        request,
        prompt_version=request.prompt_version,
    )

    assert len(client.calls) == 1
    assert selected_provider.correction_attempts == 0
    assert response.raw_output["higher_timeframe_context"]["description"] == (
        "M5 structure remains constructive. "
        "M15 context retains a bullish bias."
    )
    assert response.operational_metadata["local_shape_normalizations"] == (
        "higher_timeframe_context.summary",
    )
    attempt = selected_provider.attempts_for(request.request_id)[0]
    assert attempt["schema_valid"] is True
    assert attempt["schema_correction_triggered"] is False


@pytest.mark.asyncio
async def test_normalized_http_200_reaches_validation_persistence_and_commit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    state, quant, config, request = await _request()
    raw = compact_output(request)
    raw["higher_timeframe_context"]["summary"] = [
        "M5 is constructive.",
        "M15 remains aligned.",
    ]
    selected_provider = provider(CompactClient(raw), config)
    repository = InMemoryAIReasoningRepository()

    result = await build_service(  # type: ignore[arg-type]
        repository,
        selected_provider,
    ).process(state, quant)

    assert result is not None
    assert result.analysis.validation_passed is True
    assert await repository.analysis_for_state(state.state_id) == result.analysis
    assert selected_provider.http_calls == 1
    assert selected_provider.correction_attempts == 0
    messages = {record.getMessage() for record in caplog.records}
    assert "structured_validation.completed" in messages
    assert "ai_reasoning.persist.completed" in messages
    assert "ai_reasoning.analysis.committed" in messages


@pytest.mark.parametrize(
    "invalid_summary",
    [None, [], ["valid prose", 7], [""], {"m15": "bullish"}],
)
@pytest.mark.asyncio
async def test_semantically_invalid_higher_timeframe_shapes_remain_strict(
    invalid_summary: object,
) -> None:
    _, _, _, request = await _request()
    raw = compact_output(request)
    raw["higher_timeframe_context"]["summary"] = invalid_summary

    normalized, changes = normalize_compact_output_shapes(raw)

    assert normalized == raw
    assert changes == ()
    with pytest.raises(ValidationError):
        CompactAIAnalysisOutput.model_validate(normalized)


@pytest.mark.asyncio
async def test_market_structure_string_list_is_normalized_without_provider_correction() -> None:
    state, quant, config, request = await _request()
    raw = compact_output(request)
    raw["market_structure"]["short_term"] = [
        "M5 higher low holds.",
        "Momentum is constructive.",
    ]
    selected_provider = provider(CompactClient(raw), config)
    repository = InMemoryAIReasoningRepository()

    result = await build_service(  # type: ignore[arg-type]
        repository,
        selected_provider,
    ).process(state, quant)

    assert result is not None
    assert result.analysis.validation_passed is True
    assert selected_provider.http_calls == 1
    assert selected_provider.correction_attempts == 0
    attempt = selected_provider.request_attempts[0]
    assert "market_structure.short_term" in attempt["local_shape_normalizations"]


@pytest.mark.asyncio
async def test_executive_summary_list_is_normalized_without_provider_correction() -> None:
    state, quant, config, request = await _request()
    raw = compact_output(request)
    raw["executive_summary"] = [
        "M5 evidence is constructive.",
        "M15 context remains aligned.",
    ]
    selected_provider = provider(CompactClient(raw), config)
    repository = InMemoryAIReasoningRepository()

    result = await build_service(  # type: ignore[arg-type]
        repository,
        selected_provider,
    ).process(state, quant)

    assert result is not None
    assert selected_provider.http_calls == 1
    assert selected_provider.correction_attempts == 0
    assert "executive_summary" in selected_provider.request_attempts[0][
        "local_shape_normalizations"
    ]
    artifact = next(iter(repository.response_artifacts.values()), None)
    assert artifact is not None
    assert artifact.status.value == "COMMITTED"
    assert artifact.provider_output["executive_summary"] == raw["executive_summary"]
    assert artifact.normalized_output is not None
    assert artifact.normalized_output["executive_summary"] == (
        "M5 evidence is constructive. M15 context remains aligned."
    )
    assert artifact.analysis_id == result.analysis.analysis_id


def test_schema_wide_shape_normalizer_repairs_safe_string_variants() -> None:
    raw: dict[str, Any] = {
        "market_structure": {
            "short_term": ["M5 higher low holds.", "Momentum is constructive."],
            "medium_term": "M15 remains balanced.",
            "recent_change": "A local break occurred.",
            "evidence_refs": "E1",
        }
    }

    normalized, changes = normalize_compact_output_shapes(raw)

    assert normalized["market_structure"]["short_term"] == (
        "M5 higher low holds. Momentum is constructive."
    )
    assert normalized["market_structure"]["evidence_refs"] == ["E1"]
    assert {change["path"] for change in changes} == {
        "market_structure.short_term",
        "market_structure.evidence_refs",
    }
    assert all("received_value_hash" in change for change in changes)


@pytest.mark.parametrize(
    "invalid_value",
    (None, [], ["valid", 7], {"trend": "bullish"}),
)
def test_schema_wide_shape_normalizer_does_not_repair_unsafe_values(
    invalid_value: object,
) -> None:
    raw = {"market_structure": {"short_term": invalid_value}}

    normalized, changes = normalize_compact_output_shapes(raw)

    assert normalized == raw
    assert changes == ()


@pytest.mark.asyncio
async def test_schema_driven_normalizer_preserves_nullable_and_rejects_semantics() -> None:
    _, _, _, request = await _request()
    raw = compact_output(request)
    raw["supply_demand_analysis"]["nearest_supply_ref"] = None
    raw["supply_demand_analysis"]["nearest_demand_ref"] = None
    normalized, changes = normalize_compact_output_shapes(raw)
    assert normalized["supply_demand_analysis"]["nearest_supply_ref"] is None
    assert normalized["supply_demand_analysis"]["nearest_demand_ref"] is None
    assert changes == ()

    raw["market_regime"]["classification"] = "strongly_bullish"
    normalized, changes = normalize_compact_output_shapes(raw)
    assert normalized["market_regime"]["classification"] == "strongly_bullish"
    assert changes == ()
    with pytest.raises(ValidationError):
        CompactAIAnalysisOutput.model_validate(normalized)


def test_provider_contract_preflight_uses_actual_model_capability() -> None:
    strict_mode, strict_schema = validate_provider_contract(
        "openai/gpt-oss-120b",
        OutputProfile.COMPACT,
    )
    object_mode, object_schema = validate_provider_contract(
        "llama-3.1-8b-instant",
        OutputProfile.COMPACT,
    )

    assert strict_mode == "strict_schema"
    assert object_mode == "json_object"
    assert strict_schema == object_schema == reasoning_response_schema()
    encoded = json.dumps(strict_schema)
    assert "x-ten-normalize" not in encoded
    assert set(strict_schema["required"]) == set(strict_schema["properties"])


@pytest.mark.asyncio
async def test_higher_timeframe_summary_contract_is_explicit_and_matches_schema() -> None:
    _, _, config, request = await _request()
    selected_provider = provider(
        CompactClient(compact_output(request)),
        config,
    )

    contract = selected_provider._response_contract(  # noqa: SLF001
        OutputProfile.COMPACT,
        build_llm_analysis_context(request),
    )
    schema = CompactAIAnalysisOutput.model_json_schema()

    summary_schema = schema["$defs"]["CompactHigherTimeframe"]["properties"][
        "summary"
    ]
    assert summary_schema["type"] == "string"
    assert summary_schema["minLength"] == 1
    assert summary_schema["maxLength"] == HIGHER_TIMEFRAME_SUMMARY_LIMIT
    contract_summary = contract["json_schema"]["$defs"][
        "CompactHigherTimeframe"
    ]["properties"]["summary"]
    assert contract_summary == reasoning_response_schema()["$defs"][
        "CompactHigherTimeframe"
    ]["properties"]["summary"]


@pytest.mark.asyncio
async def test_max_items_and_non_allowlisted_reference_lengths_are_strict() -> None:
    _, _, _, request = await _request()
    raw = compact_output(request)
    raw["bullish_evidence_refs"] = ["E1", "E2", "E3", "E4"]
    with pytest.raises(ValidationError):
        CompactAIAnalysisOutput.model_validate(raw)

    raw = compact_output(request)
    raw["market_regime"]["evidence_refs"] = ["not-an-evidence-id"]
    wire = CompactAIAnalysisOutput.model_validate(raw)
    with pytest.raises(CompactOutputValidationError):
        validate_evidence_references(
            wire,
            build_llm_analysis_context(request).evidence_catalog,
        )


@pytest.mark.asyncio
async def test_market_regime_evidence_limit_matches_prompt_contract() -> None:
    _, _, config, request = await _request()
    selected_provider = provider(
        CompactClient(compact_output(request)),
        config,
    )

    contract = selected_provider._response_contract(  # noqa: SLF001
        OutputProfile.COMPACT,
        build_llm_analysis_context(request),
    )

    assert MARKET_REGIME_EVIDENCE_REF_LIMIT == 2
    regime_refs = contract["json_schema"]["$defs"]["CompactRegime"][
        "properties"
    ]["evidence_refs"]
    assert regime_refs["type"] == "array"
    raw = compact_output(request)
    valid_refs = [
        item.evidence_id
        for item in build_llm_analysis_context(request).evidence_catalog[:2]
    ]
    raw["market_regime"]["evidence_refs"] = valid_refs
    unchanged, changes = truncate_market_regime_evidence_refs(
        raw,
        frozenset(valid_refs),
    )
    assert unchanged["market_regime"]["evidence_refs"] == valid_refs
    assert changes == ()


@pytest.mark.asyncio
async def test_too_many_valid_regime_refs_are_truncated_without_correction() -> None:
    _, _, config, request = await _request()
    context = build_llm_analysis_context(request)
    refs = [item.evidence_id for item in context.evidence_catalog[:3]]
    assert len(refs) == 3
    raw = compact_output(request)
    raw["market_regime"]["evidence_refs"] = refs
    client = CompactClient(raw)
    selected_provider = provider(client, config)

    response = await selected_provider.reason(
        request,
        prompt_version=request.prompt_version,
    )

    assert len(client.calls) == 1
    assert selected_provider.correction_attempts == 0
    assert selected_provider.attempts_for(request.request_id)[0][
        "schema_correction_triggered"
    ] is False
    assert [
        item["claim"]
        for item in response.raw_output["market_regime"]["evidence"]
    ] == [
        context.evidence_catalog[0].fact,
        context.evidence_catalog[1].fact,
    ]
    assert response.operational_metadata[
        "local_evidence_ref_truncations"
    ] == ("market_regime.evidence_refs",)


@pytest.mark.asyncio
async def test_unknown_overflow_regime_refs_remain_invalid() -> None:
    _, _, _, request = await _request()
    raw = compact_output(request)
    raw["market_regime"]["evidence_refs"] = ["E1", "E2", "E99"]

    normalized, changes = truncate_market_regime_evidence_refs(
        raw,
        frozenset({"E1", "E2"}),
    )

    assert normalized["market_regime"]["evidence_refs"] == [
        "E1",
        "E2",
        "E99",
    ]
    assert changes == ()
    with pytest.raises(ValidationError):
        CompactAIAnalysisOutput.model_validate(normalized)

    wrongly_typed = compact_output(request)
    wrongly_typed["market_regime"]["evidence_refs"] = ["E1", "E2", 3]
    untouched, type_changes = truncate_market_regime_evidence_refs(
        wrongly_typed,
        frozenset({"E1", "E2", "E3"}),
    )
    assert untouched["market_regime"]["evidence_refs"] == ["E1", "E2", 3]
    assert type_changes == ()
    with pytest.raises(ValidationError):
        CompactAIAnalysisOutput.model_validate(untouched)


@pytest.mark.asyncio
async def test_valid_compact_response_is_one_call_and_persists_exact_usage_shape() -> None:
    _, _, config, request = await _request()
    client = CompactClient(compact_output(request))
    selected_provider = provider(client, config)
    response = await selected_provider.reason(
        request,
        prompt_version=request.prompt_version,
    )

    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 1400
    assert response.token_usage == {
        "input_tokens": 2100,
        "output_tokens": 620,
        "total_tokens": 2720,
    }
    assert response.raw_output["market_regime"]["classification"] == "bullish"
    attempts = selected_provider.attempts_for(request.request_id)
    assert len(attempts) == 1
    assert attempts[0]["output_profile"] == "compact"


@pytest.mark.asyncio
async def test_reference_correction_lists_allowed_ids_and_never_maps_price_locally() -> None:
    _, _, config, request = await _request()
    request = request_with_zones(request)
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        bodies.append(body)
        output = compact_output(request)
        if len(bodies) == 1:
            output["supply_demand_analysis"]["nearest_supply_ref"] = 9999.0
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps(output)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 450,
                    "total_tokens": 950,
                },
            },
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq_1",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    response = await provider(client, config).reason(  # type: ignore[arg-type]
        request,
        prompt_version=request.prompt_version,
    )
    initial = json.loads(bodies[0]["messages"][1]["content"])
    correction = json.loads(bodies[1]["messages"][1]["content"])

    assert initial["response_contract"]["reference_catalog"][
        "nearest_supply_ref"
    ] == ["SZ1"]
    assert correction["allowed_reference_values"]["nearest_supply_ref"] == [
        "SZ1"
    ]
    assert correction["previous_response"]["supply_demand_analysis"][
        "nearest_supply_ref"
    ] == 9999.0
    assert response.raw_output["supply_demand_analysis"]["nearest_supply"] == 3345.0
    assert len(bodies) == 2


@pytest.mark.asyncio
async def test_truncation_uses_one_smaller_fresh_retry_without_previous_output() -> None:
    _, _, config, request = await _request()
    bodies: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"market_regime":'},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2200,
                        "completion_tokens": 1400,
                        "total_tokens": 3600,
                    },
                },
                request=http_request,
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                compact_output(request, retry=True)
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1800,
                    "completion_tokens": 580,
                    "total_tokens": 2380,
                },
            },
            request=http_request,
        )

    client = HttpAIProviderClient(
        "groq_1",
        "safe-test-key",
        "https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    selected_provider = provider(client, config)  # type: ignore[arg-type]
    response = await selected_provider.reason(
        request,
        prompt_version=request.prompt_version,
    )

    assert len(bodies) == 2
    first_payload = json.loads(bodies[0]["messages"][1]["content"])
    retry_payload = json.loads(bodies[1]["messages"][1]["content"])
    assert "previous_response" not in retry_payload
    assert retry_payload["response_contract"] == first_payload["response_contract"]
    assert bodies[1]["max_tokens"] < bodies[0]["max_tokens"]
    assert response.fallback_reason == "output_truncated_compact_retry"
    attempts = selected_provider.attempts_for(request.request_id)
    assert [item["request_kind"] for item in attempts] == [
        "analysis",
        "compact_retry",
    ]
    assert response.token_usage == {
        "input_tokens": 4000,
        "output_tokens": 1980,
        "total_tokens": 5980,
    }


def test_compact_retry_uses_the_same_canonical_response_schema() -> None:
    compact = json.dumps(
        reasoning_response_schema(OutputProfile.COMPACT),
        separators=(",", ":"),
    )
    retry = json.dumps(
        reasoning_response_schema(OutputProfile.COMPACT_RETRY),
        separators=(",", ":"),
    )
    assert retry == compact
    assert "alternative_scenarios" in retry
    assert "executive_summary" in retry


def test_repeated_budget_failures_degrade_policy_health_not_provider_health() -> None:
    service = build_service(
        InMemoryAIReasoningRepository(),
        ValidProvider(),
        now=NOW,
    )
    service.last_eligible_cycle_at = NOW
    service.last_cycle_outcome = "provider_failure"
    service.metrics["eligible_five_minute_cycles"] = 3
    service.metrics["analysis_requests"] = 3
    service.metrics["truncated_outputs"] = 3
    service.metrics["request_policy_failures"] = 3

    health = service.health()

    assert health["operations_status"] == "unhealthy"
    assert health["provider_available"] is True
    assert health["call_control"]["truncation_rate"] == 1.0  # type: ignore[index]


def test_excess_provider_efficiency_cost_degrades_policy_health() -> None:
    service = build_service(
        InMemoryAIReasoningRepository(),
        ValidProvider(),
        now=NOW,
    )
    service.last_eligible_cycle_at = NOW
    service.last_cycle_outcome = "pool_success"
    service.metrics["eligible_five_minute_cycles"] = 2
    service.metrics["analyses_successfully_completed"] = 2
    service.metrics["analysis_requests"] = 2
    service.metrics["provider_http_calls"] = 4
    service.metrics["schema_correction_requests"] = 1
    service.metrics["provider_total_tokens"] = 13_000

    health = service.health()
    controls = health["call_control"]

    assert health["operations_status"] == "degraded"
    assert controls["provider_calls_per_completed_analysis"] == 2.0  # type: ignore[index]
    assert controls["schema_correction_rate"] == 0.5  # type: ignore[index]
    assert controls["tokens_per_completed_analysis"] == 6500.0  # type: ignore[index]
