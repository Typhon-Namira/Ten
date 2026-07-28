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
    normalize_descriptive_overflow,
    resolve_compact_output,
    validate_evidence_references,
)
from backend.app.ai_reasoning.llm_context import (
    build_llm_analysis_context,
    provider_context_payload,
)
from backend.app.ai_reasoning.provider import GroqProvider, reasoning_response_schema
from backend.app.ai_reasoning.token_budget import OutputProfile, TokenBudgetManager
from tests.ai_reasoning.test_llm_payload_boundary import _request
from tests.ai_reasoning.test_ai_reasoning_lifecycle import (
    NOW,
    InMemoryAIReasoningRepository,
    ValidProvider,
    build_service,
)


def compact_output(request: Any, *, retry: bool = False) -> dict[str, Any]:
    catalog = build_llm_analysis_context(request).evidence_catalog
    refs = [catalog[0].evidence_id] if catalog else []
    output: dict[str, Any] = {
        "analysis_schema_version": (
            "compact-retry-1.0" if retry else "compact-1.0"
        ),
        "output_profile": "compact_retry" if retry else "compact",
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
            "nearest_supply": None,
            "nearest_demand": None,
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
    if not retry:
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
async def test_descriptive_overflow_only_truncates_allowlisted_prose() -> None:
    _, _, _, request = await _request()
    raw = compact_output(request)
    raw["executive_summary"] = "x" * 400
    raw["market_regime"]["classification"] = "bullish"
    raw["market_regime"]["strength"] = 72.12345
    raw["market_regime"]["evidence_refs"] = ["E1"]

    normalized, changes = normalize_descriptive_overflow(raw)

    assert len(normalized["executive_summary"]) == 320
    assert normalized["market_regime"]["classification"] == "bullish"
    assert normalized["market_regime"]["strength"] == 72.12345
    assert normalized["market_regime"]["evidence_refs"] == ["E1"]
    assert changes == ("executive_summary",)


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
    assert (
        len(json.dumps(retry_payload["response_contract"]))
        < len(json.dumps(first_payload["response_contract"]))
    )
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


def test_compact_retry_schema_is_strictly_smaller() -> None:
    compact = json.dumps(
        reasoning_response_schema(OutputProfile.COMPACT),
        separators=(",", ":"),
    )
    retry = json.dumps(
        reasoning_response_schema(OutputProfile.COMPACT_RETRY),
        separators=(",", ":"),
    )
    assert len(retry) < len(compact)
    assert "alternative_scenarios" not in retry
    assert "executive_summary" not in retry


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
