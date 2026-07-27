import asyncio
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
from pydantic import BaseModel

from backend.app.ai_reasoning.config import AIReasoningConfig
from backend.app.ai_reasoning.lifecycle import SignalLifecycleService
from backend.app.ai_reasoning.memory import MarketMemory
from backend.app.ai_reasoning.models import (
    AIMarketForecast,
    AIResultStatus,
    AISignalProposal,
    AlternativeScenario,
    Direction,
    EntryZone,
    ManagedSignalState,
    MarketMemoryEntry,
    MarketMemorySummary,
    ProposalAction,
    SetupReadiness,
)
from backend.app.ai_reasoning.provider import AIProviderResponse, CerebrasProvider, GroqProvider
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.ai_reasoning.service import (
    AIReasoningService,
    reasoning_cycle_idempotency_key,
)
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from backend.app.ai_reasoning.validation import StructuredAIOutputValidator, structural_opportunity_key
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.config import YamlConfigRepository
from backend.app.core.exceptions import AIProviderFailureDetails, AIProviderRequestError
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.integration import CanonicalEventEnvelope
from backend.app.market_state import InMemoryUnifiedMarketStateRepository, UnifiedMarketStateService
from backend.app.quant_forecasting.config import QuantForecastingConfig
from backend.app.quant_forecasting.features import PointInTimeFeatureExtractor
from backend.app.quant_forecasting.provider import DeterministicBaselineProvider
from backend.app.quant_forecasting.repository import InMemoryQuantForecastRepository
from backend.app.quant_forecasting.service import QuantForecastService

BOUNDARY = datetime(2026, 7, 23, 12, 30, tzinfo=UTC)
NOW = BOUNDARY + timedelta(seconds=5)


class EngineOutput(BaseModel):
    snapshot_id: str
    analysis_timestamp: datetime = BOUNDARY
    created_at: datetime = NOW
    engine_version: str = "1.0.0"
    status: str = "ready"
    confidence_score: float = 85
    quality_score: float = 90
    regime: str = "trending"
    structure: dict[str, object] = {"direction": "bullish", "bos": True}


def candle(timeframe: Timeframe) -> Candle:
    return Candle(
        timestamp=BOUNDARY - timeframe.duration,
        ingestion_timestamp=NOW,
        symbol="XAUUSD",
        timeframe=timeframe,
        open=3300,
        high=3305,
        low=3298,
        close=3302,
        volume=100,
        spread=0.2,
        provider="existing-provider",
    )


async def state_and_quant():
    state_service = UnifiedMarketStateService(InMemoryUnifiedMarketStateRepository(), clock=lambda: NOW)
    outputs = {
        name: EngineOutput(snapshot_id=f"{name}-snapshot")
        for name in ("smc", "liquidity", "volume_profile", "institutional_flow", "market_regime", "economic_calendar")
    }
    state = None
    for timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15):
        envelope = CanonicalEventEnvelope.final_candle(candle(timeframe), uuid4(), NOW)
        state = await state_service.capture_cycle(envelope, outputs)
    assert state is not None
    quant_config = YamlConfigRepository().load_model("quant_forecasting", QuantForecastingConfig)
    quant = await QuantForecastService(
        InMemoryQuantForecastRepository(),
        DeterministicBaselineProvider(quant_config, clock=lambda: NOW),
        PointInTimeFeatureExtractor(quant_config.feature_schema_version, clock=lambda: NOW),
        quant_config,
        enabled=True,
        clock=lambda: NOW,
    ).forecast(state)
    assert quant is not None
    return state, quant


class ValidProvider:
    def __init__(self) -> None:
        self.calls = 0

    def metadata(self):
        return {"provider": "cerebras", "model_identifier": "configured-model", "external_ai_apis": ("cerebras",)}

    async def reason(self, request, *, prompt_version):
        self.calls += 1
        market_id = UUID(request.trend_evidence[0]["evidence_id"])
        regime_id = UUID(request.market_regime[0]["evidence_id"])
        forecast_id = uuid5(NAMESPACE_URL, f"forecast:{request.request_id}")
        forecast = AIMarketForecast(
            forecast_id=forecast_id,
            request_id=request.request_id,
            market_state_id=request.market_state_id,
            quantitative_forecast_id=request.quantitative_forecast_id,
            cycle_id=request.cycle_id,
            status=AIResultStatus.AVAILABLE,
            dominant_direction=Direction.BUY,
            buy_probability=0.62,
            sell_probability=0.18,
            neutral_probability=0.20,
            expected_horizon="10_m1",
            expected_minimum_move=0.001,
            expected_base_move=0.002,
            expected_maximum_move=0.003,
            expected_volatility=0.002,
            dominant_scenario="bullish continuation",
            dominant_scenario_probability=0.62,
            alternative_scenarios=(AlternativeScenario(name="range", probability=0.20, direction=Direction.NEUTRAL),),
            selected_setup_family="trend_continuation",
            setup_family_candidates=("trend_continuation", "pullback_continuation"),
            supporting_evidence_ids=(market_id, regime_id),
            contradicting_evidence_ids=(),
            evidence_completeness=0.90,
            evidence_agreement=0.75,
            forecast_confidence=0.68,
            execution_confidence=0.60,
            risk_quality=0.70,
            setup_readiness=SetupReadiness.READY,
            uncertainty=0.32,
            reasoning_summary="Trend structure remains constructive.",
            monitoring_conditions=("protected structure", "regime transition"),
            model_provider="cerebras",
            model_identifier=request.model_identifier,
            prompt_version=prompt_version,
            reasoning_policy_version=request.reasoning_policy_version,
            setup_family_registry_version=request.setup_family_registry_version,
            quantitative_model_version=request.quantitative_model_version,
            feature_schema_version=request.feature_schema_version,
            market_state_schema_version=request.market_state_schema_version,
            validation_passed=True,
            retry_count=0,
            generated_at=NOW,
        )
        key = structural_opportunity_key(
            "XAUUSD",
            "trend_continuation",
            "BUY",
            ("market_data:M1", "market_regime:M1"),
            "bullish continuation",
        )
        proposal = AISignalProposal(
            proposal_id=uuid5(NAMESPACE_URL, f"proposal:{request.request_id}"),
            forecast_id=forecast_id,
            market_state_id=request.market_state_id,
            structural_opportunity_key=key,
            recommended_action=ProposalAction.BUY,
            direction=Direction.BUY,
            entry_type="limit",
            entry_zone=EntryZone(low=3300, high=3301),
            stop_loss=3295,
            take_profit_levels=(3311, 3320),
            expected_risk_to_reward=2,
            invalidation_price=3295,
            invalidation_conditions=("protected low breaks",),
            expires_at=BOUNDARY + timedelta(minutes=15),
            setup_readiness=SetupReadiness.READY,
            proposal_confidence=0.64,
            supporting_evidence_ids=(market_id, regime_id),
            monitoring_conditions=("entry remains structurally valid",),
            model_identifier=request.model_identifier,
            policy_version=request.reasoning_policy_version,
            created_at=NOW,
        )
        return AIProviderResponse(
            raw_output={"forecast": forecast.model_dump(mode="json"), "proposal": proposal.model_dump(mode="json")},
            provider="cerebras",
            model_identifier=request.model_identifier,
            latency_ms=5,
            token_usage=None,
        )


class UnavailableProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        raise RuntimeError("configured LLM unavailable")


class TypedUnavailableProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="cerebras",
                reason_code="authentication_failed",
                phase="http_request",
                endpoint="https://api.cerebras.ai/v1/chat/completions",
                model=request.model_identifier,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                http_status=401,
                error_code="401",
                error_message="User not found.",
                content_type="application/json",
                body_length=42,
                elapsed_ms=12.5,
                exception_class="HTTPStatusError",
            )
        )


class TemporaryUnavailableProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        if self.calls == 1:
            raise AIProviderRequestError(
                AIProviderFailureDetails(
                    provider="cerebras",
                    reason_code="provider_unavailable",
                    phase="http_request",
                    endpoint="https://api.cerebras.ai/v1/chat/completions",
                    model=request.model_identifier,
                    request_id=str(request.request_id),
                    cycle_id=str(request.cycle_id),
                    http_status=503,
                    exception_class="HTTPStatusError",
                )
            )
        # ValidProvider increments its counter, so reproduce its one successful response while
        # preserving an exact physical-call count of two.
        self.calls -= 1
        return await super().reason(request, prompt_version=prompt_version)


class PermanentHttpFailureProvider(ValidProvider):
    def __init__(self, status: int, reason_code: str) -> None:
        super().__init__()
        self.status = status
        self.reason_code = reason_code

    async def reason(self, request, *, prompt_version):
        self.calls += 1
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="cerebras",
                reason_code=self.reason_code,
                phase="http_request",
                endpoint="https://api.cerebras.ai/v1/chat/completions",
                model=request.model_identifier,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                http_status=self.status,
                exception_class="HTTPStatusError",
            )
        )


class TimeoutProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        raise TimeoutError("simulated asyncio.wait_for timeout")


class InvalidProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        return AIProviderResponse(
            raw_output={"forecast": {"buy_probability": 2}, "proposal": {"recommended_action": "BUY"}},
            provider="cerebras",
            model_identifier=request.model_identifier,
            latency_ms=2,
            token_usage=None,
        )


class NoProposalProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        response = await super().reason(request, prompt_version=prompt_version)
        return AIProviderResponse(
            raw_output={
                "forecast": response.raw_output["forecast"],
                "proposal": None,
            },
            provider=response.provider,
            model_identifier=response.model_identifier,
            latency_ms=response.latency_ms,
            token_usage=None,
        )


class UnknownEvidenceProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        response = await super().reason(request, prompt_version=prompt_version)
        forecast = dict(response.raw_output["forecast"])
        forecast["supporting_evidence_ids"] = [str(uuid4())]
        return AIProviderResponse(
            raw_output={"forecast": forecast, "proposal": response.raw_output["proposal"]},
            provider=response.provider,
            model_identifier=response.model_identifier,
            latency_ms=response.latency_ms,
            token_usage=None,
        )


class SimplifiedWaitProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        return AIProviderResponse(
            raw_output={
                "forecast": "WAIT",
                "proposal": {
                    "horizon": "M1",
                    "timestamp": request.analysis_timestamp.isoformat(),
                    "setup_family": "WAIT",
                    "execution_levels": None,
                },
            },
            provider="cerebras",
            model_identifier=request.model_identifier,
            latency_ms=3,
            token_usage=None,
        )


class OptionalProposalFieldInvalidProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        response = await super().reason(request, prompt_version=prompt_version)
        proposal = dict(response.raw_output["proposal"])
        proposal["risk_notes"] = [object()]
        return AIProviderResponse(
            raw_output={"forecast": response.raw_output["forecast"], "proposal": proposal},
            provider=response.provider,
            model_identifier=response.model_identifier,
            latency_ms=response.latency_ms,
            token_usage=None,
        )


class UnknownCompactSetupFamilyProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        return AIProviderResponse(
            raw_output={
                "decision": "LONG",
                "confidence": 0.78,
                "rationale": "Constructive trend continuation.",
                "risk_flags": [],
                "proposal": {
                    "setup_family": "invented_smart_money_setup",
                    "entry_low": 3300,
                    "entry_high": 3301,
                    "stop_loss": 3295,
                    "take_profit_levels": [3311, 3320],
                },
            },
            provider="cerebras",
            model_identifier=request.model_identifier,
            latency_ms=3,
            token_usage=None,
        )


class CanonicalCompactSetupFamilyProvider(UnknownCompactSetupFamilyProvider):
    async def reason(self, request, *, prompt_version):
        response = await super().reason(request, prompt_version=prompt_version)
        proposal = dict(response.raw_output["proposal"])
        proposal["setup_family"] = "trend_continuation"
        return AIProviderResponse(
            raw_output={**response.raw_output, "proposal": proposal},
            provider=response.provider,
            model_identifier=response.model_identifier,
            latency_ms=response.latency_ms,
            token_usage=response.token_usage,
        )


def build_service(repository, provider, *, shadow=False, proposals=True, monitoring=False, maximum_retries=0):
    config = YamlConfigRepository().load_model("ai_reasoning", AIReasoningConfig).model_copy(update={"maximum_retries": maximum_retries})
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())
    return AIReasoningService(
        repository,
        provider,
        AIReasoningRequestBuilder(config, model_identifier="configured-model", clock=lambda: NOW),
        StructuredAIOutputValidator(registry),
        registry,
        config,
        shadow_enabled=shadow,
        proposals_enabled=proposals,
        monitoring_enabled=monitoring,
        clock=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_execution_context_revalidates_explicit_xauusd_market_status() -> None:
    state, quant = await state_and_quant()
    service = build_service(InMemoryAIReasoningRepository(), ValidProvider())

    context = service._execution_context(
        state,
        quant,
        SimpleNamespace(spread=0.2, economic_event_context=()),
        uuid4(),
        "xauusd-open-market-test",
    )

    assert state.market_schedule is not None
    assert context.market_open is True
    assert context.market_status == "OPEN"
    assert context.session == "london_new_york_overlap"
    assert context.market_timezone == "America/New_York"
    assert context.market_status_source == "ums_market_schedule_revalidated_at_guardrail"


@pytest.mark.asyncio
async def test_disabled_flags_make_no_llm_call_and_persist_nothing() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    result = await build_service(repository, provider, proposals=False, monitoring=False).process(state, quant)
    assert result is None
    assert provider.calls == 0
    assert not repository.requests and not repository.forecasts and not repository.proposals


@pytest.mark.asyncio
async def test_monitoring_flag_alone_cannot_create_new_opportunity() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    result = await build_service(repository, provider, proposals=False, monitoring=True).process(state, quant)
    assert result is None
    assert provider.calls == 0
    assert not repository.proposals and not repository.signals


@pytest.mark.asyncio
async def test_configured_provider_boundaries_are_cerebras_and_groq() -> None:
    assert CerebrasProvider.provider_name == "cerebras"
    assert GroqProvider.provider_name == "groq"
    assert ValidProvider().metadata()["external_ai_apis"] == ("cerebras",)


@pytest.mark.asyncio
async def test_reasoning_request_bounds_large_engine_collections_without_changing_market_state() -> None:
    state, quant = await state_and_quant()
    source = next(item for item in state.evidence if item.source_engine == "smc")
    zones = [
        {
            "zone_id": index,
            "created_at": (BOUNDARY - timedelta(minutes=5_000 - index)).isoformat(),
            "low": 3_000 + index / 100,
            "high": 3_001 + index / 100,
            "metadata": {"touches": list(range(20))},
        }
        for index in range(5_000)
    ]
    oversized = source.model_copy(
        update={"raw_value": {"status": "ready", "summary": {"bias": "bullish"}, "zones": zones}}
    )
    evidence = tuple(oversized if item.evidence_id == source.evidence_id else item for item in state.evidence)
    large_state = state.model_copy(update={"evidence": evidence})
    config = YamlConfigRepository().load_model("ai_reasoning", AIReasoningConfig)

    request = AIReasoningRequestBuilder(
        config,
        model_identifier="configured-model",
        clock=lambda: NOW,
    ).build(
        large_state,
        quant,
        MarketMemorySummary(entry_count=0),
        existing_signal=None,
        previous_forecast=None,
        previous_proposal=None,
    )

    encoded = json.dumps(request.model_dump(mode="json"))
    smc_raw = request.smc_evidence[0]["raw"]
    assert len(encoded) < 500_000
    assert smc_raw["summary"]["bias"] == "bullish"
    assert smc_raw["zones"]["collection_summary"]["total_count"] == 5_000
    assert [item["zone_id"] for item in smc_raw["zones"]["items"]] == [4_999]
    assert source.raw_value["structure"]["direction"] == "bullish"


@pytest.mark.asyncio
async def test_valid_structured_output_creates_auditable_shadow_proposal() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(repository, ValidProvider()).process(state, quant)
    assert result is not None and result.proposal is not None
    assert result.forecast.buy_probability + result.forecast.sell_probability + result.forecast.neutral_probability == pytest.approx(1)
    assert result.forecast.shadow_only and result.forecast.awaiting_guardrail_validation
    assert result.proposal.approved_for_publication is False
    assert len(repository.signals) == 1
    assert len(repository.transitions) == 1
    assert next(iter(repository.signals.values())).state == ManagedSignalState.PROPOSED


@pytest.mark.asyncio
async def test_llm_unavailability_is_explicit_and_never_fabricates_a_proposal() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), UnavailableProvider()
    result = await build_service(repository, provider, maximum_retries=0).process(state, quant)
    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.UNAVAILABLE
    assert result.forecast.buy_probability is None
    assert provider.calls == 1
    assert len(repository.failures) == 1
    assert not repository.proposals and not repository.signals


@pytest.mark.asyncio
async def test_provider_timeout_is_reported_distinctly_not_as_generic_llm_unavailable() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), TimeoutProvider()
    service = build_service(repository, provider, maximum_retries=0)
    result = await service.process(state, quant)

    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.UNAVAILABLE
    assert result.forecast.failure_state == "ai_reasoning_request_timeout"
    failure = next(iter(repository.failures.values()))
    assert failure.provider_failure is not None
    assert failure.provider_failure["phase"] == "provider_request_timeout"
    assert service.health()["provider_available"] is False


@pytest.mark.asyncio
async def test_health_reports_provider_availability_from_transport_outcome() -> None:
    state, quant = await state_and_quant()

    unavailable = build_service(
        InMemoryAIReasoningRepository(),
        TypedUnavailableProvider(),
        maximum_retries=0,
    )
    await unavailable.process(state, quant)
    assert unavailable.health()["provider_available"] is False

    healthy = build_service(
        InMemoryAIReasoningRepository(),
        ValidProvider(),
        maximum_retries=0,
    )
    await healthy.process(state, quant)
    assert healthy.health()["provider_available"] is True


@pytest.mark.asyncio
async def test_shadow_reasoning_runs_without_proposal_or_monitoring_flags_and_persists_typed_failure() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), TypedUnavailableProvider()
    result = await build_service(
        repository,
        provider,
        shadow=True,
        proposals=False,
        monitoring=False,
        maximum_retries=0,
    ).process(state, quant)

    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.UNAVAILABLE
    assert result.forecast.failure_state == "authentication_failed"
    assert result.forecast.failure_phase == "http_request"
    assert result.forecast.provider_http_status == 401
    assert result.forecast.provider_error_code == "401"
    assert result.forecast.provider_error_message == "User not found."
    assert provider.calls == 1
    failure = next(iter(repository.failures.values()))
    assert failure.failure_state == "authentication_failed"
    assert failure.provider_failure is not None
    assert failure.provider_failure["http_status"] == 401
    assert not repository.proposals and not repository.signals


@pytest.mark.asyncio
async def test_invalid_output_is_stored_and_cannot_create_proposal() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(repository, InvalidProvider(), maximum_retries=0).process(state, quant)
    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.INVALID
    assert result.forecast.validation_passed is False
    assert repository.failures
    assert all(failure.raw_output is None for failure in repository.failures.values())
    assert not repository.proposals


@pytest.mark.asyncio
async def test_first_structured_validation_error_is_logged_with_field_and_fragment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    logging.getLogger("backend.app.ai_reasoning.service").disabled = False

    with caplog.at_level(logging.ERROR, logger="backend.app.ai_reasoning.service"):
        await build_service(repository, InvalidProvider(), maximum_retries=0).process(state, quant)

    failure = next(record for record in caplog.records if record.message == "ai_reasoning.request.failed")
    assert failure.field_path == "forecast.status"
    assert failure.expected_type
    assert failure.validator_name == "missing"
    assert failure.offending_json_fragment == "null"


@pytest.mark.asyncio
async def test_unknown_evidence_reference_cannot_create_proposal() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(repository, UnknownEvidenceProvider(), maximum_retries=0).process(state, quant)
    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.INVALID
    assert any("unknown_evidence_reference" in error for failure in repository.failures.values() for error in failure.validation_errors)


@pytest.mark.asyncio
async def test_production_simplified_wait_shape_is_recovered_as_degraded_non_actionable_output() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(repository, SimplifiedWaitProvider(), maximum_retries=0).process(state, quant)

    assert result is not None and result.proposal is not None
    assert result.forecast.status == AIResultStatus.NON_ACTIONABLE
    assert result.forecast.validation_passed is False
    assert result.forecast.fallback_state == "recovered_simplified_wait"
    assert result.proposal.recommended_action == ProposalAction.WAIT
    assert result.proposal.entry_zone is None
    assert result.proposal.take_profit_levels == ()
    assert result.degraded_validation is True
    assert repository.forecasts and repository.proposals
    assert not repository.failures


@pytest.mark.asyncio
async def test_invalid_optional_proposal_field_preserves_valid_forecast_as_degraded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    with caplog.at_level(logging.INFO, logger="backend.app.ai_reasoning.service"):
        result = await build_service(
            repository,
            OptionalProposalFieldInvalidProvider(),
            maximum_retries=0,
        ).process(state, quant)

    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.AVAILABLE
    assert result.forecast.validation_passed is False
    assert result.forecast.failure_state == "degraded_structured_output"
    assert result.validation_issues
    assert repository.forecasts
    assert not repository.failures and not repository.proposals
    validation_log = next(
        record
        for record in caplog.records
        if record.message == "structured_validation.completed"
    )
    assert validation_log.field_path == "proposal.risk_notes.0"
    assert validation_log.expected_type
    assert validation_log.actual_value
    assert validation_log.validator_name
    assert validation_log.offending_json_fragment
    assert validation_log.recoverable is True


@pytest.mark.asyncio
async def test_expected_compact_response_persists_as_fully_validated_and_healthy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()

    with caplog.at_level(logging.INFO, logger="backend.app.ai_reasoning.service"):
        result = await build_service(
            repository,
            CanonicalCompactSetupFamilyProvider(),
            maximum_retries=0,
        ).process(state, quant)

    assert result is not None
    assert result.forecast.status == AIResultStatus.AVAILABLE
    assert result.forecast.validation_passed is True
    assert result.forecast.failure_state is None
    assert result.proposal is not None
    assert result.degraded_validation is False
    assert repository.forecasts and repository.proposals
    assert not repository.failures
    validation_log = next(
        record
        for record in caplog.records
        if record.message == "structured_validation.completed"
    )
    assert validation_log.validation_status == "valid"
    assert validation_log.validation_issue_count == 0
    assert validation_log.repaired_fields == ()


@pytest.mark.asyncio
async def test_unknown_compact_setup_family_persists_reasoning_without_failure_record() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()

    result = await build_service(
        repository,
        UnknownCompactSetupFamilyProvider(),
        maximum_retries=0,
    ).process(state, quant)

    assert result is not None
    assert result.forecast.status == AIResultStatus.AVAILABLE
    assert result.forecast.failure_state == "degraded_structured_output"
    assert result.forecast.selected_setup_family is None
    assert result.proposal is None
    assert result.degraded_validation is True
    assert repository.forecasts
    assert not repository.failures
    assert not repository.proposals


@pytest.mark.asyncio
async def test_valid_forecast_without_proposal_is_persisted_independently() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(
        repository,
        NoProposalProvider(),
        maximum_retries=0,
    ).process(state, quant)

    assert result is not None
    assert result.proposal is None
    assert result.forecast.status == AIResultStatus.AVAILABLE
    assert result.forecast.validation_passed is True
    assert repository.forecasts
    assert not repository.proposals and not repository.signals
    assert not repository.failures


@pytest.mark.asyncio
async def test_setup_family_enforces_only_its_own_mandatory_evidence() -> None:
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())
    assert registry.validate_requirements("trend_continuation", {"market_data", "market_regime"}, 0.8, "BUY") == ()
    errors = registry.validate_requirements("trend_continuation", {"market_data"}, 0.8, "BUY")
    assert errors == ("missing_setup_evidence:market_regime",)
    assert all("fvg" not in item and "order_block" not in item for item in errors)


@pytest.mark.asyncio
async def test_duplicate_opportunity_updates_preserve_managed_signal_identity() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    first = await build_service(repository, ValidProvider()).process(state, quant)
    second = await build_service(repository, ValidProvider()).process(state, quant)
    assert first and second
    assert len(repository.signals) == 1
    assert next(iter(repository.signals.values())).structural_opportunity_key == first.proposal.structural_opportunity_key


def test_reasoning_idempotency_key_uses_exact_ums_cycle_boundary_and_contract() -> None:
    first = datetime(2026, 7, 24, 12, 1, tzinfo=UTC)
    next_cycle = datetime(2026, 7, 24, 12, 2, tzinfo=UTC)

    assert reasoning_cycle_idempotency_key("xau/usd", first, "1.0", "contract-1") == (
        reasoning_cycle_idempotency_key("XAUUSD", first, "1.0", "contract-1")
    )
    assert reasoning_cycle_idempotency_key("XAU/USD", first, "1.0", "contract-1") != (
        reasoning_cycle_idempotency_key("XAU/USD", next_cycle, "1.0", "contract-1")
    )
    assert reasoning_cycle_idempotency_key("XAU/USD", first, "1.0", "contract-1") != (
        reasoning_cycle_idempotency_key("XAU/USD", first, "2.0", "contract-1")
    )
    assert reasoning_cycle_idempotency_key("XAU/USD", first, "1.0", "contract-1") != (
        reasoning_cycle_idempotency_key("XAU/USD", first, "1.0", "contract-2")
    )


@pytest.mark.asyncio
async def test_repeated_pipeline_invocations_for_same_cycle_make_one_provider_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    first_provider = ValidProvider()
    second_provider = ValidProvider()

    with caplog.at_level(logging.INFO, logger="backend.app.ai_reasoning.service"):
        first = await build_service(repository, first_provider).process(state, quant)
        second = await build_service(repository, second_provider).process(state, quant)

    assert first is not None and second is not None
    assert first.forecast.forecast_id == second.forecast.forecast_id
    assert first_provider.calls == 1
    assert second_provider.calls == 0
    assert len(repository.reasoning_cycles) == 1
    assert len(repository.forecasts) == 1
    provider_calls = [
        record
        for record in caplog.records
        if record.message == "ai_reasoning.provider_call.started"
    ]
    assert len(provider_calls) == 1
    assert provider_calls[0].trigger == "integration_worker"
    assert provider_calls[0].instrument == "XAUUSD"
    assert provider_calls[0].idempotency_key
    assert provider_calls[0].ums_boundary == state.market_data_boundary.isoformat()
    assert any(
        record.message == "ai_reasoning.result.reused"
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_successive_one_minute_ums_boundaries_each_receive_fresh_reasoning() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    provider = ValidProvider()
    service = build_service(repository, provider)

    first = await service.process(state, quant)
    next_state_id = uuid4()
    next_state = state.model_copy(
        update={
            "state_id": next_state_id,
            "state_hash": "b" * 64,
            "cycle_id": uuid4(),
            "market_data_boundary": state.market_data_boundary + timedelta(minutes=1),
            "knowledge_cutoff": state.knowledge_cutoff + timedelta(minutes=1),
            "evidence": tuple(
                item.model_copy(update={"market_state_id": next_state_id})
                for item in state.evidence
            ),
        }
    )
    next_quant = quant.model_copy(
        update={
            "result_id": uuid4(),
            "market_state_id": next_state_id,
            "cycle_id": next_state.cycle_id,
            "point_in_time": next_state.market_data_boundary,
            "generated_at": quant.generated_at + timedelta(minutes=1),
        }
    )
    second = await service.process(next_state, next_quant)

    assert first is not None and second is not None
    assert provider.calls == 2
    assert len(repository.reasoning_cycles) == 2


@pytest.mark.asyncio
async def test_concurrent_workers_share_one_durable_window_claim() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    first_provider = ValidProvider()
    second_provider = ValidProvider()
    first_service = build_service(repository, first_provider)
    second_service = build_service(repository, second_provider)

    await asyncio.gather(
        first_service.process(state, quant),
        second_service.process(state, quant),
    )
    reused = await second_service.process(state, quant)

    assert first_provider.calls + second_provider.calls == 1
    assert reused is not None
    assert len(repository.reasoning_cycles) == 1
    assert len(repository.forecasts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "reason_code"),
    (
        (401, "authentication_failed"),
        (402, "payment_blocked"),
        (403, "authentication_failed"),
        (429, "rate_limited"),
    ),
)
async def test_permanent_provider_failure_is_never_retried(
    status: int,
    reason_code: str,
) -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    provider = PermanentHttpFailureProvider(status, reason_code)

    result = await build_service(repository, provider).process(state, quant)

    assert result is not None
    assert result.forecast.provider_http_status == status
    assert provider.calls == 1
    assert len(repository.failures) == 1


@pytest.mark.asyncio
async def test_orchestrator_does_not_duplicate_router_owned_retry() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    provider = TemporaryUnavailableProvider()

    result = await build_service(repository, provider).process(state, quant)

    assert result is not None
    assert result.forecast.status == AIResultStatus.UNAVAILABLE
    assert provider.calls == 1
    assert len(repository.failures) == 1


@pytest.mark.asyncio
async def test_monitoring_persists_closed_cycle_evaluation_without_executing_signal() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    await build_service(repository, ValidProvider()).process(state, quant)
    result = await build_service(repository, ValidProvider(), proposals=True, monitoring=True).process(state, quant)
    assert result is not None
    assert len(repository.monitoring) == 1
    assert next(iter(repository.signals.values())).state == ManagedSignalState.PROPOSED
    assert result.proposal.approved_for_publication is False


@pytest.mark.asyncio
async def test_level_revisions_are_auditable_and_stop_widening_is_blocked() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(repository, ValidProvider()).process(state, quant)
    assert result and result.proposal
    signal = next(iter(repository.signals.values()))
    lifecycle = SignalLifecycleService(repository, policy_version="policy-v1", model_version="configured-model", clock=lambda: NOW)
    with pytest.raises(ValueError, match="cannot be widened"):
        await lifecycle.revise_level(signal, level_type="stop_loss", new_value=3290, reason="avoid loss", evidence_ids=())
    updated, revision = await lifecycle.revise_level(signal, level_type="stop_loss", new_value=3297, reason="protect risk", evidence_ids=result.proposal.supporting_evidence_ids)
    assert revision.old_value == 3295 and revision.new_value == 3297
    assert updated.stop_loss == 3297
    assert repository.revisions[revision.revision_id] == revision


@pytest.mark.asyncio
async def test_lifecycle_transition_requires_explicit_guardrail_rule_and_is_audited() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(repository, ValidProvider()).process(state, quant)
    assert result and result.proposal
    signal = next(iter(repository.signals.values()))
    lifecycle = SignalLifecycleService(repository, policy_version="policy-v1", model_version="configured-model", clock=lambda: NOW)
    with pytest.raises(ValueError, match="explicit guardrail"):
        await lifecycle.apply_guardrail_approved_transition(
            signal,
            ManagedSignalState.CONFIRMED,
            approval_rule="",
            forecast=result.forecast,
            proposal=result.proposal,
        )
    confirmed = await lifecycle.apply_guardrail_approved_transition(
        signal,
        ManagedSignalState.CONFIRMED,
        approval_rule="future_guardrail_test_rule",
        forecast=result.forecast,
        proposal=result.proposal,
    )
    assert confirmed.state == ManagedSignalState.CONFIRMED
    assert any(item.previous_state == ManagedSignalState.PROPOSED and item.new_state == ManagedSignalState.CONFIRMED for item in repository.transitions.values())


def test_market_memory_is_bounded_and_preserves_recent_changes() -> None:
    entries = tuple(
        MarketMemoryEntry(
            entry_id=uuid4(),
            instrument="XAUUSD",
            cycle_id=uuid4(),
            market_state_id=uuid4(),
            category="evidence_change",
            summary=f"change-{index}",
            occurred_at=NOW + timedelta(minutes=index),
        )
        for index in range(30)
    )
    summary = MarketMemory(5).summarize(entries)
    assert summary.entry_count == 5
    assert summary.evidence_changes == tuple(f"change-{index}" for index in range(25, 30))


@pytest.mark.asyncio
async def test_future_or_mismatched_quantitative_inputs_are_rejected_before_llm() -> None:
    state, quant = await state_and_quant()
    provider = ValidProvider()
    bad_quant = quant.model_copy(update={"market_state_id": uuid4()})
    with pytest.raises(ValueError, match="does not belong"):
        await build_service(InMemoryAIReasoningRepository(), provider).process(state, bad_quant)
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_monitoring_rejects_non_closed_market_state_before_llm() -> None:
    state, quant = await state_and_quant()
    provider = ValidProvider()
    non_closed = state.model_copy(update={"market_data_boundary": state.market_data_boundary - timedelta(minutes=1)})
    with pytest.raises(ValueError):
        await build_service(InMemoryAIReasoningRepository(), provider, monitoring=True).process(non_closed, quant)
    assert provider.calls == 0


def test_all_reasoning_prompts_are_versioned_files() -> None:
    loader = PromptLoader(Path("backend/app/ai_reasoning/prompts"))
    versions = (
        "new_market_analysis_v1",
        "existing_signal_monitoring_v1",
        "setup_invalidation_v1",
        "entry_refinement_v1",
        "active_signal_monitoring_v1",
        "partial_profit_analysis_v1",
        "signal_closure_analysis_v1",
    )
    assert all("strict JSON" in loader.load(version) or "JSON object" in loader.load(version) for version in versions)


def test_all_ai_signal_flags_remain_disabled_by_default() -> None:
    flags = YamlConfigRepository().load("feature_flags")["flags"]
    assert flags["ai_signal_proposals"] is False
    assert flags["ai_signal_monitoring"] is False
    assert flags["ai_signal_publication"] is False
    assert flags["ai_signal_adjustments"] is False
