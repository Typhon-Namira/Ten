from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
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
from backend.app.ai_reasoning.provider import AIProviderResponse, ExistingOpenRouterReasoningProvider
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.ai_reasoning.service import AIReasoningService
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from backend.app.ai_reasoning.validation import StructuredAIOutputValidator, structural_opportunity_key
from backend.app.ai.prompts.loader import PromptLoader
from backend.app.core.config import YamlConfigRepository
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
        return {"provider": "openrouter", "model_identifier": "configured-model", "external_ai_apis": ("openrouter",)}

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
            model_provider="openrouter",
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
            provider="openrouter",
            model_identifier=request.model_identifier,
            latency_ms=5,
            token_usage=None,
        )


class UnavailableProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        raise RuntimeError("configured LLM unavailable")


class InvalidProvider(ValidProvider):
    async def reason(self, request, *, prompt_version):
        self.calls += 1
        return AIProviderResponse(
            raw_output={"forecast": {"buy_probability": 2}, "proposal": {"recommended_action": "BUY"}},
            provider="openrouter",
            model_identifier=request.model_identifier,
            latency_ms=2,
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


def build_service(repository, provider, *, proposals=True, monitoring=False, maximum_retries=1):
    config = YamlConfigRepository().load_model("ai_reasoning", AIReasoningConfig).model_copy(update={"maximum_retries": maximum_retries})
    registry = SetupFamilyRegistry.from_yaml(YamlConfigRepository())
    return AIReasoningService(
        repository,
        provider,
        AIReasoningRequestBuilder(config, model_identifier="configured-model", clock=lambda: NOW),
        StructuredAIOutputValidator(registry),
        registry,
        config,
        proposals_enabled=proposals,
        monitoring_enabled=monitoring,
        clock=lambda: NOW,
    )


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
async def test_existing_openrouter_is_the_only_provider_boundary() -> None:
    assert ExistingOpenRouterReasoningProvider.provider_name == "openrouter"
    assert ValidProvider().metadata()["external_ai_apis"] == ("openrouter",)


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
    assert [item["zone_id"] for item in smc_raw["zones"]["items"]] == [0, 4_999]
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
    result = await build_service(repository, provider, maximum_retries=1).process(state, quant)
    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.UNAVAILABLE
    assert result.forecast.buy_probability is None
    assert provider.calls == 2
    assert len(repository.failures) == 2
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
    assert not repository.proposals


@pytest.mark.asyncio
async def test_unknown_evidence_reference_cannot_create_proposal() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    result = await build_service(repository, UnknownEvidenceProvider(), maximum_retries=0).process(state, quant)
    assert result is not None and result.proposal is None
    assert result.forecast.status == AIResultStatus.INVALID
    assert any("unknown_evidence_reference" in error for failure in repository.failures.values() for error in failure.validation_errors)


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
