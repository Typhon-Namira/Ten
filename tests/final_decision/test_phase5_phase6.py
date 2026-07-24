from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.app.ai_reasoning.models import ManagedSignalState, ProposalAction
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.core.config import YamlConfigRepository
from backend.app.api.routes.ai_reasoning import _runtime_state
from backend.app.final_decision import (
    DeterministicReplayAdapter,
    EvaluationCandle,
    ExecutionContext,
    FinalAction,
    FinalDecisionService,
    GateStatus,
    GuardrailPolicyConfig,
    HardGateRegistry,
    InMemoryFinalDecisionRepository,
    InMemoryReplayResponseStore,
    MonitoringAdjustmentPolicy,
    OperationMode,
    PerformanceReporter,
    PointInTimeReplay,
    ProbabilityCalibration,
    ProductionReadinessEvaluator,
    PublicationState,
    ReplayLLMMode,
    SignalOutcomeEvaluator,
)
from backend.app.final_decision.replay import RecordedLLMResponse, replay_request_hash
from backend.app.ai_reasoning.setup_families import SetupFamilyRegistry
from tests.ai_reasoning.test_ai_reasoning_lifecycle import (
    NOW,
    ValidProvider,
    build_service,
    state_and_quant,
)


async def proposal_fixture():
    state, quant = await state_and_quant()
    ai_repository = InMemoryAIReasoningRepository()
    provider = ValidProvider()
    await build_service(ai_repository, provider).process(state, quant)
    forecast = await ai_repository.latest_forecast("XAUUSD")
    proposal = await ai_repository.latest_proposal()
    signals = await ai_repository.active_signals("XAUUSD")
    assert forecast is not None and proposal is not None and signals
    return state, quant, forecast, proposal, signals[0], provider


def final_service(*, publication: bool, adjustments: bool = False):
    configs = YamlConfigRepository()
    config = configs.load_model("ai_guardrails", GuardrailPolicyConfig)
    repository = InMemoryFinalDecisionRepository()
    return (
        FinalDecisionService(
            repository,
            HardGateRegistry(),
            SetupFamilyRegistry.from_yaml(configs),
            config,
            publication_enabled=publication,
            adjustments_enabled=adjustments,
            clock=lambda: NOW,
        ),
        repository,
    )


def test_dashboard_runtime_metadata_is_observational_and_explicitly_analytical_only() -> None:
    flags = {
        "ai_centric_shadow_mode": True,
        "ai_signal_publication": False,
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_registry=SimpleNamespace(
                    context=SimpleNamespace(
                        feature_flags=SimpleNamespace(snapshot=lambda: flags),
                    ),
                ),
            ),
        ),
    )
    runtime = _runtime_state(request)
    assert runtime == {
        "operating_profile": "shadow",
        "feature_flags": flags,
        "analytical_only": True,
        "broker_execution_available": False,
    }

    service, _ = final_service(publication=False)
    health = service.health()
    assert health["daily_request_allowance"] == service.config.maximum_daily_llm_requests
    assert health["daily_token_allowance"] == service.config.maximum_daily_llm_tokens
    assert health["llm_concurrency_limit"] == service.config.llm_concurrency_limit


def context(state, quant, proposal, signal, **updates):
    values = {
        "context_id": uuid4(),
        "instrument": "XAUUSD",
        "evaluated_at": NOW,
        "operation_mode": OperationMode.ANALYTICAL_LIVE,
        "analytical_only": True,
        "broker_execution_available": False,
        "market_open": True,
        "current_price": quant.predictions[0].reference_price,
        "spread": 0.2,
        "session": "unknown",
        "publication_service_available": True,
        "persistence_available": True,
        "economic_context_available": True,
        "prohibited_economic_event_window": False,
        "active_opportunity_keys": (proposal.structural_opportunity_key,),
        "active_signal_id": signal.signal_id,
    }
    values.update(updates)
    return ExecutionContext(**values)


@pytest.mark.asyncio
async def test_live_analytical_publication_requires_flag_and_never_executes_broker_order() -> None:
    state, quant, forecast, proposal, signal, _ = await proposal_fixture()
    shadow_service, shadow_repository = final_service(publication=False)
    shadow = await shadow_service.evaluate(state, quant, forecast, proposal, signal, context(state, quant, proposal, signal))
    assert shadow.action.action == FinalAction.APPROVED
    assert shadow.action.publication_state == PublicationState.DISABLED
    assert not shadow_repository.publications

    live_service, live_repository = final_service(publication=True)
    live = await live_service.evaluate(state, quant, forecast, proposal, signal, context(state, quant, proposal, signal))
    assert live.action.action == FinalAction.PUBLISHED
    assert live.action.publication_state == PublicationState.PUBLISHED
    assert live.publication is not None
    assert live.publication.analytical_only and not live.publication.broker_execution
    assert len(live_repository.publications) == 1


@pytest.mark.asyncio
async def test_original_ai_proposal_is_immutable_and_policy_modification_is_audited() -> None:
    state, quant, forecast, proposal, signal, _ = await proposal_fixture()
    service, _ = final_service(publication=False)
    long_expiry = NOW + timedelta(seconds=service.config.maximum_setup_expiry_seconds + 60)
    changed = proposal.model_copy(update={"expires_at": long_expiry})
    original = changed.model_dump(mode="json")
    result = await service.evaluate(state, quant, forecast, changed, signal, context(state, quant, changed, signal))
    assert changed.model_dump(mode="json") == original
    assert result.action.modifications[0].field_name == "expires_at"
    assert result.action.modifications[0].original_value == long_expiry.isoformat()
    assert result.action.original_proposal_hash


@pytest.mark.asyncio
async def test_market_closed_stale_data_and_low_risk_reward_are_deterministically_blocked() -> None:
    state, quant, forecast, proposal, signal, _ = await proposal_fixture()
    service, _ = final_service(publication=True)
    closed = await service.evaluate(
        state,
        quant,
        forecast,
        proposal,
        signal,
        context(state, quant, proposal, signal, market_open=False),
    )
    assert closed.publication is None
    assert any(item.gate_id == "market_open" and item.status == GateStatus.FAILED for item in closed.action.gate_evaluations)

    stale_state = state.model_copy(
        update={"timeframes": tuple(item.model_copy(update={"stale": True}) for item in state.timeframes)}
    )
    stale = await service.evaluate(stale_state, quant, forecast, proposal, signal, context(stale_state, quant, proposal, signal))
    assert stale.publication is None
    assert any(item.gate_id == "authoritative_data_fresh" and item.status == GateStatus.FAILED for item in stale.action.gate_evaluations)

    low_rr = proposal.model_copy(update={"take_profit_levels": (proposal.entry_zone.high + 0.1,)})
    rejected = await service.evaluate(state, quant, forecast, low_rr, signal, context(state, quant, low_rr, signal))
    assert rejected.publication is None
    assert any(item.gate_id == "absolute_risk_to_reward" and item.status == GateStatus.FAILED for item in rejected.action.gate_evaluations)


@pytest.mark.asyncio
async def test_unknown_market_status_remains_fail_closed_with_typed_audit_context() -> None:
    state, quant, forecast, proposal, signal, _ = await proposal_fixture()
    service, _ = final_service(publication=True)
    result = await service.evaluate(
        state,
        quant,
        forecast,
        proposal,
        signal,
        context(
            state,
            quant,
            proposal,
            signal,
            market_open=None,
            market_status="UNKNOWN",
            market_status_source="unavailable",
        ),
    )

    gate = next(item for item in result.action.gate_evaluations if item.gate_id == "market_open")
    assert gate.status == GateStatus.UNAVAILABLE
    assert gate.reason_codes == ("market_status_unavailable",)
    assert gate.audit_payload["market_status"] == "UNKNOWN"
    assert result.publication is None


@pytest.mark.asyncio
async def test_duplicate_structural_opportunity_is_blocked_and_publication_is_idempotent() -> None:
    state, quant, forecast, proposal, signal, _ = await proposal_fixture()
    service, repository = final_service(publication=True)
    duplicate = await service.evaluate(
        state,
        quant,
        forecast,
        proposal,
        signal,
        context(state, quant, proposal, signal, active_signal_id=uuid4()),
    )
    assert duplicate.publication is None
    assert any(item.gate_id == "duplicate_structural_opportunity" and item.status == GateStatus.FAILED for item in duplicate.action.gate_evaluations)

    first = await service.evaluate(state, quant, forecast, proposal, signal, context(state, quant, proposal, signal))
    second = await service.evaluate(state, quant, forecast, proposal, signal, context(state, quant, proposal, signal))
    assert first.publication == second.publication
    assert len(repository.publications) == 1


def test_registry_contains_only_genuine_safety_gates_not_analytical_disagreements() -> None:
    gate_ids = {item.gate_id for item in HardGateRegistry().all()}
    assert {
        "market_state_consistent",
        "future_data_absent",
        "market_open",
        "entry_geometry_valid",
        "absolute_risk_to_reward",
        "economic_event_blackout",
    } <= gate_ids
    forbidden = {"htf_disagreement", "weak_volume", "ranging_regime", "missing_fvg", "liquidity_disagreement"}
    assert not gate_ids.intersection(forbidden)


@pytest.mark.asyncio
async def test_missing_optional_fvg_and_account_risk_do_not_block_unrelated_analytical_setup() -> None:
    state, quant, forecast, proposal, signal, _ = await proposal_fixture()
    service, _ = final_service(publication=False)
    result = await service.evaluate(state, quant, forecast, proposal, signal, context(state, quant, proposal, signal))
    mandatory = next(item for item in result.action.gate_evaluations if item.gate_id == "mandatory_setup_evidence")
    account = [item for item in result.action.gate_evaluations if item.gate_id.startswith("maximum_") and item.category == "risk"]
    assert mandatory.status == GateStatus.PASSED
    assert account and all(item.status == GateStatus.NOT_APPLICABLE for item in account)
    assert result.action.action in {FinalAction.APPROVED, FinalAction.APPROVED_REDUCED_RISK}


@pytest.mark.asyncio
async def test_invalid_or_unavailable_ai_output_never_reaches_publication() -> None:
    state, quant = await state_and_quant()
    ai_repository = InMemoryAIReasoningRepository()
    from tests.ai_reasoning.test_ai_reasoning_lifecycle import InvalidProvider

    result = await build_service(ai_repository, InvalidProvider(), maximum_retries=0).process(state, quant)
    assert result is not None and result.proposal is None
    assert not ai_repository.proposals and not ai_repository.signals


@pytest.mark.asyncio
async def test_reasoning_is_deduplicated_by_immutable_market_state_hash() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    service = build_service(repository, provider)
    first = await service.process(state, quant)
    reused = await service.process(state, quant)
    assert first is not None
    assert reused is not None
    assert reused.forecast.forecast_id == first.forecast.forecast_id
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_recorded_llm_replay_is_reproducible_and_never_requires_live_llm() -> None:
    request = {"state_hash": "abc", "prompt_version": "v1"}
    request_hash = replay_request_hash(request)
    response = {"forecast": {"status": "available"}}
    adapter = DeterministicReplayAdapter(
        InMemoryReplayResponseStore(
            (
                RecordedLLMResponse(
                    request_hash=request_hash,
                    prompt_version="v1",
                    model_identifier="configured-openrouter-model",
                    temperature=0.1,
                    generation_parameters={"max_tokens": 3200},
                    structured_response=response,
                ),
            )
        )
    )
    assert await adapter.response(request, ReplayLLMMode.RECORDED_RESPONSE) == response
    with pytest.raises(LookupError):
        await adapter.response({"different": True}, ReplayLLMMode.RECORDED_RESPONSE)


@pytest.mark.asyncio
async def test_historical_replay_rejects_future_state() -> None:
    state, _ = await state_and_quant()
    replay = PointInTimeReplay()
    assert replay.validate_state(state, state.knowledge_cutoff) == state
    with pytest.raises(ValueError, match="future"):
        replay.validate_state(state, state.market_data_boundary - timedelta(seconds=1))


@pytest.mark.asyncio
async def test_outcome_evaluation_waits_for_complete_horizon_then_measures_costs() -> None:
    state, quant, forecast, proposal, signal, _ = await proposal_fixture()
    service, _ = final_service(publication=True)
    result = await service.evaluate(state, quant, forecast, proposal, signal, context(state, quant, proposal, signal))
    assert result.publication is not None
    evaluator = SignalOutcomeEvaluator(configured_slippage=0.1)
    horizon = NOW + timedelta(minutes=10)
    pending = evaluator.evaluate(
        result.publication,
        (),
        required_horizon_end=horizon,
        spread=0.2,
        evaluated_at=NOW,
    )
    assert pending.status == "pending"
    assert pending.realized_return is None
    candles = (
        EvaluationCandle(timestamp=NOW + timedelta(minutes=1), open=3301, high=3303, low=3300, close=3302),
        EvaluationCandle(timestamp=horizon, open=3302, high=3312, low=3301, close=3311),
    )
    measured = evaluator.evaluate(
        result.publication,
        candles,
        required_horizon_end=horizon,
        spread=0.2,
        evaluated_at=horizon,
    )
    assert measured.evaluation_horizon_complete
    assert measured.tp1_result == "hit"
    assert measured.spread_adjusted_result is not None
    assert measured.slippage_adjusted_result < measured.spread_adjusted_result


def test_probability_calibration_is_measured_and_raw_confidence_is_not_claimed_calibrated() -> None:
    report = ProbabilityCalibration().calculate(
        (
            {"actual": "BUY", "probabilities": {"BUY": 0.7, "SELL": 0.2, "NEUTRAL": 0.1}, "horizon": "10_m1", "setup_family": "trend_continuation", "regime": "trend", "session": "london", "confidence_band": "high"},
            {"actual": "SELL", "probabilities": {"BUY": 0.4, "SELL": 0.4, "NEUTRAL": 0.2}, "horizon": "10_m1", "setup_family": "trend_continuation", "regime": "trend", "session": "london", "confidence_band": "medium"},
        ),
        generated_at=NOW,
    )
    assert report.status == "measured_uncalibrated"
    assert report.brier_score is not None and report.log_loss is not None
    assert report.expected_calibration_error is not None
    assert report.reliability_buckets


def test_performance_and_readiness_reports_use_measured_samples_without_profitability_claims() -> None:
    reporter = PerformanceReporter()
    report = reporter.build(
        period_start=NOW - timedelta(days=1),
        period_end=NOW,
        system_outcomes={"legacy": (), "quantitative_shadow": (), "ai_proposals": (), "guardrail_approved": ()},
        generated_at=NOW,
    )
    assert set(report.comparison) == {"legacy", "quantitative_shadow", "ai_proposals", "guardrail_approved"}
    assert report.comparison["guardrail_approved"]["expected_value"] is None

    config = YamlConfigRepository().load_model("ai_guardrails", GuardrailPolicyConfig)
    readiness = ProductionReadinessEvaluator(config).evaluate({"sample_count": 0}, generated_at=NOW)
    assert readiness.status == "not_ready"
    assert "sample_size" in readiness.blockers
    assert "profitability_not_guaranteed" in readiness.warnings


def test_only_existing_openrouter_ai_provider_is_present() -> None:
    source = "\n".join(
        item.read_text(encoding="utf-8")
        for item in Path("backend/app/final_decision").glob("*.py")
    )
    provider_source = Path("backend/app/ai_reasoning/provider.py").read_text(encoding="utf-8")
    assert "ExistingOpenRouterReasoningProvider" in provider_source
    assert "anthropic" not in source.lower()
    assert "openai" not in source.lower()


@pytest.mark.asyncio
async def test_monitoring_adjustment_policy_requires_flag_and_rejects_stop_widening() -> None:
    _, _, _, proposal, signal, _ = await proposal_fixture()
    active = signal.model_copy(update={"state": ManagedSignalState.ACTIVE})
    disabled = MonitoringAdjustmentPolicy(enabled=False, policy_version="ai_adjustments_v1")
    with pytest.raises(PermissionError):
        disabled.authorize(ProposalAction.REDUCE_RISK, active)
    enabled = MonitoringAdjustmentPolicy(enabled=True, policy_version="ai_adjustments_v1")
    assert "REDUCE_RISK" in enabled.authorize(ProposalAction.REDUCE_RISK, active)

    class Lifecycle:
        async def revise_level(self, *args, **kwargs):
            return args, kwargs

    with pytest.raises(ValueError, match="widening"):
        await enabled.revise_stop(
            Lifecycle(),  # type: ignore[arg-type]
            active,
            new_stop=active.stop_loss - 1,
            reason="not protective",
            evidence_ids=(),
        )


def test_operating_profiles_never_enable_broker_execution() -> None:
    profiles = Path("configs/ai_operating_profiles.yaml").read_text(encoding="utf-8")
    assert "safe_test:" in profiles and "shadow:" in profiles and "analytical_live:" in profiles
    assert "broker_execution: false" in profiles
    flags = Path("configs/feature_flags.yaml").read_text(encoding="utf-8")
    assert "ai_signal_publication: false" in flags
    assert "ai_signal_adjustments: false" in flags
