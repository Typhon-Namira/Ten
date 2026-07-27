from __future__ import annotations

from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from backend.app.ai_reasoning.analysis import (
    AIMarketAnalysis,
    AIAnalysisOutput,
    AIAnalysisTemporalContext,
    AIProviderMetadata,
    AnalysisStatus,
    ConsistencyClassification,
    TemporalContextAnalyzer,
    TemporalDataQuality,
    analysis_reference,
)
from backend.app.ai_reasoning.provider import reasoning_response_schema
from backend.app.ai_reasoning.provider import AIProviderResponse
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.engines.signal_decision_engine import ConservativeSignalDecisionPolicy
from backend.app.engines.signal_decision_engine.models import FinalSignalAction
from tests.engines.ai_scoring_engine.test_ai_scoring import NOW
from tests.engines.signal_decision_engine.test_signal_decision_engine import decision_input
from tests.ai_reasoning.test_ai_reasoning_lifecycle import build_service, state_and_quant


def evidence(claim: str = "Closed M15 structure event confirms the interpretation.") -> dict[str, object]:
    return {
        "claim": claim,
        "kind": "calculated_feature",
        "source_type": "market_structure",
        "source_reference": "feature.market_structure.recent_change",
        "timeframe": "M15",
        "observed_value": "bullish_change_of_character",
    }


def output(regime: str = "bullish", confidence: float = 0.8) -> AIAnalysisOutput:
    item = evidence()
    return AIAnalysisOutput.model_validate(
        {
            "market_regime": {
                "classification": regime,
                "strength": 75,
                "confidence": confidence,
                "evidence": [item],
            },
            "higher_timeframe_context": {
                "bias": regime if regime in {"bullish", "bearish"} else "mixed",
                "description": "Higher-timeframe evidence is aligned.",
                "evidence": [item],
            },
            "market_structure": {
                "short_term": "Short-term structure is constructive.",
                "medium_term": "Medium-term structure is constructive.",
                "higher_timeframe": "Higher-timeframe structure remains supported.",
                "recent_change": "A confirmed structural transition occurred.",
                "evidence": [item],
            },
            "liquidity_analysis": {
                "summary": "Liquidity remains traceable.",
                "events": ["sell_side_sweep"],
                "unresolved_liquidity": ["buy_side_pool"],
                "evidence": [item],
            },
            "supply_demand_analysis": {
                "summary": "Demand remains closer than supply.",
                "nearest_supply": 3350,
                "nearest_demand": 3300,
                "evidence": [item],
            },
            "momentum_analysis": {
                "direction": regime if regime in {"bullish", "bearish"} else "mixed",
                "strength": 70,
                "trend": "strengthening",
                "evidence": [item],
            },
            "volatility_analysis": {
                "state": "normal",
                "trend": "stable",
                "evidence": [item],
            },
            "bullish_evidence": [item] if regime == "bullish" else [],
            "bearish_evidence": [item] if regime == "bearish" else [],
            "contradictions": [],
            "key_risks": [evidence("A nearby opposing liquidity pool remains unresolved.")],
            "alternative_scenarios": [
                {
                    "name": "range_reentry",
                    "description": "Price could return to the prior range.",
                    "probability": 0.2,
                    "confirmation_evidence": ["feature.market_structure.recent_change"],
                }
            ],
            "analysis_confidence": confidence,
            "executive_summary": "Structure and liquidity support the current interpretation.",
        }
    )


def analysis(index: int, regime: str = "bullish", confidence: float = 0.8) -> AIMarketAnalysis:
    timestamp = NOW + timedelta(minutes=index)
    identifier = uuid5(NAMESPACE_URL, f"analysis:{index}:{regime}")
    return AIMarketAnalysis(
        analysis_id=identifier,
        request_id=uuid5(NAMESPACE_URL, f"request:{index}:{regime}"),
        cycle_id=uuid5(NAMESPACE_URL, f"cycle:{index}:{regime}"),
        symbol="XAUUSD",
        timeframe="M15",
        market_snapshot_id=uuid5(NAMESPACE_URL, f"state:{index}:{regime}"),
        quantitative_forecast_id=uuid5(NAMESPACE_URL, f"quant:{index}:{regime}"),
        analysis_timestamp=timestamp,
        knowledge_cutoff=timestamp,
        status=AnalysisStatus.AVAILABLE,
        output=output(regime, confidence),
        provider_metadata=AIProviderMetadata(
            provider="cerebras",
            model="test-analysis-model",
            prompt_version="deep_market_analysis_v2",
            provider_adapter_version="test-v2",
        ),
        validation_passed=True,
        created_at=timestamp,
    )


@pytest.mark.parametrize(
    "forbidden",
    [
        {"decision": "BUY"},
        {"proposal": {}},
        {"setup_family": "trend_continuation"},
        {"entry": 3300},
        {"stop_loss": 3290},
        {"take_profit": 3320},
        {"publication_eligible": True},
    ],
)
def test_analysis_contract_rejects_every_signal_field(forbidden: dict[str, object]) -> None:
    payload = output().model_dump(mode="python")
    payload.update(forbidden)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AIAnalysisOutput.model_validate(payload)


def test_provider_schema_is_strict_and_contains_no_signal_authority() -> None:
    schema = reasoning_response_schema()
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    for forbidden in (
        "decision",
        "proposal",
        "setup_family",
        "entry",
        "stop_loss",
        "take_profit",
        "publication_eligible",
    ):
        assert forbidden not in properties
    assert set(schema["required"]) == set(properties)


def temporal_context(values: list[AIMarketAnalysis], current: AIMarketAnalysis) -> AIAnalysisTemporalContext:
    references = tuple(analysis_reference(item) for item in values)
    return AIAnalysisTemporalContext(
        current_analysis_id=current.analysis_id,
        as_of=current.analysis_timestamp,
        previous_analysis_id=references[-1].analysis_id if references else None,
        lookbacks={"5m": references[-1] if references else None},
        rolling_window=references,
        data_quality=TemporalDataQuality.SUFFICIENT,
    )


def test_stable_history_is_high_consistency_and_reversal_is_penalized() -> None:
    stable_history = [analysis(index, "bullish", 0.7 + index * 0.01) for index in range(5)]
    stable_current = analysis(5, "bullish", 0.76)
    stable = TemporalContextAnalyzer().analyze(
        temporal_context(stable_history, stable_current),
        stable_current,
    )
    assert stable.historical_consistency.classification == ConsistencyClassification.HIGH

    reversal_current = analysis(5, "bearish", 0.95)
    reversal = TemporalContextAnalyzer().analyze(
        temporal_context(stable_history, reversal_current),
        reversal_current,
    )
    assert reversal.historical_consistency.score < stable.historical_consistency.score
    assert reversal.historical_consistency.contradicting_analysis_ids


def test_temporal_context_rejects_current_or_future_analysis() -> None:
    current = analysis(5)
    with pytest.raises(ValidationError, match="current or future"):
        AIAnalysisTemporalContext(
            current_analysis_id=current.analysis_id,
            as_of=current.analysis_timestamp,
            lookbacks={"5m": analysis_reference(current)},
            rolling_window=(),
            data_quality=TemporalDataQuality.LIMITED,
        )


@pytest.mark.asyncio
async def test_analysis_persistence_is_idempotent_and_point_in_time_bounded() -> None:
    repository = InMemoryAIReasoningRepository()
    value = analysis(2)
    assert await repository.save_analysis(value) == value
    assert await repository.save_analysis(value) == value
    assert len(repository.analyses) == 1
    assert await repository.analyses_before("XAUUSD", "M15", value.analysis_timestamp, 10) == ()


def test_signal_engine_is_only_action_authority_and_confidence_is_independent() -> None:
    history = [analysis(index) for index in range(5)]
    current = analysis(5)
    context = temporal_context(history, current)
    metrics = TemporalContextAnalyzer().analyze(context, current)
    base = decision_input(as_of=current.analysis_timestamp)
    value = base.model_copy(
        update={
            "current_ai_analysis": current,
            "temporal_context": context,
            "temporal_metrics": metrics,
            "market_snapshot_id": current.market_snapshot_id,
            "quantitative_forecast_id": current.quantitative_forecast_id,
            "current_price": 3320.0,
            "expected_move": 10.0,
        }
    )
    decision = ConservativeSignalDecisionPolicy().evaluate(value)
    assert decision.final_action == FinalSignalAction.BUY
    assert decision.publication_eligible is True
    assert decision.stop_loss is not None
    assert decision.take_profit_targets
    assert decision.source_lineage is not None
    assert decision.source_lineage.current_ai_analysis_id == current.analysis_id
    assert decision.confidence_score != current.output.analysis_confidence * 100


def test_missing_analysis_fails_closed_for_publication_without_breaking_audit_decision() -> None:
    decision = ConservativeSignalDecisionPolicy().evaluate(decision_input())
    assert decision.final_action == FinalSignalAction.WAIT
    assert decision.publication_eligible is False


class AnalysisProvider:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload or output().model_dump(mode="python")
        self.calls = 0

    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        self.calls += 1
        return AIProviderResponse(
            raw_output=self.payload,
            provider="cerebras",
            model_identifier="test-model",
            latency_ms=2,
            token_usage={"input_tokens": 10, "output_tokens": 20},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "cerebras",
            "model_identifier": "test-model",
            "provider_available": True,
        }


@pytest.mark.asyncio
async def test_service_persists_one_analysis_and_reuses_it_without_second_provider_call() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    provider = AnalysisProvider()
    service = build_service(
        repository,
        provider,
        shadow=True,
        proposals=False,
        monitoring=False,
    )
    first = await service.process(state, quant)
    second = await service.process(state, quant)
    assert first is not None
    assert second is not None
    assert first.analysis.analysis_id == second.analysis.analysis_id
    assert provider.calls == 1
    assert repository.proposals == {}
    assert repository.signals == {}


@pytest.mark.asyncio
async def test_invalid_analysis_is_persisted_and_never_reaches_signal_engine() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    provider = AnalysisProvider({"decision": "BUY"})
    service = build_service(
        repository,
        provider,
        shadow=True,
        proposals=False,
        monitoring=False,
    )
    assert await service.process(state, quant) is None
    persisted = tuple(repository.analyses.values())
    assert len(persisted) == 1
    assert persisted[0].status == AnalysisStatus.INVALID
    assert persisted[0].output is None
    assert repository.proposals == {}
