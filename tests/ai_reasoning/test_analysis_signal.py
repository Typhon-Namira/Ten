from __future__ import annotations

from datetime import timedelta

import pytest

from backend.app.ai_reasoning.analysis import (
    AnalysisExecutionEligibility,
    AnalysisSignalAction,
    AnalysisSignalStrength,
)
from backend.app.ai_reasoning.repository import (
    AIArtifactConflictError,
    InMemoryAIReasoningRepository,
)
from backend.app.ai_reasoning.signal import DeterministicAnalysisSignalGenerator
from tests.ai_reasoning.test_ai_reasoning_lifecycle import state_and_quant
from tests.ai_reasoning.test_analysis_architecture_v2 import analysis


def aligned_analysis(state, quant, index: int = 5, **kwargs):
    value = analysis(index, **kwargs)
    return value.model_copy(
        update={
            "cycle_id": state.cycle_id,
            "market_snapshot_id": state.state_id,
            "quantitative_forecast_id": quant.result_id,
            "analysis_timestamp": state.market_data_boundary,
            "knowledge_cutoff": state.knowledge_cutoff,
            "created_at": state.knowledge_cutoff,
        }
    )


@pytest.mark.asyncio
async def test_completed_analysis_produces_directionally_valid_signal() -> None:
    state, quant = await state_and_quant()

    signal = DeterministicAnalysisSignalGenerator().generate(
        aligned_analysis(state, quant),
        state,
        quant,
    )

    assert signal.signal == AnalysisSignalAction.BUY
    assert 0 <= signal.confidence <= 100
    if signal.confidence >= 70:
        assert signal.strength in {
            AnalysisSignalStrength.STRONG,
            AnalysisSignalStrength.VERY_STRONG,
        }
    assert signal.execution_eligibility == AnalysisExecutionEligibility.INELIGIBLE
    assert signal.entry is None
    assert signal.blocking_reasons


@pytest.mark.asyncio
async def test_weak_directional_evidence_preserves_low_confidence_direction_without_levels() -> None:
    state, quant = await state_and_quant()

    signal = DeterministicAnalysisSignalGenerator().generate(
        aligned_analysis(state, quant, regime="ranging", confidence=0.4),
        state,
        quant,
    )

    assert signal.signal in {AnalysisSignalAction.BUY, AnalysisSignalAction.SELL}
    assert signal.entry is None
    assert signal.stop_loss is None
    assert signal.take_profit is None
    assert signal.execution_eligibility == AnalysisExecutionEligibility.INELIGIBLE


@pytest.mark.asyncio
async def test_ai_bearish_claim_cannot_override_deterministic_evidence_or_create_levels() -> None:
    state, quant = await state_and_quant()
    bearish = aligned_analysis(state, quant, regime="bearish")
    assert bearish.output is not None
    structural_output = bearish.output.model_copy(
        update={
            "supply_demand_analysis": bearish.output.supply_demand_analysis.model_copy(
                update={"nearest_supply": 3304, "nearest_demand": 3290}
            )
        }
    )

    generator = DeterministicAnalysisSignalGenerator()
    generator.quality_threshold = 0
    signal = generator.generate(
        bearish.model_copy(update={"output": structural_output}),
        state,
        quant,
    )

    assert signal.signal == AnalysisSignalAction.BUY
    assert signal.take_profit is None
    assert signal.entry is None
    assert signal.stop_loss is None
    assert signal.execution_eligibility == AnalysisExecutionEligibility.INELIGIBLE


@pytest.mark.asyncio
async def test_direction_without_minimum_structural_rr_is_blocked_not_hold() -> None:
    state, quant = await state_and_quant()

    signal = DeterministicAnalysisSignalGenerator().generate(
        aligned_analysis(state, quant, regime="bearish"),
        state,
        quant,
    )

    assert signal.signal in {AnalysisSignalAction.BUY, AnalysisSignalAction.SELL}
    assert signal.entry is None
    assert signal.blocking_reasons


@pytest.mark.asyncio
async def test_tiny_expected_move_never_manufactures_geometry() -> None:
    state, quant = await state_and_quant()
    prediction = quant.predictions[0].model_copy(
        update={
            "expected_base_movement": 0.001,
            "expected_minimum_movement": 0.0005,
            "expected_maximum_movement": 0.0015,
            "expected_volatility": 0.001,
            "expected_mfe": 0.001,
            "expected_mae": 0.00075,
        }
    )
    tiny_quant = quant.model_copy(update={"predictions": (prediction,)})

    signal = DeterministicAnalysisSignalGenerator().generate(
        aligned_analysis(state, tiny_quant),
        state,
        tiny_quant,
    )

    assert signal.signal == AnalysisSignalAction.BUY
    assert signal.entry is None
    assert signal.stop_loss is None
    assert signal.take_profit is None
    assert signal.analysis_confidence == 80
    assert signal.confidence == round(signal.signal_confidence)


@pytest.mark.asyncio
async def test_replay_of_identical_point_in_time_inputs_is_byte_stable() -> None:
    state, quant = await state_and_quant()
    generator = DeterministicAnalysisSignalGenerator()

    current = aligned_analysis(state, quant)
    first = generator.generate(current, state, quant)
    second = generator.generate(current, state, quant)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.asyncio
async def test_degraded_source_data_caps_confidence_at_59() -> None:
    state, quant = await state_and_quant()
    from backend.app.market_state import EvidenceAvailability

    evidence = list(state.evidence)
    evidence[0] = evidence[0].model_copy(
        update={"availability": EvidenceAvailability.DEGRADED}
    )
    degraded = state.model_copy(
        update={
            "evidence": tuple(evidence),
            "degraded_evidence": (evidence[0].evidence_id,),
        }
    )

    baseline = DeterministicAnalysisSignalGenerator().generate(
        aligned_analysis(state, quant), state, quant
    )
    signal = DeterministicAnalysisSignalGenerator().generate(
        aligned_analysis(degraded, quant),
        degraded,
        quant,
    )

    assert signal.confidence <= baseline.confidence
    assert "timeframe_evidence_degraded" in signal.risk_flags


@pytest.mark.asyncio
async def test_signal_persistence_detects_conflicting_duplicate() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    signal = DeterministicAnalysisSignalGenerator().generate(
        aligned_analysis(state, quant),
        state,
        quant,
    )

    assert await repository.save_analysis_signal(signal) == signal
    assert await repository.save_analysis_signal(signal) == signal
    with pytest.raises(AIArtifactConflictError):
        await repository.save_analysis_signal(
            signal.model_copy(update={"reasoning_summary": "Conflicting payload."})
        )
    assert len(repository.analysis_signals) == 1


@pytest.mark.asyncio
async def test_latest_completed_cycle_ignores_newer_analysis_without_signal() -> None:
    state, quant = await state_and_quant()
    next_state, next_quant = await state_and_quant(
        state.market_data_boundary + timedelta(minutes=5)
    )
    repository = InMemoryAIReasoningRepository()
    completed_analysis = aligned_analysis(state, quant, 5)
    incomplete_newer_analysis = aligned_analysis(next_state, next_quant, 10)
    completed_signal = DeterministicAnalysisSignalGenerator().generate(
        completed_analysis,
        state,
        quant,
    )
    await repository.save_analysis(completed_analysis)
    await repository.save_analysis_signal(completed_signal)
    await repository.save_analysis(incomplete_newer_analysis)

    chart_timeframe_filtered = await repository.latest_completed_analysis_cycle(
        "XAUUSD",
        "M5",
    )
    latest = await repository.latest_completed_analysis_cycle("XAUUSD")

    assert chart_timeframe_filtered is None
    assert latest == (completed_analysis, completed_signal)


@pytest.mark.asyncio
async def test_analysis_signal_history_is_stable_and_paginated() -> None:
    state, quant = await state_and_quant()
    next_state, next_quant = await state_and_quant(
        state.market_data_boundary + timedelta(minutes=5)
    )
    repository = InMemoryAIReasoningRepository()
    first_analysis = aligned_analysis(state, quant, 5)
    second_analysis = aligned_analysis(next_state, next_quant, 10)
    first = DeterministicAnalysisSignalGenerator().generate(
        first_analysis,
        state,
        quant,
    )
    second = DeterministicAnalysisSignalGenerator().generate(
        second_analysis,
        next_state,
        next_quant,
    )
    await repository.save_analysis(first_analysis)
    await repository.save_analysis_signal(first)
    await repository.save_analysis(second_analysis)
    await repository.save_analysis_signal(second)

    page_one = await repository.list_analysis_signals(
        "XAUUSD", "M15", None, None, None, None, None, 0, 1
    )
    page_two = await repository.list_analysis_signals(
        "XAUUSD", "M15", None, None, None, None, None, 1, 1
    )

    assert page_one == (second,)
    assert page_two == (first,)
    assert await repository.count_analysis_signals("XAUUSD", "M15") == 2
    filtered = await repository.list_analysis_signals(
        "XAUUSD",
        "M15",
        None,
        None,
        second.signal.value,
        second.confidence,
        second.strength.value,
        0,
        10,
    )
    assert second in filtered
