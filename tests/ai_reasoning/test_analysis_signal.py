from __future__ import annotations

import pytest

from backend.app.ai_reasoning.analysis import (
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


@pytest.mark.asyncio
async def test_completed_analysis_produces_directionally_valid_signal() -> None:
    state, quant = await state_and_quant()

    signal = DeterministicAnalysisSignalGenerator().generate(
        analysis(5),
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
    assert signal.stop_loss is not None
    assert signal.entry is not None
    assert signal.take_profit is not None
    assert signal.stop_loss < signal.entry < signal.take_profit


@pytest.mark.asyncio
async def test_weak_directional_evidence_produces_hold_without_levels() -> None:
    state, quant = await state_and_quant()

    signal = DeterministicAnalysisSignalGenerator().generate(
        analysis(5, regime="ranging", confidence=0.4),
        state,
        quant,
    )

    assert signal.signal == AnalysisSignalAction.HOLD
    assert signal.confidence <= 39
    assert signal.entry is None
    assert signal.stop_loss is None
    assert signal.take_profit is None
    assert "insufficient_evidence_for_direction" in signal.risk_flags


@pytest.mark.asyncio
async def test_bearish_analysis_produces_directionally_valid_sell_levels() -> None:
    state, quant = await state_and_quant()
    bearish = analysis(5, regime="bearish")
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

    assert signal.signal == AnalysisSignalAction.SELL
    assert signal.take_profit is not None
    assert signal.entry is not None
    assert signal.stop_loss is not None
    assert signal.take_profit < signal.entry < signal.stop_loss
    assert signal.risk_reward_ratio is not None
    assert signal.risk_reward_ratio >= 2


@pytest.mark.asyncio
async def test_direction_without_minimum_structural_rr_produces_hold() -> None:
    state, quant = await state_and_quant()

    signal = DeterministicAnalysisSignalGenerator().generate(
        analysis(5, regime="bearish"),
        state,
        quant,
    )

    assert signal.signal == AnalysisSignalAction.HOLD
    assert signal.entry is None
    assert "no_structural_geometry_meets_minimum_risk_reward" in signal.risk_flags


@pytest.mark.asyncio
async def test_geometry_uses_structure_not_tiny_expected_move() -> None:
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
        analysis(5),
        state,
        tiny_quant,
    )

    assert signal.signal == AnalysisSignalAction.BUY
    assert signal.entry == prediction.reference_price
    assert signal.stop_loss == 3300
    assert signal.take_profit == 3350
    assert signal.risk_reward_ratio is not None
    assert signal.risk_reward_ratio > 2
    assert signal.analysis_confidence == 80
    assert signal.confidence == round(signal.signal_confidence)


@pytest.mark.asyncio
async def test_replay_of_identical_point_in_time_inputs_is_byte_stable() -> None:
    state, quant = await state_and_quant()
    generator = DeterministicAnalysisSignalGenerator()

    first = generator.generate(analysis(5), state, quant)
    second = generator.generate(analysis(5), state, quant)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.asyncio
async def test_degraded_source_data_caps_confidence_at_59() -> None:
    state, quant = await state_and_quant()
    degraded = state.model_copy(
        update={"degraded_evidence": (state.evidence[0].evidence_id,)}
    )

    signal = DeterministicAnalysisSignalGenerator().generate(
        analysis(5),
        degraded,
        quant,
    )

    assert signal.confidence <= 59
    assert "stale_or_degraded_source_data" in signal.risk_flags


@pytest.mark.asyncio
async def test_signal_persistence_detects_conflicting_duplicate() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    signal = DeterministicAnalysisSignalGenerator().generate(
        analysis(5),
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
    repository = InMemoryAIReasoningRepository()
    completed_analysis = analysis(5)
    incomplete_newer_analysis = analysis(10)
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
    repository = InMemoryAIReasoningRepository()
    first_analysis = analysis(5)
    second_analysis = analysis(10)
    first = DeterministicAnalysisSignalGenerator().generate(
        first_analysis,
        state,
        quant,
    )
    second = DeterministicAnalysisSignalGenerator().generate(
        second_analysis,
        state,
        quant,
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
