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

    signal = DeterministicAnalysisSignalGenerator().generate(
        analysis(5, regime="bearish"),
        state,
        quant,
    )

    assert signal.signal == AnalysisSignalAction.SELL
    assert signal.take_profit is not None
    assert signal.entry is not None
    assert signal.stop_loss is not None
    assert signal.take_profit < signal.entry < signal.stop_loss


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

    latest = await repository.latest_completed_analysis_cycle("XAUUSD", "M15")

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
