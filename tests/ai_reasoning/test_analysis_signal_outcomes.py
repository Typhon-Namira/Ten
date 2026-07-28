from __future__ import annotations

from datetime import timedelta

import pytest

from backend.app.ai_reasoning.analysis import AnalysisSignalLifecycle
from backend.app.ai_reasoning.signal import DeterministicAnalysisSignalGenerator
from backend.app.ai_reasoning.signal_outcomes import AnalysisSignalOutcomeEvaluator
from tests.ai_reasoning.test_ai_reasoning_lifecycle import state_and_quant
from tests.ai_reasoning.test_analysis_architecture_v2 import analysis


@pytest.mark.asyncio
async def test_target_hit_records_mfe_holding_time_rr_and_profit() -> None:
    state, quant = await state_and_quant()
    signal = DeterministicAnalysisSignalGenerator().generate(analysis(5), state, quant)
    assert signal.entry is not None
    assert signal.take_profit is not None
    evaluator = AnalysisSignalOutcomeEvaluator()
    initial = evaluator.initial(signal)

    entered = evaluator.evaluate(
        signal,
        initial,
        candle_low=signal.entry,
        candle_high=signal.entry,
        candle_close=signal.entry,
        evaluated_at=signal.generated_at + timedelta(minutes=1),
        superseded=False,
    )
    outcome = evaluator.evaluate(
        signal,
        entered,
        candle_low=signal.entry,
        candle_high=signal.take_profit,
        candle_close=signal.take_profit,
        evaluated_at=signal.generated_at + timedelta(minutes=2),
        superseded=False,
    )

    assert outcome.status == AnalysisSignalLifecycle.TARGET_HIT
    assert outcome.entry_reached
    assert outcome.target_hit
    assert outcome.profit_loss is not None and outcome.profit_loss > 0
    assert outcome.actual_risk_reward == pytest.approx(signal.risk_reward_ratio)
    assert outcome.maximum_favorable_excursion > 0


@pytest.mark.asyncio
async def test_same_candle_target_and_stop_resolves_conservatively() -> None:
    state, quant = await state_and_quant()
    signal = DeterministicAnalysisSignalGenerator().generate(analysis(5), state, quant)
    assert signal.stop_loss is not None
    assert signal.take_profit is not None
    evaluator = AnalysisSignalOutcomeEvaluator()

    outcome = evaluator.evaluate(
        signal,
        evaluator.initial(signal),
        candle_low=signal.stop_loss,
        candle_high=signal.take_profit,
        candle_close=signal.entry or signal.stop_loss,
        evaluated_at=signal.generated_at + timedelta(minutes=1),
        superseded=False,
    )

    assert outcome.status == AnalysisSignalLifecycle.STOP_HIT
    assert not outcome.target_hit
    assert outcome.actual_risk_reward == pytest.approx(-1)


@pytest.mark.asyncio
async def test_new_cycle_supersedes_unresolved_signal() -> None:
    state, quant = await state_and_quant()
    signal = DeterministicAnalysisSignalGenerator().generate(analysis(5), state, quant)
    assert signal.entry is not None
    evaluator = AnalysisSignalOutcomeEvaluator()

    outcome = evaluator.evaluate(
        signal,
        evaluator.initial(signal),
        candle_low=signal.entry,
        candle_high=signal.entry,
        candle_close=signal.entry,
        evaluated_at=signal.generated_at + timedelta(seconds=30),
        superseded=True,
    )

    assert outcome.status == AnalysisSignalLifecycle.SUPERSEDED
    assert outcome.completed_at is not None
