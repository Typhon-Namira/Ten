from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from backend.app.quant_forecasting.models import ForecastValueUnit
from backend.app.market_state.repository import InMemoryUnifiedMarketStateRepository
from backend.app.engines.signal_decision_engine import (
    ConservativeSignalDecisionPolicy,
    FinalSignalAction,
)
from backend.app.engines.market_data_engine.models import Candle, Timeframe
from backend.app.scenario_forecasting import InMemoryScenarioForecastRepository
from backend.app.scenario_forecasting.instrument import (
    InstrumentSpecification,
    convert_prediction_move,
)
from backend.app.scenario_forecasting.simulation_engine import (
    MarketSimulationConfig,
    MarketSimulationEngine,
)
from backend.app.scenario_forecasting.simulation_models import (
    CandidateDirection,
    ScenarioSignalAction,
    SelectionStatus,
    SimulationAttemptStatus,
)
from backend.app.scenario_forecasting.simulation_repository import (
    InMemoryMarketSimulationRepository,
)
from backend.app.scenario_forecasting.simulation_service import MarketSimulationService
from backend.app.scenario_forecasting.models import GeometryValidity, ScenarioValidity
from tests.scenario_forecasting.test_scenario_forecasting import scenario_inputs
from tests.engines.signal_decision_engine.test_signal_decision_engine import (
    ai_score,
    decision_input,
)
from tests.signal_synthesis.test_multi_timeframe_signal import aligned_analysis


@pytest.mark.asyncio
async def test_decimal_return_is_converted_to_xauusd_price_points() -> None:
    _, quant, _ = await scenario_inputs()
    prediction = next(
        item for item in quant.predictions if item.horizon.timeframe == "M15"
    ).model_copy(
        update={
            "reference_price": 4098.78,
            "expected_base_movement": 0.001,
            "expected_minimum_movement": 0.0005,
            "expected_maximum_movement": 0.002,
            "expected_base_movement_unit": ForecastValueUnit.DECIMAL_RETURN,
        }
    )

    conversion = convert_prediction_move(prediction, InstrumentSpecification())

    assert conversion.raw_expected_move == 0.001
    assert conversion.raw_expected_move_unit == "decimal_return"
    assert conversion.converted_expected_move == pytest.approx(4.09878)
    assert conversion.converted_expected_move_unit == "price_points"
    assert conversion.conversion_method == "reference_price_times_decimal_return"


@pytest.mark.asyncio
async def test_normalized_quant_output_is_never_treated_as_price() -> None:
    _, quant, _ = await scenario_inputs()
    prediction = quant.predictions[0].model_copy(
        update={"expected_base_movement_unit": ForecastValueUnit.NORMALIZED}
    )
    with pytest.raises(ValueError, match="unsupported_quant_movement_unit"):
        convert_prediction_move(prediction, InstrumentSpecification())


@pytest.mark.asyncio
async def test_authoritative_m15_cycle_generates_distinct_ranked_candidates() -> None:
    state, quant, synthesis = await scenario_inputs()

    simulation, selection = MarketSimulationEngine().simulate(
        state, quant, synthesis
    )

    assert 5 <= simulation.candidate_count <= 10
    assert len({item.diversity_key for item in simulation.candidates}) == len(
        simulation.candidates
    )
    assert tuple(item.rank for item in simulation.candidates) == tuple(
        range(1, simulation.candidate_count + 1)
    )
    assert {
        CandidateDirection.BULLISH,
        CandidateDirection.BEARISH,
        CandidateDirection.RANGE,
    }.issubset({item.direction for item in simulation.candidates})
    assert all(len(item.path_sequence) >= 2 for item in simulation.candidates)
    assert all(item.score_components for item in simulation.candidates)
    assert selection.status in {
        SelectionStatus.SELECTED,
        SelectionStatus.INSUFFICIENT_CONFIDENCE,
    }


@pytest.mark.asyncio
async def test_selected_primary_is_only_geometry_and_signal_authority() -> None:
    state, quant, synthesis = await scenario_inputs()
    _, selection = MarketSimulationEngine(
        MarketSimulationConfig(primary_scenario_threshold=0)
    ).simulate(state, quant, synthesis)

    assert selection.status == SelectionStatus.SELECTED
    assert selection.primary is not None
    assert selection.alternative is not None
    assert selection.primary_candidate_id != selection.alternative_candidate_id
    expected = (
        ScenarioSignalAction.BUY
        if selection.primary.direction == CandidateDirection.BULLISH
        else ScenarioSignalAction.SELL
    )
    assert selection.authoritative_action == expected
    assert selection.primary.geometry_validity == GeometryValidity.VALID
    assert selection.primary.geometry is not None


@pytest.mark.asyncio
async def test_below_threshold_fails_closed_without_geometry_publication() -> None:
    state, quant, synthesis = await scenario_inputs()
    _, selection = MarketSimulationEngine(
        MarketSimulationConfig(primary_scenario_threshold=100)
    ).simulate(state, quant, synthesis)

    assert selection.status == SelectionStatus.INSUFFICIENT_CONFIDENCE
    assert selection.authoritative_action == ScenarioSignalAction.HOLD
    assert selection.primary is None
    assert not selection.signal_eligible


@pytest.mark.asyncio
async def test_liquidity_sweep_label_requires_real_liquidity_evidence() -> None:
    state, quant, synthesis = await scenario_inputs()
    state = state.model_copy(
        update={
            "evidence": tuple(
                item.model_copy(update={"source_engine": "unavailable_source"})
                if "liquidity" in item.source_engine.lower()
                else item
                for item in state.evidence
            )
        }
    )
    simulation, _ = MarketSimulationEngine().simulate(state, quant, synthesis)
    sweeps = [
        item for item in simulation.candidates if "liquidity_sweep" in item.scenario_type
    ]

    assert len(sweeps) == 2
    for candidate in sweeps:
        assert candidate.scenario_validity == ScenarioValidity.INVALID
        assert candidate.geometry is None
        assert (
            candidate.rejection_reason
            == "liquidity_sweep_definition_missing_valid_liquidity_evidence"
        )


@pytest.mark.asyncio
async def test_repository_is_idempotent_per_market_state() -> None:
    state, quant, synthesis = await scenario_inputs()
    simulation, selection = MarketSimulationEngine(
        MarketSimulationConfig(primary_scenario_threshold=0)
    ).simulate(state, quant, synthesis)
    repository = InMemoryMarketSimulationRepository()

    first = await repository.save(simulation, selection)
    second = await repository.save(simulation, selection)

    assert first == second
    assert len(repository.simulations) == 1
    assert len(await repository.candidates(simulation.simulation_cycle_id)) == 7


@pytest.mark.asyncio
async def test_stale_input_and_future_history_cannot_affect_simulation() -> None:
    state, quant, synthesis = await scenario_inputs()
    stale = state.model_copy(
        update={
            "timeframes": tuple(
                item.model_copy(update={"stale": True})
                if item.timeframe == "M5"
                else item
                for item in state.timeframes
            )
        }
    )
    with pytest.raises(ValueError, match="fresh synchronized"):
        MarketSimulationEngine().simulate(stale, quant, synthesis)

    scenario, outcome = (await scenario_inputs())[0:2]
    # Unknown history types are intentionally ignored after their completion boundary.
    simulation, _ = MarketSimulationEngine().simulate(
        state,
        quant,
        synthesis,
        completed_history=((scenario, outcome),),
    )
    assert simulation.market_cutoff == state.market_data_boundary


@pytest.mark.asyncio
async def test_geometry_remains_display_safe_and_expires_with_m15_horizon() -> None:
    state, quant, synthesis = await scenario_inputs()
    simulation, _ = MarketSimulationEngine(
        MarketSimulationConfig(primary_scenario_threshold=0)
    ).simulate(state, quant, synthesis)
    valid = [
        item
        for item in simulation.candidates
        if item.geometry_validity == GeometryValidity.VALID
    ]
    assert valid
    for candidate in valid:
        assert candidate.expiry == state.market_data_boundary + timedelta(minutes=15)
        assert candidate.geometry is not None
        displayed = {
            round(candidate.geometry.entry, 2),
            round(candidate.geometry.stop_loss, 2),
            round(candidate.geometry.take_profit, 2),
        }
        assert len(displayed) == 3
        assert candidate.geometry.risk_reward_ratio >= 2


@pytest.mark.asyncio
async def test_expired_primary_scenario_outcome_is_persisted_without_lookahead() -> None:
    state, quant, synthesis = await scenario_inputs()
    engine = MarketSimulationEngine(
        MarketSimulationConfig(primary_scenario_threshold=0)
    )
    simulation, selection = engine.simulate(state, quant, synthesis)
    repository = InMemoryMarketSimulationRepository()
    await repository.save(simulation, selection)
    assert selection.primary is not None
    primary = selection.primary
    bullish = primary.direction == CandidateDirection.BULLISH
    close = primary.reference_price + (0.5 if bullish else -0.5)
    candle = Candle(
        timestamp=primary.market_cutoff + timedelta(minutes=5),
        timeframe=Timeframe.M5,
        open=primary.reference_price,
        high=max(primary.reference_price, close),
        low=min(primary.reference_price, close),
        close=close,
    )
    future = candle.model_copy(
        update={
            "timestamp": primary.expiry + timedelta(minutes=5),
            "high": 9999,
            "low": 1,
        }
    )
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        engine,
    )

    await service._evaluate_expired(
        state.instrument,
        (candle, future),
        primary.expiry,
    )

    outcome = repository.outcomes[primary.candidate_id]
    assert outcome.actual_high != 9999
    assert outcome.actual_low != 1
    assert outcome.status == "DIRECTION_CORRECT"


@pytest.mark.asyncio
async def test_primary_scenario_controls_final_direction_and_geometry() -> None:
    state, quant, synthesis = await scenario_inputs()
    _, selection = MarketSimulationEngine(
        MarketSimulationConfig(primary_scenario_threshold=0)
    ).simulate(state, quant, synthesis)
    analysis = aligned_analysis(state, quant)
    score = ai_score(
        as_of=state.market_data_boundary,
        calculated_at=state.market_data_boundary,
    )
    value = decision_input(
        score=score,
        as_of=state.market_data_boundary,
    ).model_copy(
        update={
            "current_ai_analysis": analysis,
            "market_snapshot_id": state.state_id,
            "quantitative_forecast_id": quant.result_id,
            "current_price": selection.primary.reference_price
            if selection.primary
            else None,
            "expected_move": selection.primary.expected_move
            if selection.primary
            else None,
            "current_primary_scenario": selection,
        }
    )

    decision = ConservativeSignalDecisionPolicy().evaluate(value)

    assert decision.final_action == FinalSignalAction(
        selection.authoritative_action.value
    )
    assert decision.source_lineage is not None
    assert (
        decision.source_lineage.primary_scenario_selection_id
        == selection.selection_id
    )
    assert decision.publication_eligible
    assert decision.entry_low is not None
    assert decision.stop_loss is not None
    assert decision.take_profit_targets
    assert decision.notification_context is not None
    assert (
        decision.notification_context["primary_scenario_id"]
        == str(selection.primary_candidate_id)
    )


@pytest.mark.asyncio
async def test_m15_close_is_terminal_and_duplicate_processing_is_idempotent() -> None:
    state, quant, synthesis = await scenario_inputs()
    repository = InMemoryMarketSimulationRepository()
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
    )

    first = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        evaluated_at=state.market_data_boundary,
    )
    second = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        evaluated_at=state.market_data_boundary + timedelta(seconds=1),
    )

    assert first is not None
    assert second == first
    assert len(repository.attempts) == 1
    attempt = await repository.latest_attempt(state.instrument)
    assert attempt is not None
    assert attempt.status.terminal
    assert attempt.candidate_count == 7
    assert len(repository.simulations) == 1


@pytest.mark.asyncio
async def test_m15_close_boundary_is_not_eligible_one_second_early() -> None:
    state, quant, synthesis = await scenario_inputs()
    repository = InMemoryMarketSimulationRepository()
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(),
    )

    result = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        evaluated_at=state.market_data_boundary - timedelta(seconds=1),
    )

    assert result is None
    attempt = await repository.latest_attempt(state.instrument)
    assert attempt is not None
    assert attempt.status == SimulationAttemptStatus.SKIPPED
    assert attempt.skip_reason == "M15_CANDLE_NOT_CLOSED"


@pytest.mark.asyncio
async def test_four_consecutive_m15_closes_each_get_one_terminal_cycle() -> None:
    base_state, base_quant, base_synthesis = await scenario_inputs()
    repository = InMemoryMarketSimulationRepository()
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
    )
    cutoffs = tuple(
        base_state.market_data_boundary + timedelta(minutes=15 * index)
        for index in range(4)
    )

    for cutoff in cutoffs:
        state_id = uuid4()
        cycle_id = uuid4()
        frames = tuple(
            frame.model_copy(
                update={
                    "frame_id": uuid4(),
                    "source_candle_open_at": cutoff
                    - timedelta(minutes=5 if frame.timeframe == "M5" else 15),
                    "source_candle_close_at": cutoff,
                    "expected_candle_close_at": cutoff,
                }
            )
            for frame in base_state.timeframes
        )
        state = base_state.model_copy(
            update={
                "state_id": state_id,
                "cycle_id": cycle_id,
                "market_data_boundary": cutoff,
                "knowledge_cutoff": cutoff,
                "created_at": cutoff,
                "timeframes": frames,
            }
        )
        quant = base_quant.model_copy(
            update={
                "result_id": uuid4(),
                "market_state_id": state_id,
                "cycle_id": cycle_id,
                "point_in_time": cutoff,
                "created_at": cutoff,
            }
        )
        synthesis = base_synthesis.model_copy(
            update={
                "synthesis_id": uuid4(),
                "market_state_id": state_id,
                "cycle_id": cycle_id,
                "analysis_id": uuid4(),
                "market_cutoff": cutoff,
                "created_at": cutoff,
            }
        )
        await service.process(
            state,
            quant,
            synthesis,
            trigger_timeframe="M15",
            evaluated_at=cutoff,
        )

    attempts = await repository.recent_attempts(base_state.instrument)
    assert tuple(item.market_cutoff for item in reversed(attempts)) == cutoffs
    assert all(item.status.terminal for item in attempts)
    assert all(item.candidate_count == 7 for item in attempts)
    assert len(repository.simulations) == 4


@pytest.mark.asyncio
async def test_startup_recovery_processes_latest_missing_m15_from_persisted_inputs() -> None:
    state, quant, synthesis = await scenario_inputs()
    state = state.model_copy(update={"trigger_timeframe": "M15"})
    repository = InMemoryMarketSimulationRepository()
    state_repository = AsyncMock()
    state_repository.latest_state.return_value = state
    quant_repository = AsyncMock()
    quant_repository.result_for_state.return_value = quant
    synthesis_repository = AsyncMock()
    synthesis_repository.for_state.return_value = synthesis
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
        market_state_repository=state_repository,
        quant_repository=quant_repository,
        synthesis_repository=synthesis_repository,
    )

    recovered = await service.recover_latest(
        state.instrument,
        now=state.market_data_boundary + timedelta(minutes=1),
    )

    assert recovered is not None
    state_repository.latest_state.assert_awaited_once_with(
        state.instrument,
        trigger_timeframe="M15",
    )
    attempt = await repository.latest_attempt(state.instrument)
    assert attempt is not None and attempt.status.terminal
    assert attempt.market_cutoff == state.market_data_boundary


@pytest.mark.asyncio
async def test_startup_recovery_reopens_simulation_not_invoked() -> None:
    state, quant, synthesis = await scenario_inputs()
    state = state.model_copy(update={"trigger_timeframe": "M15"})
    repository = InMemoryMarketSimulationRepository()
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
        market_state_repository=AsyncMock(),
        quant_repository=AsyncMock(),
        synthesis_repository=AsyncMock(),
    )
    service.market_state_repository.latest_state.return_value = state
    service.quant_repository.result_for_state.return_value = quant
    service.synthesis_repository.for_state.return_value = synthesis
    await service.record_blocked_cutoff(
        instrument=state.instrument,
        market_cutoff=state.market_data_boundary,
        server_time=state.market_data_boundary,
        reason="SIMULATION_NOT_INVOKED",
        market_state_id=state.state_id,
        quantitative_forecast_id=quant.result_id,
    )

    recovered = await service.recover_latest(
        state.instrument,
        now=state.market_data_boundary + timedelta(minutes=1),
    )

    assert recovered is not None
    attempt = await repository.latest_attempt(state.instrument)
    assert attempt is not None
    assert attempt.status.terminal
    assert attempt.failure_type is None


@pytest.mark.asyncio
async def test_newer_m5_state_does_not_hide_latest_recoverable_m15_state() -> None:
    state, _, _ = await scenario_inputs()
    m15_state = state.model_copy(
        update={
            "state_id": uuid4(),
            "trigger_timeframe": "M15",
        }
    )
    newer_m5_state = state.model_copy(
        update={
            "state_id": uuid4(),
            "market_data_boundary": state.market_data_boundary
            + timedelta(minutes=5),
            "trigger_timeframe": "M5",
        }
    )
    repository = InMemoryUnifiedMarketStateRepository()
    await repository.save_state(m15_state)
    await repository.save_state(newer_m5_state)

    latest_m15 = await repository.latest_state(
        state.instrument,
        trigger_timeframe="M15",
    )

    assert latest_m15 == m15_state


@pytest.mark.asyncio
async def test_scenario_waits_until_exact_authoritative_analysis_is_committed() -> None:
    state, quant, synthesis = await scenario_inputs()
    repository = InMemoryMarketSimulationRepository()
    ai_repository = AsyncMock()
    ai_repository.analysis_for_state.return_value = None
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(),
        ai_analysis_repository=ai_repository,
        ai_lookup_retry_delays_seconds=(),
    )

    result = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        evaluated_at=state.market_data_boundary,
    )

    assert result is None
    attempt = await repository.attempt_at_cutoff(
        state.instrument, state.market_data_boundary
    )
    assert attempt is not None
    assert attempt.status == SimulationAttemptStatus.WAITING_FOR_AI_ANALYSIS
    assert attempt.started_at is None
    assert repository.simulations == {}
    ai_repository.analysis_for_state.assert_awaited_once_with(state.state_id)


@pytest.mark.asyncio
async def test_delayed_analysis_resumes_same_idempotent_attempt_to_success() -> None:
    state, quant, synthesis = await scenario_inputs()
    analysis = aligned_analysis(state, quant)
    synthesis = synthesis.model_copy(update={"analysis_id": analysis.analysis_id})
    repository = InMemoryMarketSimulationRepository()
    ai_repository = AsyncMock()
    ai_repository.analysis_for_state.return_value = None
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
        ai_analysis_repository=ai_repository,
        ai_lookup_retry_delays_seconds=(),
    )

    await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        evaluated_at=state.market_data_boundary,
    )
    waiting = await repository.latest_attempt(state.instrument)
    assert waiting is not None
    ai_repository.analysis_for_state.return_value = analysis

    recovered = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        evaluated_at=state.market_data_boundary + timedelta(seconds=2),
    )

    assert recovered is not None
    terminal = await repository.latest_attempt(state.instrument)
    assert terminal is not None
    assert terminal.attempt_id == waiting.attempt_id
    assert terminal.status == SimulationAttemptStatus.SUCCESS
    assert terminal.ai_analysis_id == analysis.analysis_id
    assert terminal.ai_analysis_cutoff == state.market_data_boundary
    assert len(repository.attempts) == 1
    assert len(repository.simulations) == 1


@pytest.mark.asyncio
async def test_m5_cycle_is_monitoring_only_and_keeps_active_primary_authority() -> None:
    state, quant, synthesis = await scenario_inputs()
    repository = InMemoryMarketSimulationRepository()
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
    )
    active = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        evaluated_at=state.market_data_boundary,
    )

    monitored = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M5",
        evaluated_at=state.market_data_boundary + timedelta(minutes=5),
    )

    assert active is not None
    assert monitored is None
    assert await repository.latest(state.instrument) == active
    assert len(repository.selections) == 1


@pytest.mark.asyncio
async def test_concurrent_workers_create_only_one_simulation_for_cutoff() -> None:
    state, quant, synthesis = await scenario_inputs()
    analysis = aligned_analysis(state, quant)
    synthesis = synthesis.model_copy(update={"analysis_id": analysis.analysis_id})
    repository = InMemoryMarketSimulationRepository()
    ai_repository = AsyncMock()
    ai_repository.analysis_for_state.return_value = analysis
    service = MarketSimulationService(
        repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
        ai_analysis_repository=ai_repository,
        ai_lookup_retry_delays_seconds=(),
    )

    results = await asyncio.gather(
        *(
            service.process(
                state,
                quant,
                synthesis,
                trigger_timeframe="M15",
                evaluated_at=state.market_data_boundary,
            )
            for _ in range(2)
        )
    )

    assert sum(item is not None for item in results) >= 1
    assert len(repository.attempts) == 1
    assert len(repository.simulations) == 1
    attempt = await repository.latest_attempt(state.instrument)
    assert attempt is not None and attempt.status == SimulationAttemptStatus.SUCCESS
