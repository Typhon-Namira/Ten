from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from backend.app.engines.market_data_engine.models import Candle, Timeframe
from backend.app.scenario_forecasting import (
    GeometryValidity,
    InMemoryScenarioForecastRepository,
    ScenarioAgreement,
    ScenarioDirection,
    ScenarioForecastingEngine,
    ScenarioForecastingService,
)
from backend.app.scenario_forecasting.models import PriceZone
from backend.app.scenario_forecasting.outcome_evaluation import (
    evaluate_expired_scenario,
)
from backend.app.scenario_forecasting.validation import (
    GeometryCandidate,
    validate_geometry,
)
from backend.app.signal_synthesis import MultiTimeframeSignalSynthesizer
from tests.ai_reasoning.test_ai_reasoning_lifecycle import state_and_quant
from tests.signal_synthesis.test_multi_timeframe_signal import (
    aligned_analysis,
    with_smc_zones,
)


async def scenario_inputs(*, bearish: bool = False):
    state, quant = await state_and_quant()
    for timeframe in ("M5", "M15"):
        state = with_smc_zones(
            state,
            timeframe,
            [
                {
                    "id": f"active-{timeframe}",
                    "zone_type": "bullish_order_block",
                    "direction": "bullish",
                    "lifecycle_state": "active",
                    "lower_price": 3301.998,
                    "upper_price": 3302.0,
                    "mitigation_percentage": 0,
                    "quality_score": 100,
                    "source_candle_ids": [f"source-{timeframe}"],
                }
            ],
        )
    if bearish:
        predictions = tuple(
            item.model_copy(
                update={
                    "buy_probability": item.sell_probability,
                    "sell_probability": item.buy_probability,
                    "expected_return": -abs(item.expected_return),
                }
            )
            for item in quant.predictions
        )
        quant = quant.model_copy(update={"predictions": predictions})
    synthesis = MultiTimeframeSignalSynthesizer().synthesize(
        state,
        quant,
        aligned_analysis(state, quant),
    )
    return state, quant, synthesis


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeframe", "bearish", "direction"),
    [
        ("M5", False, ScenarioDirection.BULLISH),
        ("M5", True, ScenarioDirection.BEARISH),
        ("M15", False, ScenarioDirection.BULLISH),
        ("M15", True, ScenarioDirection.BEARISH),
    ],
)
async def test_completed_horizon_produces_structured_scenario(
    timeframe: str,
    bearish: bool,
    direction: ScenarioDirection,
) -> None:
    state, quant, synthesis = await scenario_inputs(bearish=bearish)
    result = ScenarioForecastingEngine().forecast(
        state, quant, synthesis, timeframe
    )

    assert result.primary_direction == direction
    assert result.expected_range.low <= result.reference_market_price <= result.expected_range.high
    assert result.expected_price_path
    assert result.invalidation_level is not None
    assert result.expiry == result.market_cutoff_time + timedelta(
        seconds={"M5": 300, "M15": 900}[timeframe]
    )
    assert result.execution_geometry_validity == GeometryValidity.VALID
    assert result.geometry is not None
    if bearish:
        assert result.geometry.take_profit < result.geometry.entry < result.geometry.stop_loss
    else:
        assert result.geometry.stop_loss < result.geometry.entry < result.geometry.take_profit


@pytest.mark.asyncio
async def test_aligned_m5_m15_produces_combined_geometry() -> None:
    state, quant, synthesis = await scenario_inputs()
    engine = ScenarioForecastingEngine()
    m5 = engine.forecast(state, quant, synthesis, "M5")
    m15 = engine.forecast(state, quant, synthesis, "M15")

    combined = engine.combine(m5, m15)

    assert combined.agreement == ScenarioAgreement.ALIGNED
    assert combined.combined_direction == ScenarioDirection.BULLISH
    assert combined.execution_geometry_validity == GeometryValidity.VALID
    assert combined.geometry is not None


@pytest.mark.asyncio
async def test_true_conflict_never_fabricates_combined_geometry() -> None:
    state, quant, synthesis = await scenario_inputs()
    engine = ScenarioForecastingEngine()
    m5 = engine.forecast(state, quant, synthesis, "M5")
    _, bearish_quant, bearish_synthesis = await scenario_inputs(bearish=True)
    m15 = engine.forecast(state, bearish_quant, bearish_synthesis, "M15")

    combined = engine.combine(m5, m15)

    assert combined.agreement == ScenarioAgreement.CONFLICT
    assert combined.geometry is None
    assert combined.geometry_rejection_reason == "true_m5_m15_directional_conflict"


@pytest.mark.asyncio
async def test_m5_pullback_inside_m15_continuation_is_not_false_conflict() -> None:
    state, quant, synthesis = await scenario_inputs()
    engine = ScenarioForecastingEngine()
    m15 = engine.forecast(state, quant, synthesis, "M15")
    m5 = engine.forecast(state, quant, synthesis, "M5").model_copy(
        update={
            "primary_direction": ScenarioDirection.BEARISH,
            "expected_low": m15.invalidation_level + 0.001,
            "expected_range": PriceZone(
                low=m15.invalidation_level + 0.001,
                high=3302.001,
            ),
            "execution_geometry_validity": GeometryValidity.UNAVAILABLE,
            "geometry_rejection_reason": "pullback_analysis_only",
            "geometry": None,
        }
    )

    combined = engine.combine(m5, m15)

    assert combined.agreement == ScenarioAgreement.PULLBACK_COMPATIBLE
    assert combined.combined_direction == ScenarioDirection.BULLISH


def candidate(**updates):
    values = {
        "direction": ScenarioDirection.BULLISH,
        "reference_price": 3300.0,
        "entry_zone": PriceZone(low=3299.8, high=3300.2),
        "entry": 3300.0,
        "stop_loss": 3299.0,
        "take_profit": 3302.2,
        "secondary_target": None,
        "expected_move": 2.0,
        "maximum_entry_distance": 1.0,
        "minimum_risk_reward": 2.0,
        "basis_fact_identifiers": ("order-block-1",),
    }
    values.update(updates)
    return GeometryCandidate(**values)


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (candidate(entry_zone=PriceZone(low=3290, high=3291), entry=3290.5), "entry_not_realistically_reachable"),
        (candidate(take_profit=3299.5), "invalid_buy_geometry_ordering"),
        (candidate(stop_loss=3300.5), "invalid_buy_geometry_ordering"),
        (candidate(take_profit=3301.5), "risk_reward_below_minimum"),
        (candidate(expected_move=0.5), "target_exceeds_expected_scenario_move"),
        (candidate(basis_fact_identifiers=()), "structural_basis_unavailable"),
    ],
)
def test_invalid_geometry_is_rejected(value: GeometryCandidate, reason: str) -> None:
    status, geometry, rejection = validate_geometry(value)
    assert status != GeometryValidity.VALID
    assert geometry is None
    assert rejection == reason


@pytest.mark.asyncio
async def test_stale_or_incomplete_source_candle_is_rejected() -> None:
    state, quant, synthesis = await scenario_inputs()
    frames = tuple(
        item.model_copy(update={"stale": True})
        if item.timeframe == "M5"
        else item
        for item in state.timeframes
    )
    with pytest.raises(ValueError, match="fresh completed"):
        ScenarioForecastingEngine().forecast(
            state.model_copy(update={"timeframes": frames}),
            quant,
            synthesis,
            "M5",
        )


@pytest.mark.asyncio
async def test_expired_outcome_uses_only_post_cutoff_pre_expiry_candles() -> None:
    state, quant, synthesis = await scenario_inputs()
    scenario = ScenarioForecastingEngine().forecast(
        state, quant, synthesis, "M5"
    )
    before = Candle(
        timestamp=scenario.market_cutoff_time - timedelta(minutes=5),
        timeframe=Timeframe.M5,
        open=3302,
        high=9999,
        low=1,
        close=3302,
    )
    realized = Candle(
        timestamp=scenario.market_cutoff_time,
        timeframe=Timeframe.M5,
        open=3302,
        high=3302.003,
        low=3301.999,
        close=3302.002,
    )
    future = Candle(
        timestamp=scenario.expiry,
        timeframe=Timeframe.M5,
        open=3302,
        high=9999,
        low=1,
        close=3302,
    )

    assert evaluate_expired_scenario(
        scenario,
        (realized,),
        evaluated_at=scenario.expiry - timedelta(seconds=1),
    ) is None
    outcome = evaluate_expired_scenario(
        scenario,
        (before, realized, future),
        evaluated_at=scenario.expiry,
    )
    assert outcome is not None
    assert outcome.actual_high == realized.high
    assert outcome.actual_low == realized.low
    assert outcome.maximum_favorable_excursion == pytest.approx(0.003)
    assert outcome.maximum_adverse_excursion == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_service_persists_one_idempotent_scenario_and_m15_combination() -> None:
    state, quant, synthesis = await scenario_inputs()
    repository = InMemoryScenarioForecastRepository()
    service = ScenarioForecastingService(repository, ScenarioForecastingEngine())

    first, _ = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M5",
        candles=(),
        evaluated_at=state.market_data_boundary,
    )
    repeated, _ = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M5",
        candles=(),
        evaluated_at=state.market_data_boundary,
    )
    _, combined = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        candles=(),
        evaluated_at=state.market_data_boundary,
    )

    assert repeated.scenario_id == first.scenario_id
    assert len(repository.scenarios) == 2
    assert combined is not None
    assert combined.m5_scenario_id == first.scenario_id


@pytest.mark.asyncio
async def test_service_accepts_pipeline_correlation_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state, quant, synthesis = await scenario_inputs()
    service = ScenarioForecastingService(
        InMemoryScenarioForecastRepository(), ScenarioForecastingEngine()
    )
    correlation_id = uuid4()
    caplog.set_level("INFO", logger="backend.app.scenario_forecasting.service")

    scenario, _ = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M5",
        candles=(),
        evaluated_at=state.market_data_boundary,
        correlation_id=correlation_id,
    )

    assert scenario is not None
    record = next(
        item
        for item in caplog.records
        if item.getMessage() == "scenario_forecast.created"
    )
    assert record.correlation_id == str(correlation_id)


@pytest.mark.asyncio
async def test_m15_cycle_synthesizes_current_m5_before_combining_when_race_is_empty() -> None:
    state, quant, synthesis = await scenario_inputs()
    repository = InMemoryScenarioForecastRepository()
    service = ScenarioForecastingService(repository, ScenarioForecastingEngine())

    m15, combined = await service.process(
        state,
        quant,
        synthesis,
        trigger_timeframe="M15",
        candles=(),
        evaluated_at=state.market_data_boundary,
    )

    m5 = await repository.latest_scenario(state.instrument, "M5")
    assert m5 is not None
    assert m5.market_cutoff_time == m15.market_cutoff_time
    assert combined is not None
    assert combined.m5_scenario_id == m5.scenario_id


@pytest.mark.asyncio
async def test_historical_scenario_cannot_replace_current_scenario() -> None:
    state, quant, synthesis = await scenario_inputs()
    repository = InMemoryScenarioForecastRepository()
    current = ScenarioForecastingEngine().forecast(state, quant, synthesis, "M5")
    await repository.save_scenario(current)

    assert await repository.latest_scenario(
        state.instrument,
        "M5",
        at_or_before=state.market_data_boundary - timedelta(seconds=1),
    ) is None
    assert await repository.latest_scenario(state.instrument, "M5") == current
