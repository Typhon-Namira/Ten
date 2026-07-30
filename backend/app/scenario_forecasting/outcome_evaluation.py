"""Evaluate expired scenarios using only post-cutoff, pre-expiry candles."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from backend.app.engines.market_data_engine.models import Candle

from .models import (
    ForwardMarketScenario,
    ScenarioDirection,
    ScenarioOutcome,
    ScenarioOutcomeStatus,
)


def evaluate_expired_scenario(
    scenario: ForwardMarketScenario,
    candles: tuple[Candle, ...],
    *,
    evaluated_at: datetime,
) -> ScenarioOutcome | None:
    evaluated_at = evaluated_at.astimezone(UTC)
    if evaluated_at < scenario.expiry:
        return None
    usable = tuple(
        candle
        for candle in candles
        if candle.timeframe.value == scenario.timeframe
        and candle.timestamp >= scenario.market_cutoff_time
        and candle.timestamp + candle.timeframe.duration <= scenario.expiry
    )
    if not usable:
        return None
    actual_high = max(item.high for item in usable)
    actual_low = min(item.low for item in usable)
    actual_close = usable[-1].close
    bullish = scenario.primary_direction == ScenarioDirection.BULLISH
    bearish = scenario.primary_direction == ScenarioDirection.BEARISH
    mfe = (
        max(0.0, actual_high - scenario.reference_market_price)
        if bullish
        else max(0.0, scenario.reference_market_price - actual_low)
        if bearish
        else max(
            actual_high - scenario.reference_market_price,
            scenario.reference_market_price - actual_low,
        )
    )
    mae = (
        max(0.0, scenario.reference_market_price - actual_low)
        if bullish
        else max(0.0, actual_high - scenario.reference_market_price)
        if bearish
        else 0.0
    )
    geometry = scenario.geometry
    entry_reached = bool(
        geometry
        and actual_low <= geometry.entry_zone.high
        and actual_high >= geometry.entry_zone.low
    )
    target_reached = bool(
        geometry
        and (
            (bullish and actual_high >= geometry.take_profit)
            or (bearish and actual_low <= geometry.take_profit)
        )
    )
    invalidated = bool(
        scenario.invalidation_level
        and (
            (bullish and actual_low <= scenario.invalidation_level)
            or (bearish and actual_high >= scenario.invalidation_level)
        )
    )
    direction_correct = (
        bullish and actual_close > scenario.reference_market_price
    ) or (bearish and actual_close < scenario.reference_market_price)
    range_correct = (
        scenario.expected_range.low <= actual_low
        and actual_high <= scenario.expected_range.high
    )
    if target_reached:
        status = ScenarioOutcomeStatus.TARGET_REACHED
    elif invalidated:
        status = ScenarioOutcomeStatus.INVALIDATED
    elif geometry and not entry_reached:
        status = ScenarioOutcomeStatus.ENTRY_NOT_REACHED
    elif direction_correct:
        status = ScenarioOutcomeStatus.DIRECTION_CORRECT
    elif range_correct:
        status = ScenarioOutcomeStatus.RANGE_CORRECT
    elif scenario.primary_direction == ScenarioDirection.INCONCLUSIVE:
        status = ScenarioOutcomeStatus.INCONCLUSIVE
    else:
        status = ScenarioOutcomeStatus.EXPIRED
    target = scenario.primary_target or scenario.reference_market_price
    return ScenarioOutcome(
        outcome_id=uuid5(NAMESPACE_URL, f"ten:scenario-outcome:{scenario.scenario_id}"),
        scenario_id=scenario.scenario_id,
        evaluated_at=evaluated_at,
        completed_at=scenario.expiry,
        status=status,
        actual_high=actual_high,
        actual_low=actual_low,
        actual_close=actual_close,
        maximum_favorable_excursion=mfe,
        maximum_adverse_excursion=mae,
        entry_reached=entry_reached,
        target_reached=target_reached,
        invalidation_occurred=invalidated,
        directional_accuracy=1.0 if direction_correct else 0.0,
        target_error=abs(actual_close - target),
        range_error=abs(actual_low - scenario.expected_low)
        + abs(actual_high - scenario.expected_high),
        high_prediction_error=abs(actual_high - scenario.expected_high),
        low_prediction_error=abs(actual_low - scenario.expected_low),
        close_prediction_error=(
            0.0
            if scenario.expected_closing_zone.low
            <= actual_close
            <= scenario.expected_closing_zone.high
            else min(
                abs(actual_close - scenario.expected_closing_zone.low),
                abs(actual_close - scenario.expected_closing_zone.high),
            )
        ),
        calibration_bucket=f"{int(scenario.raw_directional_confidence * 10) * 10:02d}",
    )
