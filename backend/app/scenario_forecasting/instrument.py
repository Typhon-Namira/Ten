"""Instrument-aware unit conversion, precision and display-safe geometry."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from pydantic import BaseModel, ConfigDict, Field

from backend.app.quant_forecasting.models import ForecastValueUnit, HorizonPrediction

from .models import GeometryValidity, PriceZone, ScenarioDirection, ScenarioGeometry
from .simulation_models import QuantMoveConversion
from .validation import GeometryCandidate, validate_geometry


class InstrumentSpecification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = "XAUUSD"
    tick_size: float = Field(default=0.01, gt=0)
    display_precision: int = Field(default=2, ge=0, le=8)
    minimum_stop_distance: float = Field(default=0.10, gt=0)
    minimum_target_distance: float = Field(default=0.20, gt=0)
    minimum_meaningful_move: float = Field(default=0.10, gt=0)
    maximum_expected_move_percent: float = Field(default=0.02, gt=0, le=0.25)

    def round_price(self, value: float) -> float:
        quantum = Decimal("1").scaleb(-self.display_precision)
        return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def convert_prediction_move(
    prediction: HorizonPrediction,
    specification: InstrumentSpecification,
) -> QuantMoveConversion:
    raw = prediction.expected_base_movement
    unit = prediction.expected_base_movement_unit
    if unit == ForecastValueUnit.DECIMAL_RETURN:
        converted = prediction.reference_price * raw
        method = "reference_price_times_decimal_return"
    elif unit == ForecastValueUnit.PERCENT:
        converted = prediction.reference_price * raw / 100
        method = "reference_price_times_percentage_divided_by_100"
    elif unit == ForecastValueUnit.PRICE_POINTS:
        converted = raw
        method = "absolute_price_points_preserved"
    else:
        raise ValueError(f"unsupported_quant_movement_unit:{unit.value}")
    if converted < specification.minimum_meaningful_move:
        raise ValueError("converted_expected_move_below_instrument_minimum")
    if converted > prediction.reference_price * specification.maximum_expected_move_percent:
        raise ValueError("converted_expected_move_above_instrument_maximum")
    return QuantMoveConversion(
        raw_expected_move=raw,
        raw_expected_move_unit=unit.value,
        converted_expected_move=converted,
        conversion_method=method,
        reference_price=prediction.reference_price,
    )


def validate_display_geometry(
    *,
    direction: ScenarioDirection,
    reference_price: float,
    entry_zone: PriceZone,
    entry: float,
    stop_loss: float,
    take_profit: float,
    secondary_target: float | None,
    expected_move: float,
    maximum_entry_distance: float,
    minimum_risk_reward: float,
    basis_fact_identifiers: tuple[str, ...],
    spread: float,
    specification: InstrumentSpecification,
) -> tuple[GeometryValidity, ScenarioGeometry | None, str | None]:
    minimum_stop = max(
        specification.minimum_stop_distance,
        specification.tick_size * 2,
        spread * 2,
    )
    if abs(entry - stop_loss) < minimum_stop:
        return GeometryValidity.UNAVAILABLE, None, "stop_distance_below_instrument_minimum"
    if abs(take_profit - entry) < specification.minimum_target_distance:
        return GeometryValidity.UNAVAILABLE, None, "target_distance_below_instrument_minimum"
    status, geometry, reason = validate_geometry(
        GeometryCandidate(
            direction=direction,
            reference_price=reference_price,
            entry_zone=entry_zone,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            secondary_target=secondary_target,
            expected_move=expected_move,
            maximum_entry_distance=maximum_entry_distance,
            minimum_risk_reward=minimum_risk_reward,
            basis_fact_identifiers=basis_fact_identifiers,
        )
    )
    if geometry is None:
        return status, None, reason
    rounded_entry = specification.round_price(geometry.entry)
    rounded_stop = specification.round_price(geometry.stop_loss)
    rounded_target = specification.round_price(geometry.take_profit)
    if len({rounded_entry, rounded_stop, rounded_target}) != 3:
        return GeometryValidity.UNAVAILABLE, None, "display_rounding_collapsed_geometry"
    if direction == ScenarioDirection.BULLISH and not (
        rounded_stop < rounded_entry < rounded_target
    ):
        return GeometryValidity.UNAVAILABLE, None, "display_rounding_invalidated_buy_ordering"
    if direction == ScenarioDirection.BEARISH and not (
        rounded_target < rounded_entry < rounded_stop
    ):
        return GeometryValidity.UNAVAILABLE, None, "display_rounding_invalidated_sell_ordering"
    displayed_rr = abs(rounded_target - rounded_entry) / abs(
        rounded_entry - rounded_stop
    )
    if displayed_rr < minimum_risk_reward:
        return GeometryValidity.UNAVAILABLE, None, "displayed_risk_reward_below_minimum"
    return status, geometry, None
