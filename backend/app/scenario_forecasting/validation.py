"""Deterministic safety validation for scenario-generated geometry."""

from __future__ import annotations

from dataclasses import dataclass

from .models import GeometryValidity, PriceZone, ScenarioDirection, ScenarioGeometry


@dataclass(frozen=True)
class GeometryCandidate:
    direction: ScenarioDirection
    reference_price: float
    entry_zone: PriceZone
    entry: float
    stop_loss: float
    take_profit: float
    secondary_target: float | None
    expected_move: float
    maximum_entry_distance: float
    minimum_risk_reward: float
    basis_fact_identifiers: tuple[str, ...]


def validate_geometry(
    candidate: GeometryCandidate,
) -> tuple[GeometryValidity, ScenarioGeometry | None, str | None]:
    if candidate.direction not in {ScenarioDirection.BULLISH, ScenarioDirection.BEARISH}:
        return GeometryValidity.UNAVAILABLE, None, "scenario_direction_is_not_tradeable"
    if not candidate.basis_fact_identifiers:
        return GeometryValidity.UNAVAILABLE, None, "structural_basis_unavailable"
    if abs(candidate.entry - candidate.reference_price) > candidate.maximum_entry_distance:
        return GeometryValidity.NOT_EXECUTABLE, None, "entry_not_realistically_reachable"
    if not candidate.entry_zone.low <= candidate.entry <= candidate.entry_zone.high:
        return GeometryValidity.NOT_EXECUTABLE, None, "entry_outside_executable_zone"
    if candidate.direction == ScenarioDirection.BULLISH:
        if not candidate.stop_loss < candidate.entry < candidate.take_profit:
            return GeometryValidity.UNAVAILABLE, None, "invalid_buy_geometry_ordering"
        if candidate.take_profit <= candidate.reference_price:
            return GeometryValidity.NOT_EXECUTABLE, None, "buy_target_already_traversed"
        if candidate.stop_loss >= candidate.reference_price:
            return GeometryValidity.NOT_EXECUTABLE, None, "buy_invalidation_already_traversed"
    else:
        if not candidate.take_profit < candidate.entry < candidate.stop_loss:
            return GeometryValidity.UNAVAILABLE, None, "invalid_sell_geometry_ordering"
        if candidate.take_profit >= candidate.reference_price:
            return GeometryValidity.NOT_EXECUTABLE, None, "sell_target_already_traversed"
        if candidate.stop_loss <= candidate.reference_price:
            return GeometryValidity.NOT_EXECUTABLE, None, "sell_invalidation_already_traversed"
    risk = abs(candidate.entry - candidate.stop_loss)
    reward = abs(candidate.take_profit - candidate.entry)
    if risk <= 0 or reward <= 0:
        return GeometryValidity.UNAVAILABLE, None, "zero_risk_or_reward"
    risk_reward = reward / risk
    if risk_reward < candidate.minimum_risk_reward:
        return GeometryValidity.UNAVAILABLE, None, "risk_reward_below_minimum"
    if reward > candidate.expected_move * 1.5:
        return GeometryValidity.UNAVAILABLE, None, "target_exceeds_expected_scenario_move"
    geometry = ScenarioGeometry(
        entry_zone=candidate.entry_zone,
        entry=round(candidate.entry, 6),
        stop_loss=round(candidate.stop_loss, 6),
        take_profit=round(candidate.take_profit, 6),
        secondary_target=(
            round(candidate.secondary_target, 6)
            if candidate.secondary_target is not None
            else None
        ),
        risk_reward_ratio=round(risk_reward, 6),
        basis_fact_identifiers=candidate.basis_fact_identifiers,
        validity=GeometryValidity.VALID,
        reason="deterministic_forward_geometry_validated",
    )
    return GeometryValidity.VALID, geometry, None
