"""Deterministic scenario forecasting over completed authoritative outputs."""

from __future__ import annotations

from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field

from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import (
    ForecastValueUnit,
    HorizonPrediction,
    QuantForecastResult,
)
from backend.app.signal_synthesis.models import (
    AnalyticalDirection,
    MultiTimeframeSignalSet,
    TimeframeAnalyticalSignal,
)

from .models import (
    AlternativeScenario,
    CombinedForwardScenario,
    ForwardMarketScenario,
    GeometryValidity,
    PriceZone,
    ScenarioAgreement,
    ScenarioDirection,
    ScenarioGeometry,
    ScenarioValidity,
)
from .validation import GeometryCandidate, validate_geometry
from .instrument import InstrumentSpecification, convert_prediction_move


class ScenarioForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_directional_edge: float = Field(default=0.12, ge=0, le=1)
    minimum_scenario_confidence: float = Field(default=55, ge=0, le=100)
    minimum_evidence_completeness: float = Field(default=0.50, ge=0, le=1)
    minimum_risk_reward: float = Field(default=2.0, gt=0)
    maximum_entry_distance_percent: float = Field(default=0.003, gt=0, le=0.02)
    entry_distance_expected_move_fraction: float = Field(default=0.35, gt=0, le=1)
    stop_expected_move_fraction: float = Field(default=0.30, gt=0, le=1)
    target_expected_move_fraction: float = Field(default=0.80, gt=0, le=1.5)
    engine_version: str = "1.0.0"


class ScenarioForecastingEngine:
    """Build forward hypotheses without changing signal or publication owners."""

    def __init__(self, config: ScenarioForecastingConfig | None = None) -> None:
        self.config = config or ScenarioForecastingConfig()

    def forecast(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        synthesis: MultiTimeframeSignalSet,
        timeframe: str,
        *,
        calibrated_probability: float | None = None,
    ) -> ForwardMarketScenario:
        if timeframe not in {"M5", "M15"}:
            raise ValueError("scenario forecasting supports only completed M5/M15 cycles")
        if (
            state.state_id != quant.market_state_id
            or state.state_id != synthesis.market_state_id
            or state.cycle_id != quant.cycle_id
            or state.cycle_id != synthesis.cycle_id
        ):
            raise ValueError("scenario inputs must share one immutable market state")
        frame = next(item for item in state.timeframes if item.timeframe == timeframe)
        if frame.source_candle_close_at != frame.expected_candle_close_at or frame.stale:
            raise ValueError("scenario requires a fresh completed source candle")
        prediction = self._prediction(quant, timeframe)
        conversion = convert_prediction_move(prediction, InstrumentSpecification())
        factor = (
            prediction.reference_price
            if prediction.expected_base_movement_unit
            == ForecastValueUnit.DECIMAL_RETURN
            else prediction.reference_price / 100
            if prediction.expected_base_movement_unit == ForecastValueUnit.PERCENT
            else 1.0
        )
        prediction = prediction.model_copy(
            update={
                "expected_minimum_movement": prediction.expected_minimum_movement
                * factor,
                "expected_base_movement": conversion.converted_expected_move,
                "expected_maximum_movement": prediction.expected_maximum_movement
                * factor,
                "expected_base_movement_unit": ForecastValueUnit.PRICE_POINTS,
            }
        )
        signal = next(
            item for item in synthesis.timeframe_signals if item.timeframe == timeframe
        )
        cutoff = state.market_data_boundary
        if prediction.reference_price <= 0:
            raise ValueError("scenario requires synchronized reference price")
        direction, raw_confidence = self._direction(prediction)
        expected_move = prediction.expected_base_movement
        maximum_move = prediction.expected_maximum_movement
        if expected_move <= 0 or maximum_move <= 0:
            direction = ScenarioDirection.INCONCLUSIVE
        expected_range, close_zone = self._ranges(
            prediction.reference_price,
            expected_move,
            maximum_move,
            direction,
        )
        supporting, contradicting = self._facts(signal, direction)
        evidence_strength = signal.confidence_decomposition.independent_confluence
        confidence = self._confidence(signal, raw_confidence, state.evidence_completeness)
        validity, validity_reason = self._validity(
            direction,
            expected_move,
            confidence,
            state.evidence_completeness,
        )
        scenario_type = self._scenario_type(direction, signal)
        invalidation = self._invalidation_level(
            prediction.reference_price,
            expected_move,
            direction,
        )
        geometry_status, geometry, geometry_reason = self._geometry(
            prediction,
            signal,
            direction,
            supporting,
            confidence,
            validity,
        )
        target = geometry.take_profit if geometry else self._target(
            prediction.reference_price, expected_move, direction
        )
        alternative_direction = {
            ScenarioDirection.BULLISH: ScenarioDirection.BEARISH,
            ScenarioDirection.BEARISH: ScenarioDirection.BULLISH,
            ScenarioDirection.RANGE: ScenarioDirection.INCONCLUSIVE,
            ScenarioDirection.INCONCLUSIVE: ScenarioDirection.RANGE,
        }[direction]
        path = self._path(direction, scenario_type, prediction.reference_price, target)
        strongest = ", ".join(supporting[:3]) or "limited validated directional evidence"
        narrative = (
            f"{timeframe} evidence supports a {scenario_type} as the most probable path. "
            f"Strongest traceable facts: {strongest}. {path} "
            f"The scenario is invalidated beyond {invalidation:.3f}. "
            + (
                "Executable geometry passed deterministic validation."
                if geometry is not None
                else f"Analytical direction remains separate from execution: {geometry_reason}."
            )
        )
        scenario_id = uuid5(
            NAMESPACE_URL,
            f"ten:forward-scenario:{state.state_id}:{timeframe}:1.0",
        )
        horizon = {"M5": 300, "M15": 900}[timeframe]
        return ForwardMarketScenario(
            scenario_id=scenario_id,
            cycle_id=state.cycle_id,
            market_state_id=state.state_id,
            synthesis_id=synthesis.synthesis_id,
            analysis_id=synthesis.analysis_id,
            quantitative_forecast_id=quant.result_id,
            instrument=state.instrument,
            timeframe=timeframe,
            created_at=max(state.knowledge_cutoff, synthesis.created_at),
            market_cutoff_time=cutoff,
            reference_market_price=prediction.reference_price,
            forecast_horizon_seconds=horizon,
            primary_direction=direction,
            scenario_type=scenario_type,
            expected_price_path=path,
            expected_range=expected_range,
            expected_closing_zone=close_zone,
            expected_move=expected_move,
            expected_high=expected_range.high,
            expected_low=expected_range.low,
            entry_zone=geometry.entry_zone if geometry else None,
            invalidation_level=invalidation,
            protective_stop=geometry.stop_loss if geometry else None,
            primary_target=target,
            secondary_target=geometry.secondary_target if geometry else None,
            raw_directional_confidence=raw_confidence,
            confidence=confidence,
            calibrated_probability=calibrated_probability,
            calibration_status=(
                "calibrated"
                if calibrated_probability is not None
                else "calibration_pending"
            ),
            evidence_strength=evidence_strength,
            supporting_fact_ids=supporting,
            contradicting_fact_ids=contradicting,
            narrative=narrative,
            alternative_scenario=AlternativeScenario(
                direction=alternative_direction,
                scenario_type=f"{alternative_direction.value.lower()} alternative",
                expected_path=(
                    f"If price crosses {invalidation:.3f}, the primary scenario fails "
                    f"and a {alternative_direction.value.lower()} path becomes more probable."
                ),
                probability=max(0.0, min(1.0, 1.0 - raw_confidence)),
                invalidation=f"Primary scenario invalidated beyond {invalidation:.3f}",
            ),
            expiry=cutoff + timedelta(seconds=horizon),
            scenario_validity=validity,
            scenario_validity_reason=validity_reason,
            execution_geometry_validity=geometry_status,
            geometry_rejection_reason=geometry_reason,
            geometry=geometry,
            source_timeframe_cycle_id=state.cycle_id,
        )

    def combine(
        self,
        m5: ForwardMarketScenario,
        m15: ForwardMarketScenario,
    ) -> CombinedForwardScenario:
        if m5.instrument != m15.instrument:
            raise ValueError("combined scenarios must use one instrument")
        if m5.market_cutoff_time > m15.market_cutoff_time:
            raise ValueError("M5 scenario cannot originate after the M15 cutoff")
        if m5.expiry <= m15.market_cutoff_time:
            agreement = ScenarioAgreement.INCONCLUSIVE
        elif m5.primary_direction == m15.primary_direction and m15.primary_direction in {
            ScenarioDirection.BULLISH,
            ScenarioDirection.BEARISH,
        }:
            agreement = ScenarioAgreement.ALIGNED
        elif self._pullback_compatible(m5, m15):
            agreement = ScenarioAgreement.PULLBACK_COMPATIBLE
        elif ScenarioDirection.INCONCLUSIVE in {
            m5.primary_direction,
            m15.primary_direction,
        }:
            agreement = ScenarioAgreement.INCONCLUSIVE
        else:
            agreement = ScenarioAgreement.CONFLICT
        direction = (
            m15.primary_direction
            if agreement
            in {ScenarioAgreement.ALIGNED, ScenarioAgreement.PULLBACK_COMPATIBLE}
            else ScenarioDirection.INCONCLUSIVE
        )
        confidence = (
            m5.confidence * 0.4 + m15.confidence * 0.6
            if agreement != ScenarioAgreement.CONFLICT
            else min(m5.confidence, m15.confidence) * 0.5
        )
        geometry = None
        geometry_status = GeometryValidity.UNAVAILABLE
        reason: str | None = "m5_m15_scenarios_not_execution_compatible"
        preferred = m5.geometry if agreement == ScenarioAgreement.PULLBACK_COMPATIBLE else m15.geometry
        if (
            agreement in {ScenarioAgreement.ALIGNED, ScenarioAgreement.PULLBACK_COMPATIBLE}
            and preferred is not None
            and confidence >= self.config.minimum_scenario_confidence
        ):
            geometry = preferred
            geometry_status = GeometryValidity.VALID
            reason = None
        elif agreement == ScenarioAgreement.CONFLICT:
            reason = "true_m5_m15_directional_conflict"
        elif confidence < self.config.minimum_scenario_confidence:
            reason = "combined_confidence_below_requirement"
        validity = (
            ScenarioValidity.VALID
            if agreement in {ScenarioAgreement.ALIGNED, ScenarioAgreement.PULLBACK_COMPATIBLE}
            else ScenarioValidity.DEGRADED
        )
        path = (
            f"{m5.expected_price_path} M15 context: {m15.expected_price_path}"
            if agreement == ScenarioAgreement.PULLBACK_COMPATIBLE
            else m15.expected_price_path
            if agreement == ScenarioAgreement.ALIGNED
            else "M5 and M15 paths are incompatible; no combined trade geometry is issued."
        )
        return CombinedForwardScenario(
            combined_scenario_id=uuid5(
                NAMESPACE_URL,
                f"ten:combined-forward-scenario:{m5.scenario_id}:{m15.scenario_id}:1.0",
            ),
            cycle_id=m15.cycle_id,
            instrument=m15.instrument,
            market_state_id=m15.market_state_id,
            m5_scenario_id=m5.scenario_id,
            m15_scenario_id=m15.scenario_id,
            created_at=max(m5.created_at, m15.created_at),
            market_cutoff_time=m15.market_cutoff_time,
            agreement=agreement,
            combined_direction=direction,
            expected_price_path=path,
            confidence=round(confidence, 4),
            scenario_validity=validity,
            scenario_validity_reason=agreement.value.lower(),
            execution_geometry_validity=geometry_status,
            geometry_rejection_reason=reason,
            geometry=geometry,
            expiry=min(m5.expiry, m15.expiry),
        )

    @staticmethod
    def _prediction(quant: QuantForecastResult, timeframe: str) -> HorizonPrediction:
        try:
            return next(
                item
                for item in quant.predictions
                if item.horizon.timeframe == timeframe
                and item.horizon.candle_count == 1
            )
        except StopIteration as exc:
            raise ValueError(
                f"quant forecast has no next-candle {timeframe} horizon"
            ) from exc

    def _direction(
        self, prediction: HorizonPrediction
    ) -> tuple[ScenarioDirection, float]:
        edge = prediction.buy_probability - prediction.sell_probability
        confidence = max(
            prediction.buy_probability,
            prediction.sell_probability,
            prediction.neutral_probability,
        )
        if prediction.neutral_probability >= confidence and abs(edge) < self.config.minimum_directional_edge:
            return ScenarioDirection.RANGE, confidence
        if abs(edge) < self.config.minimum_directional_edge:
            return ScenarioDirection.INCONCLUSIVE, confidence
        return (
            ScenarioDirection.BULLISH if edge > 0 else ScenarioDirection.BEARISH,
            prediction.buy_probability if edge > 0 else prediction.sell_probability,
        )

    @staticmethod
    def _ranges(
        price: float,
        expected: float,
        maximum: float,
        direction: ScenarioDirection,
    ) -> tuple[PriceZone, PriceZone]:
        if direction == ScenarioDirection.BULLISH:
            low, high = price - maximum * 0.25, price + maximum
            close_center = price + expected * 0.70
        elif direction == ScenarioDirection.BEARISH:
            low, high = price - maximum, price + maximum * 0.25
            close_center = price - expected * 0.70
        else:
            low, high = price - maximum * 0.5, price + maximum * 0.5
            close_center = price
        width = max(expected * 0.15, price * 0.00005)
        return (
            PriceZone(low=max(0.000001, low), high=high),
            PriceZone(
                low=max(0.000001, close_center - width),
                high=close_center + width,
            ),
        )

    @staticmethod
    def _facts(
        signal: TimeframeAnalyticalSignal,
        direction: ScenarioDirection,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        desired = (
            AnalyticalDirection.BUY
            if direction == ScenarioDirection.BULLISH
            else AnalyticalDirection.SELL
            if direction == ScenarioDirection.BEARISH
            else None
        )
        supporting: list[str] = []
        contradicting: list[str] = []
        for contribution in signal.evidence_breakdown:
            destination = (
                supporting
                if desired is not None
                and contribution.directional_contribution == desired
                else contradicting
            )
            destination.extend(contribution.source_fact_identifiers)
        return tuple(dict.fromkeys(supporting)), tuple(dict.fromkeys(contradicting))

    @staticmethod
    def _confidence(
        signal: TimeframeAnalyticalSignal,
        raw_confidence: float,
        completeness: float,
    ) -> float:
        return round(
            min(
                100.0,
                signal.confidence * 0.45
                + raw_confidence * 100 * 0.35
                + completeness * 100 * 0.20,
            ),
            4,
        )

    def _validity(
        self,
        direction: ScenarioDirection,
        expected_move: float,
        confidence: float,
        completeness: float,
    ) -> tuple[ScenarioValidity, str]:
        if expected_move <= 0:
            return ScenarioValidity.INVALID, "expected_movement_unavailable"
        if completeness < self.config.minimum_evidence_completeness:
            return ScenarioValidity.DEGRADED, "insufficient_evidence_completeness"
        if direction == ScenarioDirection.INCONCLUSIVE:
            return ScenarioValidity.DEGRADED, "directional_evidence_inconclusive"
        if confidence < self.config.minimum_scenario_confidence:
            return ScenarioValidity.DEGRADED, "scenario_confidence_below_requirement"
        return ScenarioValidity.VALID, "completed_point_in_time_evidence_validated"

    @staticmethod
    def _scenario_type(
        direction: ScenarioDirection, signal: TimeframeAnalyticalSignal
    ) -> str:
        thesis = signal.directional_thesis.lower()
        if "sweep" in thesis:
            return "liquidity sweep and reversal"
        if "break" in thesis:
            return f"{direction.value.lower()} breakout continuation"
        if direction == ScenarioDirection.RANGE:
            return "range continuation"
        if direction == ScenarioDirection.INCONCLUSIVE:
            return "inconclusive"
        return f"{direction.value.lower()} continuation"

    @staticmethod
    def _target(price: float, move: float, direction: ScenarioDirection) -> float:
        if direction == ScenarioDirection.BULLISH:
            return price + move * 0.8
        if direction == ScenarioDirection.BEARISH:
            return price - move * 0.8
        return price

    @staticmethod
    def _invalidation_level(
        price: float, move: float, direction: ScenarioDirection
    ) -> float:
        if direction == ScenarioDirection.BULLISH:
            return max(0.000001, price - move * 0.35)
        if direction == ScenarioDirection.BEARISH:
            return price + move * 0.35
        return max(0.000001, price - move)

    @staticmethod
    def _path(
        direction: ScenarioDirection,
        scenario_type: str,
        price: float,
        target: float,
    ) -> str:
        if direction == ScenarioDirection.BULLISH:
            return (
                f"A shallow pullback or hold near {price:.3f} is expected before "
                f"{scenario_type} toward {target:.3f}."
            )
        if direction == ScenarioDirection.BEARISH:
            return (
                f"A shallow retracement or rejection near {price:.3f} is expected before "
                f"{scenario_type} toward {target:.3f}."
            )
        if direction == ScenarioDirection.RANGE:
            return f"Price is expected to rotate around {price:.3f} within the projected range."
        return "Evidence does not support one dominant path; the alternative remains material."

    def _geometry(
        self,
        prediction: HorizonPrediction,
        signal: TimeframeAnalyticalSignal,
        direction: ScenarioDirection,
        supporting: tuple[str, ...],
        confidence: float,
        validity: ScenarioValidity,
    ) -> tuple[GeometryValidity, ScenarioGeometry | None, str | None]:
        if validity != ScenarioValidity.VALID:
            return GeometryValidity.UNAVAILABLE, None, "scenario_not_valid_for_execution"
        structural_facts = tuple(
            dict.fromkeys(
                fact
                for contribution in signal.evidence_breakdown
                if contribution.family
                in {"market_structure", "order_block", "imbalance", "liquidity"}
                and contribution.source_fact_identifiers
                for fact in contribution.source_fact_identifiers
            )
        )
        if not structural_facts:
            return GeometryValidity.UNAVAILABLE, None, "structural_basis_unavailable"
        price = prediction.reference_price
        expected = prediction.expected_base_movement
        max_distance = min(
            price * self.config.maximum_entry_distance_percent,
            expected * self.config.entry_distance_expected_move_fraction,
        )
        if max_distance <= 0:
            return GeometryValidity.UNAVAILABLE, None, "entry_reachability_unavailable"
        existing = signal.geometry
        if existing is not None and abs(existing.entry - price) <= max_distance:
            entry = existing.entry
            stop = existing.stop_loss
            target = existing.take_profit
            entry_zone = PriceZone(
                low=max(0.000001, entry - max_distance * 0.1),
                high=entry + max_distance * 0.1,
            )
            facts = tuple(dict.fromkeys((*existing.basis_fact_identifiers, *structural_facts)))
        else:
            entry = price
            risk = expected * self.config.stop_expected_move_fraction
            reward = expected * self.config.target_expected_move_fraction
            if direction == ScenarioDirection.BULLISH:
                stop, target = price - risk, price + reward
            elif direction == ScenarioDirection.BEARISH:
                stop, target = price + risk, price - reward
            else:
                return GeometryValidity.UNAVAILABLE, None, "non_directional_scenario"
            entry_zone = PriceZone(
                low=max(0.000001, price - max_distance * 0.1),
                high=price + max_distance * 0.1,
            )
            facts = tuple(dict.fromkeys((*structural_facts, *supporting)))
        return validate_geometry(
            GeometryCandidate(
                direction=direction,
                reference_price=price,
                entry_zone=entry_zone,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                secondary_target=None,
                expected_move=expected,
                maximum_entry_distance=max_distance,
                minimum_risk_reward=self.config.minimum_risk_reward,
                basis_fact_identifiers=facts,
            )
        )

    @staticmethod
    def _pullback_compatible(
        m5: ForwardMarketScenario, m15: ForwardMarketScenario
    ) -> bool:
        if m15.primary_direction == ScenarioDirection.BULLISH:
            return (
                m5.primary_direction == ScenarioDirection.BEARISH
                and m15.invalidation_level is not None
                and m5.expected_low >= m15.invalidation_level
            )
        if m15.primary_direction == ScenarioDirection.BEARISH:
            return (
                m5.primary_direction == ScenarioDirection.BULLISH
                and m15.invalidation_level is not None
                and m5.expected_high <= m15.invalidation_level
            )
        return False
