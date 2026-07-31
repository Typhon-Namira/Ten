"""Deterministic multi-path market simulation and authority selection."""

from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.market_state import EvidenceAvailability, UnifiedMarketState
from backend.app.quant_forecasting.models import HorizonPrediction, QuantForecastResult
from backend.app.signal_synthesis.models import AnalyticalDirection, MultiTimeframeSignalSet

from .instrument import (
    InstrumentSpecification,
    convert_prediction_move,
    validate_display_geometry,
)
from .models import GeometryValidity, PriceZone, ScenarioDirection, ScenarioValidity
from .simulation_models import (
    CandidateDirection,
    CandidateMarketScenario,
    EntryType,
    MarketSimulationCycle,
    PrimaryScenarioSelection,
    QuantMoveConversion,
    ScenarioPathStage,
    ScenarioScoreComponent,
    ScenarioSignalAction,
    SelectionStatus,
)


class MarketSimulationConfig(BaseModel):
    """Single authoritative configuration boundary for simulation and publication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_candidates: int = Field(default=5, ge=5, le=10)
    maximum_candidates: int = Field(default=7, ge=5, le=10)
    primary_scenario_threshold: float = Field(default=60, ge=0, le=100)
    email_scenario_threshold: float = Field(default=60, ge=0, le=100)
    minimum_risk_reward: float = Field(default=2.0, gt=0)
    maximum_entry_distance_percent: float = Field(default=0.003, gt=0, le=0.02)
    expiry_seconds: int = Field(default=900, ge=300, le=3600)
    calibration_minimum_sample: int = Field(default=30, ge=1)
    engine_version: str = "2.0.0"
    configuration_version: str = "primary-scenario-v1"
    instrument: InstrumentSpecification = InstrumentSpecification()

    @model_validator(mode="after")
    def candidate_bounds_are_coherent(self) -> MarketSimulationConfig:
        if self.maximum_candidates < self.minimum_candidates:
            raise ValueError("maximum_candidates cannot be below minimum_candidates")
        return self


class MarketSimulationEngine:
    """Construct competing paths, score them, and select one M15 authority."""

    _TEMPLATES = (
        ("bullish_continuation", CandidateDirection.BULLISH, EntryType.CURRENT_PRICE, 0.00),
        ("bearish_continuation", CandidateDirection.BEARISH, EntryType.CURRENT_PRICE, 0.00),
        ("bullish_pullback_continuation", CandidateDirection.BULLISH, EntryType.PULLBACK, -0.20),
        ("bearish_retracement_continuation", CandidateDirection.BEARISH, EntryType.RETEST, 0.20),
        (
            "upside_liquidity_sweep_reversal",
            CandidateDirection.BEARISH,
            EntryType.LIQUIDITY_SWEEP_REVERSAL,
            0.25,
        ),
        (
            "downside_liquidity_sweep_reversal",
            CandidateDirection.BULLISH,
            EntryType.LIQUIDITY_SWEEP_REVERSAL,
            -0.25,
        ),
        ("range_continuation", CandidateDirection.RANGE, EntryType.NONE, 0.00),
    )

    def __init__(self, config: MarketSimulationConfig | None = None) -> None:
        self.config = config or MarketSimulationConfig()

    def simulate(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        synthesis: MultiTimeframeSignalSet,
        *,
        completed_history: tuple[tuple[object, object], ...] = (),
    ) -> tuple[MarketSimulationCycle, PrimaryScenarioSelection]:
        self._validate_inputs(state, quant, synthesis)
        prediction = self._m15_prediction(quant)
        conversion = convert_prediction_move(prediction, self.config.instrument)
        completed_history = tuple(
            item
            for item in completed_history
            if getattr(item[1], "completed_at", state.market_data_boundary)
            <= state.market_data_boundary
        )
        cycle_id = uuid5(
            NAMESPACE_URL,
            f"ten:market-simulation:{state.state_id}:{self.config.configuration_version}",
        )
        candidates = [
            self._candidate(
                state,
                synthesis,
                prediction,
                conversion,
                cycle_id,
                index,
                scenario_type,
                direction,
                entry_type,
                entry_offset,
                completed_history,
            )
            for index, (scenario_type, direction, entry_type, entry_offset) in enumerate(
                self._TEMPLATES[: self.config.maximum_candidates],
                start=1,
            )
        ]
        candidates.sort(key=lambda item: (-item.final_scenario_score, item.diversity_key))
        ranked = tuple(
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(candidates, start=1)
        )
        simulation = MarketSimulationCycle(
            simulation_cycle_id=cycle_id,
            cycle_id=state.cycle_id,
            market_state_id=state.state_id,
            synthesis_id=synthesis.synthesis_id,
            analysis_id=synthesis.analysis_id,
            quantitative_forecast_id=quant.result_id,
            instrument=state.instrument,
            market_cutoff=state.market_data_boundary,
            m5_source_cycle_id=state.cycle_id,
            m15_source_cycle_id=state.cycle_id,
            candidate_count=len(ranked),
            candidates=ranked,
            created_at=max(state.knowledge_cutoff, synthesis.created_at),
            engine_version=self.config.engine_version,
            configuration_version=self.config.configuration_version,
        )
        return simulation, self._select(simulation)

    def _candidate(
        self,
        state: UnifiedMarketState,
        synthesis: MultiTimeframeSignalSet,
        prediction: HorizonPrediction,
        conversion: QuantMoveConversion,
        simulation_cycle_id: UUID,
        index: int,
        scenario_type: str,
        direction: CandidateDirection,
        entry_type: EntryType,
        entry_offset: float,
        completed_history: tuple[tuple[object, object], ...],
    ) -> CandidateMarketScenario:
        move = conversion.converted_expected_move
        reference = prediction.reference_price
        sign = 1 if direction == CandidateDirection.BULLISH else -1
        entry = reference + move * entry_offset
        zone_half = max(self.config.instrument.tick_size * 3, move * 0.04)
        entry_zone = (
            PriceZone(low=entry - zone_half, high=entry + zone_half)
            if entry_type != EntryType.NONE
            else None
        )
        stop = entry - sign * max(move * 0.25, self.config.instrument.minimum_stop_distance)
        target = entry + sign * max(move * 0.70, self.config.instrument.minimum_target_distance)
        secondary = entry + sign * max(move, self.config.instrument.minimum_target_distance * 1.5)
        evidence = self._evidence(synthesis, direction)
        contradicting = self._evidence(synthesis, self._opposite(direction))
        has_liquidity = any(
            "liquidity" in item.source_engine.lower()
            and item.availability == EvidenceAvailability.AVAILABLE
            for item in state.evidence
        )
        label_valid = "liquidity_sweep" not in scenario_type or has_liquidity
        scenario_direction = {
            CandidateDirection.BULLISH: ScenarioDirection.BULLISH,
            CandidateDirection.BEARISH: ScenarioDirection.BEARISH,
            CandidateDirection.RANGE: ScenarioDirection.RANGE,
            CandidateDirection.INCONCLUSIVE: ScenarioDirection.INCONCLUSIVE,
        }[direction]
        geometry_status = GeometryValidity.UNAVAILABLE
        geometry = None
        geometry_reason: str | None = "analytical_range_has_no_execution_geometry"
        if entry_zone is not None and label_valid:
            geometry_status, geometry, geometry_reason = validate_display_geometry(
                direction=scenario_direction,
                reference_price=reference,
                entry_zone=entry_zone,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                secondary_target=secondary,
                expected_move=move,
                maximum_entry_distance=reference
                * self.config.maximum_entry_distance_percent,
                minimum_risk_reward=self.config.minimum_risk_reward,
                basis_fact_identifiers=evidence[:6] or ("quant:m15",),
                spread=self._spread(state),
                specification=self.config.instrument,
            )
        elif not label_valid:
            geometry_reason = "liquidity_sweep_definition_missing_valid_liquidity_evidence"
        components = self._score_components(
            synthesis,
            prediction,
            direction,
            geometry_status,
            state.evidence_completeness,
            len(contradicting),
            completed_history,
        )
        raw = sum(item.contribution for item in components)
        confidence = max(0.0, min(100.0, raw))
        calibration, sample = self._calibration(scenario_type, completed_history)
        adjustment = 0.0 if calibration is None else (calibration - 0.5) * 10
        final = max(0.0, min(100.0, confidence + adjustment))
        validity = (
            ScenarioValidity.INVALID
            if not label_valid
            else ScenarioValidity.DEGRADED
            if state.evidence_completeness < 0.5
            else ScenarioValidity.VALID
        )
        cutoff = state.market_data_boundary
        range_low = reference - move
        range_high = reference + move
        close_center = reference + sign * move * 0.55 if direction not in {
            CandidateDirection.RANGE,
            CandidateDirection.INCONCLUSIVE,
        } else reference
        close_half = move * 0.15
        stages = self._stages(
            simulation_cycle_id,
            index,
            scenario_type,
            direction,
            reference,
            entry,
            target,
            range_low,
            range_high,
            evidence,
        )
        diversity_key = sha256(
            f"{direction.value}|{scenario_type}|{entry_type.value}|"
            f"{round(target / self.config.instrument.tick_size)}".encode()
        ).hexdigest()
        candidate_id = uuid5(
            NAMESPACE_URL,
            f"ten:market-simulation-candidate:{simulation_cycle_id}:{diversity_key}",
        )
        rejection = (
            "liquidity_sweep_definition_missing_valid_liquidity_evidence"
            if not label_valid
            else geometry_reason
        )
        return CandidateMarketScenario(
            candidate_id=candidate_id,
            simulation_cycle_id=simulation_cycle_id,
            cycle_id=state.cycle_id,
            instrument=state.instrument,
            market_cutoff=cutoff,
            reference_price=reference,
            forecast_horizon_seconds=self.config.expiry_seconds,
            direction=direction,
            scenario_type=scenario_type,
            path_sequence=stages,
            ai_proposed_path=tuple(stage.label for stage in stages),
            deterministically_validated_path=tuple(
                f"{stage.sequence}. {stage.label}: "
                f"{stage.expected_price_area.low:.2f}-{stage.expected_price_area.high:.2f}"
                for stage in stages
            ),
            expected_low=range_low,
            expected_high=range_high,
            likely_close_low=close_center - close_half,
            likely_close_high=close_center + close_half,
            expected_move=move,
            quant_move_conversion=conversion,
            trigger_condition=self._trigger(scenario_type),
            entry_type=entry_type,
            entry_zone=geometry.entry_zone if geometry else entry_zone,
            invalidation_level=stop if entry_zone else None,
            protective_stop=geometry.stop_loss if geometry else None,
            primary_target=geometry.take_profit if geometry else None,
            secondary_target=geometry.secondary_target if geometry else None,
            expiry=cutoff + timedelta(seconds=self.config.expiry_seconds),
            supporting_evidence_ids=evidence,
            contradicting_evidence_ids=contradicting,
            score_components=components,
            raw_model_score=raw,
            normalized_confidence=confidence,
            calibration_adjustment=adjustment,
            calibrated_probability=calibration,
            calibration_sample_size=sample,
            final_scenario_score=final,
            rank=0,
            scenario_validity=validity,
            geometry_validity=geometry_status,
            geometry=geometry,
            rejection_reason=rejection,
            diversity_key=diversity_key,
        )

    def _select(self, simulation: MarketSimulationCycle) -> PrimaryScenarioSelection:
        valid = [
            item
            for item in simulation.candidates
            if item.scenario_validity == ScenarioValidity.VALID
            and item.direction in {CandidateDirection.BULLISH, CandidateDirection.BEARISH}
        ]
        primary = valid[0] if valid else None
        alternative = next(
            (
                item
                for item in valid[1:]
                if primary is not None
                and (
                    item.direction != primary.direction
                    or item.scenario_type != primary.scenario_type
                )
            ),
            None,
        )
        selected = (
            primary is not None
            and alternative is not None
            and primary.final_scenario_score >= self.config.primary_scenario_threshold
        )
        status = (
            SelectionStatus.SELECTED
            if selected
            else SelectionStatus.INSUFFICIENT_CONFIDENCE
            if primary is not None
            else SelectionStatus.NO_VALID_CANDIDATE
        )
        action = (
            {
                CandidateDirection.BULLISH: ScenarioSignalAction.BUY,
                CandidateDirection.BEARISH: ScenarioSignalAction.SELL,
            }.get(primary.direction, ScenarioSignalAction.HOLD)
            if primary is not None
            else ScenarioSignalAction.HOLD
        )
        eligible = bool(
            selected
            and primary is not None
            and primary.geometry_validity == GeometryValidity.VALID
            and primary.geometry is not None
            and primary.final_scenario_score >= self.config.email_scenario_threshold
        )
        if not selected:
            action = ScenarioSignalAction.HOLD
        return PrimaryScenarioSelection(
            selection_id=uuid5(
                NAMESPACE_URL,
                f"ten:primary-scenario:{simulation.simulation_cycle_id}",
            ),
            simulation_cycle_id=simulation.simulation_cycle_id,
            cycle_id=simulation.cycle_id,
            market_state_id=simulation.market_state_id,
            instrument=simulation.instrument,
            market_cutoff=simulation.market_cutoff,
            selected_at=simulation.created_at,
            status=status,
            authoritative_action=action,
            primary_candidate_id=primary.candidate_id if selected and primary else None,
            alternative_candidate_id=(
                alternative.candidate_id if selected and alternative else None
            ),
            primary=primary if selected else None,
            alternative=alternative if selected else None,
            minimum_score=self.config.primary_scenario_threshold,
            signal_eligible=eligible,
            rejection_reason=(
                None
                if selected
                else "best_candidate_below_primary_scenario_threshold"
                if primary
                else "no_valid_candidate"
            ),
            ranking_explanation=(
                f"Primary ranked first at {primary.final_scenario_score:.1f}; "
                f"alternative ranked {alternative.rank} at "
                f"{alternative.final_scenario_score:.1f}."
                if selected and primary and alternative
                else "No candidate satisfied the configured Primary Scenario contract."
            ),
        )

    @staticmethod
    def _validate_inputs(
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        synthesis: MultiTimeframeSignalSet,
    ) -> None:
        if (
            state.state_id != quant.market_state_id
            or state.state_id != synthesis.market_state_id
            or state.cycle_id != quant.cycle_id
            or state.cycle_id != synthesis.cycle_id
        ):
            raise ValueError("simulation inputs must share one immutable cycle")
        m15 = next(item for item in state.timeframes if item.timeframe == "M15")
        m5 = next(item for item in state.timeframes if item.timeframe == "M5")
        if m15.stale or m5.stale:
            raise ValueError("authoritative simulation requires fresh synchronized M5/M15")
        if m15.source_candle_close_at != state.market_data_boundary:
            raise ValueError("authoritative simulation requires an M15 close boundary")
        if m5.source_candle_close_at > m15.source_candle_close_at:
            raise ValueError("M5 source cannot exceed authoritative M15 cutoff")

    @staticmethod
    def _m15_prediction(quant: QuantForecastResult) -> HorizonPrediction:
        prediction = next(
            (
                item
                for item in quant.predictions
                if item.horizon.timeframe == "M15"
            ),
            None,
        )
        if prediction is None:
            raise ValueError("M15 quantitative prediction unavailable")
        return prediction

    @staticmethod
    def _evidence(
        synthesis: MultiTimeframeSignalSet, direction: CandidateDirection
    ) -> tuple[str, ...]:
        wanted = (
            AnalyticalDirection.BUY
            if direction == CandidateDirection.BULLISH
            else AnalyticalDirection.SELL
        )
        values = [
            str(item.evidence_id)
            for signal in synthesis.timeframe_signals
            for item in signal.evidence_breakdown
            if item.directional_contribution == wanted
        ]
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _opposite(direction: CandidateDirection) -> CandidateDirection:
        return (
            CandidateDirection.BEARISH
            if direction == CandidateDirection.BULLISH
            else CandidateDirection.BULLISH
        )

    @staticmethod
    def _spread(state: UnifiedMarketState) -> float:
        for item in state.evidence:
            if "spread" in item.evidence_type.lower() and isinstance(
                item.normalized_value, (float, int)
            ):
                return max(0.0, float(item.normalized_value))
        return 0.05

    def _score_components(
        self,
        synthesis: MultiTimeframeSignalSet,
        prediction: HorizonPrediction,
        direction: CandidateDirection,
        geometry: GeometryValidity,
        completeness: float,
        contradiction_count: int,
        history: tuple[tuple[object, object], ...],
    ) -> tuple[ScenarioScoreComponent, ...]:
        m15 = synthesis.timeframe_signals[1]
        target = AnalyticalDirection.BUY if direction == CandidateDirection.BULLISH else AnalyticalDirection.SELL
        directional = (
            m15.bullish_score if target == AnalyticalDirection.BUY else m15.bearish_score
        )
        probability = (
            prediction.buy_probability
            if target == AnalyticalDirection.BUY
            else prediction.sell_probability
        )
        history_value = (
            sum(float(getattr(outcome, "directional_accuracy", 0)) for _, outcome in history)
            / len(history)
            if history
            else 0.5
        )
        data = (
            ("directional_synthesis", directional, 0.30, directional * 0.30, "M15 decomposed synthesis"),
            ("quant_compatibility", probability * 100, 0.20, probability * 20, "M15 Quant direction"),
            ("evidence_completeness", completeness * 100, 0.15, completeness * 15, "immutable UMS completeness"),
            ("temporal_alignment", m15.confidence_decomposition.timeframe_alignment, 0.10, m15.confidence_decomposition.timeframe_alignment * 0.10, "M5/M15 alignment"),
            ("historical_calibration", history_value * 100, 0.10, history_value * 10, "completed outcomes before cutoff"),
            ("geometry_quality", 100 if geometry == GeometryValidity.VALID else 0, 0.15, 15 if geometry == GeometryValidity.VALID else 0, "deterministic geometry"),
            ("contradiction_penalty", contradiction_count, 0.05, -min(10, contradiction_count * 1.5), "opposing independent evidence"),
        )
        return tuple(
            ScenarioScoreComponent(
                name=name,
                raw_value=raw,
                weight=weight,
                contribution=contribution,
                reason=reason,
            )
            for name, raw, weight, contribution, reason in data
        )

    def _calibration(
        self, scenario_type: str, history: tuple[tuple[object, object], ...]
    ) -> tuple[float | None, int]:
        completed = [
            outcome
            for scenario, outcome in history
            if getattr(scenario, "scenario_type", None) == scenario_type
        ]
        if len(completed) < self.config.calibration_minimum_sample:
            return None, len(completed)
        return (
            sum(float(getattr(item, "directional_accuracy", 0)) for item in completed)
            / len(completed),
            len(completed),
        )

    def _stages(
        self,
        cycle_id: UUID,
        index: int,
        scenario_type: str,
        direction: CandidateDirection,
        reference: float,
        entry: float,
        target: float,
        low: float,
        high: float,
        evidence: tuple[str, ...],
    ) -> tuple[ScenarioPathStage, ...]:
        labels = (
            ("Hold current structure", "Confirm trigger", "Expand toward target")
            if "continuation" in scenario_type
            else ("Test identified liquidity", "Confirm rejection", "Rotate toward target")
            if "sweep" in scenario_type
            else ("Remain inside established range", "Re-test opposing boundary")
        )
        prices = (reference, entry, target) if len(labels) == 3 else (low, high)
        return tuple(
            ScenarioPathStage(
                stage_id=uuid5(
                    NAMESPACE_URL, f"ten:simulation-stage:{cycle_id}:{index}:{sequence}"
                ),
                sequence=sequence,
                label=label,
                expected_price_area=PriceZone(
                    low=max(0.01, price - self.config.instrument.tick_size * 5),
                    high=price + self.config.instrument.tick_size * 5,
                ),
                supporting_evidence_ids=evidence[:4],
                invalidation_condition=(
                    f"Price violates the {direction.value.lower()} path structure"
                ),
                timing_seconds=sequence * 300,
            )
            for sequence, (label, price) in enumerate(zip(labels, prices, strict=True), 1)
        )

    @staticmethod
    def _trigger(scenario_type: str) -> str:
        if "sweep" in scenario_type:
            return "M5 sweep and close back through the identified liquidity boundary"
        if "pullback" in scenario_type or "retracement" in scenario_type:
            return "M5 rejection from the validated near-market entry zone"
        if "range" in scenario_type:
            return "Range boundaries remain intact through the M15 horizon"
        return "M5 close confirms continuation without structural invalidation"
