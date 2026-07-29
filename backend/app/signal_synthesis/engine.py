"""Deterministic multi-factor synthesis over immutable UMS evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import fsum
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.ai_reasoning.analysis import AIMarketAnalysis, AnalysisBias
from backend.app.market_state import EvidenceAvailability, EvidenceItem, UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult

from .models import (
    AnalyticalDirection,
    ConfidenceDecomposition,
    DirectionalContribution,
    ExecutionEligibility,
    ExecutionStatus,
    MultiTimeframeSignalSet,
    SignalGeometry,
    TimeframeAnalyticalSignal,
    TimeframeContribution,
    strength_for,
)


_TIMEFRAMES = ("M5", "M15")
_HORIZONS = {"M5": "5-15 minutes", "M15": "15-90 minutes"}
_ACTIVE_ZONE_STATES = {
    "created",
    "candidate",
    "confirmed",
    "active",
    "touched",
    "partially_mitigated",
}
_ACTIVE_LIQUIDITY_STATES = {
    "confirmed",
    "active",
    "approached",
    "touched",
    "partially_swept",
}
_INVALID_ZONE_STATES = {
    "mitigated",
    "broken",
    "invalidated",
    "archived",
    "expired",
    "superseded",
}


class SignalSynthesisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_version: str = "1.0.0"
    configuration_version: str = "multi-factor-mtf-1.0"
    minimum_execution_confidence: float = Field(default=55, ge=0, le=100)
    minimum_risk_reward: float = Field(default=2.0, gt=0)
    correlated_evidence_discount: float = Field(default=0.35, ge=0, le=1)
    strength_thresholds: tuple[float, float, float, float] = (
        40,
        55,
        70,
        85,
    )
    timeframe_weights: dict[str, float] = Field(
        default_factory=lambda: {"M5": 0.45, "M15": 0.55}
    )
    structure_event_scores: dict[str, float] = Field(
        default_factory=lambda: {
            "bos": 0.90,
            "choch": 1.0,
            "mss": 1.0,
            "structure_continuation": 0.75,
        }
    )
    structure_zone_scores: dict[str, float] = Field(
        default_factory=lambda: {
            "order_block": 1.0,
            "breaker": 0.80,
            "imbalance": 0.65,
        }
    )
    family_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "market_structure": 0.16,
            "order_block": 0.12,
            "imbalance": 0.07,
            "liquidity": 0.12,
            "volume": 0.08,
            "institutional_flow": 0.13,
            "regime": 0.12,
            "quant": 0.12,
            "ai_interpretation": 0.03,
            "price_action": 0.05,
        }
    )

    @model_validator(mode="after")
    def complete_weight_contract(self) -> SignalSynthesisConfig:
        required_families = {
            "market_structure",
            "order_block",
            "imbalance",
            "liquidity",
            "volume",
            "institutional_flow",
            "regime",
            "quant",
            "ai_interpretation",
            "price_action",
        }
        if set(self.family_weights) != required_families:
            raise ValueError("family_weights must configure every evidence family exactly once")
        if set(self.timeframe_weights) != set(_TIMEFRAMES):
            raise ValueError("timeframe_weights must configure M5 and M15")
        if any(value <= 0 for value in (*self.family_weights.values(), *self.timeframe_weights.values())):
            raise ValueError("signal synthesis weights must be positive")
        if abs(fsum(self.family_weights.values()) - 1.0) > 1e-9:
            raise ValueError("family_weights must sum to 1")
        if abs(fsum(self.timeframe_weights.values()) - 1.0) > 1e-9:
            raise ValueError("timeframe_weights must sum to 1")
        if (
            tuple(sorted(self.strength_thresholds)) != self.strength_thresholds
            or len(set(self.strength_thresholds)) != 4
            or self.strength_thresholds[0] < 0
            or self.strength_thresholds[-1] > 100
        ):
            raise ValueError("strength_thresholds must be four ascending unique percentages")
        if not {"order_block", "breaker", "imbalance"}.issubset(
            self.structure_zone_scores
        ):
            raise ValueError("structure_zone_scores are incomplete")
        return self


class _Fact:
    def __init__(
        self,
        *,
        item: EvidenceItem,
        family: str,
        correlation_group: str,
        score: float,
        quality: float,
        raw: Any,
        reason: str,
        fact_ids: Iterable[object],
    ) -> None:
        self.item = item
        self.family = family
        self.correlation_group = correlation_group
        self.score = max(-1.0, min(1.0, score))
        self.quality = max(0.0, min(1.0, quality))
        self.raw = raw
        self.reason = reason
        self.fact_ids = tuple(dict.fromkeys(str(value) for value in fact_ids if value))


class MultiTimeframeSignalSynthesizer:
    """Preserve directional analysis while applying safety only to execution."""

    def __init__(self, config: SignalSynthesisConfig | None = None) -> None:
        self.config = config or SignalSynthesisConfig()

    def synthesize(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        analysis: AIMarketAnalysis,
    ) -> MultiTimeframeSignalSet:
        if analysis.output is None or not analysis.validation_passed:
            raise ValueError("multi-timeframe synthesis requires validated AI analysis")
        if (
            quant.market_state_id != state.state_id
            or analysis.market_snapshot_id != state.state_id
            or quant.cycle_id != state.cycle_id
            or analysis.cycle_id != state.cycle_id
        ):
            raise ValueError("synthesis inputs do not share one immutable market state")
        synthesis_id = uuid5(
            NAMESPACE_URL,
            f"ten:multi-timeframe-signal:{state.state_id}:{analysis.analysis_id}:1.0",
        )
        signals = tuple(
            self._timeframe_signal(synthesis_id, timeframe, state, quant, analysis)
            for timeframe in _TIMEFRAMES
        )
        combined, contributions = self._combined(synthesis_id, state, quant, analysis, signals)
        return MultiTimeframeSignalSet(
            synthesis_id=synthesis_id,
            cycle_id=state.cycle_id,
            market_state_id=state.state_id,
            analysis_id=analysis.analysis_id,
            quantitative_forecast_id=quant.result_id,
            instrument=state.instrument,
            market_timestamp=state.market_data_boundary,
            timeframe_signals=signals,
            combined_signal=combined,
            timeframe_contributions=contributions,
            engine_version=self.config.engine_version,
            configuration_version=self.config.configuration_version,
            created_at=max(state.knowledge_cutoff, analysis.created_at),
        )

    def _timeframe_signal(
        self,
        synthesis_id: UUID,
        timeframe: str,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        analysis: AIMarketAnalysis,
    ) -> TimeframeAnalyticalSignal:
        evidence = tuple(item for item in state.evidence if item.source_timeframe == timeframe)
        facts = self._facts(timeframe, evidence, quant, analysis)
        if not facts:
            raise ValueError(f"{timeframe} has no validated directional evidence")
        contributions = self._contributions(facts)
        bullish = fsum(max(0.0, item.weighted_score) for item in contributions)
        bearish = fsum(max(0.0, -item.weighted_score) for item in contributions)
        direction = self._stronger_direction(contributions, bullish, bearish)
        confidence = self._confidence(
            direction,
            contributions,
            state,
            evidence,
            timeframe_alignment=50,
            quant_ai_alignment=self._quant_ai_alignment(timeframe, quant, analysis),
            compression=self._compression(evidence),
        )
        geometry: SignalGeometry | None = None
        geometry_reasons: tuple[str, ...] = ()
        invalidations: tuple[str, ...] = ()
        blockers: list[str] = []
        if confidence.final_confidence >= self.config.minimum_execution_confidence:
            geometry, geometry_reasons, invalidations = self._geometry(
                direction, evidence, self.config.minimum_risk_reward
            )
            blockers.extend(geometry_reasons)
        frame = next(item for item in state.timeframes if item.timeframe == timeframe)
        if frame.stale:
            blockers.append("market_data_stale")
        if any(item.availability != EvidenceAvailability.AVAILABLE for item in evidence):
            blockers.append("timeframe_evidence_degraded")
        if state.market_schedule is None:
            blockers.append("market_status_unavailable")
        elif not state.market_schedule.market_open:
            blockers.append(
                f"market_{state.market_schedule.market_status.value.lower()}"
            )
        if confidence.final_confidence < self.config.minimum_execution_confidence:
            blockers.append("directional_confidence_below_execution_threshold")
        blockers = list(dict.fromkeys(blockers))
        eligible = not blockers and geometry is not None
        dominant = sorted(
            (
                item
                for item in contributions
                if item.directional_contribution == direction
            ),
            key=lambda item: abs(item.weighted_score),
            reverse=True,
        )
        contrary = sorted(
            (
                item
                for item in contributions
                if item.directional_contribution != direction
            ),
            key=lambda item: abs(item.weighted_score),
            reverse=True,
        )
        thesis = self._thesis(timeframe, direction, dominant, contrary)
        return TimeframeAnalyticalSignal(
            signal_id=uuid5(NAMESPACE_URL, f"ten:timeframe-signal:{synthesis_id}:{timeframe}"),
            synthesis_id=synthesis_id,
            market_state_id=state.state_id,
            analysis_id=analysis.analysis_id,
            quantitative_forecast_id=quant.result_id,
            instrument=state.instrument,
            timeframe=timeframe,
            analytical_direction=direction,
            confidence=confidence.final_confidence,
            strength=strength_for(
                confidence.final_confidence,
                self.config.strength_thresholds,
            ),
            bullish_score=round(bullish, 4),
            bearish_score=round(bearish, 4),
            expected_horizon=_HORIZONS[timeframe],
            evidence_breakdown=contributions,
            confidence_decomposition=confidence,
            directional_thesis=thesis,
            invalidation_conditions=invalidations,
            execution_eligibility=(
                ExecutionEligibility.ELIGIBLE
                if eligible
                else ExecutionEligibility.INELIGIBLE
            ),
            execution_status=ExecutionStatus.READY if eligible else ExecutionStatus.BLOCKED,
            blocking_reasons=tuple(blockers),
            geometry=geometry if eligible else None,
            completed_at=max(state.knowledge_cutoff, analysis.created_at),
        )

    def _combined(
        self,
        synthesis_id: UUID,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        analysis: AIMarketAnalysis,
        signals: tuple[TimeframeAnalyticalSignal, ...],
    ) -> tuple[TimeframeAnalyticalSignal, tuple[TimeframeContribution, ...]]:
        contributions = tuple(
            TimeframeContribution(
                timeframe=signal.timeframe,
                direction=signal.analytical_direction,
                confidence=signal.confidence,
                structural_importance=self.config.timeframe_weights[signal.timeframe],
                evidence_quality=signal.confidence_decomposition.evidence_quality / 100,
                signed_contribution=round(
                    (1 if signal.analytical_direction == AnalyticalDirection.BUY else -1)
                    * signal.confidence
                    * self.config.timeframe_weights[signal.timeframe]
                    * (signal.confidence_decomposition.evidence_quality / 100),
                    4,
                ),
                explanation=(
                    f"{signal.timeframe} {signal.analytical_direction.value} contributes "
                    f"{self.config.timeframe_weights[signal.timeframe]:.0%} structural importance "
                    f"at {signal.confidence:.1f}% confidence."
                ),
            )
            for signal in signals
        )
        signed = fsum(item.signed_contribution for item in contributions)
        if signed > 0:
            direction = AnalyticalDirection.BUY
        elif signed < 0:
            direction = AnalyticalDirection.SELL
        else:
            strongest_timeframe = max(
                contributions,
                key=lambda item: (
                    abs(item.signed_contribution),
                    item.structural_importance,
                    item.timeframe,
                ),
            )
            direction = strongest_timeframe.direction
        supporting = tuple(item for item in signals if item.analytical_direction == direction)
        opposing = tuple(item for item in signals if item.analytical_direction != direction)
        weighted_confidence = fsum(
            item.confidence * self.config.timeframe_weights[item.timeframe]
            for item in signals
        )
        agreement = (
            fsum(
                self.config.timeframe_weights[item.timeframe]
                for item in supporting
            )
            * 100
        )
        contradiction = (
            fsum(
                self.config.timeframe_weights[item.timeframe]
                for item in opposing
            )
            * 100
        )
        combined_confidence = max(
            0.0,
            min(100.0, weighted_confidence * (0.75 + agreement / 400) - contradiction * 0.20),
        )
        execution_source = next(
            (
                item
                for item in sorted(
                    supporting,
                    key=lambda value: self.config.timeframe_weights[value.timeframe],
                    reverse=True,
                )
                if item.execution_eligibility == ExecutionEligibility.ELIGIBLE
            ),
            None,
        )
        blockers = (
            ()
            if execution_source is not None
            else tuple(
                dict.fromkeys(
                    reason
                    for signal in supporting
                    for reason in signal.blocking_reasons
                )
            )
            or ("no_directionally_aligned_timeframe_has_valid_execution_geometry",)
        )
        all_evidence = tuple(
            item for signal in signals for item in signal.evidence_breakdown
        )
        bullish = fsum(
            item.bullish_score * self.config.timeframe_weights[item.timeframe]
            for item in signals
        )
        bearish = fsum(
            item.bearish_score * self.config.timeframe_weights[item.timeframe]
            for item in signals
        )
        quality = fsum(
            item.confidence_decomposition.evidence_quality
            * self.config.timeframe_weights[item.timeframe]
            for item in signals
        )
        decomposition = ConfidenceDecomposition(
            score_separation=round(min(100, abs(signed)), 4),
            independent_confluence=round(agreement, 4),
            evidence_quality=round(quality, 4),
            evidence_freshness=round(
                fsum(
                    item.confidence_decomposition.evidence_freshness
                    * self.config.timeframe_weights[item.timeframe]
                    for item in signals
                ),
                4,
            ),
            evidence_completeness=round(state.evidence_completeness * 100, 4),
            timeframe_alignment=round(agreement, 4),
            quant_ai_alignment=round(
                fsum(
                    item.confidence_decomposition.quant_ai_alignment
                    * self.config.timeframe_weights[item.timeframe]
                    for item in signals
                ),
                4,
            ),
            contradiction_penalty=round(contradiction, 4),
            missing_evidence_penalty=round((1 - state.evidence_completeness) * 100, 4),
            regime_suitability_penalty=round(
                fsum(
                    item.confidence_decomposition.regime_suitability_penalty
                    * self.config.timeframe_weights[item.timeframe]
                    for item in signals
                ),
                4,
            ),
            final_confidence=round(combined_confidence, 4),
        )
        thesis = (
            f"Combined {direction.value}: "
            + "; ".join(item.explanation for item in contributions)
            + (
                " Higher-timeframe structure outweighs the opposing lower-timeframe scenario."
                if opposing
                else " All completed timeframes agree directionally."
            )
        )
        combined = TimeframeAnalyticalSignal(
            signal_id=uuid5(NAMESPACE_URL, f"ten:timeframe-signal:{synthesis_id}:COMBINED"),
            synthesis_id=synthesis_id,
            market_state_id=state.state_id,
            analysis_id=analysis.analysis_id,
            quantitative_forecast_id=quant.result_id,
            instrument=state.instrument,
            timeframe="COMBINED",
            analytical_direction=direction,
            confidence=decomposition.final_confidence,
            strength=strength_for(
                decomposition.final_confidence,
                self.config.strength_thresholds,
            ),
            bullish_score=round(bullish, 4),
            bearish_score=round(bearish, 4),
            expected_horizon="multi-timeframe intraday",
            evidence_breakdown=all_evidence,
            confidence_decomposition=decomposition,
            directional_thesis=thesis,
            invalidation_conditions=(
                execution_source.invalidation_conditions if execution_source else ()
            ),
            execution_eligibility=(
                ExecutionEligibility.ELIGIBLE
                if execution_source is not None
                else ExecutionEligibility.INELIGIBLE
            ),
            execution_status=(
                ExecutionStatus.READY
                if execution_source is not None
                else ExecutionStatus.BLOCKED
            ),
            blocking_reasons=blockers,
            geometry=execution_source.geometry if execution_source is not None else None,
            completed_at=max(state.knowledge_cutoff, analysis.created_at),
        )
        return combined, contributions

    def _facts(
        self,
        timeframe: str,
        evidence: tuple[EvidenceItem, ...],
        quant: QuantForecastResult,
        analysis: AIMarketAnalysis,
    ) -> list[_Fact]:
        facts: list[_Fact] = []
        market_item = next(
            (item for item in evidence if item.source_engine == "market_data"),
            None,
        )
        current_price = (
            _num(_map(market_item.raw_value).get("close"))
            if market_item is not None
            else 0.0
        )
        for item in evidence:
            raw = _map(item.raw_value)
            quality = _quality(item)
            if item.source_engine == "market_data":
                open_price = _num(raw.get("open"))
                high = _num(raw.get("high"))
                low = _num(raw.get("low"))
                close = _num(raw.get("close"))
                correlation_group = (
                    f"market_candle:{item.provenance.get('market_event_id')}"
                )
                if open_price and close and close != open_price:
                    facts.append(
                        self._fact(
                            item,
                            "price_action",
                            correlation_group,
                            _sign(close - open_price),
                            quality,
                            {
                                "open": open_price,
                                "high": high,
                                "low": low,
                                "close": close,
                            },
                            (
                                "completed candle closed above open"
                                if close > open_price
                                else "completed candle closed below open"
                            ),
                            (item.provenance.get("market_event_id"),),
                        )
                    )
                body = abs(close - open_price)
                upper_wick = high - max(open_price, close)
                lower_wick = min(open_price, close) - low
                if body > 0 and max(upper_wick, lower_wick) > body:
                    rejection = lower_wick - upper_wick
                    if rejection:
                        facts.append(
                            self._fact(
                                item,
                                "price_action",
                                correlation_group,
                                _sign(rejection)
                                * min(1.0, abs(rejection) / body),
                                quality,
                                {
                                    "body": body,
                                    "upper_wick": upper_wick,
                                    "lower_wick": lower_wick,
                                },
                                (
                                    "lower-wick rejection supports buyers"
                                    if rejection > 0
                                    else "upper-wick rejection supports sellers"
                                ),
                                (item.provenance.get("market_event_id"),),
                            )
                        )
            elif item.source_engine == "market_regime":
                score = _num(raw.get("net_directional_score"))
                if score:
                    facts.append(self._fact(item, "regime", "regime_structure", score, quality, {"dominant_regime": raw.get("dominant_regime"), "net_directional_score": score, "compression_score": raw.get("compression_score")}, "market-regime directional imbalance", (raw.get("snapshot_id"),)))
            elif item.source_engine == "smc":
                facts.extend(self._smc_facts(item, raw, quality, current_price))
            elif item.source_engine == "liquidity":
                facts.extend(self._liquidity_facts(item, raw, quality))
            elif item.source_engine == "institutional_flow":
                flow_state = _map(raw.get("state"))
                pressure = _map(flow_state.get("pressure"))
                score = _num(pressure.get("net_pressure"))
                if score:
                    facts.append(self._fact(item, "institutional_flow", "institutional_state", score, _percent(pressure.get("quality"), quality), pressure, "institutional directional-flow pressure", (raw.get("id"),)))
                absorption = _map(flow_state.get("absorption"))
                if absorption and not absorption.get("invalidated"):
                    direction = _direction(absorption.get("defending_side"))
                    if direction:
                        intensity = _num(absorption.get("estimated_intensity"))
                        ambiguity = _num(absorption.get("ambiguity"))
                        facts.append(self._fact(item, "institutional_flow", "institutional_state", direction * intensity * (1 - ambiguity), _percent(absorption.get("confidence"), quality), absorption, "validated absorption by the defending side", (*_items(absorption.get("evidence_ids")), raw.get("id"))))
                exhaustion = _map(flow_state.get("exhaustion"))
                exhausted = _direction(exhaustion.get("exhausted_direction"))
                if exhausted:
                    reversal = _num(exhaustion.get("reversal_evidence"))
                    facts.append(self._fact(item, "institutional_flow", "institutional_state", -exhausted * max(0.25, reversal), _percent(exhaustion.get("confidence"), quality), exhaustion, "directional exhaustion supports the opposing scenario", (*_items(exhaustion.get("evidence_ids")), raw.get("id"))))
                inventory = _map(flow_state.get("inventory"))
                inventory_direction = _direction(inventory.get("direction"))
                if inventory_direction:
                    facts.append(self._fact(item, "institutional_flow", "institutional_state", inventory_direction * _num(inventory.get("strength")), _percent(inventory.get("confidence"), quality), inventory, "accumulation/distribution inventory inference", (*_items(inventory.get("evidence_ids")), raw.get("id"))))
            elif item.source_engine == "volume_profile":
                profiles = (
                    _items(raw.get("developing"))
                    or _items(raw.get("completed"))
                    or _items(raw.get("profiles"))
                )
                profile = _map(profiles[-1]) if profiles else {}
                buckets = tuple(_map(value) for value in _items(profile.get("buckets")))
                buy_volume = fsum(_num(value.get("estimated_buy_volume")) for value in buckets)
                sell_volume = fsum(_num(value.get("estimated_sell_volume")) for value in buckets)
                allocated_volume = buy_volume + sell_volume
                if allocated_volume > 0 and buy_volume != sell_volume:
                    facts.append(
                        self._fact(
                            item,
                            "volume",
                            "volume_participation",
                            (buy_volume - sell_volume) / allocated_volume,
                            _percent(profile.get("quality_score"), quality),
                            {
                                "profile_id": profile.get("id"),
                                "estimated_buy_volume": buy_volume,
                                "estimated_sell_volume": sell_volume,
                            },
                            "volume-profile participation imbalance",
                            (profile.get("id"),),
                        )
                    )
                migrations = _items(raw.get("migrations"))
                latest = _map(migrations[-1]) if migrations else {}
                migration_type = str(latest.get("migration_type", "")).lower()
                poc_change = _num(latest.get("poc_change"))
                direction = (
                    _sign(poc_change)
                    if poc_change
                    else 1.0
                    if migration_type == "upward"
                    else -1.0
                    if migration_type == "downward"
                    else 0.0
                )
                if direction:
                    magnitude = max(
                        0.25,
                        min(1.0, abs(_num(latest.get("normalized_change")))),
                    )
                    facts.append(
                        self._fact(
                            item,
                            "volume",
                            "volume_migration",
                            direction * magnitude,
                            _percent(latest.get("quality_score"), quality),
                            latest,
                            f"{migration_type or 'directional'} volume point-of-control migration",
                            (latest.get("id"), raw.get("id")),
                        )
                    )
        facts.extend(self._quant_facts(timeframe, evidence, quant))
        facts.extend(self._ai_facts(timeframe, evidence, analysis))
        return facts

    def _smc_facts(
        self,
        item: EvidenceItem,
        raw: Mapping[str, Any],
        quality: float,
        current_price: float,
    ) -> list[_Fact]:
        facts: list[_Fact] = []
        state = _map(raw.get("structure_state"))
        direction = _direction(state.get("current_direction"))
        if direction:
            facts.append(self._fact(item, "market_structure", "structure_state", direction, quality, state, "current confirmed market structure", (state.get("last_bos_id"), state.get("last_choch_id"), raw.get("id"))))
        for scope in ("internal", "external"):
            scoped_direction = _direction(state.get(f"{scope}_direction"))
            if scoped_direction:
                facts.append(
                    self._fact(
                        item,
                        "market_structure",
                        "structure_state",
                        scoped_direction,
                        quality,
                        {
                            "scope": scope,
                            "direction": state.get(f"{scope}_direction"),
                        },
                        f"{scope} market structure",
                        (
                            state.get("last_bos_id"),
                            state.get("last_choch_id"),
                            raw.get("id"),
                        ),
                    )
                )
        for event in _items(raw.get("structure_events"))[-5:]:
            value = _map(event)
            direction = _direction(value.get("resulting_direction") or value.get("direction"))
            if not direction or value.get("event_type") == "structure_invalidated":
                continue
            score = direction * self.config.structure_event_scores.get(
                str(value.get("event_type")),
                0.5,
            )
            facts.append(self._fact(item, "market_structure", f"structure_event:{value.get('scope', 'unknown')}", score, _percent(value.get("quality_score"), quality), value, f"{value.get('event_type')} confirmed {value.get('resulting_direction')}", (value.get("id"), value.get("broken_swing_id"), value.get("confirmation_candle_id"))))
        for displacement in _items(raw.get("displacements"))[-3:]:
            value = _map(displacement)
            if str(value.get("lifecycle_state")) in _INVALID_ZONE_STATES:
                continue
            direction = _direction(value.get("direction"))
            if direction:
                strength = 1.0 if value.get("strength") == "strong" else 0.45
                facts.append(self._fact(item, "market_structure", "displacement", direction * strength, _percent(value.get("quality_score"), quality), value, "structural displacement", (value.get("id"),)))
        for zone in _items(raw.get("zones")):
            value = _map(zone)
            lifecycle = str(value.get("lifecycle_state", "")).lower()
            if lifecycle in _INVALID_ZONE_STATES or lifecycle not in _ACTIVE_ZONE_STATES:
                continue
            zone_type = str(value.get("zone_type", ""))
            direction = _direction(value.get("direction") or zone_type)
            if not direction:
                continue
            freshness = max(0.15, 1 - _num(value.get("mitigation_percentage")) / 100)
            family = "order_block" if "block" in zone_type else "imbalance"
            score_family = (
                "order_block"
                if "order_block" in zone_type
                else "breaker"
                if "breaker" in zone_type
                else "imbalance"
            )
            base = self.config.structure_zone_scores[score_family]
            midpoint = _num(value.get("midpoint")) or (
                _num(value.get("lower_price")) + _num(value.get("upper_price"))
            ) / 2
            relative_distance = (
                abs(midpoint - current_price) / current_price
                if current_price > 0 and midpoint > 0
                else 0.0
            )
            proximity = max(0.35, 1 - min(0.65, relative_distance * 20))
            score = direction * base * freshness * proximity
            facts.append(self._fact(item, family, f"{family}:{zone_type}", score, _percent(value.get("quality_score"), quality), {**value, "relative_distance": relative_distance, "freshness_factor": freshness, "proximity_factor": proximity}, f"active {zone_type} ({lifecycle}, {value.get('mitigation_percentage', 0)}% mitigated, {relative_distance:.4%} from price)", (value.get("id"), value.get("trigger_event_id"), *(_items(value.get("source_candle_ids"))))))
        return facts

    def _liquidity_facts(self, item: EvidenceItem, raw: Mapping[str, Any], quality: float) -> list[_Fact]:
        facts: list[_Fact] = []
        for sweep in _items(raw.get("sweeps"))[-3:]:
            value = _map(sweep)
            side = str(value.get("side"))
            reclaimed = value.get("reclaim_timestamp") is not None or _num(value.get("reclaim_strength")) > 0
            direction = 1.0 if side == "sell_side" else -1.0 if side == "buy_side" else 0.0
            if direction and not reclaimed and value.get("classification") == "continuation":
                direction *= -1
            if direction:
                facts.append(self._fact(item, "liquidity", f"liquidity_sweep:{value.get('pool_id')}", direction, _percent(value.get("quality_score"), quality), value, f"{side} liquidity sweep {'reclaimed' if reclaimed else 'continued'}", (value.get("id"), value.get("pool_id"))))
        for target in _items(raw.get("targets"))[:5]:
            value = _map(target)
            if value.get("status") != "active":
                continue
            side = str(value.get("side"))
            direction = 1.0 if side == "buy_side" else -1.0 if side == "sell_side" else 0.0
            if direction:
                accessibility = _percent(value.get("accessibility_score"), 0.5)
                facts.append(self._fact(item, "liquidity", f"liquidity_target:{side}", direction * accessibility, _percent(value.get("confidence_score"), quality), value, f"active {side} liquidity target", (value.get("id"), value.get("pool_id"))))
        for inducement in _items(raw.get("inducements"))[:3]:
            value = _map(inducement)
            if str(value.get("lifecycle_state", "")).lower() not in _ACTIVE_LIQUIDITY_STATES:
                continue
            side = str(value.get("side"))
            direction = _direction(side)
            if direction:
                facts.append(
                    self._fact(
                        item,
                        "liquidity",
                        f"liquidity_inducement:{side}",
                        direction * 0.45,
                        _percent(value.get("quality_score"), quality),
                        value,
                        f"active {side} inducement",
                        (value.get("id"), *(_items(value.get("source_object_ids")))),
                    )
                )
        return facts

    def _quant_facts(self, timeframe: str, evidence: tuple[EvidenceItem, ...], quant: QuantForecastResult) -> list[_Fact]:
        matching = tuple(item for item in quant.predictions if item.horizon.timeframe == timeframe)
        if not matching:
            return []
        score = fsum(item.buy_probability - item.sell_probability for item in matching) / len(matching)
        if score == 0:
            return []
        anchor = next(item for item in evidence if item.source_engine == "market_data")
        return [self._fact(anchor, "quant", f"quant:{timeframe}", score, max(item.buy_probability + item.sell_probability for item in matching), {"horizons": [item.horizon.horizon_id for item in matching], "buy_probability": max(item.buy_probability for item in matching), "sell_probability": max(item.sell_probability for item in matching)}, "point-in-time Quant probability imbalance", tuple(str(item.horizon.horizon_id) for item in matching))]

    def _ai_facts(self, timeframe: str, evidence: tuple[EvidenceItem, ...], analysis: AIMarketAnalysis) -> list[_Fact]:
        assert analysis.output is not None
        known = {str(item.evidence_id): item for item in evidence}
        result: list[_Fact] = []
        for direction, values in (
            (1.0, analysis.output.bullish_evidence),
            (-1.0, analysis.output.bearish_evidence),
        ):
            for value in values:
                if value.timeframe not in {None, timeframe}:
                    continue
                source = known.get(value.source_reference)
                if source is None:
                    continue
                result.append(
                    self._fact(
                        source,
                        "ai_interpretation",
                        _primary_correlation_group(source, direction),
                        direction,
                        analysis.output.analysis_confidence,
                        value.model_dump(mode="json"),
                        "validated AI interpretation of a catalogued fact",
                        (str(analysis.analysis_id), value.source_reference),
                    )
                )
        return result

    def _contributions(self, facts: list[_Fact]) -> tuple[DirectionalContribution, ...]:
        grouped: dict[str, list[_Fact]] = defaultdict(list)
        for fact in facts:
            grouped[fact.correlation_group].append(fact)
        result: list[DirectionalContribution] = []
        for group in sorted(grouped):
            ordered = sorted(
                grouped[group],
                key=lambda item: (
                    item.family != "ai_interpretation",
                    abs(item.score) * item.quality,
                ),
                reverse=True,
            )
            for index, fact in enumerate(ordered):
                discount = 1.0 if index == 0 else self.config.correlated_evidence_discount
                freshness = _freshness(fact.item)
                weight = self.config.family_weights[fact.family]
                weighted = fact.score * weight * fact.quality * freshness * discount * 100
                if weighted == 0:
                    continue
                result.append(
                    DirectionalContribution(
                        evidence_id=fact.item.evidence_id,
                        tool=fact.item.source_engine,
                        timeframe=fact.item.source_timeframe,
                        family=fact.family,
                        correlation_group=fact.correlation_group,
                        directional_contribution=AnalyticalDirection.BUY if weighted > 0 else AnalyticalDirection.SELL,
                        raw_value=fact.raw,
                        normalized_score=round(fact.score, 6),
                        weight=weight,
                        effective_weight=round(weight * discount, 6),
                        weighted_score=round(weighted, 6),
                        quality=round(fact.quality, 6),
                        freshness=round(freshness, 6),
                        correlated_discount=discount,
                        reason=fact.reason,
                        source_fact_identifiers=fact.fact_ids or (str(fact.item.evidence_id),),
                    )
                )
        return tuple(result)

    def _confidence(
        self,
        direction: AnalyticalDirection,
        contributions: tuple[DirectionalContribution, ...],
        state: UnifiedMarketState,
        evidence: tuple[EvidenceItem, ...],
        *,
        timeframe_alignment: float,
        quant_ai_alignment: float,
        compression: float,
    ) -> ConfidenceDecomposition:
        bullish = fsum(max(0.0, item.weighted_score) for item in contributions)
        bearish = fsum(max(0.0, -item.weighted_score) for item in contributions)
        total = bullish + bearish
        separation = abs(bullish - bearish) / total * 100 if total else 0
        selected = tuple(item for item in contributions if item.directional_contribution == direction)
        groups = {item.correlation_group for item in selected}
        families = {item.family for item in selected}
        independent = min(100.0, len(families) / 5 * 100)
        confluence = min(100.0, len(groups) / 6 * 100)
        quality = _mean(item.quality * 100 for item in contributions)
        freshness = _mean(item.freshness * 100 for item in contributions)
        completeness = len({item.tool for item in contributions}) / 7 * 100
        contradiction = min(bullish, bearish) / total * 100 if total else 100
        missing = (1 - len(evidence) / 7) * 100
        regime_penalty = compression * 20
        final = (
            separation * 0.30
            + independent * 0.18
            + confluence * 0.12
            + quality * 0.15
            + freshness * 0.10
            + max(0, completeness) * 0.07
            + timeframe_alignment * 0.04
            + quant_ai_alignment * 0.04
            - contradiction * 0.18
            - max(0, missing) * 0.10
            - regime_penalty
        )
        return ConfidenceDecomposition(
            score_separation=round(separation, 4),
            independent_confluence=round((independent + confluence) / 2, 4),
            evidence_quality=round(quality, 4),
            evidence_freshness=round(freshness, 4),
            evidence_completeness=round(max(0, min(100, completeness)), 4),
            timeframe_alignment=round(timeframe_alignment, 4),
            quant_ai_alignment=round(quant_ai_alignment, 4),
            contradiction_penalty=round(contradiction, 4),
            missing_evidence_penalty=round(max(0, missing), 4),
            regime_suitability_penalty=round(regime_penalty, 4),
            final_confidence=round(max(0, min(100, final)), 4),
        )

    @staticmethod
    def _stronger_direction(
        contributions: tuple[DirectionalContribution, ...],
        bullish: float,
        bearish: float,
    ) -> AnalyticalDirection:
        if bullish > bearish:
            return AnalyticalDirection.BUY
        if bearish > bullish:
            return AnalyticalDirection.SELL
        strongest = max(contributions, key=lambda item: (abs(item.weighted_score), item.source_fact_identifiers))
        return strongest.directional_contribution

    @staticmethod
    def _thesis(
        timeframe: str,
        direction: AnalyticalDirection,
        supporting: list[DirectionalContribution],
        contrary: list[DirectionalContribution],
    ) -> str:
        supports = ", ".join(f"{item.tool}:{item.reason}" for item in supporting[:3])
        conflicts = ", ".join(f"{item.tool}:{item.reason}" for item in contrary[:2])
        return (
            f"{timeframe} {direction.value} is the stronger validated scenario. "
            f"Independent support: {supports or 'limited directional evidence'}. "
            f"Contradiction retained: {conflicts or 'none material'}."
        )

    @staticmethod
    def _compression(evidence: tuple[EvidenceItem, ...]) -> float:
        for item in evidence:
            if item.source_engine == "market_regime":
                return max(0.0, min(1.0, _num(_map(item.raw_value).get("compression_score"))))
        return 0.0

    @staticmethod
    def _quant_ai_alignment(timeframe: str, quant: QuantForecastResult, analysis: AIMarketAnalysis) -> float:
        assert analysis.output is not None
        matching = tuple(item for item in quant.predictions if item.horizon.timeframe == timeframe)
        if not matching:
            return 50.0
        quant_score = fsum(item.buy_probability - item.sell_probability for item in matching)
        ai_bias = analysis.output.momentum_analysis.direction
        ai_score = 1 if ai_bias == AnalysisBias.BULLISH else -1 if ai_bias == AnalysisBias.BEARISH else 0
        return 100.0 if quant_score * ai_score > 0 else 25.0 if quant_score * ai_score < 0 else 50.0

    @staticmethod
    def _geometry(
        direction: AnalyticalDirection,
        evidence: tuple[EvidenceItem, ...],
        minimum_rr: float,
    ) -> tuple[SignalGeometry | None, tuple[str, ...], tuple[str, ...]]:
        current = None
        zones: list[tuple[float, float, str, str]] = []
        targets: list[tuple[float, str, str]] = []
        invalidations: list[str] = []
        for item in evidence:
            raw = _map(item.raw_value)
            if item.source_engine == "market_data":
                current = _num(raw.get("close")) or current
            elif item.source_engine == "smc":
                for zone in _items(raw.get("zones")):
                    value = _map(zone)
                    lifecycle = str(value.get("lifecycle_state", "")).lower()
                    if lifecycle not in _ACTIVE_ZONE_STATES:
                        continue
                    low, high = _num(value.get("lower_price")), _num(value.get("upper_price"))
                    if low > 0 and high >= low:
                        zones.append((low, high, str(value.get("zone_type")), str(value.get("id"))))
                for event in _items(raw.get("structure_events"))[-3:]:
                    value = _map(event)
                    invalidations.append(f"{value.get('event_type')} invalidated beyond {value.get('broken_level')}")
            elif item.source_engine == "liquidity":
                pool_prices = {
                    str(value.get("id")): (
                        (_num(value.get("lower_bound")) + _num(value.get("upper_bound")))
                        / 2,
                        str(value.get("side")),
                    )
                    for pool in _items(raw.get("pools"))
                    if (value := _map(pool))
                    and str(value.get("lifecycle_state", "")).lower()
                    in _ACTIVE_LIQUIDITY_STATES
                    and _num(value.get("lower_bound")) > 0
                    and _num(value.get("upper_bound")) > 0
                }
                for target in _items(raw.get("targets")):
                    value = _map(target)
                    if value.get("status") != "active":
                        continue
                    pool_price = pool_prices.get(str(value.get("pool_id")))
                    price = (
                        _num(
                            value.get("price")
                            or value.get("lower_bound")
                            or value.get("upper_bound")
                        )
                        or (pool_price[0] if pool_price is not None else 0.0)
                    )
                    side = str(value.get("side")) or (
                        pool_price[1] if pool_price is not None else ""
                    )
                    if price:
                        targets.append((price, side, str(value.get("id"))))
                for pool in _items(raw.get("pools")):
                    value = _map(pool)
                    if (
                        str(value.get("lifecycle_state", "")).lower()
                        not in _ACTIVE_LIQUIDITY_STATES
                    ):
                        continue
                    price = (_num(value.get("lower_bound")) + _num(value.get("upper_bound"))) / 2
                    if price:
                        targets.append((price, str(value.get("side")), str(value.get("id"))))
        if current is None:
            return None, ("current_price_unavailable",), tuple(invalidations)
        if direction == AnalyticalDirection.BUY:
            entries = [item for item in zones if item[1] <= current and "bullish" in item[2]]
            entry_zone = max(entries, key=lambda item: item[1], default=None)
            entry = entry_zone[1] if entry_zone else current
            stop = entry_zone[0] if entry_zone and entry_zone[0] < entry else None
            target = min((item for item in targets if item[0] > entry and item[1] == "buy_side"), default=None)
            if target is None:
                supply = [item for item in zones if item[0] > entry and "bearish" in item[2]]
                zone = min(supply, key=lambda item: item[0], default=None)
                target = (zone[0], "supply", zone[3]) if zone else None
        else:
            entries = [item for item in zones if item[0] >= current and "bearish" in item[2]]
            entry_zone = min(entries, key=lambda item: item[0], default=None)
            entry = entry_zone[0] if entry_zone else current
            stop = entry_zone[1] if entry_zone and entry_zone[1] > entry else None
            target = max((item for item in targets if item[0] < entry and item[1] == "sell_side"), default=None)
            if target is None:
                demand = [item for item in zones if item[1] < entry and "bullish" in item[2]]
                zone = max(demand, key=lambda item: item[1], default=None)
                target = (zone[1], "demand", zone[3]) if zone else None
        if entry_zone is None:
            return None, ("no_fresh_directionally_aligned_entry_zone",), tuple(invalidations)
        if stop is None:
            return None, ("no_structural_invalidation_level",), tuple(invalidations)
        if target is None:
            return None, ("no_valid_structural_or_liquidity_target",), tuple(invalidations)
        risk = abs(entry - stop)
        reward = abs(target[0] - entry)
        rr = reward / risk if risk else 0
        if rr < minimum_rr:
            return None, ("risk_reward_below_minimum",), tuple(invalidations)
        geometry = SignalGeometry(
            entry=entry,
            stop_loss=stop,
            take_profit=target[0],
            risk_reward_ratio=round(rr, 8),
            basis_fact_identifiers=(entry_zone[3], entry_zone[3], target[2]),
        )
        return geometry, (), tuple(invalidations)

    @staticmethod
    def _fact(
        item: EvidenceItem,
        family: str,
        correlation_group: str,
        score: float,
        quality: float,
        raw: Any,
        reason: str,
        fact_ids: Iterable[object],
    ) -> _Fact:
        return _Fact(item=item, family=family, correlation_group=correlation_group, score=score, quality=quality, raw=raw, reason=reason, fact_ids=fact_ids)


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _percent(value: Any, default: float) -> float:
    number = _num(value)
    if number == 0:
        return default
    return max(0.0, min(1.0, number / 100 if number > 1 else number))


def _sign(value: float) -> float:
    return 1.0 if value > 0 else -1.0 if value < 0 else 0.0


def _direction(value: Any) -> float:
    text = str(value).lower()
    if "bullish" in text or text in {"buy", "buy_side", "up", "positive", "accumulation"}:
        return 1.0
    if "bearish" in text or text in {"sell", "sell_side", "down", "negative", "distribution"}:
        return -1.0
    return 0.0


def _primary_correlation_group(
    item: EvidenceItem,
    direction: float,
) -> str:
    if item.source_engine == "market_data":
        return f"market_candle:{item.provenance.get('market_event_id')}"
    if item.source_engine == "market_regime":
        return "regime_structure"
    if item.source_engine == "smc":
        return "structure_state"
    if item.source_engine == "liquidity":
        side = "buy_side" if direction > 0 else "sell_side"
        return f"liquidity_target:{side}"
    if item.source_engine == "institutional_flow":
        return "institutional_state"
    if item.source_engine == "volume_profile":
        return "volume_migration"
    return f"evidence:{item.evidence_id}"


def _quality(item: EvidenceItem) -> float:
    value = item.quality if item.quality is not None else item.confidence
    return _percent(value, 0.5)


def _freshness(item: EvidenceItem) -> float:
    if item.availability == EvidenceAvailability.STALE:
        return 0.25
    if item.availability == EvidenceAvailability.UNAVAILABLE:
        return 0.0
    if item.availability == EvidenceAvailability.DEGRADED:
        return 0.6
    return max(0.5, 1 / (1 + item.freshness_seconds / 900))


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return fsum(items) / len(items) if items else 0.0
