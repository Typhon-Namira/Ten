"""Deterministic institutional signal synthesis from point-in-time evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.market_state import EvidenceAvailability, UnifiedMarketState
from backend.app.quant_forecasting.models import HorizonPrediction, QuantForecastResult

from .analysis import (
    AIAnalysisSignal,
    AIMarketAnalysis,
    AnalysisBias,
    AnalysisSignalAction,
    AnalysisSignalLifecycle,
    QuantAIAlignment,
    RegimeClassification,
    signal_strength,
)
from .config import AIReasoningConfig


QUALITY_WEIGHTS = {
    "trend_alignment": 0.10,
    "higher_timeframe_alignment": 0.09,
    "market_structure": 0.10,
    "institutional_flow": 0.08,
    "smc_confirmation": 0.08,
    "liquidity_sweep": 0.07,
    "fair_value_gap": 0.05,
    "order_block": 0.05,
    "volume_profile": 0.06,
    "event_risk": 0.07,
    "atr": 0.05,
    "volatility": 0.04,
    "time_of_day": 0.04,
    "spread": 0.04,
    "expected_move": 0.08,
}

_ACTIVE_STATES = {
    "active",
    "confirmed",
    "touched",
    "partially_mitigated",
    "approached",
}


@dataclass(frozen=True)
class _StructuralZone:
    lower: float
    upper: float
    kind: str
    source: str


@dataclass(frozen=True)
class _Geometry:
    entry: float
    stop: float
    target: float
    risk_reward: float
    basis: tuple[str, ...]


class DeterministicAnalysisSignalGenerator:
    """Generate BUY/SELL/HOLD without accepting provider trade instructions.

    Price geometry is selected only from persisted point-in-time structure. Quant
    movement estimates score accessibility but never manufacture a price level.
    """

    def __init__(self, config: AIReasoningConfig | None = None) -> None:
        self.minimum_rr = (
            config.signal_minimum_risk_reward if config is not None else 2.0
        )
        self.preferred_rr = (
            config.signal_preferred_risk_reward if config is not None else 2.5
        )
        self.exceptional_rr = (
            config.signal_exceptional_risk_reward if config is not None else 3.0
        )
        self.quality_threshold = (
            config.signal_quality_threshold if config is not None else 65.0
        )

    def generate(
        self,
        analysis: AIMarketAnalysis,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
    ) -> AIAnalysisSignal:
        if analysis.output is None or not analysis.validation_passed:
            raise ValueError("analysis signal requires a validated analysis")
        output = analysis.output
        prediction = quant.predictions[0] if quant.predictions else None
        bullish_votes, bearish_votes = self._direction_votes(analysis, prediction)
        candidate = self._candidate(bullish_votes, bearish_votes)
        compression = _state_contains(state, "compression")
        engines = {
            item.source_engine.lower(): item
            for item in state.evidence
        }
        quant_action, quant_confidence = self._quant_direction(prediction)
        alignment, alignment_explanation = self._quant_alignment(
            candidate,
            quant_action,
            output.liquidity_analysis.events,
        )
        geometry = (
            self._geometry(candidate, state, analysis, prediction)
            if candidate != AnalysisSignalAction.HOLD
            else None
        )
        components = self._quality_components(
            candidate,
            state,
            engines,
            analysis,
            prediction,
            geometry,
            bullish_votes,
            bearish_votes,
        )
        quality_score = sum(
            components[name] * weight for name, weight in QUALITY_WEIGHTS.items()
        )
        analysis_confidence = output.analysis_confidence * 100
        signal_confidence = (
            quality_score * 0.50
            + analysis_confidence * 0.25
            + (quant_confidence if quant_confidence is not None else quality_score)
            * 0.25
        )
        risk_flags: list[str] = []
        if compression and candidate != AnalysisSignalAction.HOLD:
            candidate = AnalysisSignalAction.HOLD
            geometry = None
            risk_flags.append("compression_regime_prefers_hold")
        if state.stale_evidence or state.degraded_evidence:
            signal_confidence = min(signal_confidence, 59)
            risk_flags.append("stale_or_degraded_source_data")
        if len(output.contradictions) >= 2:
            signal_confidence = min(signal_confidence, 69)
            risk_flags.append("major_contradictory_evidence")
        if alignment == QuantAIAlignment.DISAGREEMENT:
            signal_confidence = min(signal_confidence, 69)
            risk_flags.append("quant_ai_directional_disagreement")
        if geometry is None and candidate != AnalysisSignalAction.HOLD:
            candidate = AnalysisSignalAction.HOLD
            risk_flags.append("no_structural_geometry_meets_minimum_risk_reward")
        if quality_score < self.quality_threshold:
            candidate = AnalysisSignalAction.HOLD
            geometry = None
            risk_flags.append("signal_quality_below_threshold")
        if max(bullish_votes, bearish_votes) < 3:
            candidate = AnalysisSignalAction.HOLD
            geometry = None
            signal_confidence = min(signal_confidence, 39)
            risk_flags.append("insufficient_evidence_for_direction")

        signal_confidence = max(0.0, min(100.0, signal_confidence))
        overall_confidence = (
            signal_confidence * 0.60
            + analysis_confidence * 0.20
            + (quant_confidence if quant_confidence is not None else signal_confidence)
            * 0.20
        )
        rounded_confidence = round(signal_confidence)
        holding_seconds = (
            prediction.horizon.duration_seconds if prediction is not None else 300
        )
        valid_from = analysis.analysis_timestamp
        valid_until = valid_from + timedelta(seconds=holding_seconds)
        supporting = (
            output.bullish_evidence
            if candidate == AnalysisSignalAction.BUY
            else output.bearish_evidence
            if candidate == AnalysisSignalAction.SELL
            else ()
        )
        reason = output.executive_summary
        if candidate == AnalysisSignalAction.HOLD:
            hold_reason = (
                risk_flags[-1].replace("_", " ")
                if risk_flags
                else "no directional setup passed deterministic quality controls"
            )
            reason = (
                f"HOLD: {hold_reason}. "
                f"{output.executive_summary}"
            )
        return AIAnalysisSignal(
            signal_id=uuid5(
                NAMESPACE_URL,
                f"ten:analysis-signal:{analysis.analysis_id}:2.0",
            ),
            cycle_id=analysis.cycle_id,
            snapshot_id=analysis.market_snapshot_id,
            analysis_id=analysis.analysis_id,
            instrument=analysis.symbol,
            timeframe=analysis.timeframe,
            signal=candidate,
            confidence=rounded_confidence,
            strength=signal_strength(rounded_confidence),
            entry=geometry.entry if geometry else None,
            stop_loss=geometry.stop if geometry else None,
            take_profit=geometry.target if geometry else None,
            risk_reward_ratio=geometry.risk_reward if geometry else None,
            evidence_refs=_unique(item.source_reference for item in supporting),
            reasoning_summary=reason,
            risk_flags=tuple(dict.fromkeys(risk_flags)),
            scoring_components={
                **{key: round(value, 2) for key, value in components.items()},
                "signal_quality": round(quality_score, 2),
            },
            analysis_confidence=round(analysis_confidence, 2),
            signal_confidence=round(signal_confidence, 2),
            quant_confidence=(
                round(quant_confidence, 2)
                if quant_confidence is not None
                else None
            ),
            overall_confidence=round(max(0, min(100, overall_confidence)), 2),
            quant_ai_alignment=alignment,
            quant_ai_explanation=alignment_explanation,
            quality_threshold=self.quality_threshold,
            geometry_basis=geometry.basis if geometry else (),
            valid_from=valid_from,
            valid_until=valid_until,
            expected_holding_seconds=holding_seconds,
            lifecycle_status=(
                AnalysisSignalLifecycle.ACTIVE
                if candidate != AnalysisSignalAction.HOLD
                else AnalysisSignalLifecycle.COMPLETED
            ),
            generated_at=analysis.created_at,
        )

    @staticmethod
    def _direction_votes(
        analysis: AIMarketAnalysis,
        prediction: HorizonPrediction | None,
    ) -> tuple[int, int]:
        assert analysis.output is not None
        output = analysis.output
        bullish = sum(
            (
                output.market_regime.classification
                == RegimeClassification.BULLISH,
                output.higher_timeframe_context.bias == AnalysisBias.BULLISH,
                output.momentum_analysis.direction == AnalysisBias.BULLISH,
                bool(
                    prediction
                    and prediction.buy_probability
                    > max(
                        prediction.sell_probability,
                        prediction.neutral_probability,
                    )
                ),
            )
        )
        bearish = sum(
            (
                output.market_regime.classification
                == RegimeClassification.BEARISH,
                output.higher_timeframe_context.bias == AnalysisBias.BEARISH,
                output.momentum_analysis.direction == AnalysisBias.BEARISH,
                bool(
                    prediction
                    and prediction.sell_probability
                    > max(
                        prediction.buy_probability,
                        prediction.neutral_probability,
                    )
                ),
            )
        )
        return bullish, bearish

    @staticmethod
    def _candidate(bullish: int, bearish: int) -> AnalysisSignalAction:
        if bullish >= 3 and bullish - bearish >= 2:
            return AnalysisSignalAction.BUY
        if bearish >= 3 and bearish - bullish >= 2:
            return AnalysisSignalAction.SELL
        return AnalysisSignalAction.HOLD

    @staticmethod
    def _quant_direction(
        prediction: HorizonPrediction | None,
    ) -> tuple[AnalysisSignalAction, float | None]:
        if prediction is None:
            return AnalysisSignalAction.HOLD, None
        probabilities = {
            AnalysisSignalAction.BUY: prediction.buy_probability,
            AnalysisSignalAction.SELL: prediction.sell_probability,
            AnalysisSignalAction.HOLD: prediction.neutral_probability,
        }
        action = max(probabilities, key=probabilities.__getitem__)
        return action, probabilities[action] * 100

    @staticmethod
    def _quant_alignment(
        candidate: AnalysisSignalAction,
        quant_action: AnalysisSignalAction,
        liquidity_events: tuple[str, ...],
    ) -> tuple[QuantAIAlignment, str]:
        if candidate == AnalysisSignalAction.HOLD:
            return (
                QuantAIAlignment.NEUTRAL,
                "Deterministic evidence did not support a directional signal.",
            )
        if quant_action == AnalysisSignalAction.HOLD:
            return (
                QuantAIAlignment.NEUTRAL,
                "Quant is neutral; structure determines whether the candidate survives guardrails.",
            )
        if candidate == quant_action:
            return (
                QuantAIAlignment.AGREEMENT,
                f"Quant momentum and validated structural analysis both support {candidate.value}.",
            )
        structural_reason = (
            ", ".join(liquidity_events[:2])
            if liquidity_events
            else "higher-timeframe structure and regime evidence"
        )
        return (
            QuantAIAlignment.DISAGREEMENT,
            f"Quant favors {quant_action.value} short-term momentum, while deterministic "
            f"analysis favors {candidate.value} because of {structural_reason}; confidence "
            "is capped and guardrails retain final authority.",
        )

    def _geometry(
        self,
        action: AnalysisSignalAction,
        state: UnifiedMarketState,
        analysis: AIMarketAnalysis,
        prediction: HorizonPrediction | None,
    ) -> _Geometry | None:
        if prediction is None or analysis.output is None:
            return None
        current = prediction.reference_price
        zones = _structural_zones(state)
        output = analysis.output
        if action == AnalysisSignalAction.BUY:
            supports = [
                item
                for item in zones
                if item.upper <= current
                and item.upper > item.lower
                and any(token in item.kind for token in ("bullish", "demand"))
            ]
            support = max(supports, key=lambda item: item.upper, default=None)
            entry = support.upper if support else current
            stop_values = {
                item.lower: item.source
                for item in zones
                if item.lower < entry
                and any(token in item.kind for token in ("bullish", "demand", "low", "sell_side"))
            }
            nearest_demand = output.supply_demand_analysis.nearest_demand
            if nearest_demand is not None and nearest_demand < entry:
                stop_values[nearest_demand] = "ai_analysis.nearest_demand"
            stop = max(stop_values, default=None)
            stop_source = stop_values.get(stop, "structural_support") if stop is not None else "missing"
            target_values = {
                item.lower: item.source
                for item in zones
                if item.lower > entry
                and any(token in item.kind for token in ("bearish", "supply", "high", "buy_side"))
            }
            nearest_supply = output.supply_demand_analysis.nearest_supply
            if nearest_supply is not None and nearest_supply > entry:
                target_values[nearest_supply] = "ai_analysis.nearest_supply"
            return self._select_geometry(
                action,
                entry,
                stop,
                target_values,
                (
                    support.source if support else "quant.reference_price",
                    stop_source,
                ),
                extended_target=(
                    output.market_regime.classification
                    == RegimeClassification.BULLISH
                ),
            )
        supports = [
            item
            for item in zones
            if item.lower >= current
            and item.upper > item.lower
            and any(token in item.kind for token in ("bearish", "supply"))
        ]
        support = min(supports, key=lambda item: item.lower, default=None)
        entry = support.lower if support else current
        stop_values = {
            item.upper: item.source
            for item in zones
            if item.upper > entry
            and any(token in item.kind for token in ("bearish", "supply", "high", "buy_side"))
        }
        nearest_supply = output.supply_demand_analysis.nearest_supply
        if nearest_supply is not None and nearest_supply > entry:
            stop_values[nearest_supply] = "ai_analysis.nearest_supply"
        stop = min(stop_values, default=None)
        stop_source = stop_values.get(stop, "structural_resistance") if stop is not None else "missing"
        target_values = {
            item.upper: item.source
            for item in zones
            if item.upper < entry
            and any(token in item.kind for token in ("bullish", "demand", "low", "sell_side"))
        }
        nearest_demand = output.supply_demand_analysis.nearest_demand
        if nearest_demand is not None and nearest_demand < entry:
            target_values[nearest_demand] = "ai_analysis.nearest_demand"
        return self._select_geometry(
            action,
            entry,
            stop,
            target_values,
            (
                support.source if support else "quant.reference_price",
                stop_source,
            ),
            extended_target=(
                output.market_regime.classification
                == RegimeClassification.BEARISH
            ),
        )

    def _select_geometry(
        self,
        action: AnalysisSignalAction,
        entry: float,
        stop: float | None,
        targets: Mapping[float, str],
        basis: tuple[str, str],
        *,
        extended_target: bool,
    ) -> _Geometry | None:
        if stop is None or min(entry, stop) <= 0:
            return None
        risk = entry - stop if action == AnalysisSignalAction.BUY else stop - entry
        if risk <= 0:
            return None
        ordered = sorted(targets, reverse=action == AnalysisSignalAction.SELL)
        valid: list[tuple[float, float, str]] = []
        for target in ordered:
            reward = (
                target - entry
                if action == AnalysisSignalAction.BUY
                else entry - target
            )
            rr = reward / risk
            if reward > 0 and rr >= self.minimum_rr:
                valid.append((target, rr, targets[target]))
        if not valid:
            return None
        preferred_values = [
            item for item in valid if item[1] >= self.preferred_rr
        ]
        exceptional_values = [
            item for item in preferred_values if item[1] >= self.exceptional_rr
        ]
        preferred = (
            max(exceptional_values, key=lambda item: item[1])
            if extended_target and exceptional_values
            else preferred_values[0]
            if preferred_values
            else None
        )
        target, rr, target_source = preferred or valid[0]
        return _Geometry(
            entry=entry,
            stop=stop,
            target=target,
            risk_reward=round(rr, 4),
            basis=(*basis, target_source),
        )

    def _quality_components(
        self,
        action: AnalysisSignalAction,
        state: UnifiedMarketState,
        engines: Mapping[str, Any],
        analysis: AIMarketAnalysis,
        prediction: HorizonPrediction | None,
        geometry: _Geometry | None,
        bullish_votes: int,
        bearish_votes: int,
    ) -> dict[str, float]:
        assert analysis.output is not None
        output = analysis.output

        def availability(*names: str) -> float:
            matches = [
                value
                for name, value in engines.items()
                if any(token in name for token in names)
            ]
            if not matches:
                return 0.0
            if any(item.availability == EvidenceAvailability.AVAILABLE for item in matches):
                return 100.0
            if any(item.availability == EvidenceAvailability.DEGRADED for item in matches):
                return 40.0
            return 0.0

        direction = (
            "bullish"
            if action == AnalysisSignalAction.BUY
            else "bearish"
            if action == AnalysisSignalAction.SELL
            else "neutral"
        )
        smc_match = _raw_contains(engines.get("smc"), direction)
        flow_match = _raw_contains(
            engines.get("institutional_flow"),
            "buying" if action == AnalysisSignalAction.BUY else "selling",
        )
        liquidity_text = " ".join(
            (*output.liquidity_analysis.events, *output.liquidity_analysis.unresolved_liquidity)
        ).lower()
        schedule = state.market_schedule
        market_open = bool(schedule and schedule.market_open)
        spread_score = _spread_score(state, prediction)
        expected_score = 0.0
        if prediction is not None and geometry is not None:
            target_distance = abs(geometry.target - geometry.entry)
            expected = prediction.expected_maximum_movement
            expected_score = (
                100.0
                if expected > 0 and target_distance <= expected
                else 60.0
                if expected > 0 and target_distance <= expected * 2
                else 25.0
            )
        return {
            "trend_alignment": max(bullish_votes, bearish_votes) / 4 * 100,
            "higher_timeframe_alignment": (
                100.0
                if output.higher_timeframe_context.bias.value == direction
                else 50.0
                if output.higher_timeframe_context.bias
                in {AnalysisBias.MIXED, AnalysisBias.NEUTRAL}
                else 0.0
            ),
            "market_structure": min(
                100.0,
                availability("smc") * 0.5
                + len(output.market_structure.evidence) * 20,
            ),
            "institutional_flow": (
                100.0 if flow_match else availability("institutional_flow") * 0.5
            ),
            "smc_confirmation": (
                100.0 if smc_match else availability("smc") * 0.5
            ),
            "liquidity_sweep": (
                100.0
                if any(token in liquidity_text for token in ("sweep", "grab", "raid", "reclaim"))
                else availability("liquidity") * 0.5
            ),
            "fair_value_gap": (
                100.0
                if any("fvg" in item.kind for item in _structural_zones(state))
                else 0.0
            ),
            "order_block": (
                100.0
                if any("order_block" in item.kind for item in _structural_zones(state))
                else 0.0
            ),
            "volume_profile": availability("volume_profile", "volume profile"),
            "event_risk": (
                0.0
                if any(
                    "high_impact" in code.lower()
                    for item in state.evidence
                    if "economic" in item.source_engine.lower()
                    for code in item.reason_codes
                )
                else availability("economic")
            ),
            "atr": 100.0 if _has_numeric_key(state, "atr") else 50.0 if prediction else 0.0,
            "volatility": (
                25.0
                if output.volatility_analysis.state.value == "extreme"
                else 75.0
                if output.volatility_analysis.state.value != "uncertain"
                else 0.0
            ),
            "time_of_day": 100.0 if market_open else 0.0,
            "spread": spread_score,
            "expected_move": expected_score,
        }


def _structural_zones(state: UnifiedMarketState) -> tuple[_StructuralZone, ...]:
    zones: list[_StructuralZone] = []
    for evidence in state.evidence:
        if evidence.availability not in {
            EvidenceAvailability.AVAILABLE,
            EvidenceAvailability.DEGRADED,
        }:
            continue
        raw = evidence.raw_value
        if not isinstance(raw, Mapping):
            continue
        source = f"{evidence.source_engine}:{evidence.source_timeframe}"
        if "smc" in evidence.source_engine.lower():
            snapshot = raw.get("snapshot")
            if isinstance(snapshot, Mapping):
                for index, item in enumerate(_sequence(snapshot.get("zones"))):
                    if not isinstance(item, Mapping) or not _active(item):
                        continue
                    _append_zone(
                        zones,
                        item.get("lower_price"),
                        item.get("upper_price"),
                        str(item.get("zone_type", "smc_zone")),
                        f"{source}.snapshot.zones[{index}]",
                    )
                for index, item in enumerate(_sequence(snapshot.get("swings"))):
                    if not isinstance(item, Mapping):
                        continue
                    price = _positive(item.get("price"))
                    if price is not None:
                        zones.append(
                            _StructuralZone(
                                price,
                                price,
                                str(item.get("swing_type", "swing")),
                                f"{source}.snapshot.swings[{index}]",
                            )
                        )
        if "liquidity" in evidence.source_engine.lower():
            snapshot = raw.get("snapshot")
            if isinstance(snapshot, Mapping):
                for collection in ("pools", "levels", "sessions", "map_bands"):
                    for index, item in enumerate(_sequence(snapshot.get(collection))):
                        if not isinstance(item, Mapping) or not _active(item):
                            continue
                        lower = item.get("lower_bound", item.get("low", item.get("price")))
                        upper = item.get("upper_bound", item.get("high", item.get("price")))
                        side = str(item.get("side", item.get("level_type", collection)))
                        _append_zone(
                            zones,
                            lower,
                            upper,
                            side,
                            f"{source}.snapshot.{collection}[{index}]",
                        )
        if "volume_profile" in evidence.source_engine.lower():
            for key, kind in (("val", "volume_low"), ("vah", "volume_high"), ("poc", "volume_poc")):
                price = _positive(raw.get(key))
                if price is not None:
                    zones.append(_StructuralZone(price, price, kind, f"{source}.{key}"))
    return tuple(dict.fromkeys(zones))


def _append_zone(
    output: list[_StructuralZone],
    lower_value: object,
    upper_value: object,
    kind: str,
    source: str,
) -> None:
    lower = _positive(lower_value)
    upper = _positive(upper_value)
    if lower is not None and upper is not None and lower <= upper:
        output.append(_StructuralZone(lower, upper, kind.lower(), source))


def _active(value: Mapping[str, object]) -> bool:
    state = str(value.get("lifecycle_state", "active")).lower()
    return state in _ACTIVE_STATES


def _sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else ()


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if numeric > 0 else None


def _raw_contains(evidence: Any, expected: str) -> bool:
    if evidence is None:
        return False
    return expected.lower() in str(evidence.raw_value).lower()


def _state_contains(state: UnifiedMarketState, expected: str) -> bool:
    token = expected.lower()
    return any(token in str(item.raw_value).lower() for item in state.evidence)


def _has_numeric_key(state: UnifiedMarketState, key: str) -> bool:
    def walk(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(
                str(name).lower() == key and _positive(item) is not None
                or walk(item)
                for name, item in value.items()
            )
        if isinstance(value, (tuple, list)):
            return any(walk(item) for item in value)
        return False

    return any(walk(item.raw_value) for item in state.evidence)


def _spread_score(
    state: UnifiedMarketState,
    prediction: HorizonPrediction | None,
) -> float:
    spreads: list[float] = []
    for item in state.evidence:
        if "market_data" not in item.source_engine.lower():
            continue
        raw = item.raw_value
        if isinstance(raw, Mapping):
            value = raw.get("spread")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                spreads.append(float(value))
    if not spreads or prediction is None or prediction.expected_base_movement <= 0:
        return 0.0
    spread = max(spreads)
    if spread == 0:
        return 0.0
    ratio = spread / prediction.expected_base_movement
    return 100.0 if ratio <= 0.1 else 70.0 if ratio <= 0.25 else 30.0 if ratio <= 0.5 else 0.0


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
