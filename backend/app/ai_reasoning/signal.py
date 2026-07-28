"""Deterministic confidence calibration and analysis-signal generation."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import NAMESPACE_URL, uuid5

from backend.app.market_state import EvidenceAvailability, UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult

from .analysis import (
    AIAnalysisSignal,
    AIMarketAnalysis,
    AnalysisBias,
    AnalysisSignalAction,
    RegimeClassification,
    signal_strength,
)


CONFIDENCE_WEIGHTS = {
    "market_structure_alignment": 0.16,
    "trend_regime_alignment": 0.14,
    "liquidity_evidence": 0.08,
    "volume_profile_confirmation": 0.08,
    "institutional_flow_confirmation": 0.08,
    "multi_timeframe_agreement": 0.10,
    "economic_event_safety": 0.08,
    "data_freshness": 0.08,
    "evidence_coverage": 0.10,
    "contradiction_control": 0.06,
    "risk_reward_quality": 0.04,
}


class DeterministicAnalysisSignalGenerator:
    """Generate BUY/SELL/HOLD without accepting provider trade instructions."""

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
        bullish_votes = sum(
            (
                output.market_regime.classification == RegimeClassification.BULLISH,
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
        bearish_votes = sum(
            (
                output.market_regime.classification == RegimeClassification.BEARISH,
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
        candidate = (
            AnalysisSignalAction.BUY
            if bullish_votes >= 3 and bullish_votes - bearish_votes >= 2
            else AnalysisSignalAction.SELL
            if bearish_votes >= 3 and bearish_votes - bullish_votes >= 2
            else AnalysisSignalAction.HOLD
        )
        direction_votes = max(bullish_votes, bearish_votes)
        engines = {
            item.source_engine.lower(): item
            for item in state.evidence
        }

        def engine_score(*names: str) -> float:
            matches = [
                value
                for name, value in engines.items()
                if any(candidate_name in name for candidate_name in names)
            ]
            if not matches:
                return 0.0
            if any(item.availability == EvidenceAvailability.AVAILABLE for item in matches):
                return 100.0
            if any(item.availability == EvidenceAvailability.DEGRADED for item in matches):
                return 50.0
            return 0.0

        major_contradiction = len(output.contradictions) >= 2
        high_impact_risk = any(
            "high_impact" in code.lower()
            for item in state.evidence
            if "economic" in item.source_engine.lower()
            for code in item.reason_codes
        )
        stale_or_degraded = bool(
            state.stale_evidence or state.degraded_evidence
        )
        missing_components = any(
            engine_score(*names) == 0
            for names in (
                ("liquidity",),
                ("volume_profile", "volume profile"),
                ("institutional_flow", "institutional flow"),
            )
        )
        components = {
            "market_structure_alignment": direction_votes / 4 * 100,
            "trend_regime_alignment": (
                100.0
                if output.market_regime.classification.value
                == output.higher_timeframe_context.bias.value
                else 50.0
                if output.higher_timeframe_context.bias
                in {AnalysisBias.MIXED, AnalysisBias.NEUTRAL}
                else 0.0
            ),
            "liquidity_evidence": engine_score("liquidity"),
            "volume_profile_confirmation": engine_score(
                "volume_profile", "volume profile"
            ),
            "institutional_flow_confirmation": engine_score(
                "institutional_flow", "institutional flow"
            ),
            "multi_timeframe_agreement": (
                sum(not frame.stale for frame in state.timeframes)
                / len(state.timeframes)
                * 100
            ),
            "economic_event_safety": 0.0 if high_impact_risk else 100.0,
            "data_freshness": 0.0 if state.stale_evidence else 100.0,
            "evidence_coverage": state.evidence_completeness * 100,
            "contradiction_control": max(
                0.0,
                100.0 - len(output.contradictions) * 25,
            ),
            "risk_reward_quality": (
                100.0
                if prediction and prediction.expected_base_movement > 0
                else 0.0
            ),
        }
        confidence = round(
            sum(
                components[name] * weight
                for name, weight in CONFIDENCE_WEIGHTS.items()
            )
        )
        risk_flags: list[str] = []
        if missing_components:
            confidence = min(confidence, 59)
            risk_flags.append("missing_required_market_components")
        if stale_or_degraded:
            confidence = min(confidence, 59)
            risk_flags.append("stale_or_degraded_source_data")
        if major_contradiction:
            confidence = min(confidence, 69)
            risk_flags.append("major_contradictory_evidence")
        if high_impact_risk:
            confidence = min(confidence, 69)
            risk_flags.append("unresolved_high_impact_economic_event")

        entry = stop = target = risk_reward = None
        if candidate != AnalysisSignalAction.HOLD:
            if prediction is None or prediction.expected_base_movement <= 0:
                candidate = AnalysisSignalAction.HOLD
                risk_flags.append("invalid_risk_reward")
            else:
                entry = prediction.reference_price
                movement = prediction.expected_base_movement
                if candidate == AnalysisSignalAction.BUY:
                    stop, target = entry - movement, entry + 2 * movement
                else:
                    stop, target = entry + movement, entry - 2 * movement
                if min(entry, stop, target) <= 0:
                    candidate = AnalysisSignalAction.HOLD
                    entry = stop = target = None
                    risk_flags.append("invalid_risk_reward")
                else:
                    risk_reward = 2.0
        if candidate == AnalysisSignalAction.HOLD:
            entry = stop = target = risk_reward = None
            if direction_votes < 3:
                confidence = min(confidence, 39)
                risk_flags.append("insufficient_evidence_for_direction")

        supporting = (
            output.bullish_evidence
            if candidate == AnalysisSignalAction.BUY
            else output.bearish_evidence
            if candidate == AnalysisSignalAction.SELL
            else ()
        )
        return AIAnalysisSignal(
            signal_id=uuid5(
                NAMESPACE_URL,
                f"ten:analysis-signal:{analysis.analysis_id}:1.0",
            ),
            cycle_id=analysis.cycle_id,
            snapshot_id=analysis.market_snapshot_id,
            analysis_id=analysis.analysis_id,
            instrument=analysis.symbol,
            timeframe=analysis.timeframe,
            signal=candidate,
            confidence=max(0, min(100, confidence)),
            strength=signal_strength(max(0, min(100, confidence))),
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            risk_reward_ratio=risk_reward,
            evidence_refs=_unique(
                item.source_reference for item in supporting
            ),
            reasoning_summary=output.executive_summary,
            risk_flags=tuple(dict.fromkeys(risk_flags)),
            scoring_components={
                key: round(value, 2) for key, value in components.items()
            },
            generated_at=analysis.created_at,
        )


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
