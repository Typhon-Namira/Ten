"""Deterministic institutional signal synthesis from point-in-time evidence."""

from __future__ import annotations

from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult
from backend.app.signal_synthesis import MultiTimeframeSignalSynthesizer

from .analysis import (
    AIAnalysisSignal,
    AIMarketAnalysis,
    AnalysisExecutionEligibility,
    AnalysisExecutionStatus,
    AnalysisSignalAction,
    AnalysisSignalLifecycle,
    QuantAIAlignment,
    signal_strength,
)
from .config import AIReasoningConfig


class DeterministicAnalysisSignalGenerator:
    """Project deterministic multi-timeframe analysis into the legacy read model."""

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
        """Project the combined multi-timeframe forecast into the legacy read contract.

        The complete M5/M15 matrix is persisted by the signal-synthesis repository.
        This projection keeps existing API consumers working without allowing execution
        blockers to erase the analytical BUY/SELL direction.
        """
        if analysis.output is None or not analysis.validation_passed:
            raise ValueError("analysis signal requires a validated analysis")
        bundle = MultiTimeframeSignalSynthesizer().synthesize(state, quant, analysis)
        combined = bundle.combined_signal
        candidate = AnalysisSignalAction(combined.analytical_direction.value)
        geometry = combined.geometry
        analysis_confidence = analysis.output.analysis_confidence * 100
        quant_confidence = combined.confidence_decomposition.quant_ai_alignment
        alignment = (
            QuantAIAlignment.AGREEMENT
            if quant_confidence >= 75
            else QuantAIAlignment.DISAGREEMENT
            if quant_confidence < 50
            else QuantAIAlignment.NEUTRAL
        )
        alignment_explanation = (
            f"Persisted M5/M15 synthesis reports {quant_confidence:.1f}% "
            "Quant/AI alignment; disagreement is retained in confidence."
        )
        rounded_confidence = round(combined.confidence)
        holding_seconds = 900
        valid_from = analysis.analysis_timestamp
        valid_until = valid_from + timedelta(seconds=holding_seconds)
        evidence_refs = tuple(
            dict.fromkeys(
                str(item.evidence_id)
                for item in combined.evidence_breakdown
                if item.directional_contribution.value == candidate.value
            )
        )
        components = {
            "bullish_score": combined.bullish_score,
            "bearish_score": combined.bearish_score,
            "score_separation": combined.confidence_decomposition.score_separation,
            "independent_confluence": combined.confidence_decomposition.independent_confluence,
            "evidence_quality": combined.confidence_decomposition.evidence_quality,
            "timeframe_alignment": combined.confidence_decomposition.timeframe_alignment,
            "contradiction_penalty": combined.confidence_decomposition.contradiction_penalty,
        }
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
            stop_loss=geometry.stop_loss if geometry else None,
            take_profit=geometry.take_profit if geometry else None,
            risk_reward_ratio=geometry.risk_reward_ratio if geometry else None,
            evidence_refs=evidence_refs,
            reasoning_summary=combined.directional_thesis,
            risk_flags=combined.blocking_reasons,
            scoring_components={key: round(value, 2) for key, value in components.items()},
            analysis_confidence=round(analysis_confidence, 2),
            signal_confidence=round(combined.confidence, 2),
            quant_confidence=round(quant_confidence, 2),
            overall_confidence=round(combined.confidence, 2),
            quant_ai_alignment=alignment,
            quant_ai_explanation=alignment_explanation,
            quality_threshold=self.quality_threshold,
            geometry_basis=geometry.basis_fact_identifiers if geometry else (),
            valid_from=valid_from,
            valid_until=valid_until,
            expected_holding_seconds=holding_seconds,
            lifecycle_status=(
                AnalysisSignalLifecycle.ACTIVE
                if combined.execution_eligibility.value
                == AnalysisExecutionEligibility.ELIGIBLE.value
                else AnalysisSignalLifecycle.COMPLETED
            ),
            execution_eligibility=AnalysisExecutionEligibility(
                combined.execution_eligibility.value
            ),
            execution_status=AnalysisExecutionStatus(combined.execution_status.value),
            blocking_reasons=combined.blocking_reasons,
            generated_at=analysis.created_at,
        )
