from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from hashlib import sha256
import math

from backend.app.engines.ai_scoring_engine import ScoreMode, ScoreStatus
from backend.app.ai_reasoning.analysis import AnalysisSignalAction

from .config import SignalDecisionConfig
from .models import (
    DecisionDirection,
    DecisionExplanation,
    DecisionLifecycleStatus,
    DecisionMetadata,
    DecisionReason,
    DecisionState,
    DependencyCriticality,
    DependencyState,
    RuleEvaluation,
    RuleOutcome,
    RuleSeverity,
    FinalSignalAction,
    SignalEvidence,
    SignalReadiness,
    SignalSourceLineage,
    SignalDecision,
    SignalDecisionInput,
    stable_id,
)
from .rules import RuleRegistry, production_rule_registry


class ConservativeSignalDecisionPolicy:
    name = "conservative_signal_policy"
    version = "1.0.0"

    def __init__(self, config: SignalDecisionConfig | None = None, rules: RuleRegistry | None = None) -> None:
        self.config = config or SignalDecisionConfig()
        self.name = self.config.policy_name
        self.version = self.config.policy_version
        self.rules = rules or production_rule_registry()
        self.configuration_hash = sha256(self.config.model_dump_json().encode()).hexdigest()

    def evaluate(self, decision_input: SignalDecisionInput) -> SignalDecision:
        score = decision_input.ai_score
        direction = DecisionDirection(self.config.direction_mapping[score.directional_label.value])
        strength = abs(score.directional_score)
        age = (decision_input.as_of - score.as_of).total_seconds()
        window = self.config.freshness[decision_input.timeframe]
        evaluations: list[RuleEvaluation] = []

        def rule(rule_id: str, outcome: RuleOutcome, severity: RuleSeverity, reason: str, observed: str | int | float | bool | None = None, threshold: str | int | float | bool | None = None, contribution: float | None = None) -> None:
            definition = self.rules.get(rule_id)
            evaluations.append(
                RuleEvaluation(
                    rule_id=rule_id,
                    rule_version=definition.version,
                    category=definition.category,
                    severity=severity,
                    outcome=outcome,
                    observed_value=observed,
                    threshold=threshold,
                    reason_code=reason,
                    contribution=contribution,
                    evaluated_at=decision_input.as_of,
                )
            )

        policy_valid = decision_input.policy_name == self.name and decision_input.policy_version == self.version
        rule("policy.integrity", RuleOutcome.PASSED if policy_valid else RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "policy_valid" if policy_valid else "policy_mismatch", f"{decision_input.policy_name}:{decision_input.policy_version}", f"{self.name}:{self.version}")
        metadata_valid = len(score.metadata.configuration_hash) == 64 and len(score.metadata.input_fingerprint) == 64
        score_mode_valid = decision_input.mode.value == score.mode.value or decision_input.mode.value == "replay" and score.mode == ScoreMode.LIVE
        source_valid = score.status != ScoreStatus.INVALID and metadata_valid and score_mode_valid
        rule("source.snapshot_integrity", RuleOutcome.PASSED if source_valid else RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "snapshot_integrity_valid" if source_valid else "snapshot_integrity_invalid", score.status.value)
        point_in_time = score.as_of <= decision_input.as_of and score.calculated_at <= decision_input.requested_at
        rule("source.point_in_time", RuleOutcome.PASSED if point_in_time else RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "point_in_time_valid" if point_in_time else "future_snapshot", score.as_of.isoformat(), decision_input.as_of.isoformat())

        analysis = decision_input.current_ai_analysis
        analysis_signal = decision_input.current_ai_signal
        analysis_valid = bool(
            analysis is not None
            and analysis.validation_passed
            and analysis.output is not None
            and analysis.analysis_timestamp <= decision_input.as_of
        )
        rule(
            "source.snapshot_integrity",
            RuleOutcome.PASSED,
            RuleSeverity.INFORMATIONAL,
            "ai_analysis_valid" if analysis_valid else "ai_analysis_missing_or_invalid",
            str(analysis.analysis_id) if analysis else None,
        )
        ai_alignment = True
        if analysis_valid and analysis is not None and analysis.output is not None:
            analysis_regime = analysis.output.market_regime.classification.value
            ai_alignment = (
                direction == DecisionDirection.NEUTRAL
                or analysis_regime not in {"bullish", "bearish"}
                or analysis_regime == direction.value
            )
            rule(
                "alignment.minimum",
                RuleOutcome.PASSED if ai_alignment else RuleOutcome.WARNING,
                RuleSeverity.INFORMATIONAL if ai_alignment else RuleSeverity.SOFT_GATE,
                "ai_analysis_aligned" if ai_alignment else "ai_analysis_contradicts_market_score",
                analysis_regime,
                direction.value,
            )
        else:
            rule(
                "alignment.minimum",
                RuleOutcome.NOT_EVALUATED,
                RuleSeverity.INFORMATIONAL,
                "ai_analysis_unavailable",
            )
        if analysis_signal is None:
            rule(
                "ai.signal_authority",
                RuleOutcome.NOT_EVALUATED,
                RuleSeverity.INFORMATIONAL,
                "ai_signal_unavailable",
            )
        elif analysis_signal.signal == AnalysisSignalAction.HOLD:
            rule(
                "ai.signal_authority",
                RuleOutcome.PASSED,
                RuleSeverity.INFORMATIONAL,
                "ai_signal_hold_preserved",
                str(analysis_signal.signal_id),
            )
        else:
            signal_aligned = (
                analysis_signal.signal == AnalysisSignalAction.BUY
                and direction == DecisionDirection.BULLISH
                or analysis_signal.signal == AnalysisSignalAction.SELL
                and direction == DecisionDirection.BEARISH
            )
            rule(
                "ai.signal_authority",
                RuleOutcome.PASSED if signal_aligned else RuleOutcome.WARNING,
                RuleSeverity.INFORMATIONAL,
                (
                    "ai_signal_accepted"
                    if signal_aligned
                    else "ai_signal_rejected_direction_mismatch"
                ),
                analysis_signal.signal.value,
                direction.value,
            )
        temporal = decision_input.temporal_metrics
        if temporal is None:
            rule(
                "temporal.validity",
                RuleOutcome.PASSED,
                RuleSeverity.INFORMATIONAL,
                "temporal_history_building",
            )
        else:
            consistency = temporal.historical_consistency
            temporal_unstable = consistency.classification.value == "unstable"
            rule(
                "temporal.validity",
                RuleOutcome.WARNING if temporal_unstable else RuleOutcome.PASSED,
                RuleSeverity.SOFT_GATE if temporal_unstable else RuleSeverity.INFORMATIONAL,
                "temporal_analysis_unstable" if temporal_unstable else "temporal_analysis_acceptable",
                consistency.score,
                50,
            )

        if age > window.observe_max_age_seconds or score.status == ScoreStatus.STALE:
            rule("freshness.ai_score", RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "ai_score_stale", age, window.observe_max_age_seconds)
            freshness_factor = 0.0
        elif age > window.eligible_max_age_seconds:
            rule("freshness.ai_score", RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, "ai_score_aging", age, window.eligible_max_age_seconds)
            freshness_factor = max(0.0, 1 - age / window.observe_max_age_seconds)
        else:
            rule("freshness.ai_score", RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, "ai_score_fresh", age, window.eligible_max_age_seconds)
            freshness_factor = 1.0

        previous = decision_input.history.active
        stay_eligible = bool(previous and previous.state == DecisionState.ELIGIBLE and previous.direction == direction and self.config.hysteresis.enabled)
        strength_eligible = max(0.0, self.config.directional_strength.minimum_for_eligibility - (self.config.hysteresis.strength_relief if stay_eligible else 0))
        confidence_eligible = max(0.0, self.config.confidence.minimum_for_eligibility - (self.config.hysteresis.confidence_relief if stay_eligible else 0))
        self._threshold_rule(rule, "strength.minimum", strength, self.config.directional_strength.minimum_for_observation, strength_eligible, "directional_strength")
        self._threshold_rule(rule, "confidence.minimum", score.confidence_score, self.config.confidence.minimum_for_observation, confidence_eligible, "confidence")

        if score.market_risk_score >= self.config.risk.hard_block_minimum:
            rule("risk.maximum", RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "risk_hard_block", score.market_risk_score, self.config.risk.hard_block_minimum)
        elif score.market_risk_score >= self.config.risk.preferred_maximum:
            rule("risk.maximum", RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, "risk_elevated", score.market_risk_score, self.config.risk.preferred_maximum)
        else:
            rule("risk.maximum", RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, "risk_preferred", score.market_risk_score, self.config.risk.preferred_maximum)

        if score.data_quality_score < self.config.data_quality.invalid_below:
            rule("quality.minimum", RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "data_quality_invalid", score.data_quality_score, self.config.data_quality.invalid_below)
        else:
            self._threshold_rule(rule, "quality.minimum", score.data_quality_score, self.config.data_quality.minimum_for_observation, self.config.data_quality.minimum_for_eligibility, "data_quality")
        self._threshold_rule(rule, "alignment.minimum", score.evidence_alignment_score, self.config.alignment.minimum_for_observation, self.config.alignment.minimum_for_eligibility, "alignment")

        penalties = math.fsum(item.confidence_penalty for item in score.conflicts)
        severe = any(item.severity == "severe" for item in score.conflicts) or penalties >= self.config.conflicts.severe_penalty_threshold
        moderate = bool(score.conflicts)
        if severe:
            rule("conflict.maximum", RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "severe_evidence_conflict", penalties, self.config.conflicts.severe_penalty_threshold)
        elif moderate:
            rule("conflict.maximum", RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, "moderate_evidence_conflict", penalties, self.config.conflicts.severe_penalty_threshold)
        else:
            rule("conflict.maximum", RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, "no_material_conflict", 0, self.config.conflicts.severe_penalty_threshold)

        economic = decision_input.economic_risk
        if not self.config.economic_event.enabled:
            rule("economic_event.window", RuleOutcome.NOT_APPLICABLE, RuleSeverity.INFORMATIONAL, "economic_policy_disabled")
        elif economic is None or economic.degraded:
            # `degraded` is only ever true for a genuine data-unavailability category (provider
            # unreachable/timeout/auth-failed/rate-limited, or no calendar data at all) — "no
            # relevant events" and "outside the risk window" are routine, healthy states that
            # never reach this branch. The reason code is the specific category, not a generic
            # "unavailable" label, so explainability/diagnostics can say exactly what failed.
            fail_closed = self.config.economic_event.fail_closed_when_source_unavailable
            reason = economic.context_state if economic and economic.context_state else "economic_context_unavailable"
            rule("economic_event.window", RuleOutcome.FAILED if fail_closed else RuleOutcome.WARNING, RuleSeverity.HARD_BLOCK if fail_closed else RuleSeverity.SOFT_GATE, reason)
        elif economic.phase in self.config.economic_event.hard_phases:
            rule("economic_event.window", RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "economic_hard_block_window", economic.phase)
        elif economic.phase in self.config.economic_event.caution_phases:
            rule("economic_event.window", RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, "economic_caution_window", economic.phase)
        else:
            rule("economic_event.window", RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, "economic_window_clear", economic.phase)

        regime = decision_input.market_regime
        if regime is None or regime.degraded:
            rule("regime.allowed", RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, "regime_context_unavailable")
        else:
            outcome = self.config.market_regimes.get(regime.regime, "blocked")
            if outcome == "blocked":
                rule("regime.allowed", RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "regime_blocked", regime.regime)
            elif outcome == "observe_only":
                rule("regime.allowed", RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, "regime_observe_only", regime.regime)
            else:
                rule("regime.allowed", RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, "regime_allowed", regime.regime)

        critical_failure = any(item.state == DependencyState.UNAVAILABLE and item.criticality == DependencyCriticality.CRITICAL for item in decision_input.dependency_health)
        eligibility_failure = any(item.state != DependencyState.AVAILABLE and item.criticality == DependencyCriticality.REQUIRED_FOR_ELIGIBILITY for item in decision_input.dependency_health)
        optional_degraded = any(item.state != DependencyState.AVAILABLE and item.criticality == DependencyCriticality.OPTIONAL for item in decision_input.dependency_health)
        if critical_failure:
            rule("dependency.health", RuleOutcome.FAILED, RuleSeverity.HARD_BLOCK, "critical_dependency_unavailable")
        elif eligibility_failure or optional_degraded:
            rule("dependency.health", RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, "dependency_degraded")
        else:
            rule("dependency.health", RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, "dependencies_available")

        rule("duplicate.active_equivalent", RuleOutcome.NOT_APPLICABLE, RuleSeverity.INFORMATIONAL, "duplicate_checked_by_repository")
        same = decision_input.history.recent_same_direction
        cooldown_seconds = self._cooldown_seconds(same.state if same else None)
        cooldown_active = bool(self.config.cooldown.enabled and same and (decision_input.as_of - same.decided_at).total_seconds() < cooldown_seconds)
        rule("cooldown.instrument_direction", RuleOutcome.FAILED if cooldown_active else RuleOutcome.PASSED, RuleSeverity.HARD_BLOCK if cooldown_active else RuleSeverity.INFORMATIONAL, "cooldown_active" if cooldown_active else "cooldown_clear", (decision_input.as_of - same.decided_at).total_seconds() if same else None, cooldown_seconds)

        opposite = decision_input.history.recent_opposite_eligible
        reversal_active = bool(self.config.reversal.enabled and opposite and direction != DecisionDirection.NEUTRAL and (decision_input.as_of - opposite.decided_at).total_seconds() < self.config.reversal.lock_seconds and strength < opposite.directional_strength + self.config.reversal.additional_strength_required and score.confidence_score < opposite.confidence_score + self.config.reversal.additional_confidence_required)
        rule("reversal.opposite_direction_lock", RuleOutcome.FAILED if reversal_active else RuleOutcome.PASSED, RuleSeverity.HARD_BLOCK if reversal_active else RuleSeverity.INFORMATIONAL, "reversal_lock_active" if reversal_active else "reversal_clear", (decision_input.as_of - opposite.decided_at).total_seconds() if opposite else None, self.config.reversal.lock_seconds)
        rule("temporal.validity", RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, "validity_configured")

        invalid = any(item.outcome == RuleOutcome.FAILED and item.rule_id in {"policy.integrity", "source.snapshot_integrity", "source.point_in_time"} for item in evaluations) or score.data_quality_score < self.config.data_quality.invalid_below
        hard = [item for item in evaluations if item.outcome == RuleOutcome.FAILED and item.severity == RuleSeverity.HARD_BLOCK]
        insufficient = [item for item in evaluations if item.outcome == RuleOutcome.FAILED and item.severity == RuleSeverity.INSUFFICIENT]
        soft = [item for item in evaluations if item.outcome == RuleOutcome.WARNING and item.severity == RuleSeverity.SOFT_GATE]
        if score.status == ScoreStatus.INSUFFICIENT_EVIDENCE:
            insufficient.append(self._synthetic_limit(decision_input, "ai_score_insufficient_evidence"))
        if score.status == ScoreStatus.DEGRADED:
            soft.append(self._synthetic_warning(decision_input, "ai_score_degraded"))
        if invalid:
            state = DecisionState.INVALID
        elif hard:
            state = DecisionState.BLOCKED
        elif insufficient:
            state = DecisionState.INSUFFICIENT_EVIDENCE
        elif soft:
            state = DecisionState.OBSERVE_ONLY
        else:
            state = DecisionState.ELIGIBLE

        ai_confidence = (
            float(analysis_signal.confidence)
            if analysis_signal is not None
            else 0.0
        )
        temporal_confidence = (
            decision_input.temporal_metrics.historical_consistency.score
            if decision_input.temporal_metrics is not None
            else 50.0
        )
        independent_confidence = (
            score.confidence_score * 0.55
            + ai_confidence * 0.20
            + temporal_confidence * 0.15
            + score.data_quality_score * 0.10
            - (10.0 if not ai_alignment else 0.0)
        )
        independent_confidence = max(0.0, min(100.0, independent_confidence))
        eligibility = self._eligibility(strength, independent_confidence, score.data_quality_score, score.evidence_alignment_score, score.market_risk_score, freshness_factor)
        validity = self.config.validity_seconds(state.value, decision_input.timeframe)
        fingerprint = decision_input.fingerprint(self.configuration_hash)
        blockers = tuple(self._reason(item) for item in evaluations if item.outcome == RuleOutcome.FAILED)
        if score.status == ScoreStatus.INSUFFICIENT_EVIDENCE:
            blockers += (self._reason(insufficient[-1]),)
        warnings = tuple(self._reason(item) for item in evaluations if item.outcome == RuleOutcome.WARNING)
        if score.status == ScoreStatus.DEGRADED:
            warnings += (self._reason(soft[-1]),)
        supporting = tuple(self._reason(item) for item in evaluations if item.outcome == RuleOutcome.PASSED)
        explanation = DecisionExplanation(
            summary_code=f"{direction.value}_{state.value}",
            decision_state=state,
            direction=direction,
            passed_rules=tuple(item.rule_id for item in evaluations if item.outcome == RuleOutcome.PASSED),
            failed_rules=tuple(item.rule_id for item in evaluations if item.outcome == RuleOutcome.FAILED),
            warning_rules=tuple(item.rule_id for item in evaluations if item.outcome == RuleOutcome.WARNING),
            evidence_limitations=tuple(item.reason_code for item in (*insufficient, *soft)),
            policy_name=self.name,
            policy_version=self.version,
        )
        previous = decision_input.history.latest
        active = decision_input.history.active
        actionable = (
            state == DecisionState.ELIGIBLE
            and direction != DecisionDirection.NEUTRAL
            and analysis_valid
            and analysis_signal is not None
            and analysis_signal.signal
            in {AnalysisSignalAction.BUY, AnalysisSignalAction.SELL}
            and (
                analysis_signal.signal == AnalysisSignalAction.BUY
                and direction == DecisionDirection.BULLISH
                or analysis_signal.signal == AnalysisSignalAction.SELL
                and direction == DecisionDirection.BEARISH
            )
            and decision_input.current_price is not None
            and decision_input.expected_move is not None
        )
        final_action = (
            FinalSignalAction.BUY
            if actionable and direction == DecisionDirection.BULLISH
            else FinalSignalAction.SELL
            if actionable and direction == DecisionDirection.BEARISH
            else FinalSignalAction.WAIT
        )
        entry_low: float | None = None
        entry_high: float | None = None
        stop_loss: float | None = None
        take_profit_targets: tuple[float, ...] = ()
        risk_reward: float | None = None
        invalidation: str | None = None
        if actionable:
            assert decision_input.current_price is not None
            assert decision_input.expected_move is not None
            price = decision_input.current_price
            movement = decision_input.expected_move
            entry_low = price - movement * 0.05
            entry_high = price + movement * 0.05
            if final_action == FinalSignalAction.BUY:
                stop_loss = entry_low - movement * 0.5
                take_profit_targets = (entry_high + movement, entry_high + movement * 1.5)
                invalidation = f"Price closes below {stop_loss:.5f}"
            else:
                stop_loss = entry_high + movement * 0.5
                take_profit_targets = (entry_low - movement, entry_low - movement * 1.5)
                invalidation = f"Price closes above {stop_loss:.5f}"
            entry_mid = (entry_low + entry_high) / 2
            risk_reward = abs(take_profit_targets[0] - entry_mid) / abs(entry_mid - stop_loss)
        ai_evidence: tuple[SignalEvidence, ...] = ()
        contradictory_ai_evidence: tuple[SignalEvidence, ...] = ()
        if analysis_valid and analysis is not None and analysis.output is not None:
            regime_evidence = SignalEvidence(
                evidence_id=stable_id("signal-evidence", fingerprint, "ai-regime"),
                category="market_regime",
                side="supporting" if ai_alignment else "contradicting",
                claim=(
                    f"Validated AI analysis classifies regime as "
                    f"{analysis.output.market_regime.classification.value}."
                ),
                source_type="current_ai_analysis",
                source_reference=str(analysis.analysis_id),
                observed_value=analysis.output.market_regime.strength,
                timestamp=analysis.analysis_timestamp,
                timeframe=decision_input.timeframe,
                reliability=analysis.output.market_regime.confidence,
            )
            if ai_alignment:
                ai_evidence = (regime_evidence,)
            else:
                contradictory_ai_evidence = (regime_evidence,)
        temporal_evidence: tuple[SignalEvidence, ...] = ()
        if decision_input.temporal_metrics is not None:
            consistency = decision_input.temporal_metrics.historical_consistency
            temporal_evidence = (
                SignalEvidence(
                    evidence_id=stable_id("signal-evidence", fingerprint, "temporal"),
                    category="historical_consistency",
                    side="supporting",
                    claim=consistency.reason,
                    source_type="temporal_metric",
                    source_reference=decision_input.temporal_context.version if decision_input.temporal_context else "1.0",
                    observed_value=consistency.score,
                    timestamp=decision_input.as_of,
                    timeframe=decision_input.timeframe,
                    reliability=min(1.0, consistency.sample_size / 10),
                ),
            )
        historical_ids = (
            tuple(item.analysis_id for item in decision_input.temporal_context.rolling_window)
            if decision_input.temporal_context is not None
            else ()
        )
        return SignalDecision(
            decision_id=stable_id("decision", fingerprint, decision_input.mode.value),
            decision_key=f"{decision_input.instrument}:{decision_input.timeframe}:{direction.value}:{self.version}:{decision_input.mode.value}",
            input_fingerprint=fingerprint,
            instrument=decision_input.instrument,
            timeframe=decision_input.timeframe,
            direction=direction,
            state=state,
            as_of=decision_input.as_of,
            decided_at=decision_input.requested_at,
            valid_from=decision_input.as_of,
            valid_until=decision_input.as_of + timedelta(seconds=validity),
            ai_score_snapshot_id=score.snapshot_id,
            ai_score_policy_name=score.policy_name,
            ai_score_policy_version=score.policy_version,
            decision_policy_name=self.name,
            decision_policy_version=self.version,
            eligibility_score=eligibility,
            directional_strength=self._rounded(strength),
            confidence_score=self._rounded(independent_confidence),
            market_risk_score=score.market_risk_score,
            data_quality_score=score.data_quality_score,
            evidence_alignment_score=score.evidence_alignment_score,
            rules=tuple(evaluations),
            blockers=blockers,
            warnings=warnings,
            supporting_reasons=supporting,
            previous_decision_id=previous.decision_id if previous else None,
            supersedes_decision_id=active.decision_id if active else None,
            status=DecisionLifecycleStatus.ACTIVE,
            mode=decision_input.mode,
            explanation=explanation,
            metadata=DecisionMetadata(
                engine_version=self.config.engine_version,
                schema_version=self.config.schema_version,
                configuration_version=self.config.configuration_version,
                configuration_hash=self.configuration_hash,
                ai_score_configuration_hash=score.metadata.configuration_hash if len(score.metadata.configuration_hash) == 64 else sha256(score.metadata.configuration_hash.encode()).hexdigest(),
            ),
            final_action=final_action,
            setup_family=(
                "trend_continuation"
                if actionable and ai_alignment
                else "pullback_continuation"
                if actionable
                else None
            ),
            readiness=SignalReadiness.READY if actionable else SignalReadiness.WAITING,
            entry_low=entry_low,
            entry_high=entry_high,
            invalidation=invalidation,
            stop_loss=stop_loss,
            take_profit_targets=take_profit_targets,
            risk_reward=risk_reward,
            decision_reason=(
                f"Deterministic synthesis produced {final_action.value} from market score, "
                "validated AI interpretation, temporal consistency, and risk rules."
            ),
            opposite_direction_rejection=(
                f"Opposite direction rejected because deterministic directional evidence is {direction.value}."
                if direction != DecisionDirection.NEUTRAL
                else "Neither direction has sufficient deterministic support."
            ),
            supporting_evidence=ai_evidence,
            contradicting_evidence=contradictory_ai_evidence,
            temporal_evidence=temporal_evidence,
            source_lineage=SignalSourceLineage(
                market_snapshot_id=decision_input.market_snapshot_id,
                feature_snapshot_id=score.snapshot_id,
                current_ai_analysis_id=analysis.analysis_id if analysis is not None else None,
                current_ai_signal_id=(
                    analysis_signal.signal_id
                    if analysis_signal is not None
                    else None
                ),
                historical_ai_analysis_ids=historical_ids,
                quantitative_forecast_id=decision_input.quantitative_forecast_id,
                strategy_evaluation_ids=tuple(item.rule_id for item in evaluations),
                temporal_context_version=(
                    decision_input.temporal_context.version
                    if decision_input.temporal_context is not None
                    else None
                ),
                signal_engine_version=self.config.engine_version,
                strategy_configuration_version=self.config.configuration_version,
                risk_policy_version=self.config.policy_version,
            ),
            publication_eligible=actionable,
        )

    def _threshold_rule(self, add: Callable[..., None], rule_id: str, value: float, observe: float, eligible: float, name: str) -> None:
        callback = add
        if value < observe:
            callback(rule_id, RuleOutcome.FAILED, RuleSeverity.INSUFFICIENT, f"{name}_insufficient", value, observe)
        elif value < eligible:
            callback(rule_id, RuleOutcome.WARNING, RuleSeverity.SOFT_GATE, f"{name}_below_eligibility", value, eligible)
        else:
            callback(rule_id, RuleOutcome.PASSED, RuleSeverity.INFORMATIONAL, f"{name}_passed", value, eligible)

    def _cooldown_seconds(self, state: DecisionState | None) -> int:
        if state == DecisionState.ELIGIBLE:
            return self.config.cooldown.eligible_repeat_seconds
        if state == DecisionState.OBSERVE_ONLY:
            return self.config.cooldown.observe_repeat_seconds
        return self.config.cooldown.blocked_repeat_seconds

    def _eligibility(self, strength: float, confidence: float, quality: float, alignment: float, risk: float, freshness: float) -> float:
        value = 100 * (strength / 100) * (confidence / 100) * (quality / 100) * (alignment / 100) * (1 - risk / 100) * freshness
        return self._rounded(max(0.0, min(100.0, value)))

    def _rounded(self, value: float) -> float:
        rounded = round(value, self.config.output_precision)
        return 0.0 if rounded == 0 else rounded

    @staticmethod
    def _reason(item: RuleEvaluation) -> DecisionReason:
        return DecisionReason(reason_code=item.reason_code, severity=item.severity, source="signal_decision", message_key=f"signal_decision.{item.reason_code}", rule_id=item.rule_id)

    @staticmethod
    def _synthetic_limit(decision_input: SignalDecisionInput, reason: str) -> RuleEvaluation:
        return RuleEvaluation(rule_id="source.snapshot_integrity", category="source_integrity", severity=RuleSeverity.INSUFFICIENT, outcome=RuleOutcome.FAILED, reason_code=reason, evaluated_at=decision_input.as_of)

    @staticmethod
    def _synthetic_warning(decision_input: SignalDecisionInput, reason: str) -> RuleEvaluation:
        return RuleEvaluation(rule_id="source.snapshot_integrity", category="source_integrity", severity=RuleSeverity.SOFT_GATE, outcome=RuleOutcome.WARNING, reason_code=reason, evaluated_at=decision_input.as_of)
