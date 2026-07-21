from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from hashlib import sha256
import math

from backend.app.engines.ai_scoring_engine import ScoreMode, ScoreStatus

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

        eligibility = self._eligibility(strength, score.confidence_score, score.data_quality_score, score.evidence_alignment_score, score.market_risk_score, freshness_factor)
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
            confidence_score=score.confidence_score,
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
