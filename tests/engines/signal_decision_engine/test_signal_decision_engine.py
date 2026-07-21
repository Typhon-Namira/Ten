from datetime import timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from backend.app.engines.ai_scoring_engine import DeterministicAIScoringEngine, FixedClock, ScoreMode, ScoreStatus
from backend.app.engines.signal_decision_engine import (
    ConservativeSignalDecisionPolicy,
    DecisionDirection,
    DecisionHistory,
    DecisionHistoryReference,
    DecisionPolicyRegistry,
    DecisionState,
    DependencyCriticality,
    DependencyHealth,
    DependencyState,
    EconomicRiskReference,
    MarketRegimeReference,
    RuleDefinition,
    RuleRegistry,
    SignalDecisionConfig,
    SignalDecisionConfigurationError,
    SignalDecisionInput,
    production_rule_registry,
)
from backend.app.engines.signal_decision_engine.models import DecisionMode, RuleCategory
from tests.engines.ai_scoring_engine.test_ai_scoring import NOW, aligned_input

ECONOMIC_ID = UUID("11111111-1111-1111-1111-111111111111")
REGIME_ID = UUID("22222222-2222-2222-2222-222222222222")
HISTORY_ID = UUID("33333333-3333-3333-3333-333333333333")


def ai_score(**updates: object):
    value = DeterministicAIScoringEngine(clock=FixedClock(NOW)).score(aligned_input())
    return value.model_copy(update=updates)


def available_dependencies() -> tuple[DependencyHealth, ...]:
    return (
        DependencyHealth(name="ai_scoring", state=DependencyState.AVAILABLE, criticality=DependencyCriticality.CRITICAL, checked_at=NOW),
        DependencyHealth(name="persistence", state=DependencyState.AVAILABLE, criticality=DependencyCriticality.CRITICAL, checked_at=NOW),
        DependencyHealth(name="economic_calendar", state=DependencyState.AVAILABLE, criticality=DependencyCriticality.REQUIRED_FOR_ELIGIBILITY, checked_at=NOW),
        DependencyHealth(name="market_regime", state=DependencyState.AVAILABLE, criticality=DependencyCriticality.OPTIONAL, checked_at=NOW),
    )


def decision_input(*, score=None, economic_phase: str | None = "outside", regime: str | None = "trending_bull", history: DecisionHistory | None = None, dependencies: tuple[DependencyHealth, ...] | None = None, as_of=NOW, policy_name: str = "conservative_signal_policy") -> SignalDecisionInput:
    economic = None if economic_phase is None else EconomicRiskReference(context_id=ECONOMIC_ID, phase=economic_phase, risk_score=0, as_of=as_of)
    market_regime = None if regime is None else MarketRegimeReference(snapshot_id=REGIME_ID, regime=regime, as_of=as_of)
    return SignalDecisionInput(
        instrument="XAUUSD",
        timeframe="M15",
        as_of=as_of,
        requested_at=max(as_of, NOW),
        ai_score=score or ai_score(),
        economic_risk=economic,
        market_regime=market_regime,
        dependency_health=dependencies if dependencies is not None else available_dependencies(),
        history=history or DecisionHistory(),
        mode=DecisionMode.LIVE,
        policy_name=policy_name,
        policy_version="1.0.0",
    )


def history(direction: DecisionDirection, state: DecisionState, *, age: int = 60, strength: float = 70, confidence: float = 90, expired: bool = False) -> DecisionHistoryReference:
    return DecisionHistoryReference(
        decision_id=HISTORY_ID,
        direction=direction,
        state=state,
        decided_at=NOW - timedelta(seconds=age),
        valid_until=NOW - timedelta(seconds=1) if expired else NOW + timedelta(minutes=30),
        directional_strength=strength,
        confidence_score=confidence,
    )


def reason_codes(decision) -> set[str]:
    return {item.reason_code for item in (*decision.blockers, *decision.warnings, *decision.supporting_reasons)}


def test_eligible_is_deterministic_explainable_and_non_executing() -> None:
    policy = ConservativeSignalDecisionPolicy()
    first = policy.evaluate(decision_input())
    second = policy.evaluate(decision_input())
    assert first == second
    assert first.state == DecisionState.ELIGIBLE
    assert first.direction == DecisionDirection.BULLISH
    assert first.directional_strength == 70
    assert first.valid_until == NOW + timedelta(minutes=15)
    assert first.metadata.trade_execution is False
    assert first.metadata.order_generation is False
    assert first.explanation.financial_safety_notice == "analytical_decision_only"
    assert [item.rule_id for item in first.rules] == [item.rule_id for item in second.rules]


@pytest.mark.parametrize(
    ("updates", "state", "reason"),
    [
        ({"directional_score": 10.0}, DecisionState.INSUFFICIENT_EVIDENCE, "directional_strength_insufficient"),
        ({"confidence_score": 44.0}, DecisionState.INSUFFICIENT_EVIDENCE, "confidence_insufficient"),
        ({"data_quality_score": 44.0}, DecisionState.INSUFFICIENT_EVIDENCE, "data_quality_insufficient"),
        ({"evidence_alignment_score": 34.0}, DecisionState.INSUFFICIENT_EVIDENCE, "alignment_insufficient"),
        ({"directional_score": 30.0}, DecisionState.OBSERVE_ONLY, "directional_strength_below_eligibility"),
        ({"confidence_score": 60.0}, DecisionState.OBSERVE_ONLY, "confidence_below_eligibility"),
        ({"market_risk_score": 40.0}, DecisionState.OBSERVE_ONLY, "risk_elevated"),
        ({"data_quality_score": 60.0}, DecisionState.OBSERVE_ONLY, "data_quality_below_eligibility"),
        ({"evidence_alignment_score": 50.0}, DecisionState.OBSERVE_ONLY, "alignment_below_eligibility"),
        ({"market_risk_score": 65.0}, DecisionState.BLOCKED, "risk_hard_block"),
        ({"data_quality_score": 9.0}, DecisionState.INVALID, "data_quality_invalid"),
    ],
)
def test_numeric_threshold_precedence(updates: dict[str, object], state: DecisionState, reason: str) -> None:
    result = ConservativeSignalDecisionPolicy().evaluate(decision_input(score=ai_score(**updates)))
    assert result.state == state
    assert reason in reason_codes(result)


def test_freshness_boundaries_and_stale_status() -> None:
    policy = ConservativeSignalDecisionPolicy()
    aging_at = NOW + timedelta(seconds=901)
    aging = policy.evaluate(decision_input(as_of=aging_at, score=ai_score(), economic_phase="outside"))
    assert aging.state == DecisionState.OBSERVE_ONLY
    assert "ai_score_aging" in reason_codes(aging)
    stale_at = NOW + timedelta(seconds=1801)
    stale = policy.evaluate(decision_input(as_of=stale_at, score=ai_score(), economic_phase="outside"))
    assert stale.state == DecisionState.BLOCKED
    status_stale = policy.evaluate(decision_input(score=ai_score(status=ScoreStatus.STALE)))
    assert status_stale.state == DecisionState.BLOCKED


def test_invalid_snapshot_policy_and_replay_mode_integrity() -> None:
    policy = ConservativeSignalDecisionPolicy()
    invalid = policy.evaluate(decision_input(score=ai_score(status=ScoreStatus.INVALID)))
    assert invalid.state == DecisionState.INVALID
    invalid_hash = policy.evaluate(decision_input(score=ai_score(metadata=ai_score().metadata.model_copy(update={"configuration_hash": "bad"}))))
    assert invalid_hash.state == DecisionState.INVALID
    mismatch = policy.evaluate(decision_input(policy_name="unknown_policy"))
    assert mismatch.state == DecisionState.INVALID
    replay_score = ai_score().model_copy(update={"mode": ScoreMode.REPLAY})
    assert policy.evaluate(decision_input(score=replay_score)).state == DecisionState.INVALID


def test_ai_status_degraded_and_insufficient_are_preserved() -> None:
    policy = ConservativeSignalDecisionPolicy()
    degraded = policy.evaluate(decision_input(score=ai_score(status=ScoreStatus.DEGRADED)))
    insufficient = policy.evaluate(decision_input(score=ai_score(status=ScoreStatus.INSUFFICIENT_EVIDENCE)))
    assert degraded.state == DecisionState.OBSERVE_ONLY
    assert "ai_score_degraded" in reason_codes(degraded)
    assert insufficient.state == DecisionState.INSUFFICIENT_EVIDENCE
    assert "ai_score_insufficient_evidence" in reason_codes(insufficient)


def test_economic_regime_conflict_and_dependency_gates() -> None:
    policy = ConservativeSignalDecisionPolicy()
    assert policy.evaluate(decision_input(economic_phase="imminent")).state == DecisionState.BLOCKED
    assert policy.evaluate(decision_input(economic_phase="pre_event")).state == DecisionState.OBSERVE_ONLY
    assert policy.evaluate(decision_input(economic_phase=None)).state == DecisionState.BLOCKED
    assert policy.evaluate(decision_input(regime="ranging")).state == DecisionState.OBSERVE_ONLY
    assert policy.evaluate(decision_input(regime="uncertain")).state == DecisionState.BLOCKED
    assert policy.evaluate(decision_input(regime="future_unknown")).state == DecisionState.BLOCKED
    assert policy.evaluate(decision_input(regime=None)).state == DecisionState.OBSERVE_ONLY

    conflict = ai_score().conflicts[0].model_copy(update={"severity": "moderate", "confidence_penalty": 8.0}) if ai_score().conflicts else None
    if conflict is None:
        from backend.app.engines.ai_scoring_engine.models import EvidenceConflict

        conflict = EvidenceConflict(conflict_id=HISTORY_ID, conflict_type="directional", severity="moderate", sources=("smc", "flow"), directional_gap=1.3, description_code="conflict", confidence_penalty=8)
    assert policy.evaluate(decision_input(score=ai_score(conflicts=(conflict,)))).state == DecisionState.OBSERVE_ONLY
    assert policy.evaluate(decision_input(score=ai_score(conflicts=(conflict.model_copy(update={"severity": "severe", "confidence_penalty": 25.0}),)))).state == DecisionState.BLOCKED

    critical = (DependencyHealth(name="database", state=DependencyState.UNAVAILABLE, criticality=DependencyCriticality.CRITICAL, checked_at=NOW),)
    optional = (DependencyHealth(name="event_bus", state=DependencyState.DEGRADED, criticality=DependencyCriticality.OPTIONAL, checked_at=NOW),)
    required = (DependencyHealth(name="calendar", state=DependencyState.UNAVAILABLE, criticality=DependencyCriticality.REQUIRED_FOR_ELIGIBILITY, checked_at=NOW),)
    assert policy.evaluate(decision_input(dependencies=critical)).state == DecisionState.BLOCKED
    assert policy.evaluate(decision_input(dependencies=optional)).state == DecisionState.OBSERVE_ONLY
    assert policy.evaluate(decision_input(dependencies=required)).state == DecisionState.OBSERVE_ONLY


def test_disabled_or_fail_open_economic_policy_and_degraded_context() -> None:
    disabled = ConservativeSignalDecisionPolicy(SignalDecisionConfig(economic_event={"enabled": False}))
    assert disabled.evaluate(decision_input(economic_phase=None)).state == DecisionState.ELIGIBLE
    fail_open = ConservativeSignalDecisionPolicy(SignalDecisionConfig(economic_event={"fail_closed_when_source_unavailable": False}))
    assert fail_open.evaluate(decision_input(economic_phase=None)).state == DecisionState.OBSERVE_ONLY
    degraded = EconomicRiskReference(context_id=ECONOMIC_ID, phase="outside", risk_score=0, as_of=NOW, degraded=True)
    value = decision_input().model_copy(update={"economic_risk": degraded})
    assert fail_open.evaluate(value).state == DecisionState.OBSERVE_ONLY


def test_economic_reason_category_never_blocks_on_no_news_but_blocks_on_genuine_provider_failure() -> None:
    """Regression test: `degraded` alone used to be the only signal this rule saw, which made "no
    relevant events" indistinguishable from "the provider is actually down." `context_state` must
    carry the specific reason through to the rule's `reason_code`, and routine states
    (no_relevant_events / outside_risk_window / inside_risk_window) must never set `degraded`."""
    policy = ConservativeSignalDecisionPolicy()

    for routine_state in ("no_relevant_events", "outside_risk_window"):
        routine = EconomicRiskReference(context_id=ECONOMIC_ID, phase="outside", risk_score=0, as_of=NOW, degraded=False, context_state=routine_state)
        value = decision_input().model_copy(update={"economic_risk": routine})
        decision = policy.evaluate(value)
        assert decision.state == DecisionState.ELIGIBLE
        economic_rule = next(item for item in decision.rules if item.rule_id == "economic_event.window")
        assert economic_rule.outcome.value == "passed"

    for failure_state in ("provider_unreachable", "provider_timeout", "provider_auth_failed", "provider_rate_limited", "no_calendar_data"):
        failure = EconomicRiskReference(context_id=ECONOMIC_ID, phase="outside", risk_score=0, as_of=NOW, degraded=True, context_state=failure_state)
        value = decision_input().model_copy(update={"economic_risk": failure})
        decision = policy.evaluate(value)
        assert decision.state == DecisionState.BLOCKED
        assert failure_state in reason_codes(decision)


def test_cooldown_reversal_and_hysteresis() -> None:
    policy = ConservativeSignalDecisionPolicy()
    same = history(DecisionDirection.BULLISH, DecisionState.ELIGIBLE)
    cooldown = policy.evaluate(decision_input(history=DecisionHistory(latest=same, active=same, recent_same_direction=same)))
    assert cooldown.state == DecisionState.BLOCKED
    assert "cooldown_active" in reason_codes(cooldown)

    old_same = history(DecisionDirection.BULLISH, DecisionState.ELIGIBLE, age=901)
    assert policy.evaluate(decision_input(history=DecisionHistory(recent_same_direction=old_same))).state == DecisionState.ELIGIBLE
    observe_same = history(DecisionDirection.BULLISH, DecisionState.OBSERVE_ONLY, age=301)
    assert policy.evaluate(decision_input(history=DecisionHistory(recent_same_direction=observe_same))).state == DecisionState.ELIGIBLE
    blocked_same = history(DecisionDirection.BULLISH, DecisionState.BLOCKED, age=301)
    assert policy.evaluate(decision_input(history=DecisionHistory(recent_same_direction=blocked_same))).state == DecisionState.ELIGIBLE

    opposite = history(DecisionDirection.BEARISH, DecisionState.ELIGIBLE, strength=80, confidence=95)
    reversal = policy.evaluate(decision_input(history=DecisionHistory(recent_opposite_eligible=opposite)))
    assert reversal.state == DecisionState.BLOCKED
    stronger = policy.evaluate(decision_input(score=ai_score(directional_score=95), history=DecisionHistory(recent_opposite_eligible=opposite)))
    assert stronger.state == DecisionState.ELIGIBLE

    active = history(DecisionDirection.BULLISH, DecisionState.ELIGIBLE, age=1000)
    hysteresis = policy.evaluate(decision_input(score=ai_score(directional_score=42, confidence_score=67), history=DecisionHistory(active=active, latest=active)))
    assert hysteresis.state == DecisionState.ELIGIBLE
    hard = policy.evaluate(decision_input(score=ai_score(directional_score=42, confidence_score=67, market_risk_score=90), history=DecisionHistory(active=active)))
    assert hard.state == DecisionState.BLOCKED


def test_model_validation_fingerprints_and_invariants() -> None:
    value = decision_input()
    assert value.fingerprint("a" * 64) == value.model_copy(update={"requested_at": NOW + timedelta(seconds=1)}).fingerprint("a" * 64)
    assert value.fingerprint("a" * 64) != value.fingerprint("b" * 64)
    with pytest.raises(ValidationError, match="identity"):
        SignalDecisionInput.model_validate(value.model_dump() | {"instrument": "EURUSD"})
    with pytest.raises(ValidationError, match="future AI score"):
        SignalDecisionInput.model_validate(value.model_dump() | {"as_of": NOW - timedelta(seconds=1)})
    with pytest.raises(ValidationError, match="future economic"):
        SignalDecisionInput.model_validate(value.model_dump() | {"economic_risk": value.economic_risk.model_copy(update={"as_of": NOW + timedelta(seconds=1)})})
    with pytest.raises(ValidationError, match="future market regime"):
        SignalDecisionInput.model_validate(value.model_dump() | {"market_regime": value.market_regime.model_copy(update={"as_of": NOW + timedelta(seconds=1)})})

    decision = ConservativeSignalDecisionPolicy().evaluate(value)
    assert decision.history_reference().decision_id == decision.decision_id
    with pytest.raises(ValidationError, match="eligible decisions"):
        decision.model_copy(update={"direction": DecisionDirection.NEUTRAL}).__class__.model_validate(decision.model_dump() | {"direction": "neutral"})
    with pytest.raises(ValidationError, match="timezone-aware"):
        SignalDecisionInput.model_validate(value.model_dump() | {"as_of": NOW.replace(tzinfo=None)})
    with pytest.raises(ValidationError, match="valid_until"):
        decision.__class__.model_validate(decision.model_dump() | {"valid_until": NOW - timedelta(seconds=1)})


def test_configuration_and_registries_fail_closed() -> None:
    with pytest.raises(ValidationError, match="observation threshold"):
        SignalDecisionConfig(directional_strength={"minimum_for_observation": 80, "minimum_for_eligibility": 70})
    with pytest.raises(ValidationError, match="invalid quality"):
        SignalDecisionConfig(data_quality={"invalid_below": 50, "minimum_for_observation": 45, "minimum_for_eligibility": 70})
    with pytest.raises(ValidationError, match="preferred risk"):
        SignalDecisionConfig(risk={"preferred_maximum": 65, "hard_block_minimum": 65})
    with pytest.raises(ValidationError, match="eligible freshness"):
        SignalDecisionConfig(freshness={**SignalDecisionConfig().freshness, "M1": {"eligible_max_age_seconds": 180, "observe_max_age_seconds": 180}})
    with pytest.raises(ValidationError, match="direction mapping"):
        SignalDecisionConfig(direction_mapping={"neutral": "neutral"})
    with pytest.raises(ValidationError, match="every supported timeframe"):
        SignalDecisionConfig(freshness={"M1": {"eligible_max_age_seconds": 1, "observe_max_age_seconds": 2}})
    with pytest.raises(ValidationError, match="market-regime"):
        SignalDecisionConfig(market_regimes={"unknown": "execute"})
    with pytest.raises(ValidationError, match="validity"):
        SignalDecisionConfig(validity={"eligible": {"M15": 0}})
    assert SignalDecisionConfig().validity_seconds("blocked", "M15") == 300

    rules = RuleRegistry()
    definition = RuleDefinition("test.rule", RuleCategory.RISK)
    rules.register(definition)
    assert rules.get("test.rule") == definition
    assert rules.definitions() == (definition,)
    with pytest.raises(SignalDecisionConfigurationError, match="duplicate"):
        rules.register(definition)
    with pytest.raises(SignalDecisionConfigurationError, match="unknown"):
        rules.get("missing")
    assert len(production_rule_registry().definitions()) == 17

    policies = DecisionPolicyRegistry()
    policy = ConservativeSignalDecisionPolicy()
    policies.register(policy)
    assert policies.get(policy.name, policy.version) is policy
    with pytest.raises(SignalDecisionConfigurationError, match="duplicate"):
        policies.register(policy)
    with pytest.raises(SignalDecisionConfigurationError, match="unknown"):
        policies.get("missing", "1.0.0")
