from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.signal_decision_engine import (
    ConservativeSignalDecisionPolicy,
    DecisionHistoryReference,
    DecisionState,
    DependencyHealth,
    EconomicRiskReference,
    MarketRegimeReference,
    RuleEvaluation,
)
from backend.app.engines.signal_decision_engine import registration
from backend.app.engines.signal_decision_engine.models import DecisionDirection, DecisionMode, RuleCategory, RuleOutcome, RuleSeverity
from backend.app.engines.signal_decision_engine.repository import InMemorySignalDecisionRepository
from backend.app.engines.ai_scoring_engine import ScoreMode
from backend.app.features import InMemoryFeatureStore
from backend.app.services.pipeline_contracts import PipelineExecutionContext
from tests.engines.signal_decision_engine.test_signal_decision_engine import NOW, ai_score, decision_input
from tests.engines.signal_decision_engine.test_signal_decision_service import EconomicService, RegimeService, build_service


def test_timestamp_validators_and_decision_invariants() -> None:
    naive = NOW.replace(tzinfo=None)
    with pytest.raises(ValueError, match="dependency"):
        DependencyHealth.aware(naive)
    with pytest.raises(ValueError, match="economic"):
        EconomicRiskReference.aware(naive)
    with pytest.raises(ValueError, match="regime"):
        MarketRegimeReference.aware(naive)
    with pytest.raises(ValueError, match="history"):
        DecisionHistoryReference.aware(naive)
    with pytest.raises(ValueError, match="rule"):
        RuleEvaluation.aware(naive)

    decision = ConservativeSignalDecisionPolicy().evaluate(decision_input())
    with pytest.raises(ValueError, match="decision timestamps"):
        decision.aware(naive)
    blocker = decision.supporting_reasons[0]
    with pytest.raises(ValidationError, match="blocked decisions"):
        decision.__class__.model_validate(decision.model_dump() | {"state": "blocked", "blockers": []})
    with pytest.raises(ValidationError, match="observe-only"):
        decision.__class__.model_validate(decision.model_dump() | {"state": "observe_only", "warnings": []})
    with pytest.raises(ValidationError, match="non-actionable"):
        decision.__class__.model_validate(decision.model_dump() | {"state": "invalid", "blockers": []})
    with pytest.raises(ValueError, match="finite"):
        decision.model_copy(update={"eligibility_score": float("nan")}).invariants()
    assert blocker.rule_id


@pytest.mark.asyncio
async def test_registration_execute_validation_and_replay_branch() -> None:
    policy = registration._build(SimpleNamespace(), {})
    context = PipelineExecutionContext(correlation_id=uuid4(), candles=[], events=[], feature_store=InMemoryFeatureStore(), now=NOW)
    with pytest.raises(TypeError, match="approved"):
        await registration._execute(object(), context)
    with pytest.raises(ValueError, match="AI Scoring"):
        await registration._execute(policy, context)
    context.results["ai_scoring"] = ai_score()
    result = await registration._execute(policy, context)
    assert result.namespace == "signal_decision"
    assert result.features["trade_execution"] is False
    context.results["ai_scoring"] = ai_score().model_copy(update={"mode": ScoreMode.REPLAY})
    replay = await registration._execute(policy, context)
    assert replay.output.mode == DecisionMode.REPLAY
    factory = SimpleNamespace(register=Mock())
    registration.register(factory)
    assert factory.register.call_count == 1


@pytest.mark.asyncio
async def test_repository_duplicate_context_branches_and_publication_no_event_state() -> None:
    repository = InMemorySignalDecisionRepository()
    decision = ConservativeSignalDecisionPolicy().evaluate(decision_input())
    assert await repository.save_decision(decision) == decision
    assert await repository.save_decision(decision) == decision

    service, _, score = await build_service()
    service.economic_calendar = None
    service.market_regime = None
    collected = await service.collect_input(SimpleNamespace(
        instrument="XAUUSD", timeframe="M15", ai_score_snapshot_id=score.snapshot_id, as_of=NOW, mode=DecisionMode.LIVE,
        decision_policy_name=None, decision_policy_version=None,
    ))
    assert collected.economic_risk is None
    assert collected.market_regime is None

    service.economic_calendar = SimpleNamespace(context=AsyncMock(return_value=SimpleNamespace(
        context_id=uuid4(), risk_window_phase=SimpleNamespace(value="outside"), risk_score=0.1, analysis_timestamp=NOW,
        unavailable_context=(), active_relevant_events=(), next_relevant_event=SimpleNamespace(event_id=uuid4()),
        context_state=SimpleNamespace(value="outside_risk_window"),
    )))
    service.market_regime = SimpleNamespace(state=AsyncMock(return_value=None))
    assert await service._economic_context("XAUUSD", NOW) is not None
    assert await service._regime_context("XAUUSD", "M15", NOW) is None

    expired = decision.model_copy(update={"state": DecisionState.EXPIRED})
    await service._publish(expired, True)
    await service._publish(decision, False)


@pytest.mark.asyncio
async def test_persistence_failure_and_healthy_status_branch() -> None:
    service, _, score = await build_service(economic=EconomicService(), regime=RegimeService())
    service.repository.save_decision = AsyncMock(side_effect=RuntimeError("write failed"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="write failed"):
        await service.evaluate(SimpleNamespace(
            instrument="XAUUSD", timeframe="M15", ai_score_snapshot_id=score.snapshot_id, as_of=NOW, mode=DecisionMode.LIVE,
            decision_policy_name=None, decision_policy_version=None, persist=True, publish_events=False,
        ))
    assert service.metrics.persistence_failures_total == 1

    healthy, _, _ = await build_service()
    healthy.ai_scoring.health = Mock(return_value={"status": "healthy"})  # type: ignore[method-assign]
    assert healthy.health()["status"] == "healthy"


def test_rule_construction_accepts_all_declared_values() -> None:
    value = RuleEvaluation(
        rule_id="test",
        category=RuleCategory.RISK,
        severity=RuleSeverity.WARNING,
        outcome=RuleOutcome.NOT_EVALUATED,
        reason_code="not_evaluated",
        evaluated_at=NOW,
        contribution=0,
    )
    assert value.outcome == RuleOutcome.NOT_EVALUATED
    assert DecisionDirection.NEUTRAL.value == "neutral"
