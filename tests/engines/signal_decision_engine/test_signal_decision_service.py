from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest

from backend.app.engines.ai_scoring_engine import AIScoringConfig, AIScoringService, FixedClock, InMemoryAIScoringRepository
from backend.app.engines.signal_decision_engine import (
    DecisionMode,
    DecisionRequest,
    DecisionState,
    InMemorySignalDecisionRepository,
    SignalDecisionConfig,
    SignalDecisionInputError,
    SignalDecisionService,
    SignalDecisionSnapshotNotFound,
)
from backend.app.engines.signal_decision_engine.models import stable_id
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from tests.engines.signal_decision_engine.test_signal_decision_engine import NOW, ai_score


class EconomicService:
    async def context(self, *_: object, **__: object) -> object:
        return SimpleNamespace(
            context_id=UUID("44444444-4444-4444-4444-444444444444"),
            risk_window_phase=SimpleNamespace(value="outside"),
            risk_score=0.1,
            analysis_timestamp=NOW,
            unavailable_context=(),
            active_relevant_events=(),
            next_relevant_event=None,
            context_state=SimpleNamespace(value="outside_risk_window"),
        )


class RegimeService:
    async def state(self, *_: object, **__: object) -> object:
        return SimpleNamespace(
            snapshot_id=UUID("55555555-5555-5555-5555-555555555555"),
            dominant_regime=SimpleNamespace(value="trending_bull"),
            analysis_timestamp=NOW,
            degradation=SimpleNamespace(is_degraded=False),
        )


class BrokenContextService:
    async def context(self, *_: object, **__: object) -> object:
        raise RuntimeError("provider secret must not escape")

    async def state(self, *_: object, **__: object) -> object:
        raise RuntimeError("provider secret must not escape")


class FailingFeatureStore(InMemoryFeatureStore):
    async def write(self, feature: object) -> None:
        raise RuntimeError("feature unavailable")


class FailingEventBus(InMemoryEventBus):
    async def publish(self, event: object) -> None:
        raise RuntimeError("event unavailable")


async def build_service(*, events=None, features=None, economic=None, regime=None, decision_config=None, repository_mode: str = "postgresql"):
    ai_repository = InMemoryAIScoringRepository()
    ai = AIScoringService(ai_repository, InMemoryEventBus(), InMemoryFeatureStore(), AIScoringConfig(), repository_mode="postgresql", clock=FixedClock(NOW))
    await ai.start()
    score = await ai_repository.save_snapshot(ai_score())
    repository = InMemorySignalDecisionRepository()
    service = SignalDecisionService(
        repository,
        ai,
        events or InMemoryEventBus(),
        features or InMemoryFeatureStore(),
        decision_config or SignalDecisionConfig(),
        economic_calendar=economic if economic is not None else EconomicService(),
        market_regime=regime if regime is not None else RegimeService(),
        repository_mode=repository_mode,
        clock=FixedClock(NOW),
    )
    await service.start()
    return service, repository, score


@pytest.mark.asyncio
async def test_service_persists_publishes_and_suppresses_duplicate() -> None:
    events = InMemoryEventBus()
    features = InMemoryFeatureStore()
    service, repository, score = await build_service(events=events, features=features)
    request = DecisionRequest(ai_score_snapshot_id=score.snapshot_id)
    first = await service.evaluate(request)
    second = await service.evaluate(request)
    assert first == second
    assert first.state == DecisionState.ELIGIBLE
    assert service.metrics.requests_total == 2
    assert service.metrics.completed_total == 1
    assert service.metrics.duplicates_total == 1
    assert len(events.history()) == 1
    snapshot = await features.snapshot(stable_id("decision-correlation", first.input_fingerprint))
    assert snapshot.features["signal_decision"]["is_eligible"] is True
    assert snapshot.features["signal_decision"]["trade_execution"] is False
    assert await repository.get_decision(first.decision_id) == first
    assert await repository.get_active_decision("XAUUSD", "M15", NOW) == first
    assert await repository.get_latest_decision("XAUUSD", "M15") == first
    assert await service.get_decision(uuid4()) is None
    service.clock = FixedClock(first.valid_until)
    expired = await service.get_decision(first.decision_id)
    assert expired is not None and expired.state == DecisionState.EXPIRED
    assert expired.explanation.decision_state == DecisionState.EXPIRED


@pytest.mark.asyncio
async def test_replay_isolated_and_point_in_time_safe() -> None:
    events = InMemoryEventBus()
    features = InMemoryFeatureStore()
    service, _, score = await build_service(events=events, features=features)
    replay = await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id, as_of=NOW, mode=DecisionMode.REPLAY, persist=False, publish_events=True))
    assert replay.mode == DecisionMode.REPLAY
    assert events.history() == ()
    snapshot = await features.snapshot(stable_id("decision-correlation", replay.input_fingerprint))
    assert snapshot.features == {}
    live_unpersisted = await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id, persist=False, publish_events=True))
    assert live_unpersisted.state == DecisionState.ELIGIBLE
    assert events.history() == ()
    with pytest.raises(SignalDecisionInputError, match="future"):
        await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id, as_of=NOW + timedelta(minutes=1)))
    with pytest.raises(SignalDecisionInputError, match="timezone-aware"):
        await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id, as_of=NOW.replace(tzinfo=None)))


@pytest.mark.asyncio
async def test_missing_snapshot_policy_selection_context_failure_and_cleanup() -> None:
    service, repository, score = await build_service(economic=BrokenContextService(), regime=BrokenContextService())
    with pytest.raises(SignalDecisionSnapshotNotFound):
        await service.evaluate(DecisionRequest(ai_score_snapshot_id=uuid4()))
    with pytest.raises(Exception, match="unknown decision policy"):
        await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id, decision_policy_name="missing_policy"))
    blocked = await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id))
    assert blocked.state == DecisionState.BLOCKED
    assert await repository.get_active_decision("XAUUSD", "M15", blocked.valid_until) is None
    assert await repository.get_latest_decision("XAUUSD", "M15") == blocked
    assert await service.cleanup() == 0
    assert service.metrics.expiration_runs_total == 1


@pytest.mark.asyncio
async def test_publication_failures_are_observable_without_corrupting_decision() -> None:
    service, _, score = await build_service(events=FailingEventBus(), features=FailingFeatureStore())
    decision = await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id))
    assert decision.state == DecisionState.ELIGIBLE
    assert service.metrics.feature_store_failures_total == 1
    assert service.metrics.event_publish_failures_total == 1
    assert service.health()["status"] == "degraded"


@pytest.mark.asyncio
async def test_health_lifecycle_memory_policy_metrics_and_failure_counter() -> None:
    config = SignalDecisionConfig(persistence_required_in_production=False)
    service, _, score = await build_service(decision_config=config, repository_mode="memory")
    assert service.health()["status"] == "degraded"
    await service.stop()
    assert service.health()["status"] == "unavailable"
    await service.stop()
    assert service.closed is True
    service.policy.evaluate = Mock(side_effect=ValueError("invalid policy"))  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="invalid policy"):
        await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id))
    assert service.metrics.failed_total == 1
    assert service.metrics.snapshot()["maximum_duration_ms"] >= 0


@pytest.mark.asyncio
async def test_memory_repository_filters_pagination_history_and_retention() -> None:
    service, repository, score = await build_service()
    first = await service.evaluate(DecisionRequest(ai_score_snapshot_id=score.snapshot_id))
    assert await repository.find_by_fingerprint(first.input_fingerprint, first.mode) == first
    assert await repository.list_decisions("XAUUSD", "M15", direction=first.direction, state=first.state, policy_version="1.0.0", ai_score_policy_version="1.0.0", mode=DecisionMode.LIVE, limit=1) == (first,)
    assert await repository.list_decisions("OTHER", "M15") == ()
    assert await repository.find_recent_decisions("XAUUSD", "M15", NOW) == (first,)
    assert await repository.prune(first.valid_until + timedelta(days=1), DecisionMode.LIVE, 1) == 1
    assert await repository.get_decision(first.decision_id) is None
