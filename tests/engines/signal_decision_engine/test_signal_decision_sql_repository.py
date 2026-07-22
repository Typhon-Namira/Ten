from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.app.engines.signal_decision_engine import DecisionMode, DecisionState, SqlAlchemySignalDecisionRepository
from backend.app.engines.signal_decision_engine.exceptions import SignalDecisionPersistenceError
from tests.conftest import FakeSessionFactory
from tests.engines.signal_decision_engine.test_signal_decision_engine import NOW, ConservativeSignalDecisionPolicy, decision_input


class Scalars:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def first(self) -> object | None:
        return self.values[0] if self.values else None

    def all(self) -> list[object]:
        return self.values


class InsertResult:
    def __init__(self, value: object | None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> object | None:
        return self.value


def session() -> SimpleNamespace:
    return SimpleNamespace(execute=AsyncMock(), scalars=AsyncMock(), get=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())


def decision():
    return ConservativeSignalDecisionPolicy().evaluate(decision_input())


@pytest.mark.asyncio
async def test_sql_save_duplicate_success_and_rollback() -> None:
    value = decision()
    db = session()
    repository = SqlAlchemySignalDecisionRepository(FakeSessionFactory(db))
    repository.find_by_fingerprint = AsyncMock(return_value=value)  # type: ignore[method-assign]
    assert await repository.save_decision(value) == value
    db.execute.assert_not_called()

    repository.find_by_fingerprint = AsyncMock(side_effect=[None, value])  # type: ignore[method-assign]
    db.execute.side_effect = [InsertResult(value.decision_id), None, None]
    assert await repository.save_decision(value) == value
    assert db.execute.await_count == 3
    db.commit.assert_awaited_once()

    raced = session()
    raced.execute.return_value = InsertResult(None)
    repository = SqlAlchemySignalDecisionRepository(FakeSessionFactory(raced))
    repository.find_by_fingerprint = AsyncMock(side_effect=[None, value])  # type: ignore[method-assign]
    assert await repository.save_decision(value) == value
    raced.rollback.assert_awaited_once()

    unresolved = session()
    unresolved.execute.return_value = InsertResult(None)
    repository = SqlAlchemySignalDecisionRepository(FakeSessionFactory(unresolved))
    repository.find_by_fingerprint = AsyncMock(side_effect=[None, None])  # type: ignore[method-assign]
    with pytest.raises(SignalDecisionPersistenceError, match="persistence failed"):
        await repository.save_decision(value)

    failing = session()
    failing.execute.side_effect = RuntimeError("database unavailable")
    repository = SqlAlchemySignalDecisionRepository(FakeSessionFactory(failing))
    repository.find_by_fingerprint = AsyncMock(return_value=None)  # type: ignore[method-assign]
    with pytest.raises(SignalDecisionPersistenceError, match="persistence failed"):
        await repository.save_decision(value)
    failing.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_sql_reads_active_filters_history_and_pruning() -> None:
    value = decision()
    record = SimpleNamespace(payload=value.model_dump(mode="json"))
    db = session()
    repository = SqlAlchemySignalDecisionRepository(FakeSessionFactory(db))
    db.get.side_effect = [record, None]
    assert await repository.get_decision(value.decision_id) == value
    assert await repository.get_decision(value.decision_id) is None

    db.scalars.side_effect = [Scalars([record]), Scalars([]), Scalars([record]), Scalars([record]), Scalars([record]), Scalars([record]), Scalars([]), Scalars([value.decision_id])]
    assert await repository.find_by_fingerprint(value.input_fingerprint, DecisionMode.LIVE) == value
    assert await repository.find_by_fingerprint("missing", DecisionMode.LIVE) is None
    assert await repository.get_active_decision("XAUUSD", "M15", NOW, value.direction, DecisionState.ELIGIBLE) == value
    assert await repository.get_latest_decision("XAUUSD", "M15", value.direction, DecisionState.ELIGIBLE) == value
    listed = await repository.list_decisions(
        "XAUUSD",
        "M15",
        start=NOW - timedelta(days=1),
        end=NOW + timedelta(days=1),
        direction=value.direction,
        state=value.state,
        policy_version="1.0.0",
        ai_score_policy_version="1.0.0",
        mode=DecisionMode.LIVE,
        offset=1,
        limit=2,
    )
    assert listed == (value,)
    assert await repository.find_recent_decisions("XAUUSD", "M15", NOW) == (value,)
    assert await repository.prune(NOW, DecisionMode.LIVE, 10) == 0
    assert await repository.prune(NOW + timedelta(days=1), DecisionMode.LIVE, 10) == 1
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()
