from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.app.engines.ai_scoring_engine import DeterministicAIScoringEngine, FixedClock, ScoreMode, SqlAlchemyAIScoringRepository
from backend.app.engines.ai_scoring_engine.exceptions import AIScoringPersistenceError
from tests.engines.ai_scoring_engine.test_ai_scoring import NOW, aligned_input, evidence, scoring_input


class Scalars:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def first(self) -> object | None:
        return self.values[0] if self.values else None

    def all(self) -> list[object]:
        return self.values


def session() -> SimpleNamespace:
    return SimpleNamespace(execute=AsyncMock(), scalars=AsyncMock(), get=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())


def snapshots() -> tuple[object, object]:
    engine = DeterministicAIScoringEngine(clock=FixedClock(NOW))
    aligned = engine.score(aligned_input())
    conflict = engine.score(scoring_input(evidence("smc", "structure", 0.9), evidence("institutional_flow", "participation", -0.9)))
    return aligned, conflict


@pytest.mark.asyncio
async def test_sql_save_duplicate_success_and_rollback() -> None:
    aligned, conflict = snapshots()
    db = session()
    repository = SqlAlchemyAIScoringRepository(db)
    repository.find_by_fingerprint = AsyncMock(return_value=aligned)  # type: ignore[method-assign]
    assert await repository.save_snapshot(aligned) == aligned
    db.execute.assert_not_called()

    repository.find_by_fingerprint = AsyncMock(side_effect=[None, conflict])  # type: ignore[method-assign]
    assert await repository.save_snapshot(conflict) == conflict
    assert db.execute.await_count == 3
    db.commit.assert_awaited_once()

    failing = session()
    failing.execute.side_effect = RuntimeError("database unavailable")
    repository = SqlAlchemyAIScoringRepository(failing)
    repository.find_by_fingerprint = AsyncMock(return_value=None)  # type: ignore[method-assign]
    with pytest.raises(AIScoringPersistenceError, match="persistence failed"):
        await repository.save_snapshot(aligned)
    failing.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_sql_reads_filters_and_pruning() -> None:
    aligned, _ = snapshots()
    record = SimpleNamespace(payload=aligned.model_dump(mode="json"))
    db = session()
    repository = SqlAlchemyAIScoringRepository(db)

    db.get.side_effect = [record, None]
    assert await repository.get_snapshot(aligned.snapshot_id) == aligned
    assert await repository.get_snapshot(aligned.snapshot_id) is None

    db.scalars.side_effect = [Scalars([record]), Scalars([]), Scalars([record]), Scalars([record]), Scalars([]), Scalars([aligned.snapshot_id])]
    assert await repository.get_latest_snapshot("XAUUSD", "M15", "1.0.0") == aligned
    assert await repository.get_latest_snapshot("XAUUSD", "M15") is None
    listed = await repository.list_snapshots(
        "XAUUSD",
        "M15",
        start=NOW - timedelta(days=1),
        end=NOW + timedelta(days=1),
        status=aligned.status,
        policy_version="1.0.0",
        mode=ScoreMode.LIVE,
        offset=1,
        limit=2,
    )
    assert listed == (aligned,)
    assert await repository.find_by_fingerprint(aligned.metadata.input_fingerprint, ScoreMode.LIVE) == aligned
    assert await repository.prune(NOW, ScoreMode.LIVE, 10) == 0
    assert await repository.prune(NOW + timedelta(days=1), ScoreMode.LIVE, 10) == 1
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()
