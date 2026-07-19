from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.app.engines.replay_engine import (
    HistoricalEventQuery,
    ProductionReplayEngine,
    ReplayCheckpoint,
    ReplayConfig,
    ReplayOutputReference,
    ReplayStatus,
    ReplayTraceRecord,
    ReplayWorker,
    SqlAlchemyEconomicRevisionSource,
    SqlAlchemyHistoricalCandleSource,
    SqlAlchemyReplayRepository,
    stable_hash,
)
from backend.app.engines.replay_engine.registration import _build, _execute
from backend.app.features import InMemoryFeatureStore
from backend.app.services.pipeline_contracts import PipelineExecutionContext
from tests.engines.replay_engine.test_replay_engine import NOW, build_service, dataset, historical_event, replay_request


class ScalarResult:
    def __init__(self, values):
        self.values = list(values)

    def all(self):
        return self.values


class ExecuteResult:
    def __init__(self, value=None, rowcount=1):
        self.value = value
        self.rowcount = rowcount

    def scalar_one_or_none(self):
        return self.value


def db_session():
    return SimpleNamespace(execute=AsyncMock(), scalars=AsyncMock(), get=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())


@pytest.mark.asyncio
async def test_sql_repository_all_operations_and_failure_paths() -> None:
    service, memory = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    transition = (await memory.list_transitions(session.replay_id))[0]
    checkpoint = ReplayCheckpoint(checkpoint_id=uuid4(), replay_id=session.replay_id, sequence=1, cursor_at=NOW, last_ordering_key=None, processed_events=0, generated_events=0, semantic_output_hash="0" * 64, state_hash="0" * 64, created_at=NOW, reason="manual")
    checkpoint = checkpoint.model_copy(update={"state_hash": checkpoint.calculated_state_hash()})
    trace = ReplayTraceRecord(replay_id=session.replay_id, sequence=1, virtual_time=NOW, event_id=uuid4(), event_type="market.candle.closed", source="historical_candles", processing_status="processed")
    output = ReplayOutputReference(output_id=uuid4(), replay_id=session.replay_id, output_type="ai_score", source_engine="ai_scoring", source_id="score", fingerprint=stable_hash({"score": 1}), as_of=NOW)
    record = SimpleNamespace(payload=session.model_dump(mode="json"))
    transition_record = SimpleNamespace(payload=transition.model_dump(mode="json"))
    checkpoint_record = SimpleNamespace(payload=checkpoint.model_dump(mode="json"))
    trace_record = SimpleNamespace(payload=trace.model_dump(mode="json"))
    output_record = SimpleNamespace(payload=output.model_dump(mode="json"))

    db = db_session()
    repository = SqlAlchemyReplayRepository(db)
    db.execute.return_value = ExecuteResult(session.replay_id)
    assert await repository.create_session(session) == session
    db.get.side_effect = [record, None]
    assert await repository.get_session(session.replay_id) == session
    assert await repository.get_session(uuid4()) is None
    db.scalars.side_effect = [ScalarResult([record]), ScalarResult([transition_record]), ScalarResult([checkpoint_record]), ScalarResult([checkpoint_record]), ScalarResult([trace_record]), ScalarResult([output_record]), ScalarResult([session.replay_id])]
    assert await repository.list_sessions(status=ReplayStatus.READY) == (session,)
    assert await repository.list_transitions(session.replay_id) == (transition,)
    assert await repository.list_checkpoints(session.replay_id) == (checkpoint,)
    assert await repository.latest_checkpoint(session.replay_id) == checkpoint
    assert await repository.list_trace(session.replay_id) == (trace,)
    assert await repository.list_outputs(session.replay_id, "ai_score") == (output,)
    db.execute.return_value = ExecuteResult(rowcount=1)
    updated = session.model_copy(update={"row_version": session.row_version + 1})
    assert await repository.save_session(updated, session.row_version) == updated
    await repository.save_transition(transition)
    db.execute.return_value = ExecuteResult(checkpoint.checkpoint_id)
    assert await repository.save_checkpoint(checkpoint) == checkpoint
    await repository.save_trace(())
    await repository.save_trace((trace,))
    await repository.save_outputs(())
    await repository.save_outputs((output,))
    assert await repository.cleanup(NOW + timedelta(days=1), 10) == 1

    db.execute.return_value = ExecuteResult(None)
    with pytest.raises(Exception, match="already exists"):
        await repository.create_session(session)
    db.execute.return_value = ExecuteResult(rowcount=0)
    with pytest.raises(Exception, match="version conflict"):
        await repository.save_session(updated, session.row_version)
    db.execute.return_value = ExecuteResult(None)
    with pytest.raises(Exception, match="sequence conflict"):
        await repository.save_checkpoint(checkpoint)
    with pytest.raises(Exception, match="state hash"):
        await repository.save_checkpoint(checkpoint.model_copy(update={"state_hash": "1" * 64}))

    for operation, arguments, message in (
        (repository.create_session, (session,), "persistence failed"),
        (repository.save_session, (updated, session.row_version), "update failed"),
        (repository.save_checkpoint, (checkpoint,), "checkpoint persistence failed"),
    ):
        db.execute.side_effect = RuntimeError("database secret")
        with pytest.raises(Exception, match=message):
            await operation(*arguments)
        db.execute.side_effect = None


@pytest.mark.asyncio
async def test_sql_repository_lease_wrappers() -> None:
    service, _ = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    repository = SqlAlchemyReplayRepository(db_session())
    repository.get_session = AsyncMock(return_value=session)
    repository.save_session = AsyncMock(side_effect=lambda value, _: value)
    leased = await repository.acquire_lease(session.replay_id, "worker", NOW, 30, session.row_version)
    repository.get_session.return_value = leased
    renewed = await repository.renew_lease(session.replay_id, "worker", NOW + timedelta(seconds=1), 30)
    repository.get_session.return_value = renewed
    released = await repository.release_lease(session.replay_id, "worker")
    assert released.worker_id is None
    repository.get_session.return_value = None
    with pytest.raises(Exception, match="does not exist"):
        await repository.acquire_lease(session.replay_id, "worker", NOW, 30, 1)
    with pytest.raises(Exception, match="lost"):
        await repository.renew_lease(session.replay_id, "worker", NOW, 30)
    with pytest.raises(Exception, match="does not own"):
        await repository.release_lease(session.replay_id, "worker")
    repository.get_session.return_value = leased.model_copy(update={"worker_id": "other", "lease_expires_at": NOW + timedelta(seconds=20)})
    with pytest.raises(Exception, match="active worker"):
        await repository.acquire_lease(session.replay_id, "worker", NOW, 30, leased.row_version)


@pytest.mark.asyncio
async def test_sql_historical_candle_and_economic_sources() -> None:
    candle = SimpleNamespace(id=1, symbol="XAUUSD", timeframe="M1", timestamp=NOW, open=2000.0, high=2002.0, low=1999.0, close=2001.0, volume=10.0, spread=0.2, provider="test", quality_score=100.0, quality_level="native", ingestion_timestamp=NOW + timedelta(minutes=1))
    revision = SimpleNamespace(event_id=uuid4(), revision_number=1, revision_type="actual", available_at=NOW + timedelta(minutes=2), payload={"actual": 5.0})
    db = db_session()
    candle_source = SqlAlchemyHistoricalCandleSource(db)
    assert (await candle_source.validate(replay_request())).valid
    mismatch = replay_request().model_copy(update={"dataset": dataset().model_copy(update={"source_name": "other"})})
    assert not (await candle_source.validate(mismatch)).valid
    db.scalars.side_effect = [ScalarResult([candle]), ScalarResult([])]
    candle_events = [item async for item in candle_source.stream(HistoricalEventQuery(uuid4(), replay_request(), batch_size=1))]
    assert len(candle_events) == 1 and candle_events[0].available_at == NOW + timedelta(minutes=1)

    economic = SqlAlchemyEconomicRevisionSource(db)
    assert (await economic.validate(replay_request())).valid
    db.scalars.side_effect = [ScalarResult([revision]), ScalarResult([])]
    economic_events = [item async for item in economic.stream(HistoricalEventQuery(uuid4(), replay_request().model_copy(update={"source_filters": {"source_names": ("economic_calendar",)}}), batch_size=1))]
    assert len(economic_events) == 1 and economic_events[0].event_type == "economic.revision.published"


@pytest.mark.asyncio
async def test_worker_engine_and_registration_contracts() -> None:
    service, _ = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    await service.command_start(session.replay_id)
    worker = ReplayWorker(service, ReplayConfig(), "worker-test")
    assert await worker.run_once() == 1
    assert (await service.get(session.replay_id)).status == ReplayStatus.COMPLETED
    worker._stop.set()
    assert await worker.run_once() == 0
    worker.start()
    worker.start()
    await worker.stop()
    await worker.stop()

    engine = ProductionReplayEngine(service)
    created = await engine.create(replay_request())
    events = [item async for item in engine.events(created)]
    assert len(events) == 1
    config = _build(SimpleNamespace(), ReplayConfig().model_dump())
    result = await _execute(config, PipelineExecutionContext(correlation_id=uuid4(), candles=[], events=[], feature_store=InMemoryFeatureStore()))
    assert result.output["trade_execution"] is False
