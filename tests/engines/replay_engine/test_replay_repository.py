from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from backend.app.engines.replay_engine import (
    InMemoryReplayRepository,
    ReplayCheckpoint,
    ReplayOutputReference,
    ReplayStatus,
    ReplayTraceRecord,
    ReplayTransition,
    stable_hash,
    stable_id,
)
from tests.engines.replay_engine.test_replay_engine import NOW, build_service, historical_event, replay_request


@pytest.mark.asyncio
async def test_repository_optimistic_concurrency_leases_filters_and_cleanup() -> None:
    service, repository = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    with pytest.raises(Exception, match="version conflict"):
        await repository.save_session(session.model_copy(update={"row_version": session.row_version + 1}), session.row_version - 1)
    leased = await repository.acquire_lease(session.replay_id, "worker-a", NOW, 30, session.row_version)
    with pytest.raises(Exception, match="active worker"):
        await repository.acquire_lease(session.replay_id, "worker-b", NOW, 30, leased.row_version)
    renewed = await repository.renew_lease(session.replay_id, "worker-a", NOW + timedelta(seconds=1), 30)
    released = await repository.release_lease(session.replay_id, "worker-a")
    assert renewed.lease_expires_at and released.worker_id is None
    assert await repository.list_sessions(status=ReplayStatus.READY) == (released,)

    cancelled = await service.cancel(session.replay_id)
    assert cancelled.status == ReplayStatus.CANCELLED
    assert await repository.cleanup(NOW + timedelta(days=1), 10) == 1
    assert await repository.get_session(session.replay_id) is None


@pytest.mark.asyncio
async def test_repository_checkpoint_trace_output_and_transition_idempotency() -> None:
    repository = InMemoryReplayRepository()
    service, built = await build_service((historical_event(5, 1),))
    repository = built
    session = await service.create(replay_request())
    checkpoint = ReplayCheckpoint(
        checkpoint_id=uuid4(),
        replay_id=session.replay_id,
        sequence=1,
        cursor_at=NOW,
        last_ordering_key=None,
        processed_events=0,
        generated_events=0,
        semantic_output_hash="0" * 64,
        state_hash="0" * 64,
        created_at=NOW,
        reason="manual",
    )
    checkpoint = checkpoint.model_copy(update={"state_hash": checkpoint.calculated_state_hash()})
    await repository.save_checkpoint(checkpoint)
    assert await repository.latest_checkpoint(session.replay_id) == checkpoint
    invalid = checkpoint.model_copy(update={"checkpoint_id": uuid4(), "sequence": 2, "state_hash": "1" * 64})
    with pytest.raises(Exception, match="hash"):
        await repository.save_checkpoint(invalid)

    transition = ReplayTransition(transition_id=uuid4(), replay_id=session.replay_id, from_status=ReplayStatus.READY, to_status=ReplayStatus.RUNNING, reason_code="test", occurred_at=NOW)
    await repository.save_transition(transition)
    await repository.save_transition(transition)
    assert transition in await repository.list_transitions(session.replay_id)

    trace = ReplayTraceRecord(replay_id=session.replay_id, sequence=1, virtual_time=NOW, event_id=uuid4(), event_type="market.candle.closed", source="historical_candles", processing_status="processed")
    await repository.save_trace((trace, trace))
    assert await repository.list_trace(session.replay_id) == (trace,)

    output = ReplayOutputReference(output_id=stable_id("output", session.replay_id), replay_id=session.replay_id, output_type="ai_score", source_engine="ai_scoring", source_id="score-1", fingerprint=stable_hash({"score": 1}), as_of=NOW)
    await repository.save_outputs((output, output))
    assert await repository.list_outputs(session.replay_id, "ai_score") == (output,)


@pytest.mark.asyncio
async def test_repository_rejects_duplicate_session_and_checkpoint_sequence() -> None:
    service, built = await build_service((historical_event(5, 1),))
    session = await service.create(replay_request())
    with pytest.raises(Exception, match="already exists"):
        await built.create_session(session)
