from __future__ import annotations

import pytest

from backend.app.engines.replay_engine import (
    HistoricalEvent,
    ReplayGeneratedEvent,
    ReplayMode,
    ReplayOutputReference,
    ReplayProcessingContext,
    ReplayProcessingResult,
    ReplayStatus,
    stable_hash,
    stable_id,
)
from tests.engines.replay_engine.test_replay_engine import build_service, historical_event, replay_request


class AnalyticalProcessor:
    engine_name = "market_data"
    engine_version = "1.0.0"

    async def process(self, event: HistoricalEvent | ReplayGeneratedEvent, context: ReplayProcessingContext) -> ReplayProcessingResult:
        if isinstance(event, HistoricalEvent):
            fingerprint = stable_hash({"source": str(event.replay_event_id), "type": "ai_score"})
            payload = {"output_type": "ai_score", "fingerprint": fingerprint}
            generated = ReplayGeneratedEvent(
                event_id=stable_id(context.session.replay_id, context.clock.now().isoformat(), "ai_score.completed", "ai_scoring", "1.0.0", fingerprint, stable_hash(payload), 0),
                replay_id=context.session.replay_id,
                virtual_time=context.clock.now(),
                event_type="ai_score.completed",
                source_engine="ai_scoring",
                source_engine_version="1.0.0",
                input_fingerprint=fingerprint,
                payload=payload,
            )
            output = ReplayOutputReference(output_id=stable_id("output", context.session.replay_id, fingerprint), replay_id=context.session.replay_id, output_type="ai_score", source_engine="ai_scoring", source_id=str(generated.event_id), fingerprint=fingerprint, as_of=context.clock.now())
            return ReplayProcessingResult((generated,), (output,))
        if event.event_type == "ai_score.completed":
            fingerprint = stable_hash({"score": event.input_fingerprint, "type": "signal_decision"})
            output = ReplayOutputReference(output_id=stable_id("output", context.session.replay_id, fingerprint), replay_id=context.session.replay_id, output_type="signal_decision", source_engine="signal_decision", source_id=str(event.event_id), fingerprint=fingerprint, as_of=context.clock.now(), state="eligible")
            return ReplayProcessingResult(outputs=(output,))
        return ReplayProcessingResult()


@pytest.mark.asyncio
async def test_processor_collects_replay_scoped_scores_decisions_and_summary() -> None:
    service, repository = await build_service((historical_event(5, 1),))
    service.coordinator.processors = {"market_data": AnalyticalProcessor()}
    session = await service.create(replay_request())
    await service.command_start(session.replay_id)
    await service.run(session.replay_id, "worker")
    outputs = await repository.list_outputs(session.replay_id)
    assert {item.output_type for item in outputs} == {"ai_score", "signal_decision"}
    summary = await service.summary(session.replay_id)
    assert summary.ai_scores_generated == 1
    assert summary.signal_decisions_generated == 1
    assert summary.eligible_decisions == 1
    assert summary.trade_execution is False


@pytest.mark.asyncio
async def test_resume_from_checkpoint_matches_uninterrupted_replay() -> None:
    events = (historical_event(5, 1), historical_event(10, 2), historical_event(15, 3))
    uninterrupted, _ = await build_service(events)
    full = await uninterrupted.create(replay_request())
    await uninterrupted.command_start(full.replay_id)
    await uninterrupted.run(full.replay_id, "worker-full")
    full_hash = (await uninterrupted.get(full.replay_id)).semantic_output_hash

    resumed, repository = await build_service(events)
    partial = await resumed.create(replay_request(mode=ReplayMode.STEP))
    await resumed.step(partial.replay_id, 1, "worker-step")
    checkpoint = await repository.latest_checkpoint(partial.replay_id)
    assert checkpoint is not None and checkpoint.processed_events == 1
    await resumed.resume(partial.replay_id)
    await resumed.run(partial.replay_id, "worker-resume")
    assert (await resumed.get(partial.replay_id)).semantic_output_hash == full_hash


class CycleProcessor:
    engine_name = "market_data"
    engine_version = "1.0.0"

    async def process(self, event: HistoricalEvent | ReplayGeneratedEvent, context: ReplayProcessingContext) -> ReplayProcessingResult:
        if isinstance(event, ReplayGeneratedEvent):
            return ReplayProcessingResult((event,), ())
        fingerprint = stable_hash({"event": str(event.replay_event_id)})
        payload = {"cycle": True}
        generated = ReplayGeneratedEvent(
            event_id=stable_id(context.session.replay_id, context.clock.now().isoformat(), "cycle.event", "market_data", "1.0.0", fingerprint, stable_hash(payload), 0),
            replay_id=context.session.replay_id,
            virtual_time=context.clock.now(),
            event_type="cycle.event",
            source_engine="market_data",
            source_engine_version="1.0.0",
            input_fingerprint=fingerprint,
            payload=payload,
        )
        return ReplayProcessingResult((generated,), ())


@pytest.mark.asyncio
async def test_generated_event_cycle_fails_closed_without_live_mutation() -> None:
    service, _ = await build_service((historical_event(5, 1),))
    service.coordinator.processors = {"market_data": CycleProcessor()}
    session = await service.create(replay_request())
    await service.command_start(session.replay_id)
    await service.run(session.replay_id, "worker")
    failed = await service.get(session.replay_id)
    assert failed.status == ReplayStatus.FAILED
    assert failed.failure is not None and failed.failure.detail == "Replay processing failed safely"
