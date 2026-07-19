from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import logging
from typing import Protocol
from uuid import UUID

from .clock import ReplayClock
from .config import ReplayConfig
from .exceptions import ReplayCheckpointError, ReplayConfigurationError, ReplayPointInTimeError, ReplayValidationError
from .isolation import ReplayEventBus, ReplayFeatureStore
from .models import (
    HistoricalEvent,
    ReplayCheckpoint,
    ReplayFailure,
    ReplayFailureCategory,
    ReplayGeneratedEvent,
    ReplayMode,
    ReplayOutputReference,
    ReplayRequest,
    ReplaySession,
    ReplayStatus,
    ReplayTraceRecord,
    ReplayTransition,
    stable_hash,
    stable_id,
)
from .ordering import merge_event_sources, timestamp_groups
from .registry import ReplayCompatibilityRegistry
from .repository import ReplayRepository
from .sources import HistoricalEventQuery, HistoricalSourceRegistry, ReplayDatasetRegistry
from .state_machine import validate_transition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayProcessingContext:
    session: ReplaySession
    clock: ReplayClock
    event_bus: ReplayEventBus
    feature_store: ReplayFeatureStore


@dataclass(frozen=True)
class ReplayProcessingResult:
    generated_events: tuple[ReplayGeneratedEvent, ...] = ()
    outputs: tuple[ReplayOutputReference, ...] = ()


class ReplayProcessor(Protocol):
    engine_name: str
    engine_version: str

    async def process(self, event: HistoricalEvent | ReplayGeneratedEvent, context: ReplayProcessingContext) -> ReplayProcessingResult: ...


class ReplayCoordinator:
    def __init__(
        self,
        repository: ReplayRepository,
        datasets: ReplayDatasetRegistry,
        sources: HistoricalSourceRegistry,
        engines: ReplayCompatibilityRegistry,
        config: ReplayConfig,
        processors: tuple[ReplayProcessor, ...] = (),
        *,
        now: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.repository = repository
        self.datasets = datasets
        self.sources = sources
        self.engines = engines
        self.config = config
        self.processors = {item.engine_name: item for item in processors}
        self.now = now or (lambda: datetime.now(UTC))
        self.sleeper = sleeper

    async def create(self, request: ReplayRequest) -> ReplaySession:
        self._validate_request(request)
        dataset = self.datasets.resolve(request.dataset)
        if dataset.mutable and self.config.determinism.fail_on_mutable_dataset:
            raise ReplayValidationError("mutable replay dataset is prohibited")
        graph = self.engines.resolve(request.engine_selection, request.engine_versions)
        selected_sources = self.sources.resolve(request.source_filters.source_names)
        required_sources = {source for item in graph for source in item.required_sources}
        if required_sources - set(request.source_filters.source_names):
            raise ReplayConfigurationError(f"missing historical sources: {sorted(required_sources - set(request.source_filters.source_names))}")
        validations = tuple([await source.validate(request) for source in selected_sources])
        invalid = [item.source_name for item in validations if not item.valid]
        if invalid:
            raise ReplayValidationError(f"historical source validation failed: {invalid}")
        now = self._aware_now()
        request_fingerprint = request.fingerprint(self.config.engine.ordering_version)
        engine_manifest = [{"name": item.engine_name, "version": item.engine_version, "compatibility": item.compatibility_version} for item in graph]
        session = ReplaySession(
            replay_id=request.replay_id,
            request=request,
            request_fingerprint=request_fingerprint,
            status=ReplayStatus.CREATED,
            created_at=now,
            virtual_cursor_at=request.start_at,
            engine_graph_version=self.config.engine.graph_version,
            ordering_version=self.config.engine.ordering_version,
            replay_engine_version=self.config.engine.version,
            configuration_hash=stable_hash(self.config.model_dump(mode="json")),
            policy_manifest_hash=stable_hash(request.policy_versions),
            engine_manifest_hash=stable_hash(engine_manifest),
            total_events_estimate=sum(item.event_estimate for item in validations if item.event_estimate is not None) if all(item.event_estimate is not None for item in validations) else None,
        )
        await self.repository.create_session(session)
        session = await self.transition(session, ReplayStatus.VALIDATING, "request_received", now)
        session = await self.transition(session, ReplayStatus.READY, "validation_passed", now)
        logger.info("Replay session created", extra={"engine": "replay", "replay_id": str(session.replay_id), "mode": session.request.mode.value, "instrument_count": len(request.instruments), "timeframe_count": len(request.timeframes), "request_fingerprint": request_fingerprint[:12]})
        return session

    async def transition(self, session: ReplaySession, target: ReplayStatus, reason: str, occurred_at: datetime | None = None) -> ReplaySession:
        validate_transition(session.status, target)
        timestamp = occurred_at or self._aware_now()
        updates: dict[str, object] = {"status": target, "row_version": session.row_version + 1}
        if target == ReplayStatus.RUNNING and session.started_at is None:
            updates["started_at"] = timestamp
        elif target == ReplayStatus.PAUSED:
            updates["paused_at"] = timestamp
        elif target == ReplayStatus.COMPLETED:
            updates.update(completed_at=timestamp, progress_percent=Decimal("100"), virtual_cursor_at=session.request.end_at)
        elif target == ReplayStatus.FAILED:
            updates["failed_at"] = timestamp
        elif target == ReplayStatus.CANCELLED:
            updates["cancelled_at"] = timestamp
        updated = session.model_copy(update=updates)
        persisted = await self.repository.save_session(updated, session.row_version)
        transition = ReplayTransition(
            transition_id=stable_id(session.replay_id, session.row_version, session.status.value, target.value, reason),
            replay_id=session.replay_id,
            from_status=session.status,
            to_status=target,
            reason_code=reason,
            occurred_at=timestamp,
        )
        await self.repository.save_transition(transition)
        return persisted

    async def checkpoint(self, session: ReplaySession, reason: str) -> ReplaySession:
        return await self._checkpoint(session, reason)

    async def run(self, replay_id: UUID, worker_id: str, *, max_groups: int | None = None) -> ReplaySession:
        session = await self._require_session(replay_id)
        if session.status != ReplayStatus.RUNNING:
            raise ReplayValidationError("replay must be running before worker execution")
        session = await self.repository.acquire_lease(replay_id, worker_id, self._aware_now(), self.config.worker.lease_seconds, session.row_version)
        clock = ReplayClock(session.request.start_at, session.request.end_at)
        clock.restore(session.virtual_cursor_at)
        event_bus = ReplayEventBus(session.replay_id, session.request.dataset.dataset_id, session.request.dataset.dataset_version, clock)
        feature_store = ReplayFeatureStore(session.replay_id)
        context = ReplayProcessingContext(session, clock, event_bus, feature_store)
        source_adapters = self.sources.resolve(session.request.source_filters.source_names)
        streams = [source.stream(HistoricalEventQuery(session.replay_id, session.request, session.last_ordering_key, self.config.processing.source_batch_size)) for source in source_adapters]
        groups = timestamp_groups(merge_event_sources(streams), self.config.processing.timestamp_group_limit)
        completed_groups = 0
        previous_virtual_time = clock.now()
        try:
            async for group in groups:
                current = await self._require_session(replay_id)
                if current.worker_id != worker_id:
                    raise ReplayValidationError("replay worker lease was lost")
                if current.status == ReplayStatus.CANCELLING:
                    current = await self._checkpoint(current, "cancel")
                    return await self.transition(current, ReplayStatus.CANCELLED, "cancelled_at_safe_boundary", clock.now())
                if current.status == ReplayStatus.PAUSING:
                    current = await self._checkpoint(current, "pause")
                    return await self.transition(current, ReplayStatus.PAUSED, "paused_at_safe_boundary", clock.now())
                await self._speed_wait(current.request.mode, previous_virtual_time, group[0].available_at, current.request.speed_multiplier)
                clock.advance_to(group[0].available_at)
                context = ReplayProcessingContext(current, clock, event_bus, feature_store)
                current = await self._process_group(current, group, context)
                previous_virtual_time = clock.now()
                completed_groups += 1
                if self._checkpoint_due(current):
                    current = await self._checkpoint(current, "policy")
                if max_groups is not None and completed_groups >= max_groups:
                    current = await self.transition(current, ReplayStatus.PAUSING, "step_units_complete", clock.now())
                    current = await self._checkpoint(current, "step")
                    return await self.transition(current, ReplayStatus.PAUSED, "step_paused", clock.now())
                session = current
            session = await self._require_session(replay_id)
            session = await self._checkpoint(session, "completed")
            return await self.transition(session, ReplayStatus.COMPLETED, "source_exhausted", clock.now())
        except Exception as exc:
            current = await self._require_session(replay_id)
            failure = ReplayFailure(
                category=self._failure_category(exc),
                reason_code=type(exc).__name__,
                cursor_at=clock.now(),
                detail="Replay processing failed safely",
            )
            failed = current.model_copy(update={"failure": failure, "failed_events": current.failed_events + 1, "row_version": current.row_version + 1})
            current = await self.repository.save_session(failed, current.row_version)
            if current.status in {ReplayStatus.RUNNING, ReplayStatus.PAUSING, ReplayStatus.CANCELLING, ReplayStatus.RESUMING, ReplayStatus.RECOVERING}:
                current = await self.transition(current, ReplayStatus.FAILED, "processing_failed", clock.now())
            logger.warning("Replay failed", extra={"engine": "replay", "replay_id": str(replay_id), "failure_code": failure.category.value, "virtual_cursor": clock.now().isoformat()})
            return current
        finally:
            await event_bus.close()
            await feature_store.close()
            final_current = await self.repository.get_session(replay_id)
            if final_current is not None and final_current.worker_id == worker_id:
                await self.repository.release_lease(replay_id, worker_id)

    async def _process_group(self, session: ReplaySession, group: tuple[HistoricalEvent, ...], context: ReplayProcessingContext) -> ReplaySession:
        trace: list[ReplayTraceRecord] = []
        outputs: list[ReplayOutputReference] = []
        generated_count = 0
        semantic_hash = session.semantic_output_hash
        last_key = session.last_ordering_key
        sequence = session.processed_events
        for event in group:
            if event.available_at > context.clock.now():
                raise ReplayPointInTimeError("future historical event reached processor")
            generated = list(await context.event_bus.publish(event))
            for engine_name in session.request.engine_selection:
                processor = self.processors.get(engine_name)
                if processor is None:
                    continue
                result = await asyncio.wait_for(processor.process(event, context), timeout=self.config.processing.event_processing_timeout_seconds)
                generated.extend(result.generated_events)
                outputs.extend(result.outputs)
            emitted = await self._drain_generated(generated, context, session.request.engine_selection, outputs)
            generated_count += emitted
            sequence += 1
            last_key = event.ordering_key_text()
            semantic_hash = self._hash_chain(semantic_hash, f"source:{event.replay_event_id}")
            trace.append(ReplayTraceRecord(replay_id=session.replay_id, sequence=sequence, virtual_time=context.clock.now(), event_id=event.replay_event_id, event_type=event.event_type, source=event.source_name, processing_status="processed", generated_event_count=emitted))
        for output in outputs:
            semantic_hash = self._hash_chain(semantic_hash, f"output:{output.output_type}:{output.fingerprint}")
        progress = self._progress(session.request.start_at, session.request.end_at, context.clock.now())
        updated = session.model_copy(update={
            "virtual_cursor_at": context.clock.now(),
            "last_ordering_key": last_key,
            "processed_events": session.processed_events + len(group),
            "generated_events": session.generated_events + generated_count,
            "progress_percent": progress,
            "semantic_output_hash": semantic_hash,
            "row_version": session.row_version + 1,
        })
        persisted = await self.repository.save_session(updated, session.row_version)
        if self.config.trace.enabled:
            await self.repository.save_trace(tuple(trace[: self.config.trace.max_records_per_session]))
        await self.repository.save_outputs(tuple(outputs))
        return persisted

    async def _drain_generated(self, initial: list[ReplayGeneratedEvent], context: ReplayProcessingContext, engines: tuple[str, ...], outputs: list[ReplayOutputReference]) -> int:
        queue = list(initial)
        seen: set[UUID] = set()
        emitted = 0
        while queue:
            event = queue.pop(0)
            if event.event_id in seen:
                raise ReplayValidationError("generated event cycle detected")
            seen.add(event.event_id)
            if event.chain_depth > self.config.processing.max_chain_depth or emitted >= self.config.processing.generated_event_limit_per_timestamp:
                raise ReplayValidationError("generated event chain exceeds configured bound")
            queue.extend(await context.event_bus.publish(event))
            for engine_name in engines:
                processor = self.processors.get(engine_name)
                if processor is None:
                    continue
                result = await processor.process(event, context)
                queue.extend(result.generated_events)
                outputs.extend(result.outputs)
            emitted += 1
        return emitted

    async def _checkpoint(self, session: ReplaySession, reason: str) -> ReplaySession:
        latest = await self.repository.latest_checkpoint(session.replay_id)
        sequence = (latest.sequence + 1) if latest else 1
        checkpoint_id = stable_id(session.replay_id, sequence, session.virtual_cursor_at.isoformat(), session.last_ordering_key or "", session.semantic_output_hash)
        candidate = ReplayCheckpoint(
            checkpoint_id=checkpoint_id,
            replay_id=session.replay_id,
            sequence=sequence,
            cursor_at=session.virtual_cursor_at,
            last_ordering_key=session.last_ordering_key,
            processed_events=session.processed_events,
            generated_events=session.generated_events,
            source_cursors={name: session.last_ordering_key or "" for name in session.request.source_filters.source_names},
            engine_state_references={},
            semantic_output_hash=session.semantic_output_hash,
            state_hash="0" * 64,
            created_at=session.virtual_cursor_at,
            reason=reason,
        )
        checkpoint = candidate.model_copy(update={"state_hash": candidate.calculated_state_hash()})
        if checkpoint.calculated_state_hash() != checkpoint.state_hash:
            raise ReplayCheckpointError("checkpoint hash construction failed")
        await self.repository.save_checkpoint(checkpoint)
        updated = session.model_copy(update={"latest_checkpoint_id": checkpoint.checkpoint_id, "row_version": session.row_version + 1})
        return await self.repository.save_session(updated, session.row_version)

    def _validate_request(self, request: ReplayRequest) -> None:
        if len(request.instruments) > self.config.limits.max_instruments or len(request.timeframes) > self.config.limits.max_timeframes:
            raise ReplayValidationError("replay scope exceeds configured series limits")
        if (request.end_at - request.start_at).days > self.config.limits.max_duration_days:
            raise ReplayValidationError("replay duration exceeds configured limit")
        if set(request.instruments) - self.config.approved_instruments or set(request.timeframes) - self.config.approved_timeframes:
            raise ReplayValidationError("replay series is not approved")
        if set(request.source_filters.source_names) - self.config.approved_sources:
            raise ReplayValidationError("historical source is not approved")
        if request.speed_multiplier is not None and request.speed_multiplier > Decimal(str(self.config.speed.max_multiplier)):
            raise ReplayValidationError("replay speed exceeds configured limit")
        if len(str(request.metadata).encode()) > self.config.limits.max_metadata_bytes:
            raise ReplayValidationError("replay metadata exceeds configured limit")

    async def _require_session(self, replay_id: UUID) -> ReplaySession:
        session = await self.repository.get_session(replay_id)
        if session is None:
            raise ReplayValidationError("replay session does not exist")
        return session

    def _checkpoint_due(self, session: ReplaySession) -> bool:
        if not self.config.checkpoint.enabled:
            return False
        latest_count = 0
        return session.request.checkpoint_policy.after_timestamp_group or session.processed_events - latest_count >= session.request.checkpoint_policy.every_events

    async def _speed_wait(self, mode: ReplayMode, previous: datetime, current: datetime, multiplier: Decimal | None) -> None:
        if mode in {ReplayMode.MAXIMUM_SPEED, ReplayMode.STEP}:
            return
        historical_delay = max(0.0, (current - previous).total_seconds())
        if mode == ReplayMode.ACCELERATED:
            historical_delay /= float(multiplier or Decimal("1"))
        delay = min(historical_delay, self.config.speed.max_real_time_idle_wait_seconds)
        if delay:
            await self.sleeper(delay)

    @staticmethod
    def _hash_chain(previous: str, token: str) -> str:
        return sha256(f"{previous}|{token}".encode()).hexdigest()

    @staticmethod
    def _progress(start: datetime, end: datetime, cursor: datetime) -> Decimal:
        ratio = (cursor - start).total_seconds() / (end - start).total_seconds()
        return Decimal(str(round(max(0.0, min(1.0, ratio)) * 100, 4)))

    @staticmethod
    def _failure_category(exc: Exception) -> ReplayFailureCategory:
        if isinstance(exc, ReplayPointInTimeError):
            return ReplayFailureCategory.POINT_IN_TIME_VIOLATION
        if isinstance(exc, ReplayConfigurationError):
            return ReplayFailureCategory.ENGINE_INCOMPATIBLE
        if isinstance(exc, ReplayCheckpointError):
            return ReplayFailureCategory.CHECKPOINT_FAILURE
        return ReplayFailureCategory.ENGINE_FAILURE

    def _aware_now(self) -> datetime:
        value = self.now()
        if value.tzinfo is None:
            raise ReplayConfigurationError("operational clock must be timezone-aware")
        return value.astimezone(UTC)
