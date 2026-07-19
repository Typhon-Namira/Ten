from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from typing import Any
from uuid import UUID

from backend.app.events import EventBus

from .config import ReplayConfig
from .coordinator import ReplayCoordinator
from .events import ReplayCancelled, ReplayCompleted, ReplayCreated, ReplayFailed, ReplayPaused, ReplayResumed, ReplayStarted
from .exceptions import ReplayNotFound, ReplayTransitionError, ReplayValidationError
from .models import (
    ManifestDifference,
    ReplayComparison,
    ReplayMode,
    ReplayOutputReference,
    ReplayRequest,
    ReplaySession,
    ReplayStatus,
    ReplaySummary,
    stable_id,
)
from .repository import ReplayRepository

logger = logging.getLogger(__name__)


@dataclass
class ReplayMetrics:
    sessions_created_total: int = 0
    sessions_started_total: int = 0
    sessions_completed_total: int = 0
    sessions_failed_total: int = 0
    sessions_cancelled_total: int = 0
    sessions_paused_total: int = 0
    sessions_resumed_total: int = 0
    events_processed_total: int = 0
    generated_events_total: int = 0
    checkpoints_created_total: int = 0
    lease_acquired_total: int = 0
    lease_lost_total: int = 0
    lifecycle_event_publish_failures_total: int = 0
    comparisons_total: int = 0
    semantic_hash_mismatch_total: int = 0
    failure_categories: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, object]:
        return {
            "replay_sessions_created_total": self.sessions_created_total,
            "replay_sessions_started_total": self.sessions_started_total,
            "replay_sessions_completed_total": self.sessions_completed_total,
            "replay_sessions_failed_total": self.sessions_failed_total,
            "replay_sessions_cancelled_total": self.sessions_cancelled_total,
            "replay_sessions_paused_total": self.sessions_paused_total,
            "replay_sessions_resumed_total": self.sessions_resumed_total,
            "replay_events_processed_total": self.events_processed_total,
            "replay_generated_events_total": self.generated_events_total,
            "replay_checkpoints_created_total": self.checkpoints_created_total,
            "replay_worker_lease_acquired_total": self.lease_acquired_total,
            "replay_worker_lease_lost_total": self.lease_lost_total,
            "replay_lifecycle_event_publish_failures_total": self.lifecycle_event_publish_failures_total,
            "replay_comparison_total": self.comparisons_total,
            "replay_semantic_hash_mismatch_total": self.semantic_hash_mismatch_total,
            "failure_categories": dict(sorted(self.failure_categories.items())),
        }


class ReplayService:
    def __init__(self, repository: ReplayRepository, coordinator: ReplayCoordinator, event_bus: EventBus, config: ReplayConfig, *, repository_mode: str = "memory") -> None:
        self.repository = repository
        self.coordinator = coordinator
        self.event_bus = event_bus
        self.config = config
        self.repository_mode = repository_mode
        self.metrics = ReplayMetrics()
        self.initialized = False
        self.closed = False

    async def start(self) -> None:
        self.initialized = True
        self.closed = False
        logger.info("Replay Engine initialized", extra={"engine": "replay", "version": self.config.engine.version, "repository_mode": self.repository_mode})

    async def stop(self) -> None:
        self.closed = True
        logger.info("Replay Engine stopped", extra={"engine": "replay"})

    async def create(self, request: ReplayRequest) -> ReplaySession:
        self._available()
        active = await self.repository.list_sessions(0, self.config.limits.max_concurrent_sessions + 1)
        active_count = sum(item.status not in {ReplayStatus.COMPLETED, ReplayStatus.CANCELLED, ReplayStatus.FAILED} for item in active)
        if active_count >= self.config.limits.max_concurrent_sessions:
            raise ReplayValidationError("maximum concurrent replay sessions reached")
        session = await self.coordinator.create(request)
        self.metrics.sessions_created_total += 1
        await self._publish(ReplayCreated, session)
        return session

    async def command_start(self, replay_id: UUID) -> ReplaySession:
        session = await self.get(replay_id)
        if session.status == ReplayStatus.RUNNING:
            return session
        updated = await self.coordinator.transition(session, ReplayStatus.RUNNING, "start_requested")
        self.metrics.sessions_started_total += 1
        await self._publish(ReplayStarted, updated)
        return updated

    async def pause(self, replay_id: UUID) -> ReplaySession:
        session = await self.get(replay_id)
        if session.status in {ReplayStatus.PAUSING, ReplayStatus.PAUSED}:
            return session
        updated = await self.coordinator.transition(session, ReplayStatus.PAUSING, "pause_requested")
        if updated.worker_id is None:
            updated = await self.coordinator.checkpoint(updated, "pause")
            updated = await self.coordinator.transition(updated, ReplayStatus.PAUSED, "paused_without_active_worker")
            self.metrics.sessions_paused_total += 1
            await self._publish(ReplayPaused, updated)
        return updated

    async def resume(self, replay_id: UUID) -> ReplaySession:
        session = await self.get(replay_id)
        if session.status == ReplayStatus.RUNNING:
            return session
        resumed = await self.coordinator.transition(session, ReplayStatus.RESUMING, "resume_requested")
        updated = await self.coordinator.transition(resumed, ReplayStatus.RUNNING, "resume_validated")
        self.metrics.sessions_resumed_total += 1
        await self._publish(ReplayResumed, updated)
        return updated

    async def cancel(self, replay_id: UUID) -> ReplaySession:
        session = await self.get(replay_id)
        if session.status == ReplayStatus.CANCELLED:
            return session
        if session.status in {ReplayStatus.COMPLETED, ReplayStatus.FAILED}:
            raise ReplayTransitionError("terminal replay cannot be cancelled")
        updated = await self.coordinator.transition(session, ReplayStatus.CANCELLING, "cancel_requested")
        if updated.worker_id is None:
            updated = await self.coordinator.checkpoint(updated, "cancel")
            updated = await self.coordinator.transition(updated, ReplayStatus.CANCELLED, "cancelled_without_active_worker")
            self.metrics.sessions_cancelled_total += 1
            await self._publish(ReplayCancelled, updated)
        return updated

    async def step(self, replay_id: UUID, units: int = 1, worker_id: str = "api-step") -> ReplaySession:
        if not 1 <= units <= self.config.limits.max_step_units:
            raise ReplayValidationError("step units exceed configured bound")
        session = await self.get(replay_id)
        if session.request.mode != ReplayMode.STEP:
            raise ReplayTransitionError("step is available only for step-mode replay")
        if session.status == ReplayStatus.READY:
            session = await self.command_start(replay_id)
        elif session.status == ReplayStatus.PAUSED:
            session = await self.resume(replay_id)
        elif session.status != ReplayStatus.RUNNING:
            raise ReplayTransitionError("step replay is not ready to advance")
        result = await self.coordinator.run(session.replay_id, worker_id, max_groups=units)
        final = await self.get(replay_id)
        self._record_result(final)
        if final.status == ReplayStatus.PAUSED:
            self.metrics.sessions_paused_total += 1
            await self._publish(ReplayPaused, final)
        return result.model_copy(update={"worker_id": final.worker_id, "lease_expires_at": final.lease_expires_at})

    async def run(self, replay_id: UUID, worker_id: str) -> ReplaySession:
        session = await self.coordinator.run(replay_id, worker_id)
        final = await self.get(replay_id)
        self._record_result(final)
        event_type = ReplayCompleted if final.status == ReplayStatus.COMPLETED else ReplayCancelled if final.status == ReplayStatus.CANCELLED else ReplayFailed
        await self._publish(event_type, final)
        return session.model_copy(update={"worker_id": final.worker_id, "lease_expires_at": final.lease_expires_at})

    async def get(self, replay_id: UUID) -> ReplaySession:
        session = await self.repository.get_session(replay_id)
        if session is None:
            raise ReplayNotFound("replay session not found")
        return session

    async def list(self, offset: int = 0, limit: int = 100, status: ReplayStatus | None = None) -> tuple[ReplaySession, ...]:
        return await self.repository.list_sessions(offset, min(limit, 200), status)

    async def summary(self, replay_id: UUID) -> ReplaySummary:
        session = await self.get(replay_id)
        outputs = await self.repository.list_outputs(replay_id, limit=1000)
        decisions = [item for item in outputs if item.output_type == "signal_decision"]
        states = [item.state for item in decisions]
        return ReplaySummary(
            replay_id=session.replay_id,
            status=session.status,
            start_at=session.request.start_at,
            end_at=session.request.end_at,
            final_cursor_at=session.virtual_cursor_at,
            processed_source_events=session.processed_events,
            generated_events=session.generated_events,
            ai_scores_generated=sum(item.output_type == "ai_score" for item in outputs),
            signal_decisions_generated=len(decisions),
            eligible_decisions=states.count("eligible"),
            observe_only_decisions=states.count("observe_only"),
            blocked_decisions=states.count("blocked"),
            insufficient_decisions=states.count("insufficient_evidence"),
            invalid_decisions=states.count("invalid"),
            warnings=0,
            failures=session.failed_events,
            no_lookahead_violations=int(session.failure is not None and session.failure.category.value == "point_in_time_violation"),
            semantic_output_hash=session.semantic_output_hash,
            completed_at=session.completed_at,
        )

    async def compare(self, left_id: UUID, right_id: UUID) -> ReplayComparison:
        left = await self.get(left_id)
        right = await self.get(right_id)
        differences = []
        for field_name in ("request_fingerprint", "engine_manifest_hash", "policy_manifest_hash", "configuration_hash", "ordering_version"):
            left_value = str(getattr(left, field_name))
            right_value = str(getattr(right, field_name))
            if left_value != right_value:
                differences.append(ManifestDifference(field=field_name, left=left_value, right=right_value))
        left_outputs = await self.repository.list_outputs(left_id, limit=1000)
        right_outputs = await self.repository.list_outputs(right_id, limit=1000)
        left_fingerprints = [(item.output_type, item.fingerprint) for item in left_outputs]
        right_fingerprints = [(item.output_type, item.fingerprint) for item in right_outputs]
        first_divergence = next((f"output[{index}]" for index, pair in enumerate(zip(left_fingerprints, right_fingerprints, strict=False)) if pair[0] != pair[1]), None)
        self.metrics.comparisons_total += 1
        equal = left.semantic_output_hash == right.semantic_output_hash
        self.metrics.semantic_hash_mismatch_total += not equal
        return ReplayComparison(
            left_replay_id=left_id,
            right_replay_id=right_id,
            comparable=not differences,
            manifest_differences=tuple(differences),
            semantic_hash_equal=equal,
            first_divergence=first_divergence,
            score_difference_count=self._difference_count(left_outputs, right_outputs, "ai_score"),
            decision_difference_count=self._difference_count(left_outputs, right_outputs, "signal_decision"),
            state_transition_difference_count=abs(len(await self.repository.list_transitions(left_id)) - len(await self.repository.list_transitions(right_id))),
        )

    async def cleanup(self) -> int:
        return await self.repository.cleanup(datetime.now(UTC) - timedelta(days=self.config.retention.completed_days), self.config.retention.cleanup_batch_size)

    def health(self) -> dict[str, object]:
        reasons = []
        if not self.initialized or self.closed:
            reasons.append("service_not_running")
        if self.repository_mode == "memory":
            reasons.append("ephemeral_persistence")
        if not self.coordinator.sources.names():
            reasons.append("historical_sources_unavailable")
        status = "unavailable" if not self.initialized or self.closed else "degraded" if reasons else "healthy"
        return {
            "status": status,
            "engine": "replay",
            "version": self.config.engine.version,
            "persistence": {"status": "healthy" if self.repository_mode != "memory" else "degraded", "mode": self.repository_mode},
            "event_bus_isolation": {"status": "healthy"},
            "feature_store_isolation": {"status": "healthy"},
            "historical_sources": {"status": "healthy" if self.coordinator.sources.names() else "unavailable", "sources": self.coordinator.sources.names()},
            "worker_coordination": {"status": "healthy" if self.repository_mode != "memory" else "degraded", "embedded_api_worker": self.config.worker.embedded_api_worker},
            "degradation_reasons": reasons,
            "historical_reconstruction": True,
            "live_state_mutation": False,
            "trade_execution": False,
            "timestamp": datetime.now(UTC),
        }

    def _available(self) -> None:
        if not self.initialized or self.closed:
            raise ReplayValidationError("Replay Engine is unavailable")

    async def _publish(self, event_type: type[Any], session: ReplaySession) -> None:
        if not self.config.events.publish_lifecycle_events:
            return
        payload = {
            "schema_version": "1.0",
            "replay_id": str(session.replay_id),
            "status": session.status.value,
            "mode": session.request.mode.value,
            "virtual_cursor": session.virtual_cursor_at.isoformat(),
            "processed_events": session.processed_events,
            "generated_events": session.generated_events,
            "progress": float(session.progress_percent) if session.progress_percent is not None else None,
            "dataset_id": session.request.dataset.dataset_id,
            "dataset_version": session.request.dataset.dataset_version,
            "request_fingerprint": session.request_fingerprint,
            "semantic_output_hash": session.semantic_output_hash if session.status == ReplayStatus.COMPLETED else None,
            "trade_execution": False,
        }
        try:
            await self.event_bus.publish(event_type(event_id=stable_id("replay-lifecycle", session.replay_id, session.status.value, session.row_version), correlation_id=session.replay_id, occurred_at=datetime.now(UTC), source="replay", payload=payload))
        except Exception:
            self.metrics.lifecycle_event_publish_failures_total += 1

    def _record_result(self, session: ReplaySession) -> None:
        self.metrics.events_processed_total += session.processed_events
        self.metrics.generated_events_total += session.generated_events
        if session.status == ReplayStatus.COMPLETED:
            self.metrics.sessions_completed_total += 1
        elif session.status == ReplayStatus.CANCELLED:
            self.metrics.sessions_cancelled_total += 1
        elif session.status == ReplayStatus.FAILED:
            self.metrics.sessions_failed_total += 1
            if session.failure:
                key = session.failure.category.value
                self.metrics.failure_categories[key] = self.metrics.failure_categories.get(key, 0) + 1

    @staticmethod
    def _difference_count(left: tuple[ReplayOutputReference, ...], right: tuple[ReplayOutputReference, ...], output_type: str) -> int:
        left_values = [item.fingerprint for item in left if item.output_type == output_type]
        right_values = [item.fingerprint for item in right if item.output_type == output_type]
        shared = sum(a == b for a, b in zip(left_values, right_values, strict=False))
        return max(len(left_values), len(right_values)) - shared
