from __future__ import annotations

import asyncio
from datetime import timedelta
from hashlib import sha256
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from backend.app.events import Event, EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .analyzer import build_snapshot, clusters, explain, instrument_context, reconcile, revision_between, surprise
from .clock import Clock, SystemClock
from .config import EconomicCalendarConfig
from .engine import BaselineEconomicCalendarEngine
from .events import (
    EconomicCalendarDegraded,
    EconomicCalendarProviderRecovered,
    EconomicCalendarProviderRequestCompleted,
    EconomicCalendarRecoveryCompleted,
    EconomicCalendarReplayCompleted,
    EconomicCalendarSnapshotCreated,
    EconomicCalendarSyncCompleted,
    EconomicCalendarSyncFailed,
    EconomicCalendarSyncStarted,
    EconomicEventCancelled,
    EconomicEventConflictDetected,
    EconomicEventCorrected,
    EconomicEventDiscovered,
    EconomicEventReleased,
    EconomicEventRescheduled,
    EconomicEventRevised,
    EconomicEventUpdated,
)
from .models import (
    ConnectionState,
    EconomicCalendarCheckpoint,
    EconomicCalendarExplanation,
    EconomicCalendarSnapshot,
    EconomicEvent,
    EconomicEventStatus,
    EventCluster,
    InstrumentEventContext,
    ProviderStatus,
    RevisionType,
    stable_id,
)
from .normalization import normalize_observation
from .providers import DisabledProvider, EconomicCalendarProvider, ProviderFetchRequest
from .repository import EconomicCalendarRepository, InMemoryEconomicCalendarRepository, _checkpoint_bytes


class EconomicCalendarMetrics:
    def __init__(self) -> None:
        self.provider_requests = 0
        self.provider_request_failures = 0
        self.events_fetched = 0
        self.events_inserted = 0
        self.events_updated = 0
        self.events_deduplicated = 0
        self.events_released = 0
        self.events_revised = 0
        self.events_cancelled = 0
        self.events_rescheduled = 0
        self.parse_failures = 0
        self.provider_conflicts = 0
        self.revision_count = 0
        self.snapshots_created = 0
        self.instrument_contexts_created = 0
        self.synchronization_failures = 0
        self.replay_count = 0
        self.replay_failures = 0
        self.recovery_count = 0
        self.recovery_failures = 0
        self.checkpoint_failures = 0
        self.feature_publication_failures = 0
        self.event_publication_failures = 0
        self.last_sync_duration_ms = 0.0
        self.last_successful_sync: str | None = None
        self.latest_snapshot_id: str | None = None

    def snapshot(self) -> dict[str, int | float | str | None]:
        return dict(vars(self))


class EconomicCalendarService:
    def __init__(
        self,
        event_bus: EventBus,
        feature_store: FeatureStore,
        config: EconomicCalendarConfig | None = None,
        repository: EconomicCalendarRepository | None = None,
        providers: tuple[EconomicCalendarProvider, ...] | None = None,
        repository_mode: str = "memory",
        clock: Clock | None = None,
    ) -> None:
        self.config = config or EconomicCalendarConfig()
        self.repository = repository or InMemoryEconomicCalendarRepository()
        self.providers = providers or (DisabledProvider(),)
        self.event_bus, self.feature_store = event_bus, feature_store
        self.repository_mode, self.clock = repository_mode, clock or SystemClock()
        self.engine = BaselineEconomicCalendarEngine(self.config)
        self.metrics = EconomicCalendarMetrics()
        self.recovery_state = "not_attempted"
        self.initialized = False
        self._identity: dict[str, UUID] = {}
        self._published: set[UUID] = set()
        self._scheduler: asyncio.Task[None] | None = None
        self._sync_lock = asyncio.Lock()
        # The last ACTUALLY-OBSERVED provider health, refreshed on every synchronize()/snapshot()
        # call — `health()` reads this instead of the static configured `provider.mode`, so it can
        # never claim "healthy" while every configured provider is genuinely unreachable.
        self._last_provider_statuses: tuple[ProviderStatus, ...] = ()
        self._provider_connection_state: dict[str, ConnectionState] = {}

    async def restore(self) -> bool:
        try:
            checkpoint = await self.repository.load_checkpoint()
            if checkpoint is None:
                self.recovery_state = "clean_start"
                self.initialized = True
                return False
            if (
                checkpoint.engine_version != self.engine.version
                or checkpoint.schema_version != self.config.versions.schema_version
                or checkpoint.configuration_version != self.config.version
                or checkpoint.normalization_version != self.config.versions.normalization_version
            ):
                raise ValueError("checkpoint versions are incompatible")
            if sha256(_checkpoint_bytes(checkpoint.state_payload)).hexdigest() != checkpoint.payload_hash:
                raise ValueError("checkpoint payload integrity validation failed")
            self._identity = {key: UUID(value) for key, value in checkpoint.state_payload.get("identity", {}).items()}
            self.recovery_state = "recovered"
            self.metrics.recovery_count += 1
            self.initialized = True
            await self._publish_event(
                EconomicCalendarRecoveryCompleted,
                stable_id("calendar-recovery", checkpoint.checkpoint_id),
                {"checkpoint_id": str(checkpoint.checkpoint_id), "quality": 1.0},
            )
            return True
        except Exception:
            self.recovery_state = "failed"
            self.metrics.recovery_failures += 1
            raise

    async def start(self) -> None:
        if not self.initialized:
            await self.restore()
        if self._scheduler is None and any(provider.mode.value == "live_provider" for provider in self.providers):
            self._scheduler = asyncio.create_task(self._poll(), name="economic-calendar-sync")

    async def stop(self) -> None:
        if self._scheduler:
            self._scheduler.cancel()
            try:
                await self._scheduler
            except asyncio.CancelledError:
                pass
            self._scheduler = None
        for provider in self.providers:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    async def _poll(self) -> None:
        while True:
            try:
                now = self.clock.now()
                await self.synchronize(now - timedelta(days=7), now + timedelta(days=30), boundary=now)
            except Exception:
                self.metrics.synchronization_failures += 1
            await asyncio.sleep(self.config.processing.polling_interval_seconds)

    async def synchronize(self, start: Any, end: Any, *, boundary: Any | None = None, incremental: bool = False) -> EconomicCalendarSnapshot:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("synchronization boundaries must be timezone-aware")
        now = boundary or self.clock.now()
        started = perf_counter()
        async with self._sync_lock:
            await self._publish_event(
                EconomicCalendarSyncStarted,
                stable_id("sync-start", now.isoformat(), start.isoformat(), end.isoformat()),
                {"start": start.isoformat(), "end": end.isoformat(), "mode": "incremental" if incremental else "full"},
            )
            observations: list[Any] = []
            statuses = []
            try:
                for provider in self.providers:
                    self.metrics.provider_requests += 1
                    try:
                        state = await self.repository.load_sync_state(provider.name)
                        request = ProviderFetchRequest(
                            start=start, end=end, cursor=state.get("cursor") if incremental and state else None, limit=self.config.processing.maximum_batch_size
                        )
                        result = await (
                            provider.fetch_updates(request, state.get("update_token") if state else None) if incremental else provider.fetch_events(request)
                        )
                        observations.extend(result.observations)
                        self.metrics.events_fetched += result.success_count
                        self.metrics.parse_failures += result.failure_count
                        # Live HTTP providers never raise for an HTTP-level failure (a bad status
                        # or connection error becomes an empty result + a warning, so one failing
                        # provider never aborts the whole sync loop) — `result.warnings` is the
                        # only signal that attempt actually failed, so it must count here too or
                        # `provider_request_failures` silently under-reports every HTTP failure.
                        if result.warnings:
                            self.metrics.provider_request_failures += 1
                        await self.repository.save_sync_state(
                            provider.name, {"cursor": result.cursor, "update_token": result.update_token, "updated_at": now.isoformat()}
                        )
                    except Exception:
                        self.metrics.provider_request_failures += 1
                    status = await provider.health()
                    statuses.append(status)
                    await self._publish_event(
                        EconomicCalendarProviderRequestCompleted,
                        stable_id("provider-request", provider.name, now.isoformat(), status.last_request.isoformat() if status.last_request else "unknown"),
                        {
                            "provider_name": status.provider_name,
                            "connection_state": status.connection_state.value,
                            "http_status": status.http_status,
                            "response_time_ms": status.response_time_ms,
                            "retry_count": status.retry_count,
                            "reachable": status.reachable,
                            "failure_reason": status.failure_reason,
                            "quality": 1.0 if status.reachable else 0.0,
                        },
                    )
                    previous_state = self._provider_connection_state.get(provider.name)
                    if previous_state is not None and previous_state != ConnectionState.CONNECTED and status.connection_state == ConnectionState.CONNECTED:
                        await self._publish_event(
                            EconomicCalendarProviderRecovered,
                            stable_id("provider-recovered", provider.name, now.isoformat()),
                            {"provider_name": status.provider_name, "previous_state": previous_state.value, "quality": 1.0},
                        )
                    self._provider_connection_state[provider.name] = status.connection_state
                inserted = await self.repository.save_provider_observations(tuple(observations))
                self.metrics.events_inserted += inserted
                normalized = []
                for observation in sorted(
                    observations, key=lambda item: (item.available_at, item.provider_name, item.provider_event_id, str(item.observation_id))
                ):
                    item = normalize_observation(observation, self.config)
                    key = f"{observation.provider_name}:{observation.provider_event_id}"
                    previous_id = self._identity.get(key)
                    if previous_id:
                        previous = await self.repository.get_event(previous_id)
                        if previous and item.scheduled_at_utc != previous.scheduled_at_utc:
                            item = item.model_copy(
                                update={
                                    "event_id": previous_id,
                                    "is_rescheduled": True,
                                    "original_scheduled_at": previous.original_scheduled_at or previous.scheduled_at_utc,
                                    "rescheduled_from": previous.scheduled_at_utc,
                                    "rescheduled_to": item.scheduled_at_utc,
                                    "status": EconomicEventStatus.RESCHEDULED,
                                }
                            )
                        else:
                            item = item.model_copy(update={"event_id": previous_id})
                    self._identity[key] = item.event_id
                    normalized.append(item)
                reconciled = reconcile(tuple(normalized), self.config.provider_priority)
                for item in reconciled:
                    previous = await self.repository.get_event(item.event_id)
                    revisions = await self.repository.list_revisions(item.event_id, self.config.api.maximum_revision_records)
                    revision = revision_between(previous, item, len(revisions) + 1)
                    if revision:
                        item = item.model_copy(
                            update={
                                "revision_count": len(revisions) + 1,
                                "latest_revision_id": revision.revision_id,
                                "is_revised": revision.revision_type == RevisionType.VALUE_REVISION,
                                "is_corrected": revision.revision_type == RevisionType.CORRECTION,
                            }
                        )
                        await self.repository.save_revision(revision)
                        self.metrics.revision_count += 1
                    await self.repository.save_event(item)
                    self.metrics.events_updated += previous is not None
                    self.metrics.events_deduplicated += previous is not None and revision is None
                    await self._publish_lifecycle(previous, item, revision)
                events = await self.repository.list_events(start, end, now, self.config.processing.maximum_batch_size)
                self._last_provider_statuses = tuple(statuses)
                snapshot = build_snapshot(events, now, start, end, tuple(statuses), self.config)
                await self.repository.save_snapshot(snapshot)
                await self._checkpoint(now, observations)
                await self.repository.prune_history(
                    self.config.processing.retention_events, self.config.processing.retention_observations, self.config.processing.retention_snapshots
                )
                self.metrics.snapshots_created += 1
                self.metrics.latest_snapshot_id = str(snapshot.snapshot_id)
                self.metrics.last_successful_sync = now.isoformat()
                self.metrics.last_sync_duration_ms = (perf_counter() - started) * 1000
                await self._publish_event(EconomicCalendarSnapshotCreated, stable_id("snapshot-event", snapshot.snapshot_id), self._snapshot_payload(snapshot))
                await self._publish_event(EconomicCalendarSyncCompleted, stable_id("sync-complete", snapshot.snapshot_id), self._snapshot_payload(snapshot))
                if snapshot.degradation.is_degraded:
                    await self._publish_event(EconomicCalendarDegraded, stable_id("degraded", snapshot.snapshot_id), self._snapshot_payload(snapshot))
                return snapshot
            except Exception as exc:
                self.metrics.synchronization_failures += 1
                await self._publish_event(
                    EconomicCalendarSyncFailed,
                    stable_id("sync-failed", now.isoformat(), type(exc).__name__),
                    {"error_type": type(exc).__name__, "quality": 0.0},
                )
                raise

    async def snapshot(self, start: Any, end: Any, *, as_of: Any | None = None) -> EconomicCalendarSnapshot:
        boundary = as_of or self.clock.now()
        events = await self.repository.list_events(start, end, boundary, self.config.processing.maximum_batch_size)
        statuses = tuple([await provider.health() for provider in self.providers])
        self._last_provider_statuses = statuses
        result = build_snapshot(events, boundary, start, end, statuses, self.config)
        await self.repository.save_snapshot(result)
        return result

    async def event(self, event_id: UUID, as_of: Any | None = None) -> EconomicEvent | None:
        return await (self.repository.get_event_at_boundary(event_id, as_of) if as_of else self.repository.get_event(event_id))

    async def events(self, start: Any | None = None, end: Any | None = None, as_of: Any | None = None, limit: int = 100) -> tuple[EconomicEvent, ...]:
        return await self.repository.list_events(start, end, as_of, min(limit, self.config.api.maximum_page_size))

    async def revisions(self, event_id: UUID, limit: int = 100) -> Any:
        return await self.repository.list_revisions(event_id, min(limit, self.config.api.maximum_revision_records))

    async def observations(self, event_id: UUID, limit: int = 100) -> Any:
        return await self.repository.list_provider_observations(event_id, min(limit, self.config.api.maximum_observation_records))

    async def context(
        self, symbol: str, *, as_of: Any | None = None, start: Any | None = None, end: Any | None = None, publish: bool = True
    ) -> InstrumentEventContext:
        boundary = as_of or self.clock.now()
        start = start or boundary - timedelta(days=2)
        end = end or boundary + timedelta(days=7)
        snapshot = await self.snapshot(start, end, as_of=boundary)
        result = instrument_context(symbol, snapshot, self.config)
        await self.repository.save_instrument_context(result)
        self.metrics.instrument_contexts_created += 1
        if publish:
            await self._publish_features(result, snapshot)
        return result

    async def replay(self, symbol: str, boundary: Any) -> InstrumentEventContext:
        try:
            result = await self.context(symbol, as_of=boundary)
            self.metrics.replay_count += 1
            await self._publish_event(
                EconomicCalendarReplayCompleted,
                stable_id("calendar-replay", result.context_id),
                {"context_id": str(result.context_id), "symbol": symbol, "historical_boundary": boundary.isoformat(), "quality": result.quality_score},
            )
            return result
        except Exception:
            self.metrics.replay_failures += 1
            raise

    async def event_clusters(self, start: Any, end: Any, as_of: Any | None = None) -> tuple[EventCluster, ...]:
        return clusters(await self.events(start, end, as_of, self.config.api.maximum_page_size), self.config)

    async def explanation(self, symbol: str, as_of: Any | None = None) -> EconomicCalendarExplanation:
        return explain(await self.context(symbol, as_of=as_of, publish=False))

    async def _checkpoint(self, now: Any, observations: list[Any]) -> None:
        state = {
            "last_successful_sync_at": now.isoformat(),
            "last_provider_cursor": {},
            "last_provider_update_token": {},
            "last_processed_observation": str(observations[-1].observation_id) if observations else None,
            "identity": {key: str(value) for key, value in sorted(self._identity.items())},
        }
        digest = sha256(_checkpoint_bytes(state)).hexdigest()
        checkpoint = EconomicCalendarCheckpoint(
            checkpoint_id=stable_id("calendar-checkpoint", self.config.version),
            configuration_version=self.config.version,
            schema_version=self.config.versions.schema_version,
            normalization_version=self.config.versions.normalization_version,
            last_successful_sync_at=now,
            state_payload=state,
            payload_hash=digest,
            created_at=now,
        )
        try:
            await self.repository.save_checkpoint(checkpoint)
        except Exception:
            self.metrics.checkpoint_failures += 1
            raise

    async def _publish_lifecycle(self, previous: EconomicEvent | None, current: EconomicEvent, revision: Any) -> None:
        event_type: type[Event] = EconomicEventDiscovered if previous is None else EconomicEventUpdated
        if revision:
            event_type = {
                RevisionType.FIRST_RELEASE: EconomicEventReleased,
                RevisionType.VALUE_REVISION: EconomicEventRevised,
                RevisionType.CORRECTION: EconomicEventCorrected,
                RevisionType.CANCELLATION: EconomicEventCancelled,
                RevisionType.RESCHEDULE: EconomicEventRescheduled,
                RevisionType.PROVIDER_CONFLICT: EconomicEventConflictDetected,
            }.get(revision.revision_type, event_type)
        await self._publish_event(
            event_type,
            stable_id("calendar-event-message", current.event_id, revision.revision_id if revision else current.available_at.isoformat()),
            {
                "economic_event_id": str(current.event_id),
                "provider_observation_ids": [str(item) for item in current.provider_records],
                "analysis_timestamp": current.available_at.isoformat(),
                "available_at": current.available_at.isoformat(),
                "previous_state": previous.model_dump(mode="json") if previous else None,
                "current_state": current.model_dump(mode="json"),
                "changed_fields": list(revision.changed_fields) if revision else ["initial_discovery"],
                "quality": current.source_quality,
                "freshness": "unknown",
                "conflict_state": current.conflict_state.value,
            },
        )

    async def _publish_event(self, event_type: type[Event], event_id: UUID, payload: dict[str, Any]) -> None:
        if event_id in self._published:
            return
        full = {
            "engine_name": "economic_calendar",
            "engine_version": self.engine.version,
            "schema_version": self.config.versions.schema_version,
            "configuration_version": self.config.version,
            "normalization_version": self.config.versions.normalization_version,
            "probabilistic_context": True,
            "trading_instruction": False,
            **payload,
        }
        try:
            await self.event_bus.publish(event_type(event_id=event_id, correlation_id=uuid4(), source="economic_calendar", payload=full))
            self._published.add(event_id)
        except Exception:
            self.metrics.event_publication_failures += 1

    async def _publish_features(self, context: InstrumentEventContext, snapshot: EconomicCalendarSnapshot) -> None:
        next_event, previous = context.next_relevant_event, context.previous_relevant_event
        latest = previous
        latest_surprise = surprise(latest, self.config) if latest else {"raw": None, "normalized": None}
        values: dict[str, Any] = {
            "feature_version": "1.0.0",
            "engine_name": "economic_calendar",
            "engine_version": self.engine.version,
            "schema_version": self.config.versions.schema_version,
            "configuration_version": self.config.version,
            "normalization_version": self.config.versions.normalization_version,
            "symbol": context.symbol,
            "analysis_timestamp": context.analysis_timestamp.isoformat(),
            "historical_boundary": context.historical_boundary.isoformat(),
            "source_event_ids": [str(item.event_id) for item in snapshot.events],
            "next_event_id": str(next_event.event_id) if next_event else None,
            "next_event_name": next_event.display_name if next_event else None,
            "next_event_time": next_event.scheduled_at_utc.isoformat() if next_event and next_event.scheduled_at_utc else None,
            "next_event_category": next_event.category.value if next_event else None,
            "next_event_importance": next_event.importance.value if next_event else None,
            "previous_event_id": str(previous.event_id) if previous else None,
            "previous_event_time": previous.scheduled_at_utc.isoformat() if previous and previous.scheduled_at_utc else None,
            "previous_event_importance": previous.importance.value if previous else None,
            "minutes_until_next_event": context.minutes_until_next_event,
            "minutes_since_previous_event": context.minutes_since_previous_event,
            "active_event_count": len(context.active_relevant_events),
            "upcoming_event_count": len(snapshot.upcoming_events),
            "high_importance_event_count": snapshot.high_importance_count,
            "critical_event_count": snapshot.critical_importance_count,
            "risk_window_phase": context.risk_window_phase.value,
            "risk_score": context.risk_score,
            "cluster_score": context.cluster_score,
            "relevance_score": context.relevance_score,
            "quality": context.quality_score,
            "freshness": snapshot.freshness.value,
            "is_degraded": snapshot.degradation.is_degraded,
            "provider_conflict_count": snapshot.conflicting_count,
            "revision_count": snapshot.revised_count,
            "latest_actual": latest.actual_value if latest else None,
            "latest_forecast": latest.forecast_value if latest else None,
            "latest_previous": latest.previous_value if latest else None,
            "latest_surprise": latest_surprise["raw"],
            "latest_normalized_surprise": latest_surprise["normalized"],
            "probabilistic_context": True,
            "trading_instruction": False,
        }
        try:
            await self.feature_store.write(
                FeatureRecord(
                    feature_id=stable_id("calendar-features", context.context_id),
                    correlation_id=context.context_id,
                    namespace="economic_calendar",
                    engine_name="economic_calendar",
                    engine_version=self.engine.version,
                    compatibility_version="1.0",
                    values=values,
                    created_at=context.analysis_timestamp,
                )
            )
        except Exception:
            self.metrics.feature_publication_failures += 1

    def _snapshot_payload(self, snapshot: EconomicCalendarSnapshot) -> dict[str, Any]:
        return {
            "snapshot_id": str(snapshot.snapshot_id),
            "analysis_timestamp": snapshot.analysis_timestamp.isoformat(),
            "available_at": snapshot.historical_boundary.isoformat(),
            "quality": snapshot.quality,
            "freshness": snapshot.freshness.value,
            "conflict_state": "material" if snapshot.conflicting_count else "none",
        }

    async def provider_status(self) -> tuple[ProviderStatus, ...]:
        statuses = tuple([await provider.health() for provider in self.providers])
        self._last_provider_statuses = statuses
        return statuses

    def health(self) -> dict[str, object]:
        live_configured = any(provider.mode.value == "live_provider" for provider in self.providers)
        if not live_configured:
            degraded = True
        elif self._last_provider_statuses:
            # Reflects what the provider ACTUALLY did on its last attempt, not just whether one is
            # configured — a live provider that is timing out or returning 401s must never read as
            # "healthy" here just because `mode == live_provider`.
            degraded = not any(item.enabled and item.reachable for item in self._last_provider_statuses)
        else:
            degraded = False  # configured but no attempt has completed yet; not yet proven either way
        primary_status = self._last_provider_statuses[0] if self._last_provider_statuses else None
        return {
            "status": "degraded" if degraded else "healthy",
            "engine_name": "economic_calendar",
            "engine_version": self.engine.version,
            "schema_version": self.config.versions.schema_version,
            "configuration_version": self.config.version,
            "normalization_version": self.config.versions.normalization_version,
            "enabled": self.config.enabled,
            "initialized": self.initialized,
            "repository_mode": self.repository_mode,
            "recovery_state": self.recovery_state,
            "is_degraded": degraded,
            "primary_provider": primary_status.provider_name if primary_status else None,
            "primary_provider_connection_state": primary_status.connection_state.value if primary_status else "unknown",
            "last_successful_sync": self.metrics.last_successful_sync,
            "latest_snapshot_id": self.metrics.latest_snapshot_id,
            "event_statistics": {"fetched": self.metrics.events_fetched, "inserted": self.metrics.events_inserted},
            "revision_statistics": {"count": self.metrics.revision_count},
            "failure_statistics": {"sync": self.metrics.synchronization_failures},
            "replay_statistics": {"count": self.metrics.replay_count, "failures": self.metrics.replay_failures},
            "probabilistic_context": True,
            "trading_instruction": False,
        }
