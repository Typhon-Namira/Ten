from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
from typing import Any
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPLAY_NAMESPACE = UUID("28ddcfd6-99c1-55c5-bcba-664b6d7172e8")
SUPPORTED_TIMEFRAMES = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D1"})


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(value: object) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def stable_id(*parts: object) -> UUID:
    return uuid5(REPLAY_NAMESPACE, "|".join(str(part) for part in parts))


def aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _validate_json(value: object, *, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("metadata nesting exceeds limit")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("NaN and Infinity are prohibited")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            _validate_json(item, depth=depth + 1)
        return
    raise ValueError("metadata must contain JSON values only")


class ReplayModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReplayMode(StrEnum):
    MAXIMUM_SPEED = "maximum_speed"
    ACCELERATED = "accelerated"
    REAL_TIME = "real_time"
    STEP = "step"


class ReplayStepUnit(StrEnum):
    TIMESTAMP_GROUP = "timestamp_group"


class ReplayStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


class ReplayFailurePolicy(StrEnum):
    FAIL_FAST = "fail_fast"
    CHECKPOINT_AND_FAIL = "checkpoint_and_fail"
    SKIP_OPTIONAL_SOURCE_EVENT = "skip_optional_source_event"
    CONTINUE_WITH_WARNING = "continue_with_warning"


class ReplayFailureCategory(StrEnum):
    INVALID_REQUEST = "invalid_request"
    DATASET_UNAVAILABLE = "dataset_unavailable"
    DATASET_INCOMPLETE = "dataset_incomplete"
    SOURCE_FAILURE = "source_failure"
    ORDERING_VIOLATION = "ordering_violation"
    POINT_IN_TIME_VIOLATION = "point_in_time_violation"
    ENGINE_INCOMPATIBLE = "engine_incompatible"
    ENGINE_FAILURE = "engine_failure"
    EVENT_LOOP_DETECTED = "event_loop_detected"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    PERSISTENCE_FAILURE = "persistence_failure"
    LEASE_LOST = "lease_lost"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    INTERNAL_INVARIANT = "internal_invariant"


class ReplayDatasetReference(ReplayModel):
    dataset_id: str = Field(min_length=3, max_length=128, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]+$")
    dataset_version: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    source_name: str = Field(default="historical_candles", pattern=r"^[a-z][a-z0-9_]+$")
    created_at: datetime
    available_from: datetime
    available_until: datetime
    manifest_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    revision_policy: str = Field(default="append_only_query_cutoff", max_length=64)
    mutable: bool = False
    instruments: tuple[str, ...] = ("XAUUSD",)
    timeframes: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")

    @field_validator("created_at", "available_from", "available_until")
    @classmethod
    def timestamps(cls, value: datetime) -> datetime:
        return aware(value)

    @field_validator("instruments")
    @classmethod
    def canonical_instruments(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({value.upper().replace("/", "").replace("-", "") for value in values}))
        if not normalized or any(not value.isalnum() for value in normalized):
            raise ValueError("dataset instruments are invalid")
        return normalized

    @field_validator("timeframes")
    @classmethod
    def supported_timeframes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(values)))
        if not normalized or set(normalized) - SUPPORTED_TIMEFRAMES:
            raise ValueError("dataset timeframes are invalid")
        return normalized

    @model_validator(mode="after")
    def coverage(self) -> ReplayDatasetReference:
        if self.available_from >= self.available_until:
            raise ValueError("dataset coverage must have positive duration")
        if self.created_at < self.available_from:
            raise ValueError("dataset creation cannot precede its coverage")
        return self


class ReplaySourceFilters(ReplayModel):
    source_names: tuple[str, ...] = ("historical_candles",)
    event_types: tuple[str, ...] = ()

    @field_validator("source_names", "event_types")
    @classmethod
    def bounded_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(values)))
        if len(normalized) > 32 or any(len(value) > 96 for value in normalized):
            raise ValueError("source filter exceeds bounds")
        return normalized


class ReplayOutputOptions(ReplayModel):
    collect_ai_scores: bool = True
    collect_signal_decisions: bool = True
    collect_trace: bool = False


class ReplayCheckpointPolicy(ReplayModel):
    every_events: int = Field(default=10_000, ge=1, le=10_000_000)
    every_virtual_seconds: int = Field(default=3600, ge=1, le=31_536_000)
    after_timestamp_group: bool = False
    on_pause: bool = True
    on_shutdown: bool = True


class ReplayRequest(ReplayModel):
    replay_id: UUID = Field(default_factory=uuid4)
    name: str | None = Field(default=None, max_length=128)
    instruments: tuple[str, ...] = ()
    timeframes: tuple[str, ...] = ()
    start_at: datetime
    end_at: datetime
    mode: ReplayMode = ReplayMode.MAXIMUM_SPEED
    speed_multiplier: Decimal | None = Field(default=None, gt=0, le=100_000)
    step_unit: ReplayStepUnit | None = None
    dataset: ReplayDatasetReference
    engine_selection: tuple[str, ...] = ("market_data",)
    engine_versions: dict[str, str] = Field(default_factory=lambda: {"market_data": "1.0.0"})
    policy_versions: dict[str, str] = Field(default_factory=dict)
    configuration_versions: dict[str, str] = Field(default_factory=dict)
    source_filters: ReplaySourceFilters = ReplaySourceFilters()
    output_options: ReplayOutputOptions = ReplayOutputOptions()
    checkpoint_policy: ReplayCheckpointPolicy = ReplayCheckpointPolicy()
    failure_policy: ReplayFailurePolicy = ReplayFailurePolicy.CHECKPOINT_AND_FAIL
    created_by: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_single_series(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        symbol = data.pop("symbol", None)
        timeframe = data.pop("timeframe", None)
        if symbol is not None and not data.get("instruments"):
            data["instruments"] = (symbol,)
        if timeframe is not None and not data.get("timeframes"):
            data["timeframes"] = (timeframe,)
        return data

    @field_validator("start_at", "end_at")
    @classmethod
    def request_times(cls, value: datetime) -> datetime:
        return aware(value)

    @field_validator("instruments")
    @classmethod
    def request_instruments(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({value.upper().replace("/", "").replace("-", "") for value in values}))
        if not normalized or len(normalized) > 100 or any(not value.isalnum() for value in normalized):
            raise ValueError("instruments are invalid")
        return normalized

    @field_validator("timeframes")
    @classmethod
    def request_timeframes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(values)))
        if not normalized or set(normalized) - SUPPORTED_TIMEFRAMES:
            raise ValueError("unsupported timeframe")
        return normalized

    @field_validator("engine_selection")
    @classmethod
    def engines(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(values))
        if not normalized or len(normalized) > 32 or any(not value.replace("_", "").isalnum() for value in normalized):
            raise ValueError("engine selection is invalid")
        return normalized

    @model_validator(mode="after")
    def invariants(self) -> ReplayRequest:
        if self.start_at >= self.end_at:
            raise ValueError("start_at must precede end_at")
        if self.start_at < self.dataset.available_from or self.end_at > self.dataset.available_until:
            raise ValueError("requested range is outside dataset coverage")
        if not set(self.instruments).issubset(self.dataset.instruments) or not set(self.timeframes).issubset(self.dataset.timeframes):
            raise ValueError("requested series is outside dataset coverage")
        if set(self.engine_selection) != set(self.engine_versions):
            raise ValueError("every selected engine must have one pinned version")
        if self.mode == ReplayMode.ACCELERATED and self.speed_multiplier is None:
            raise ValueError("accelerated mode requires a speed multiplier")
        if self.mode != ReplayMode.ACCELERATED and self.speed_multiplier is not None:
            raise ValueError("speed multiplier is valid only for accelerated mode")
        if self.mode == ReplayMode.STEP and self.step_unit is None:
            object.__setattr__(self, "step_unit", ReplayStepUnit.TIMESTAMP_GROUP)
        if self.mode != ReplayMode.STEP and self.step_unit is not None:
            raise ValueError("step unit is valid only in step mode")
        _validate_json(self.metadata)
        if len(canonical_json(self.metadata).encode()) > 65_536:
            raise ValueError("metadata exceeds absolute size limit")
        return self

    def fingerprint(self, ordering_version: str) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"replay_id", "name", "created_by", "metadata", "speed_multiplier"},
        )
        payload["ordering_version"] = ordering_version
        return stable_hash(payload)


class HistoricalEvent(ReplayModel):
    replay_event_id: UUID
    source_event_id: str | None = Field(default=None, max_length=256)
    event_type: str = Field(min_length=3, max_length=96, pattern=r"^[a-z][a-z0-9_.-]+$")
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    instrument: str | None = Field(default=None, max_length=32)
    timeframe: str | None = Field(default=None, max_length=16)
    occurred_at: datetime
    published_at: datetime
    available_at: datetime
    source_name: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    source_version: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    source_sequence: int | None = Field(default=None, ge=0)
    priority: int = Field(default=100, ge=0, le=10_000)
    payload: dict[str, Any]
    payload_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    dataset_id: str
    dataset_version: str
    ordering_version: str = "1.0"

    @field_validator("occurred_at", "published_at", "available_at")
    @classmethod
    def event_times(cls, value: datetime) -> datetime:
        return aware(value)

    @model_validator(mode="after")
    def event_integrity(self) -> HistoricalEvent:
        _validate_json(self.payload)
        if len(canonical_json(self.payload).encode()) > 65_536:
            raise ValueError("historical event payload exceeds limit")
        if self.published_at < self.occurred_at or self.available_at < self.published_at:
            raise ValueError("event timestamp semantics are invalid")
        if stable_hash(self.payload) != self.payload_hash:
            raise ValueError("historical event payload hash mismatch")
        expected = stable_id(
            self.dataset_id,
            self.dataset_version,
            self.source_name,
            self.source_event_id or "",
            self.event_type,
            self.available_at.isoformat(),
            self.instrument or "",
            self.timeframe or "",
            self.payload_hash,
            self.ordering_version,
        )
        if self.replay_event_id != expected:
            raise ValueError("historical event identity mismatch")
        return self

    def ordering_key(self) -> tuple[datetime, int, int, str, str, str]:
        sequence = self.source_sequence if self.source_sequence is not None else 2**63 - 1
        return (self.available_at, self.priority, sequence, self.source_name, self.source_event_id or "", self.payload_hash)

    def ordering_key_text(self) -> str:
        key = self.ordering_key()
        return canonical_json((key[0].isoformat(), *key[1:]))


class ReplayGeneratedEvent(ReplayModel):
    event_id: UUID
    replay_id: UUID
    virtual_time: datetime
    event_type: str
    source_engine: str
    source_engine_version: str
    input_fingerprint: str
    schema_version: str = "1.0"
    payload: dict[str, Any] = Field(default_factory=dict)
    chain_depth: int = Field(default=0, ge=0, le=256)

    @field_validator("virtual_time")
    @classmethod
    def generated_time(cls, value: datetime) -> datetime:
        return aware(value)

    @model_validator(mode="after")
    def generated_integrity(self) -> ReplayGeneratedEvent:
        _validate_json(self.payload)
        expected = stable_id(self.replay_id, self.virtual_time.isoformat(), self.event_type, self.source_engine, self.source_engine_version, self.input_fingerprint, stable_hash(self.payload), self.chain_depth)
        if self.event_id != expected:
            raise ValueError("generated event identity mismatch")
        return self


class ReplayFailure(ReplayModel):
    category: ReplayFailureCategory
    reason_code: str = Field(max_length=96)
    cursor_at: datetime
    retryable: bool = False
    detail: str = Field(default="Replay processing failed", max_length=256)

    @field_validator("cursor_at")
    @classmethod
    def failure_time(cls, value: datetime) -> datetime:
        return aware(value)


class ReplaySession(ReplayModel):
    replay_id: UUID
    request: ReplayRequest
    request_fingerprint: str
    status: ReplayStatus
    created_at: datetime
    validated_at: datetime | None = None
    started_at: datetime | None = None
    paused_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    cancelled_at: datetime | None = None
    virtual_cursor_at: datetime
    engine_graph_version: str
    ordering_version: str
    replay_engine_version: str
    configuration_hash: str
    policy_manifest_hash: str
    engine_manifest_hash: str
    total_events_estimate: int | None = Field(default=None, ge=0)
    processed_events: int = Field(default=0, ge=0)
    generated_events: int = Field(default=0, ge=0)
    failed_events: int = Field(default=0, ge=0)
    progress_percent: Decimal | None = Field(default=Decimal("0"), ge=0, le=100)
    latest_checkpoint_id: UUID | None = None
    last_ordering_key: str | None = None
    semantic_output_hash: str = Field(default_factory=lambda: "0" * 64, min_length=64, max_length=64)
    worker_id: str | None = Field(default=None, max_length=128)
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    failure: ReplayFailure | None = None
    row_version: int = Field(default=1, ge=1)

    @field_validator("created_at", "validated_at", "started_at", "paused_at", "completed_at", "failed_at", "cancelled_at", "virtual_cursor_at", "lease_expires_at", "heartbeat_at")
    @classmethod
    def session_times(cls, value: datetime | None) -> datetime | None:
        return aware(value) if value is not None else None

    @model_validator(mode="after")
    def session_invariants(self) -> ReplaySession:
        if not self.request.start_at <= self.virtual_cursor_at <= self.request.end_at:
            raise ValueError("virtual cursor is outside replay bounds")
        if self.status == ReplayStatus.COMPLETED and self.progress_percent != Decimal("100"):
            raise ValueError("completed replay progress must be 100")
        if self.lease_expires_at is not None and self.worker_id is None:
            raise ValueError("lease requires a worker identity")
        return self


class ReplayCheckpoint(ReplayModel):
    checkpoint_id: UUID
    replay_id: UUID
    sequence: int = Field(ge=1)
    cursor_at: datetime
    last_ordering_key: str | None
    processed_events: int = Field(ge=0)
    generated_events: int = Field(ge=0)
    source_cursors: dict[str, str] = Field(default_factory=dict)
    engine_state_references: dict[str, str] = Field(default_factory=dict)
    feature_store_snapshot_reference: str | None = None
    semantic_output_hash: str = Field(min_length=64, max_length=64)
    state_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime
    reason: str = Field(max_length=64)

    @field_validator("cursor_at", "created_at")
    @classmethod
    def checkpoint_times(cls, value: datetime) -> datetime:
        return aware(value)

    def calculated_state_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"state_hash"}))


class ReplayTransition(ReplayModel):
    transition_id: UUID
    replay_id: UUID
    from_status: ReplayStatus
    to_status: ReplayStatus
    reason_code: str = Field(max_length=96)
    occurred_at: datetime
    actor: str = Field(default="system", max_length=128)

    @field_validator("occurred_at")
    @classmethod
    def transition_time(cls, value: datetime) -> datetime:
        return aware(value)


class ReplayTraceRecord(ReplayModel):
    replay_id: UUID
    sequence: int = Field(ge=1)
    virtual_time: datetime
    event_id: UUID
    event_type: str
    source: str
    processing_status: str
    generated_event_count: int = Field(default=0, ge=0)
    checkpoint_id: UUID | None = None
    error_code: str | None = None

    @field_validator("virtual_time")
    @classmethod
    def trace_time(cls, value: datetime) -> datetime:
        return aware(value)


class ReplayOutputReference(ReplayModel):
    output_id: UUID
    replay_id: UUID
    output_type: str
    source_engine: str
    source_id: str
    fingerprint: str = Field(min_length=64, max_length=64)
    as_of: datetime
    state: str | None = None

    @field_validator("as_of")
    @classmethod
    def output_time(cls, value: datetime) -> datetime:
        return aware(value)


class ReplaySummary(ReplayModel):
    replay_id: UUID
    status: ReplayStatus
    start_at: datetime
    end_at: datetime
    final_cursor_at: datetime
    processed_source_events: int
    generated_events: int
    ai_scores_generated: int
    signal_decisions_generated: int
    eligible_decisions: int
    observe_only_decisions: int
    blocked_decisions: int
    insufficient_decisions: int
    invalid_decisions: int
    warnings: int
    failures: int
    no_lookahead_violations: int
    semantic_output_hash: str
    completed_at: datetime | None
    trade_execution: bool = False


class ManifestDifference(ReplayModel):
    field: str
    left: str
    right: str


class ReplayComparison(ReplayModel):
    left_replay_id: UUID
    right_replay_id: UUID
    comparable: bool
    manifest_differences: tuple[ManifestDifference, ...]
    semantic_hash_equal: bool
    first_divergence: str | None
    score_difference_count: int = Field(ge=0)
    decision_difference_count: int = Field(ge=0)
    state_transition_difference_count: int = Field(ge=0)


ReplayState = ReplaySession
