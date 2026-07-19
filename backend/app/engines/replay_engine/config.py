from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReplayConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReplayEngineSettings(ReplayConfigModel):
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    ordering_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    graph_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")


class ReplayLimits(ReplayConfigModel):
    max_instruments: int = Field(default=10, ge=1, le=100)
    max_timeframes: int = Field(default=8, ge=1, le=16)
    max_duration_days: int = Field(default=90, ge=1, le=3650)
    max_events_per_session: int = Field(default=10_000_000, ge=1, le=100_000_000)
    max_concurrent_sessions: int = Field(default=4, ge=1, le=64)
    max_sessions_per_owner: int = Field(default=2, ge=1, le=32)
    max_step_units: int = Field(default=100, ge=1, le=1000)
    max_history_range_days: int = Field(default=365, ge=1, le=3650)
    max_metadata_bytes: int = Field(default=4096, ge=64, le=65536)


class ReplayProcessing(ReplayConfigModel):
    source_batch_size: int = Field(default=1000, ge=1, le=100_000)
    event_queue_capacity: int = Field(default=5000, ge=1, le=1_000_000)
    timestamp_group_limit: int = Field(default=50_000, ge=1, le=1_000_000)
    generated_event_limit_per_timestamp: int = Field(default=10_000, ge=1, le=100_000)
    max_chain_depth: int = Field(default=32, ge=1, le=256)
    event_processing_timeout_seconds: float = Field(default=30, gt=0, le=3600)
    timestamp_group_timeout_seconds: float = Field(default=120, gt=0, le=7200)


class ReplaySpeed(ReplayConfigModel):
    default_mode: str = Field(default="maximum_speed", pattern=r"^(maximum_speed|accelerated|real_time|step)$")
    max_multiplier: float = Field(default=1000, gt=0, le=100_000)
    max_real_time_idle_wait_seconds: float = Field(default=60, gt=0, le=3600)


class ReplayCheckpointSettings(ReplayConfigModel):
    enabled: bool = True
    every_events: int = Field(default=10_000, ge=1, le=10_000_000)
    every_virtual_seconds: int = Field(default=3600, ge=1, le=31_536_000)
    on_pause: bool = True
    on_shutdown: bool = True
    after_timestamp_group: bool = False
    retain_latest: int = Field(default=20, ge=1, le=1000)


class ReplayWorkerSettings(ReplayConfigModel):
    enabled: bool = True
    embedded_api_worker: bool = False
    max_concurrency: int = Field(default=2, ge=1, le=32)
    poll_interval_seconds: float = Field(default=2, gt=0, le=300)
    lease_seconds: int = Field(default=30, ge=3, le=3600)
    heartbeat_seconds: int = Field(default=10, ge=1, le=1200)
    graceful_shutdown_seconds: int = Field(default=30, ge=1, le=3600)

    @model_validator(mode="after")
    def heartbeat_precedes_expiry(self) -> ReplayWorkerSettings:
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("worker heartbeat must be shorter than lease")
        return self


class ReplayIsolationSettings(ReplayConfigModel):
    require_replay_event_bus: bool = True
    require_replay_feature_store: bool = True
    allow_live_event_publication: bool = False
    allow_live_feature_writes: bool = False

    @model_validator(mode="after")
    def deny_live_mutation(self) -> ReplayIsolationSettings:
        if self.allow_live_event_publication or self.allow_live_feature_writes:
            raise ValueError("production replay cannot publish into live analytical state")
        return self


class ReplayDeterminismSettings(ReplayConfigModel):
    fail_on_unpinned_engine_version: bool = True
    fail_on_unpinned_policy_version: bool = True
    fail_on_mutable_dataset: bool = True
    produce_semantic_hash: bool = True


class ReplayPointInTimeSettings(ReplayConfigModel):
    strict: bool = True
    fail_on_latest_query: bool = True
    future_clock_skew_seconds: int = Field(default=0, ge=0, le=300)


class ReplayTraceSettings(ReplayConfigModel):
    enabled: bool = False
    max_records_per_session: int = Field(default=100_000, ge=1, le=1_000_000)
    payload_storage: str = Field(default="metadata_only", pattern=r"^(none|metadata_only)$")


class ReplayRetentionSettings(ReplayConfigModel):
    completed_days: int = Field(default=90, ge=1, le=3650)
    failed_days: int = Field(default=30, ge=1, le=3650)
    cancelled_days: int = Field(default=30, ge=1, le=3650)
    cleanup_batch_size: int = Field(default=100, ge=1, le=10_000)


class ReplayEventSettings(ReplayConfigModel):
    publish_lifecycle_events: bool = True
    publish_progress_events: bool = True
    progress_event_interval_seconds: int = Field(default=5, ge=1, le=3600)
    publish_full_trace_events: bool = False


class ReplayConfig(ReplayConfigModel):
    enabled: bool = True
    engine: ReplayEngineSettings = ReplayEngineSettings()
    limits: ReplayLimits = ReplayLimits()
    processing: ReplayProcessing = ReplayProcessing()
    speed: ReplaySpeed = ReplaySpeed()
    checkpoint: ReplayCheckpointSettings = ReplayCheckpointSettings()
    worker: ReplayWorkerSettings = ReplayWorkerSettings()
    isolation: ReplayIsolationSettings = ReplayIsolationSettings()
    determinism: ReplayDeterminismSettings = ReplayDeterminismSettings()
    point_in_time: ReplayPointInTimeSettings = ReplayPointInTimeSettings()
    trace: ReplayTraceSettings = ReplayTraceSettings()
    retention: ReplayRetentionSettings = ReplayRetentionSettings()
    events: ReplayEventSettings = ReplayEventSettings()
    approved_instruments: frozenset[str] = frozenset({"XAUUSD"})
    approved_timeframes: frozenset[str] = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D1"})
    approved_sources: frozenset[str] = frozenset({"historical_candles", "economic_calendar"})

    @model_validator(mode="after")
    def system_limits(self) -> ReplayConfig:
        if self.worker.max_concurrency > self.limits.max_concurrent_sessions:
            raise ValueError("worker concurrency exceeds replay concurrency limit")
        if not self.approved_instruments or not self.approved_timeframes or not self.approved_sources:
            raise ValueError("approved instruments, timeframes and sources cannot be empty")
        return self
