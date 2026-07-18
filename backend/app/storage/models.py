"""PostgreSQL-compatible persistence models."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database.base import Base


class SignalRecord(Base):
    """Persisted composite market scenario."""

    __tablename__ = "signals"
    __table_args__ = (Index("ix_signals_symbol_timeframe_created", "symbol", "timeframe", "created_at"),)

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(16))
    entry_low: Mapped[float] = mapped_column(Float)
    entry_high: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[list[str]] = mapped_column(JSONB)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MarketDataRecord(Base):
    """Normalized OHLCV storage."""

    __tablename__ = "market_data"
    __table_args__ = (Index("ux_market_data_series", "symbol", "timeframe", "timestamp", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)


class HistoricalCandleRecord(Base):
    __tablename__ = "historical_candles"
    __table_args__ = (Index("ux_historical_candle_series", "symbol", "timeframe", "timestamp", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    spread: Mapped[float] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    quality_score: Mapped[float] = mapped_column(Float)
    quality_level: Mapped[str] = mapped_column(String(32))
    ingestion_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RealtimeCandleRecord(Base):
    __tablename__ = "realtime_candles"
    __table_args__ = (Index("ix_realtime_candle_series", "symbol", "timeframe", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class ProviderMetricRecord(Base):
    __tablename__ = "provider_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    healthy: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float)
    uptime_ratio: Mapped[float] = mapped_column(Float)
    quota_remaining: Mapped[int | None] = mapped_column(Integer, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LatencyHistoryRecord(Base):
    __tablename__ = "market_latency_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class QualityHistoryRecord(Base):
    __tablename__ = "market_quality_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    score: Mapped[float] = mapped_column(Float)
    level: Mapped[str] = mapped_column(String(32))
    anomalies: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)


class GapHistoryRecord(Base):
    __tablename__ = "market_gap_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    missing_count: Mapped[int] = mapped_column(Integer)
    classification: Mapped[str] = mapped_column(String(32))
    repaired: Mapped[bool] = mapped_column(Boolean, default=False)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SynchronizationHistoryRecord(Base):
    __tablename__ = "market_synchronization_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class CacheMetadataRecord(Base):
    __tablename__ = "market_cache_metadata"

    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    layer: Mapped[str] = mapped_column(String(32))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    candle_count: Mapped[int] = mapped_column(Integer)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SMCObjectRecord(Base):
    """Immutable swing, leg, and structural-event audit record."""

    __tablename__ = "smc_objects"
    __table_args__ = (Index("ix_smc_objects_series_time", "symbol", "timeframe", "analytical_timestamp"), Index("ix_smc_objects_type_state", "object_type", "lifecycle_state"))

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    object_type: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analytical_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    availability_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    lifecycle_state: Mapped[str] = mapped_column(String(32), default="confirmed")
    confidence_score: Mapped[float] = mapped_column(Float)
    quality_score: Mapped[float] = mapped_column(Float)
    algorithm_version: Mapped[str] = mapped_column(String(32))
    configuration_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SMCAnalysisSnapshotRecord(Base):
    """Replayable SMC state at an exact market-data boundary."""

    __tablename__ = "smc_analysis_snapshots"
    __table_args__ = (Index("ux_smc_snapshot_boundary", "symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode", unique=True), Index("ix_smc_snapshot_series_time", "symbol", "timeframe", "analysis_timestamp"))

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    market_data_boundary: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), index=True)
    processing_mode: Mapped[str] = mapped_column(String(32), index=True)
    engine_version: Mapped[str] = mapped_column(String(32))
    configuration_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SMCCheckpointRecord(Base):
    """Bounded recovery pointer for incremental SMC processing."""

    __tablename__ = "smc_checkpoints"
    __table_args__ = (Index("ux_smc_checkpoint_series", "symbol", "timeframe", "configuration_version", unique=True),)

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    last_processed_candle: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    state_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class LiquidityObjectRecord(Base):
    __tablename__ = "liquidity_objects"
    __table_args__ = (Index("ix_liquidity_objects_series_time", "symbol", "timeframe", "availability_timestamp"), Index("ix_liquidity_objects_type_state", "object_type", "lifecycle_state"))
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    logical_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    object_type: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    availability_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    lifecycle_state: Mapped[str] = mapped_column(String(32), index=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    quality_score: Mapped[float] = mapped_column(Float)
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LiquiditySnapshotRecord(Base):
    __tablename__ = "liquidity_snapshots"
    __table_args__ = (Index("ux_liquidity_snapshot_boundary", "symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    processing_mode: Mapped[str] = mapped_column(String(32))
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LiquidityCheckpointRecord(Base):
    __tablename__ = "liquidity_checkpoints"
    __table_args__ = (Index("ux_liquidity_checkpoint_series", "symbol", "timeframe", "configuration_version", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    last_processed_candle: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    state_hash: Mapped[str] = mapped_column(String(64))
    state_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class VolumeProfileObjectRecord(Base):
    __tablename__ = "volume_profile_objects"
    __table_args__ = (Index("ix_volume_profile_objects_series_time", "symbol", "timeframe", "availability_timestamp"), Index("ix_volume_profile_objects_type_state", "object_type", "lifecycle_state"))
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    logical_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    object_type: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    availability_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    lifecycle_state: Mapped[str] = mapped_column(String(32), index=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    quality_score: Mapped[float] = mapped_column(Float)
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class VolumeProfileSnapshotRecord(Base):
    __tablename__ = "volume_profile_snapshots"
    __table_args__ = (Index("ux_volume_profile_snapshot_boundary", "symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    processing_mode: Mapped[str] = mapped_column(String(32))
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class VolumeProfileCheckpointRecord(Base):
    __tablename__ = "volume_profile_checkpoints"
    __table_args__ = (Index("ux_volume_profile_checkpoint_series", "symbol", "timeframe", "configuration_version", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    last_processed_candle: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    state_hash: Mapped[str] = mapped_column(String(64))
    state_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class InstitutionalFlowEvidenceRecord(Base):
    __tablename__ = "institutional_flow_evidence"
    __table_args__ = (Index("ix_institutional_flow_evidence_series_time", "symbol", "timeframe", "availability_timestamp"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    evidence_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    source_engine: Mapped[str] = mapped_column(String(32), index=True)
    evidence_type: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    availability_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    direction: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    quality: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class InstitutionalFlowSnapshotRecord(Base):
    __tablename__ = "institutional_flow_snapshots"
    __table_args__ = (Index("ux_institutional_flow_snapshot_boundary", "symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    processing_mode: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class InstitutionalFlowCheckpointRecord(Base):
    __tablename__ = "institutional_flow_checkpoints"
    __table_args__ = (Index("ux_institutional_flow_checkpoint_series", "symbol", "timeframe", "configuration_version", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    last_processed_candle: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    state_hash: Mapped[str] = mapped_column(String(64))
    state_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MarketRegimeSnapshotRecord(Base):
    __tablename__ = "market_regime_snapshots"
    __table_args__ = (Index("ux_market_regime_snapshot_boundary", "symbol", "timeframe", "analysis_timestamp", "configuration_version", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    dominant_regime: Mapped[str] = mapped_column(String(64), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MarketRegimeEvidenceRecord(Base):
    __tablename__ = "market_regime_evidence"
    __table_args__ = (Index("ix_market_regime_evidence_series_time", "symbol", "timeframe", "available_at"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    evidence_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    source_engine: Mapped[str] = mapped_column(String(32), index=True)
    family: Mapped[str] = mapped_column(String(32), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    accepted: Mapped[bool] = mapped_column(Boolean)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MarketRegimeTransitionRecord(Base):
    __tablename__ = "market_regime_transitions"
    __table_args__ = (Index("ix_market_regime_transition_series_time", "symbol", "timeframe", "started_at"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    from_regime: Mapped[str] = mapped_column(String(64), index=True)
    to_regime: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class MarketRegimeCheckpointRecord(Base):
    __tablename__ = "market_regime_checkpoints"
    __table_args__ = (Index("ux_market_regime_checkpoint_series", "symbol", "timeframe", "configuration_version", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    engine_name: Mapped[str] = mapped_column(String(32))
    engine_version: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[str] = mapped_column(String(32))
    configuration_version: Mapped[str] = mapped_column(String(32))
    algorithm_version: Mapped[str] = mapped_column(String(32))
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    analysis_boundary: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload_hash: Mapped[str] = mapped_column(String(64))
    state_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EconomicCalendarObservationRecord(Base):
    __tablename__ = "economic_calendar_provider_observations"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    provider_name: Mapped[str] = mapped_column(String(64), index=True)
    provider_event_id: Mapped[str] = mapped_column(String(256), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload_hash: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class EconomicCalendarEventRecord(Base):
    __tablename__ = "economic_calendar_events"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(256), index=True)
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    country_code: Mapped[str | None] = mapped_column(String(2), nullable=True, index=True)
    currency_codes: Mapped[list[str]] = mapped_column(JSONB)
    category: Mapped[str] = mapped_column(String(64), index=True)
    importance: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class EconomicCalendarRevisionRecord(Base):
    __tablename__ = "economic_calendar_event_revisions"
    __table_args__ = (Index("ux_economic_calendar_revision_number", "event_id", "revision_number", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    event_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    revision_number: Mapped[int] = mapped_column(Integer)
    revision_type: Mapped[str] = mapped_column(String(32), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload_hash: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class EconomicCalendarSnapshotRecord(Base):
    __tablename__ = "economic_calendar_snapshots"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    historical_boundary: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class EconomicCalendarContextRecord(Base):
    __tablename__ = "economic_calendar_instrument_contexts"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    historical_boundary: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class EconomicCalendarSyncStateRecord(Base):
    __tablename__ = "economic_calendar_sync_state"
    provider_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class EconomicCalendarCheckpointRecord(Base):
    __tablename__ = "economic_calendar_checkpoints"
    __table_args__ = (Index("ux_economic_calendar_checkpoint_engine", "engine_name", "configuration_version", unique=True),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    engine_name: Mapped[str] = mapped_column(String(64))
    engine_version: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[str] = mapped_column(String(32))
    configuration_version: Mapped[str] = mapped_column(String(32))
    normalization_version: Mapped[str] = mapped_column(String(32))
    payload_hash: Mapped[str] = mapped_column(String(64))
    state_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AnalysisResultRecord(Base):
    """Versioned engine output for reproducibility and audit."""

    __tablename__ = "analysis_results"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    engine: Mapped[str] = mapped_column(String(64), index=True)
    engine_version: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    result: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class EngineLogRecord(Base):
    """Structured engine lifecycle and failure log."""

    __tablename__ = "engine_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    engine: Mapped[str] = mapped_column(String(64), index=True)
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    context: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
