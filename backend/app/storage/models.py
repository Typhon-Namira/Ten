"""PostgreSQL-compatible persistence models."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
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


class AIScoreSnapshotRecord(Base):
    __tablename__ = "ai_score_snapshots"
    __table_args__ = (
        Index("ux_ai_score_fingerprint_mode", "input_fingerprint", "mode", unique=True),
        Index("ix_ai_score_series_time", "instrument", "timeframe", "as_of"),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    mode: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    policy_name: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(32), index=True)
    configuration_version: Mapped[str] = mapped_column(String(32))
    configuration_hash: Mapped[str] = mapped_column(String(64))
    input_fingerprint: Mapped[str] = mapped_column(String(64))
    directional_score: Mapped[float] = mapped_column(Float)
    confidence_score: Mapped[float] = mapped_column(Float)
    market_risk_score: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class AIScoreComponentRecord(Base):
    __tablename__ = "ai_score_components"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_score_snapshots.id", ondelete="CASCADE"), index=True)
    source_engine: Mapped[str] = mapped_column(String(32), index=True)
    source_group: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class AIScoreConflictRecord(Base):
    __tablename__ = "ai_score_conflicts"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_score_snapshots.id", ondelete="CASCADE"), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class SignalDecisionRecord(Base):
    __tablename__ = "signal_decisions"
    __table_args__ = (
        Index("ux_signal_decision_fingerprint_mode", "input_fingerprint", "mode", unique=True),
        Index("ix_signal_decision_active", "instrument", "timeframe", "valid_until"),
        Index("ix_signal_decision_history", "instrument", "timeframe", "as_of"),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    decision_key: Mapped[str] = mapped_column(String(256), index=True)
    input_fingerprint: Mapped[str] = mapped_column(String(64))
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    direction: Mapped[str] = mapped_column(String(16), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    mode: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ai_score_snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_score_snapshots.id", ondelete="RESTRICT"), index=True)
    decision_policy_version: Mapped[str] = mapped_column(String(32), index=True)
    eligibility_score: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class SignalDecisionRuleRecord(Base):
    __tablename__ = "signal_decision_rules"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    decision_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("signal_decisions.id", ondelete="CASCADE"), index=True)
    rule_id: Mapped[str] = mapped_column(String(96), index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    outcome: Mapped[str] = mapped_column(String(24), index=True)
    severity: Mapped[str] = mapped_column(String(24), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class SignalDecisionReasonRecord(Base):
    __tablename__ = "signal_decision_reasons"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    decision_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("signal_decisions.id", ondelete="CASCADE"), index=True)
    reason_type: Mapped[str] = mapped_column(String(24), index=True)
    reason_code: Mapped[str] = mapped_column(String(96), index=True)
    severity: Mapped[str] = mapped_column(String(24), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ReplaySessionRecord(Base):
    __tablename__ = "replay_sessions"
    __table_args__ = (
        Index("ix_replay_session_request_fingerprint", "request_fingerprint"),
        Index("ix_replay_session_status_created", "status", "created_at"),
        Index("ix_replay_session_worker_lease", "worker_id", "lease_expires_at"),
        Index("ix_replay_session_dataset", "dataset_id", "dataset_version"),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), index=True)
    mode: Mapped[str] = mapped_column(String(24), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    virtual_cursor_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    dataset_id: Mapped[str] = mapped_column(String(128), index=True)
    dataset_version: Mapped[str] = mapped_column(String(64))
    processed_events: Mapped[int] = mapped_column(Integer, default=0)
    generated_events: Mapped[int] = mapped_column(Integer, default=0)
    progress_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    semantic_output_hash: Mapped[str] = mapped_column(String(64))
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    row_version: Mapped[int] = mapped_column(Integer, default=1)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ReplayCheckpointRecord(Base):
    __tablename__ = "replay_checkpoints"
    __table_args__ = (
        Index("ux_replay_checkpoint_sequence", "replay_id", "sequence", unique=True),
        Index("ix_replay_checkpoint_latest", "replay_id", "sequence"),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    replay_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("replay_sessions.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    cursor_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    state_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ReplayTransitionRecord(Base):
    __tablename__ = "replay_transitions"
    __table_args__ = (Index("ix_replay_transition_history", "replay_id", "occurred_at"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    replay_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("replay_sessions.id", ondelete="CASCADE"), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    from_status: Mapped[str] = mapped_column(String(24))
    to_status: Mapped[str] = mapped_column(String(24), index=True)
    reason_code: Mapped[str] = mapped_column(String(96))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ReplayTraceRecordModel(Base):
    __tablename__ = "replay_event_trace"
    __table_args__ = (Index("ux_replay_trace_sequence", "replay_id", "sequence", unique=True),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    replay_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("replay_sessions.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    virtual_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    event_type: Mapped[str] = mapped_column(String(96), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ReplayOutputRecord(Base):
    __tablename__ = "replay_outputs"
    __table_args__ = (
        Index("ix_replay_output_lookup", "replay_id", "output_type", "as_of"),
        Index("ux_replay_output_fingerprint", "replay_id", "fingerprint", unique=True),
    )
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    replay_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("replay_sessions.id", ondelete="CASCADE"), index=True)
    output_type: Mapped[str] = mapped_column(String(64), index=True)
    source_engine: Mapped[str] = mapped_column(String(64), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


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


class IntegrationEventRecord(Base):
    __tablename__ = "integration_events"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(96), index=True)
    trace_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    correlation_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    mode: Mapped[str] = mapped_column(String(16), index=True)
    instrument: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    timeframe: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class IntegrationOutboxRecord(Base):
    __tablename__ = "integration_outbox"
    outbox_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), ForeignKey("integration_events.event_id", ondelete="CASCADE"), unique=True, index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_error_code: Mapped[str | None] = mapped_column(String(96), nullable=True)
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)


class IntegrationProcessedEventRecord(Base):
    __tablename__ = "integration_processed_events"
    event_id: Mapped[str] = mapped_column(String(64), ForeignKey("integration_events.event_id", ondelete="CASCADE"), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class IntegrationSnapshotRecord(Base):
    __tablename__ = "integration_snapshots"
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    semantic_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    trace_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    mode: Mapped[str] = mapped_column(String(16), index=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analytical_boundary: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class OperationalSignalRecord(Base):
    __tablename__ = "operational_signals"
    operational_signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    semantic_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    decision_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("signal_decisions.id", ondelete="RESTRICT"), index=True)
    ai_score_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_score_snapshots.id", ondelete="RESTRICT"), index=True)
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("integration_snapshots.snapshot_id", ondelete="RESTRICT"), index=True)
    trace_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    mode: Mapped[str] = mapped_column(String(16), index=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class IntegrationEventTraceRecord(Base):
    __tablename__ = "integration_event_trace"
    trace_record_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    trace_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class IntegrationDataQualityIssueRecord(Base):
    __tablename__ = "integration_data_quality_issues"
    issue_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class MarketEvidenceFrameRecord(Base):
    """Full pre-normalization engine outputs for one closed timeframe candle."""

    __tablename__ = "market_evidence_frames"
    __table_args__ = (
        Index("ux_market_evidence_frames_hash", "frame_hash", unique=True),
        Index("ix_market_evidence_frames_series_boundary", "instrument", "timeframe", "candle_close_at"),
    )

    frame_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    frame_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    candle_close_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    knowledge_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class UnifiedMarketStateRecord(Base):
    """Immutable, synchronized M1/M5/M15 state for an AI-centric shadow cycle."""

    __tablename__ = "unified_market_states"
    __table_args__ = (
        Index("ux_unified_market_states_hash", "state_hash", unique=True),
        Index("ix_unified_market_states_series_boundary", "instrument", "market_data_boundary"),
    )

    state_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    trigger_timeframe: Mapped[str] = mapped_column(String(16), index=True)
    market_data_boundary: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    knowledge_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class UnifiedMarketStateCurrentRecord(Base):
    """Single compact pointer to the latest immutable state for each instrument."""

    __tablename__ = "unified_market_state_current"

    instrument: Mapped[str] = mapped_column(String(32), primary_key=True)
    state_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("unified_market_states.state_id", ondelete="CASCADE"),
        index=True,
    )
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class PipelineStageCurrentRecord(Base):
    __tablename__ = "pipeline_stage_current"

    instrument: Mapped[str] = mapped_column(String(32), primary_key=True)
    stage: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    reason: Mapped[str] = mapped_column(String(160), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    record_id: Mapped[str | None] = mapped_column(String(64))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class PipelineStageHistoryRecord(Base):
    __tablename__ = "pipeline_stage_history"
    __table_args__ = (
        Index(
            "ux_pipeline_stage_history_meaningful_change",
            "instrument",
            "stage",
            "fingerprint",
            unique=True,
        ),
        Index("ix_pipeline_stage_history_lookup", "instrument", "observed_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    reason: Mapped[str] = mapped_column(String(160), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    record_id: Mapped[str | None] = mapped_column(String(64))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class StorageRetentionPolicyRecord(Base):
    __tablename__ = "storage_retention_policies"

    relation_name: Mapped[str] = mapped_column(String(96), primary_key=True)
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False)
    cleanup_batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    protected: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str] = mapped_column(String(256), nullable=False)


class UnifiedMarketStateTimeframeRecord(Base):
    """The exact evidence frame selected for one timeframe in a market state."""

    __tablename__ = "unified_market_state_timeframes"
    __table_args__ = (Index("ix_unified_market_state_timeframes_frame", "frame_id"),)

    state_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("unified_market_states.state_id", ondelete="CASCADE"),
        primary_key=True,
    )
    timeframe: Mapped[str] = mapped_column(String(16), primary_key=True)
    frame_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("market_evidence_frames.frame_id", ondelete="RESTRICT"),
    )
    source_candle_close_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    expected_candle_close_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stale: Mapped[bool] = mapped_column(Boolean, default=False)


class EvidenceItemRecord(Base):
    """State-specific evidence preserving the complete raw analytical output."""

    __tablename__ = "evidence_items"
    __table_args__ = (
        Index("ix_evidence_items_engine_timeframe", "source_engine", "source_timeframe"),
        Index("ix_evidence_items_availability_time", "availability", "available_at"),
    )

    evidence_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    source_frame_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("market_evidence_frames.frame_id", ondelete="RESTRICT"),
        index=True,
    )
    source_engine: Mapped[str] = mapped_column(String(64), index=True)
    source_timeframe: Mapped[str] = mapped_column(String(16), index=True)
    availability: Mapped[str] = mapped_column(String(32), index=True)
    source_candle_close_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class UnifiedMarketStateEvidenceLinkRecord(Base):
    """Ordered many-to-many relationship between states and evidence items."""

    __tablename__ = "unified_market_state_evidence_links"
    __table_args__ = (
        Index("ux_unified_market_state_evidence_ordinal", "state_id", "ordinal", unique=True),
        Index("ix_unified_market_state_evidence_item", "evidence_id"),
    )

    state_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("unified_market_states.state_id", ondelete="CASCADE"),
        primary_key=True,
    )
    evidence_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_items.evidence_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer)


class QuantModelMetadataRecord(Base):
    __tablename__ = "quant_forecast_model_metadata"

    model_name: Mapped[str] = mapped_column(String(96), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(48), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class QuantFeatureVectorRecord(Base):
    __tablename__ = "quant_feature_vectors"
    __table_args__ = (Index("ix_quant_feature_vectors_series_boundary", "instrument", "point_in_time"),)

    vector_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    market_state_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("unified_market_states.state_id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    point_in_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    schema_version: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class QuantFeatureReferenceRecord(Base):
    __tablename__ = "quant_feature_references"

    vector_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quant_feature_vectors.vector_id", ondelete="CASCADE"),
        primary_key=True,
    )
    feature_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    evidence_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_items.evidence_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    source_paths: Mapped[list[str]] = mapped_column(JSONB)


class QuantForecastRequestRecord(Base):
    __tablename__ = "quant_forecast_requests"
    __table_args__ = (Index("ix_quant_forecast_requests_series_boundary", "instrument", "point_in_time"),)

    request_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    market_state_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("unified_market_states.state_id", ondelete="CASCADE"),
        index=True,
    )
    cycle_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    point_in_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_name: Mapped[str] = mapped_column(String(96), index=True)
    model_version: Mapped[str] = mapped_column(String(48))
    mode: Mapped[str] = mapped_column(String(16), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class QuantForecastResultRecord(Base):
    __tablename__ = "quantitative_forecasts"
    __table_args__ = (Index("ix_quantitative_forecasts_series_boundary", "instrument", "point_in_time"),)

    result_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    request_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quant_forecast_requests.request_id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    market_state_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("unified_market_states.state_id", ondelete="CASCADE"),
        index=True,
    )
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    point_in_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str] = mapped_column(String(96), index=True)
    model_version: Mapped[str] = mapped_column(String(48))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class QuantForecastHorizonRecord(Base):
    __tablename__ = "quantitative_forecast_horizons"

    result_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quantitative_forecasts.result_id", ondelete="CASCADE"),
        primary_key=True,
    )
    horizon_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    duration_seconds: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class QuantForecastOutcomeRecord(Base):
    __tablename__ = "quant_forecast_outcomes"
    __table_args__ = (Index("ux_quant_forecast_outcomes_result_horizon", "result_id", "horizon_id", unique=True),)

    outcome_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    result_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quantitative_forecasts.result_id", ondelete="CASCADE"),
        index=True,
    )
    horizon_id: Mapped[str] = mapped_column(String(24), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class QuantCalibrationReportRecord(Base):
    __tablename__ = "quant_calibration_runs"

    report_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(96), index=True)
    model_version: Mapped[str] = mapped_column(String(48), index=True)
    sample_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class QuantCalibrationBucketRecord(Base):
    __tablename__ = "quant_calibration_buckets"

    report_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quant_calibration_runs.report_id", ondelete="CASCADE"),
        primary_key=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    horizon_id: Mapped[str] = mapped_column(String(24), index=True)
    dimension: Mapped[str] = mapped_column(String(48), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class AIReasoningRequestRecord(Base):
    __tablename__ = "ai_reasoning_requests"

    request_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    cycle_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    market_state_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), index=True)
    quantitative_forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"), index=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    prompt_version: Mapped[str] = mapped_column(String(64), index=True)
    model_identifier: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AIReasoningCycleLockRecord(Base):
    """Durable distributed claim for one synchronized analytical cycle."""

    __tablename__ = "ai_reasoning_cycle_locks"
    __table_args__ = (
        Index(
            "ix_ai_reasoning_cycle_instrument_boundary",
            "instrument",
            "ums_boundary",
        ),
        Index(
            "ux_ai_reasoning_five_minute_cycle",
            "instrument",
            "analysis_timeframe",
            "five_minute_window_start",
            "analysis_contract_version",
            unique=True,
        ),
        Index(
            "ux_ai_reasoning_market_state_contract",
            "instrument",
            "market_state_hash",
            "analysis_contract_version",
            unique=True,
        ),
    )

    idempotency_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    ums_boundary: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cycle_version: Mapped[str] = mapped_column(String(32))
    provider_contract_version: Mapped[str] = mapped_column(String(128))
    analysis_timeframe: Mapped[str | None] = mapped_column(String(16), nullable=True)
    five_minute_window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    market_state_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    analysis_contract_version: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(24))
    request_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ai_reasoning_requests.request_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    forecast_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ai_market_forecasts.forecast_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    analysis_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ai_market_analyses.analysis_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AIMarketForecastRecord(Base):
    __tablename__ = "ai_market_forecasts"

    forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    request_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_reasoning_requests.request_id", ondelete="CASCADE"), unique=True, index=True)
    market_state_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), index=True)
    quantitative_forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    dominant_direction: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    selected_setup_family: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AIMarketAnalysisRecord(Base):
    """Immutable analysis-only provider result introduced by architecture v2."""

    __tablename__ = "ai_market_analyses"
    __table_args__ = (
        Index(
            "ux_ai_market_analysis_cycle",
            "symbol",
            "timeframe",
            "cycle_id",
            "schema_version",
            unique=True,
        ),
        Index(
            "ix_ai_market_analysis_history",
            "symbol",
            "timeframe",
            "analysis_timestamp",
        ),
    )

    analysis_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    request_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ai_reasoning_requests.request_id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    cycle_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    market_snapshot_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"),
        index=True,
    )
    quantitative_forecast_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"),
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    schema_version: Mapped[str] = mapped_column(String(32), index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    validation_passed: Mapped[bool] = mapped_column(Boolean, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AIForecastScenarioRecord(Base):
    __tablename__ = "ai_forecast_scenarios"

    forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_name: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class AIForecastEvidenceLinkRecord(Base):
    __tablename__ = "ai_forecast_evidence_links"

    forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), primary_key=True)
    evidence_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("evidence_items.evidence_id", ondelete="RESTRICT"), primary_key=True)
    role: Mapped[str] = mapped_column(String(32), primary_key=True)


class AISignalProposalRecord(Base):
    __tablename__ = "ai_signal_proposals"
    __table_args__ = (Index("ix_ai_signal_proposals_opportunity_created", "structural_opportunity_key", "created_at"),)

    proposal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), index=True)
    market_state_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), index=True)
    structural_opportunity_key: Mapped[str] = mapped_column(String(64), index=True)
    recommended_action: Mapped[str] = mapped_column(String(48), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AISetupFamilyVersionRecord(Base):
    __tablename__ = "ai_setup_family_versions"

    setup_family_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    registry_version: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ManagedSignalRecord(Base):
    __tablename__ = "managed_signals"

    signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    structural_opportunity_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    setup_family: Mapped[str] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(16), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    current_proposal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_signal_proposals.proposal_id", ondelete="RESTRICT"), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SignalStateTransitionRecord(Base):
    __tablename__ = "signal_state_transitions"

    transition_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), index=True)
    previous_state: Mapped[str] = mapped_column(String(32), index=True)
    new_state: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SignalLevelRevisionRecord(Base):
    __tablename__ = "signal_level_revisions"

    revision_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), index=True)
    level_type: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SignalMonitoringEvaluationRecord(Base):
    __tablename__ = "signal_monitoring_evaluations"

    evaluation_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), index=True)
    forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), index=True)
    thesis_valid: Mapped[bool] = mapped_column(Boolean, index=True)
    recommended_action: Mapped[str] = mapped_column(String(48), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class ManagedSignalOutcomeRecord(Base):
    __tablename__ = "managed_signal_outcomes"

    outcome_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), unique=True, index=True)
    final_state: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MarketMemoryEntryRecord(Base):
    __tablename__ = "market_memory_entries"
    __table_args__ = (Index("ix_market_memory_entries_series_time", "instrument", "occurred_at"),)

    entry_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    cycle_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    market_state_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(48), index=True)
    opportunity_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    signal_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="SET NULL"), nullable=True, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LLMStructuredOutputFailureRecord(Base):
    __tablename__ = "llm_structured_output_failures"

    failure_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    request_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_reasoning_requests.request_id", ondelete="CASCADE"), index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    model_identifier: Mapped[str] = mapped_column(String(128), index=True)
    failure_state: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class HardGateVersionRecord(Base):
    __tablename__ = "hard_gate_versions"

    gate_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    gate_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    registry_version: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class FinalSystemActionRecord(Base):
    __tablename__ = "final_system_actions"
    __table_args__ = (Index("ix_final_system_actions_signal_created", "managed_signal_id", "created_at"),)

    final_action_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    ai_proposal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_signal_proposals.proposal_id", ondelete="RESTRICT"), index=True)
    managed_signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), index=True)
    market_state_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"), index=True)
    quantitative_forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"), index=True)
    ai_forecast_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_market_forecasts.forecast_id", ondelete="RESTRICT"), index=True)
    action: Mapped[str] = mapped_column(String(48), index=True)
    approval_state: Mapped[str] = mapped_column(String(32), index=True)
    publication_state: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class GuardrailEvaluationRecord(Base):
    __tablename__ = "guardrail_evaluations"
    __table_args__ = (Index("ux_guardrail_evaluation_action_gate", "final_action_id", "gate_id", unique=True),)

    evaluation_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    final_action_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("final_system_actions.final_action_id", ondelete="CASCADE"), index=True)
    gate_id: Mapped[str] = mapped_column(String(96), index=True)
    gate_version: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class PublishedAnalyticalSignalRecord(Base):
    __tablename__ = "published_analytical_signals"
    __table_args__ = (Index("ux_published_analytical_signal_signal", "signal_id", unique=True),)

    publication_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), index=True)
    final_action_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("final_system_actions.final_action_id", ondelete="RESTRICT"), unique=True, index=True)
    proposal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("ai_signal_proposals.proposal_id", ondelete="RESTRICT"), index=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(16), index=True)
    setup_family: Mapped[str] = mapped_column(String(64), index=True)
    lifecycle_state: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LLMUsageMetricRecord(Base):
    __tablename__ = "llm_usage_metrics"
    __table_args__ = (Index("ix_llm_usage_metrics_date_model", "usage_date", "model_identifier"),)

    metric_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    usage_date: Mapped[str] = mapped_column(String(10), index=True)
    request_hash: Mapped[str] = mapped_column(String(64), index=True)
    market_state_hash: Mapped[str] = mapped_column(String(64), index=True)
    model_identifier: Mapped[str] = mapped_column(String(128), index=True)
    success: Mapped[bool] = mapped_column(Boolean, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class DetailedSignalOutcomeRecord(Base):
    __tablename__ = "detailed_signal_outcomes"

    outcome_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AIPerformanceReportRecord(Base):
    __tablename__ = "ai_performance_reports"

    report_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    sample_count: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AIProductionReadinessReportRecord(Base):
    __tablename__ = "ai_production_readiness_reports"

    report_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    sample_count: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
