CREATE TABLE IF NOT EXISTS volume_profile_snapshots (
    id UUID PRIMARY KEY, symbol VARCHAR(32) NOT NULL, timeframe VARCHAR(16) NOT NULL,
    analysis_timestamp TIMESTAMPTZ NOT NULL, processing_mode VARCHAR(32) NOT NULL,
    configuration_version VARCHAR(32) NOT NULL, engine_version VARCHAR(32) NOT NULL,
    payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_volume_profile_snapshot_boundary ON volume_profile_snapshots(symbol, timeframe, analysis_timestamp, configuration_version, processing_mode);
CREATE INDEX IF NOT EXISTS ix_volume_profile_snapshot_series_time ON volume_profile_snapshots(symbol, timeframe, analysis_timestamp);

CREATE TABLE IF NOT EXISTS volume_profile_objects (
    id UUID PRIMARY KEY, logical_id UUID NOT NULL, object_type VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NOT NULL, timeframe VARCHAR(16) NOT NULL,
    availability_timestamp TIMESTAMPTZ NOT NULL, lifecycle_state VARCHAR(32) NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL, quality_score DOUBLE PRECISION NOT NULL,
    configuration_version VARCHAR(32) NOT NULL, engine_version VARCHAR(32) NOT NULL,
    payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_volume_profile_objects_series_time ON volume_profile_objects(symbol, timeframe, availability_timestamp);
CREATE INDEX IF NOT EXISTS ix_volume_profile_objects_type_state ON volume_profile_objects(object_type, lifecycle_state);

CREATE TABLE IF NOT EXISTS volume_profile_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(), symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL, configuration_version VARCHAR(32) NOT NULL,
    engine_version VARCHAR(32) NOT NULL, snapshot_id UUID NOT NULL,
    last_processed_candle TIMESTAMPTZ NOT NULL, state_hash VARCHAR(64) NOT NULL,
    state_payload JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_volume_profile_checkpoint_series ON volume_profile_checkpoints(symbol, timeframe, configuration_version);
