-- TEN SMC Engine production schema. Idempotent PostgreSQL migration.
CREATE TABLE IF NOT EXISTS smc_analysis_snapshots (
    id UUID PRIMARY KEY, symbol VARCHAR(32) NOT NULL, timeframe VARCHAR(16) NOT NULL,
    analysis_timestamp TIMESTAMPTZ NOT NULL, market_data_boundary VARCHAR(512) NOT NULL,
    status VARCHAR(32) NOT NULL, processing_mode VARCHAR(32) NOT NULL,
    engine_version VARCHAR(32) NOT NULL, configuration_version VARCHAR(32) NOT NULL,
    payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_smc_snapshot_boundary ON smc_analysis_snapshots(symbol, timeframe, analysis_timestamp, configuration_version, processing_mode);
CREATE INDEX IF NOT EXISTS ix_smc_snapshot_series_time ON smc_analysis_snapshots(symbol, timeframe, analysis_timestamp DESC);

CREATE TABLE IF NOT EXISTS smc_objects (
    id UUID PRIMARY KEY, object_type VARCHAR(32) NOT NULL, symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL, analytical_timestamp TIMESTAMPTZ NOT NULL,
    availability_timestamp TIMESTAMPTZ NOT NULL, lifecycle_state VARCHAR(32) NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL, quality_score DOUBLE PRECISION NOT NULL,
    algorithm_version VARCHAR(32) NOT NULL, configuration_version VARCHAR(32) NOT NULL,
    payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_smc_objects_series_time ON smc_objects(symbol, timeframe, analytical_timestamp DESC);
CREATE INDEX IF NOT EXISTS ix_smc_objects_type_state ON smc_objects(object_type, lifecycle_state);

CREATE TABLE IF NOT EXISTS smc_checkpoints (
    id UUID PRIMARY KEY, symbol VARCHAR(32) NOT NULL, timeframe VARCHAR(16) NOT NULL,
    configuration_version VARCHAR(32) NOT NULL, snapshot_id UUID NOT NULL,
    last_processed_candle TIMESTAMPTZ NOT NULL, state_payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_smc_checkpoint_series ON smc_checkpoints(symbol, timeframe, configuration_version);
