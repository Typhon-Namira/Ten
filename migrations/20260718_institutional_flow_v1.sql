-- TEN Institutional Flow Engine Production 1.0 (idempotent PostgreSQL migration)
CREATE TABLE IF NOT EXISTS institutional_flow_snapshots (
    id UUID PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    analysis_timestamp TIMESTAMPTZ NOT NULL,
    processing_mode VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    configuration_version VARCHAR(32) NOT NULL,
    engine_version VARCHAR(32) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_institutional_flow_snapshot_boundary
ON institutional_flow_snapshots(symbol, timeframe, analysis_timestamp, configuration_version, processing_mode);

CREATE TABLE IF NOT EXISTS institutional_flow_evidence (
    id UUID PRIMARY KEY,
    evidence_id UUID NOT NULL,
    source_engine VARCHAR(32) NOT NULL,
    evidence_type VARCHAR(64) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    availability_timestamp TIMESTAMPTZ NOT NULL,
    direction VARCHAR(32) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    quality DOUBLE PRECISION NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_institutional_flow_evidence_series_time
ON institutional_flow_evidence(symbol, timeframe, availability_timestamp);

CREATE TABLE IF NOT EXISTS institutional_flow_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    configuration_version VARCHAR(32) NOT NULL,
    engine_version VARCHAR(32) NOT NULL,
    snapshot_id UUID NOT NULL,
    last_processed_candle TIMESTAMPTZ NOT NULL,
    state_hash VARCHAR(64) NOT NULL,
    state_payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_institutional_flow_checkpoint_series
ON institutional_flow_checkpoints(symbol, timeframe, configuration_version);
