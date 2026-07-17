CREATE TABLE IF NOT EXISTS liquidity_snapshots (
 id UUID PRIMARY KEY, symbol VARCHAR(32) NOT NULL, timeframe VARCHAR(16) NOT NULL,
 analysis_timestamp TIMESTAMPTZ NOT NULL, processing_mode VARCHAR(32) NOT NULL,
 configuration_version VARCHAR(32) NOT NULL, engine_version VARCHAR(32) NOT NULL,
 payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_liquidity_snapshot_boundary ON liquidity_snapshots(symbol,timeframe,analysis_timestamp,configuration_version,processing_mode);
CREATE INDEX IF NOT EXISTS ix_liquidity_snapshot_series_time ON liquidity_snapshots(symbol,timeframe,analysis_timestamp DESC);
CREATE TABLE IF NOT EXISTS liquidity_objects (
 id UUID PRIMARY KEY, logical_id UUID NOT NULL, object_type VARCHAR(32) NOT NULL,
 symbol VARCHAR(32) NOT NULL, timeframe VARCHAR(16) NOT NULL, availability_timestamp TIMESTAMPTZ NOT NULL,
 lifecycle_state VARCHAR(32) NOT NULL, confidence_score DOUBLE PRECISION NOT NULL CHECK(confidence_score BETWEEN 0 AND 100),
 quality_score DOUBLE PRECISION NOT NULL CHECK(quality_score BETWEEN 0 AND 100), configuration_version VARCHAR(32) NOT NULL,
 engine_version VARCHAR(32) NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_liquidity_objects_series_time ON liquidity_objects(symbol,timeframe,availability_timestamp DESC);
CREATE INDEX IF NOT EXISTS ix_liquidity_objects_type_state ON liquidity_objects(object_type,lifecycle_state);
CREATE INDEX IF NOT EXISTS ix_liquidity_objects_logical ON liquidity_objects(logical_id,created_at);
CREATE TABLE IF NOT EXISTS liquidity_checkpoints (
 id UUID PRIMARY KEY, symbol VARCHAR(32) NOT NULL, timeframe VARCHAR(16) NOT NULL,
 configuration_version VARCHAR(32) NOT NULL, engine_version VARCHAR(32) NOT NULL, snapshot_id UUID NOT NULL,
 last_processed_candle TIMESTAMPTZ NOT NULL, state_hash VARCHAR(64) NOT NULL, state_payload JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_liquidity_checkpoint_series ON liquidity_checkpoints(symbol,timeframe,configuration_version);
