BEGIN;

CREATE TABLE IF NOT EXISTS ai_score_snapshots (
    id UUID PRIMARY KEY,
    instrument VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL,
    mode VARCHAR(16) NOT NULL CHECK (mode IN ('live', 'replay')),
    status VARCHAR(32) NOT NULL CHECK (status IN ('ready', 'degraded', 'insufficient_evidence', 'stale', 'invalid', 'replay')),
    policy_name VARCHAR(64) NOT NULL,
    policy_version VARCHAR(32) NOT NULL,
    configuration_version VARCHAR(32) NOT NULL,
    configuration_hash VARCHAR(64) NOT NULL,
    input_fingerprint VARCHAR(64) NOT NULL,
    directional_score DOUBLE PRECISION NOT NULL CHECK (directional_score BETWEEN -100 AND 100),
    confidence_score DOUBLE PRECISION NOT NULL CHECK (confidence_score BETWEEN 0 AND 100),
    market_risk_score DOUBLE PRECISION NOT NULL CHECK (market_risk_score BETWEEN 0 AND 100),
    payload JSONB NOT NULL,
    UNIQUE (input_fingerprint, mode)
);

CREATE TABLE IF NOT EXISTS ai_score_components (
    id VARCHAR(64) PRIMARY KEY,
    snapshot_id UUID NOT NULL REFERENCES ai_score_snapshots(id) ON DELETE CASCADE,
    source_engine VARCHAR(32) NOT NULL,
    source_group VARCHAR(32) NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_score_conflicts (
    id UUID PRIMARY KEY,
    snapshot_id UUID NOT NULL REFERENCES ai_score_snapshots(id) ON DELETE CASCADE,
    severity VARCHAR(16) NOT NULL CHECK (severity IN ('moderate', 'severe')),
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_ai_score_series_time ON ai_score_snapshots (instrument, timeframe, as_of DESC);
CREATE INDEX IF NOT EXISTS ix_ai_score_policy ON ai_score_snapshots (policy_version, status, mode);
CREATE INDEX IF NOT EXISTS ix_ai_score_components_snapshot ON ai_score_components (snapshot_id);
CREATE INDEX IF NOT EXISTS ix_ai_score_components_source ON ai_score_components (source_engine, source_group);
CREATE INDEX IF NOT EXISTS ix_ai_score_conflicts_snapshot ON ai_score_conflicts (snapshot_id);

COMMIT;
