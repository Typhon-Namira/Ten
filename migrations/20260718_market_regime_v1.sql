-- TEN Market Regime Engine Production 1.0 (PostgreSQL, additive and idempotent)
CREATE TABLE IF NOT EXISTS market_regime_snapshots (
    id UUID PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    analysis_timestamp TIMESTAMPTZ NOT NULL,
    dominant_regime VARCHAR(64) NOT NULL,
    configuration_version VARCHAR(32) NOT NULL,
    engine_version VARCHAR(32) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_market_regime_snapshot_boundary ON market_regime_snapshots(symbol, timeframe, analysis_timestamp, configuration_version);
CREATE INDEX IF NOT EXISTS ix_market_regime_snapshot_latest ON market_regime_snapshots(symbol, timeframe, analysis_timestamp DESC);

CREATE TABLE IF NOT EXISTS market_regime_evidence (
    id UUID PRIMARY KEY,
    evidence_id UUID NOT NULL,
    snapshot_id UUID NOT NULL REFERENCES market_regime_snapshots(id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    source_engine VARCHAR(32) NOT NULL,
    family VARCHAR(32) NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    accepted BOOLEAN NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_market_regime_evidence_series_time ON market_regime_evidence(symbol, timeframe, available_at);
CREATE INDEX IF NOT EXISTS ix_market_regime_evidence_snapshot ON market_regime_evidence(snapshot_id);

CREATE TABLE IF NOT EXISTS market_regime_transitions (
    id UUID PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    from_regime VARCHAR(64) NOT NULL,
    to_regime VARCHAR(64) NOT NULL,
    state VARCHAR(32) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    confirmed_at TIMESTAMPTZ,
    payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_market_regime_transition_series_time ON market_regime_transitions(symbol, timeframe, started_at);
CREATE INDEX IF NOT EXISTS ix_market_regime_transition_state ON market_regime_transitions(state, confirmed_at);

CREATE TABLE IF NOT EXISTS market_regime_checkpoints (
    id UUID PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    engine_name VARCHAR(32) NOT NULL,
    engine_version VARCHAR(32) NOT NULL,
    schema_version VARCHAR(32) NOT NULL,
    configuration_version VARCHAR(32) NOT NULL,
    algorithm_version VARCHAR(32) NOT NULL,
    snapshot_id UUID NOT NULL REFERENCES market_regime_snapshots(id),
    analysis_boundary TIMESTAMPTZ NOT NULL,
    payload_hash VARCHAR(64) NOT NULL,
    state_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_market_regime_checkpoint_series ON market_regime_checkpoints(symbol, timeframe, configuration_version);
CREATE INDEX IF NOT EXISTS ix_market_regime_checkpoint_boundary ON market_regime_checkpoints(symbol, timeframe, analysis_boundary DESC);
