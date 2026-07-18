-- TEN Economic Calendar Engine Production 1.0 (PostgreSQL, additive and idempotent)
CREATE TABLE IF NOT EXISTS economic_calendar_provider_observations (
 id UUID PRIMARY KEY, provider_name VARCHAR(64) NOT NULL, provider_event_id VARCHAR(256) NOT NULL,
 available_at TIMESTAMPTZ NOT NULL, ingested_at TIMESTAMPTZ NOT NULL, payload_hash VARCHAR(64) NOT NULL, payload JSONB NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_economic_calendar_provider_observation_hash ON economic_calendar_provider_observations(provider_name, provider_event_id, payload_hash);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_provider_observation_available ON economic_calendar_provider_observations(available_at);

CREATE TABLE IF NOT EXISTS economic_calendar_events (
 id UUID PRIMARY KEY, canonical_name VARCHAR(256) NOT NULL, scheduled_at TIMESTAMPTZ, available_at TIMESTAMPTZ NOT NULL,
 country_code VARCHAR(2), currency_codes JSONB NOT NULL, category VARCHAR(64) NOT NULL, importance VARCHAR(16) NOT NULL,
 status VARCHAR(32) NOT NULL, configuration_version VARCHAR(32) NOT NULL, payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_event_schedule ON economic_calendar_events(scheduled_at, importance, canonical_name);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_event_available ON economic_calendar_events(available_at);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_event_country ON economic_calendar_events(country_code);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_event_category ON economic_calendar_events(category);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_event_importance ON economic_calendar_events(importance);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_event_status ON economic_calendar_events(status);

CREATE TABLE IF NOT EXISTS economic_calendar_event_revisions (
 id UUID PRIMARY KEY, event_id UUID NOT NULL REFERENCES economic_calendar_events(id), revision_number INTEGER NOT NULL,
 revision_type VARCHAR(32) NOT NULL, available_at TIMESTAMPTZ NOT NULL, payload_hash VARCHAR(64) NOT NULL, payload JSONB NOT NULL,
 UNIQUE(event_id, revision_number)
);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_revision_available ON economic_calendar_event_revisions(event_id, available_at);

CREATE TABLE IF NOT EXISTS economic_calendar_snapshots (
 id UUID PRIMARY KEY, analysis_timestamp TIMESTAMPTZ NOT NULL, historical_boundary TIMESTAMPTZ NOT NULL,
 configuration_version VARCHAR(32) NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_snapshot_boundary ON economic_calendar_snapshots(historical_boundary DESC);

CREATE TABLE IF NOT EXISTS economic_calendar_instrument_contexts (
 id UUID PRIMARY KEY, symbol VARCHAR(32) NOT NULL, historical_boundary TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_economic_calendar_context_boundary ON economic_calendar_instrument_contexts(symbol, historical_boundary, id);

CREATE TABLE IF NOT EXISTS economic_calendar_sync_state (
 provider_name VARCHAR(64) PRIMARY KEY, state JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS economic_calendar_checkpoints (
 id UUID PRIMARY KEY, engine_name VARCHAR(64) NOT NULL, engine_version VARCHAR(32) NOT NULL, schema_version VARCHAR(32) NOT NULL,
 configuration_version VARCHAR(32) NOT NULL, normalization_version VARCHAR(32) NOT NULL, payload_hash VARCHAR(64) NOT NULL,
 state_payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_economic_calendar_checkpoint_engine ON economic_calendar_checkpoints(engine_name, configuration_version);
CREATE INDEX IF NOT EXISTS ix_economic_calendar_checkpoint_created ON economic_calendar_checkpoints(created_at DESC);
