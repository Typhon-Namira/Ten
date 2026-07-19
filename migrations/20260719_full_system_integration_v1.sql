BEGIN;

CREATE TABLE IF NOT EXISTS integration_events (
    event_id varchar(64) PRIMARY KEY,
    event_type varchar(96) NOT NULL,
    trace_id uuid NOT NULL,
    correlation_id uuid NOT NULL,
    mode varchar(16) NOT NULL CHECK (mode IN ('live', 'replay')),
    instrument varchar(32),
    timeframe varchar(16),
    occurred_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    payload_hash varchar(64) NOT NULL,
    payload jsonb NOT NULL,
    CHECK (available_at >= occurred_at)
);

CREATE TABLE IF NOT EXISTS integration_outbox (
    outbox_id uuid PRIMARY KEY,
    event_id varchar(64) NOT NULL UNIQUE REFERENCES integration_events(event_id) ON DELETE CASCADE,
    available_at timestamptz NOT NULL,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    published_at timestamptz,
    last_error_code varchar(96)
);
CREATE INDEX IF NOT EXISTS ix_integration_outbox_pending ON integration_outbox (published_at, available_at);

CREATE TABLE IF NOT EXISTS integration_processed_events (
    event_id varchar(64) PRIMARY KEY REFERENCES integration_events(event_id) ON DELETE CASCADE,
    processed_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS integration_snapshots (
    snapshot_id uuid PRIMARY KEY,
    semantic_hash varchar(64) NOT NULL UNIQUE,
    trace_id uuid NOT NULL,
    mode varchar(16) NOT NULL CHECK (mode IN ('live', 'replay')),
    instrument varchar(32) NOT NULL,
    timeframe varchar(16) NOT NULL,
    analytical_boundary timestamptz NOT NULL,
    status varchar(32) NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_integration_snapshot_key ON integration_snapshots (mode, instrument, timeframe, analytical_boundary DESC);

CREATE TABLE IF NOT EXISTS operational_signals (
    operational_signal_id uuid PRIMARY KEY,
    semantic_hash varchar(64) NOT NULL UNIQUE,
    decision_id uuid NOT NULL REFERENCES signal_decisions(id) ON DELETE RESTRICT,
    ai_score_id uuid NOT NULL REFERENCES ai_score_snapshots(id) ON DELETE RESTRICT,
    snapshot_id uuid NOT NULL REFERENCES integration_snapshots(snapshot_id) ON DELETE RESTRICT,
    trace_id uuid NOT NULL,
    mode varchar(16) NOT NULL CHECK (mode IN ('live', 'replay')),
    instrument varchar(32) NOT NULL,
    timeframe varchar(16) NOT NULL,
    effective_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_operational_signal_latest ON operational_signals (mode, instrument, timeframe, effective_at DESC);
CREATE INDEX IF NOT EXISTS ix_operational_signal_trace ON operational_signals (trace_id);

CREATE TABLE IF NOT EXISTS integration_event_trace (
    trace_record_id uuid PRIMARY KEY,
    trace_id uuid NOT NULL,
    event_id varchar(64) NOT NULL,
    status varchar(24) NOT NULL,
    started_at timestamptz NOT NULL,
    completed_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_integration_trace_lookup ON integration_event_trace (trace_id, started_at);

CREATE TABLE IF NOT EXISTS integration_data_quality_issues (
    issue_id uuid PRIMARY KEY,
    event_id varchar(64) NOT NULL,
    provider varchar(64) NOT NULL,
    status varchar(24) NOT NULL,
    observed_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_integration_quality_provider ON integration_data_quality_issues (provider, status, observed_at DESC);

COMMIT;
