CREATE TABLE IF NOT EXISTS replay_sessions (
    id UUID PRIMARY KEY,
    request_fingerprint VARCHAR(64) NOT NULL,
    status VARCHAR(24) NOT NULL,
    mode VARCHAR(24) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    virtual_cursor_at TIMESTAMPTZ NOT NULL,
    dataset_id VARCHAR(128) NOT NULL,
    dataset_version VARCHAR(64) NOT NULL,
    processed_events INTEGER NOT NULL DEFAULT 0 CHECK (processed_events >= 0),
    generated_events INTEGER NOT NULL DEFAULT 0 CHECK (generated_events >= 0),
    progress_percent DOUBLE PRECISION NULL CHECK (progress_percent IS NULL OR progress_percent BETWEEN 0 AND 100),
    semantic_output_hash VARCHAR(64) NOT NULL,
    worker_id VARCHAR(128) NULL,
    lease_expires_at TIMESTAMPTZ NULL,
    heartbeat_at TIMESTAMPTZ NULL,
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version >= 1),
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_replay_session_status_created ON replay_sessions (status, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_replay_session_request_fingerprint ON replay_sessions (request_fingerprint);
CREATE INDEX IF NOT EXISTS ix_replay_session_worker_lease ON replay_sessions (worker_id, lease_expires_at);
CREATE INDEX IF NOT EXISTS ix_replay_session_dataset ON replay_sessions (dataset_id, dataset_version);

CREATE TABLE IF NOT EXISTS replay_checkpoints (
    id UUID PRIMARY KEY,
    replay_id UUID NOT NULL REFERENCES replay_sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    cursor_at TIMESTAMPTZ NOT NULL,
    state_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (replay_id, sequence)
);
CREATE INDEX IF NOT EXISTS ix_replay_checkpoint_latest ON replay_checkpoints (replay_id, sequence DESC);

CREATE TABLE IF NOT EXISTS replay_transitions (
    id UUID PRIMARY KEY,
    replay_id UUID NOT NULL REFERENCES replay_sessions(id) ON DELETE CASCADE,
    occurred_at TIMESTAMPTZ NOT NULL,
    from_status VARCHAR(24) NOT NULL,
    to_status VARCHAR(24) NOT NULL,
    reason_code VARCHAR(96) NOT NULL,
    payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_replay_transition_history ON replay_transitions (replay_id, occurred_at, id);

CREATE TABLE IF NOT EXISTS replay_event_trace (
    id BIGSERIAL PRIMARY KEY,
    replay_id UUID NOT NULL REFERENCES replay_sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    virtual_time TIMESTAMPTZ NOT NULL,
    event_id UUID NOT NULL,
    event_type VARCHAR(96) NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (replay_id, sequence)
);
CREATE INDEX IF NOT EXISTS ix_replay_trace_virtual_time ON replay_event_trace (replay_id, virtual_time, sequence);

CREATE TABLE IF NOT EXISTS replay_outputs (
    id UUID PRIMARY KEY,
    replay_id UUID NOT NULL REFERENCES replay_sessions(id) ON DELETE CASCADE,
    output_type VARCHAR(64) NOT NULL,
    source_engine VARCHAR(64) NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    fingerprint VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (replay_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS ix_replay_output_lookup ON replay_outputs (replay_id, output_type, as_of);
