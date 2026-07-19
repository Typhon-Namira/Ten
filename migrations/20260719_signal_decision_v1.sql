BEGIN;

CREATE TABLE IF NOT EXISTS signal_decisions (
    id UUID PRIMARY KEY,
    decision_key VARCHAR(256) NOT NULL,
    input_fingerprint VARCHAR(64) NOT NULL,
    instrument VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    direction VARCHAR(16) NOT NULL,
    state VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL,
    mode VARCHAR(16) NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ NOT NULL,
    ai_score_snapshot_id UUID NOT NULL REFERENCES ai_score_snapshots(id) ON DELETE RESTRICT,
    decision_policy_version VARCHAR(32) NOT NULL,
    eligibility_score DOUBLE PRECISION NOT NULL CHECK (eligibility_score >= 0 AND eligibility_score <= 100),
    payload JSONB NOT NULL,
    CONSTRAINT ux_signal_decision_fingerprint_mode UNIQUE (input_fingerprint, mode),
    CONSTRAINT ck_signal_decision_validity CHECK (valid_until >= valid_from)
);

CREATE TABLE IF NOT EXISTS signal_decision_rules (
    id UUID PRIMARY KEY,
    decision_id UUID NOT NULL REFERENCES signal_decisions(id) ON DELETE CASCADE,
    rule_id VARCHAR(96) NOT NULL,
    category VARCHAR(32) NOT NULL,
    outcome VARCHAR(24) NOT NULL,
    severity VARCHAR(24) NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_decision_reasons (
    id UUID PRIMARY KEY,
    decision_id UUID NOT NULL REFERENCES signal_decisions(id) ON DELETE CASCADE,
    reason_type VARCHAR(24) NOT NULL,
    reason_code VARCHAR(96) NOT NULL,
    severity VARCHAR(24) NOT NULL,
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_signal_decision_key ON signal_decisions (decision_key);
CREATE INDEX IF NOT EXISTS ix_signal_decision_active ON signal_decisions (instrument, timeframe, valid_until DESC);
CREATE INDEX IF NOT EXISTS ix_signal_decision_history ON signal_decisions (instrument, timeframe, as_of DESC);
CREATE INDEX IF NOT EXISTS ix_signal_decision_state ON signal_decisions (state);
CREATE INDEX IF NOT EXISTS ix_signal_decision_ai_score ON signal_decisions (ai_score_snapshot_id);
CREATE INDEX IF NOT EXISTS ix_signal_decision_rule_decision ON signal_decision_rules (decision_id);
CREATE INDEX IF NOT EXISTS ix_signal_decision_rule_id ON signal_decision_rules (rule_id);
CREATE INDEX IF NOT EXISTS ix_signal_decision_reason_decision ON signal_decision_reasons (decision_id);
CREATE INDEX IF NOT EXISTS ix_signal_decision_reason_code ON signal_decision_reasons (reason_code);

COMMIT;
