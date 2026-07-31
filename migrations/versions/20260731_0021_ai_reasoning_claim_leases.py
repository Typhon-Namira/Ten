"""Replace permanent AI cycle claims with expiring ownership leases.

Revision ID: 20260731_0021
Revises: 20260731_0020
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "20260731_0021"
down_revision = "20260731_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.alter_column(
        "ai_reasoning_cycle_locks",
        "status",
        existing_type=sa.String(24),
        type_=sa.String(32),
        existing_nullable=False,
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("claim_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("market_state_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("snapshot_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("claimed_by", sa.String(128), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("failure_reason", sa.String(256), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("expired_claim_count", sa.Integer(), nullable=True),
    )
    op.execute(
        "UPDATE ai_reasoning_cycle_locks SET "
        "claim_id = gen_random_uuid(), "
        "market_state_id = r.market_state_id, snapshot_id = r.market_state_id "
        "FROM ai_reasoning_requests r "
        "WHERE ai_reasoning_cycle_locks.request_id = r.request_id"
    )
    op.execute(
        "UPDATE ai_reasoning_cycle_locks SET "
        "claim_id = COALESCE(claim_id, gen_random_uuid()), "
        "claimed_by = CASE WHEN status = 'claimed' THEN 'legacy-orphan' ELSE NULL END, "
        "heartbeat_at = claimed_at, "
        "lease_expires_at = CASE WHEN status = 'claimed' "
        "THEN claimed_at + INTERVAL '90 seconds' "
        "ELSE COALESCE(completed_at, claimed_at) END, "
        "released_at = CASE WHEN status = 'claimed' THEN NULL "
        "ELSE COALESCE(completed_at, claimed_at) END, "
        "expired_claim_count = 0, "
        "status = CASE "
        "WHEN status = 'claimed' THEN 'ACTIVE_CLAIM' "
        "WHEN status IN ('completed','COMPLETED') AND analysis_id IS NOT NULL THEN 'COMMITTED' "
        "WHEN status IN ('failed','FAILED_PROVIDER','FAILED_SCHEMA','FAILED_PERSISTENCE','TIMED_OUT') THEN 'FAILED' "
        "ELSE 'RELEASED' END"
    )
    for column in ("claim_id", "heartbeat_at", "lease_expires_at", "expired_claim_count"):
        op.alter_column("ai_reasoning_cycle_locks", column, nullable=False)
    op.create_foreign_key(
        "fk_ai_reasoning_cycle_market_state",
        "ai_reasoning_cycle_locks",
        "unified_market_states",
        ["market_state_id"],
        ["state_id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_ai_reasoning_cycle_snapshot",
        "ai_reasoning_cycle_locks",
        "unified_market_states",
        ["snapshot_id"],
        ["state_id"],
        ondelete="SET NULL",
    )
    for name, columns, unique in (
        ("ix_ai_reasoning_cycle_locks_claim_id", ["claim_id"], True),
        ("ix_ai_reasoning_cycle_locks_market_state_id", ["market_state_id"], False),
        ("ix_ai_reasoning_cycle_locks_snapshot_id", ["snapshot_id"], False),
        ("ix_ai_reasoning_cycle_locks_claimed_by", ["claimed_by"], False),
        ("ix_ai_reasoning_cycle_locks_heartbeat_at", ["heartbeat_at"], False),
        ("ix_ai_reasoning_cycle_locks_lease_expires_at", ["lease_expires_at"], False),
    ):
        op.create_index(name, "ai_reasoning_cycle_locks", columns, unique=unique)
    op.create_check_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('ACTIVE_CLAIM','COMMITTED','FAILED','RELEASED','EXPIRED','RECOVERED')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.execute(
        "UPDATE ai_reasoning_cycle_locks SET status = CASE "
        "WHEN status = 'ACTIVE_CLAIM' THEN 'claimed' "
        "WHEN status IN ('COMMITTED','RECOVERED') THEN 'COMPLETED' "
        "WHEN status = 'RELEASED' THEN 'skipped' ELSE 'failed' END"
    )
    for name in (
        "ix_ai_reasoning_cycle_locks_lease_expires_at",
        "ix_ai_reasoning_cycle_locks_heartbeat_at",
        "ix_ai_reasoning_cycle_locks_claimed_by",
        "ix_ai_reasoning_cycle_locks_snapshot_id",
        "ix_ai_reasoning_cycle_locks_market_state_id",
        "ix_ai_reasoning_cycle_locks_claim_id",
    ):
        op.drop_index(name, table_name="ai_reasoning_cycle_locks")
    op.drop_constraint("fk_ai_reasoning_cycle_snapshot", "ai_reasoning_cycle_locks", type_="foreignkey")
    op.drop_constraint("fk_ai_reasoning_cycle_market_state", "ai_reasoning_cycle_locks", type_="foreignkey")
    for column in (
        "expired_claim_count", "released_at", "failure_reason", "lease_expires_at",
        "heartbeat_at", "claimed_by", "snapshot_id", "market_state_id", "claim_id",
    ):
        op.drop_column("ai_reasoning_cycle_locks", column)
    op.alter_column(
        "ai_reasoning_cycle_locks",
        "status",
        existing_type=sa.String(32),
        type_=sa.String(24),
        existing_nullable=False,
    )
    op.create_check_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('claimed','completed','failed','skipped','COMPLETED',"
        "'FAILED_PROVIDER','FAILED_SCHEMA','FAILED_PERSISTENCE','TIMED_OUT',"
        "'SKIPPED_WITH_REASON')",
    )
