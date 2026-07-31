"""Persist the authoritative AI provider retry schedule.

Revision ID: 20260731_0022
Revises: 20260731_0021
"""

from alembic import op
import sqlalchemy as sa


revision = "20260731_0022"
down_revision = "20260731_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_ai_reasoning_cycle_locks_next_retry_at",
        "ai_reasoning_cycle_locks",
        ["next_retry_at"],
        unique=False,
    )
    op.create_check_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('ACTIVE_CLAIM','COMMITTED','FAILED','FAILED_SCHEMA',"
        "'WAITING_PROVIDER','RELEASED','EXPIRED','RECOVERED')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.execute(
        "UPDATE ai_reasoning_cycle_locks SET status = 'FAILED' "
        "WHERE status IN ('FAILED_SCHEMA','WAITING_PROVIDER')"
    )
    op.drop_index(
        "ix_ai_reasoning_cycle_locks_next_retry_at",
        table_name="ai_reasoning_cycle_locks",
    )
    op.drop_column("ai_reasoning_cycle_locks", "next_retry_at")
    op.create_check_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('ACTIVE_CLAIM','COMMITTED','FAILED','RELEASED','EXPIRED','RECOVERED')",
    )
