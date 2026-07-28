"""Track deterministic analysis-signal lifecycle and outcomes.

Revision ID: 20260728_0013
Revises: 20260728_0012
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260728_0013"
down_revision = "20260728_0012"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "ai_analysis_signal_outcomes",
        sa.Column("outcome_id", UUID, primary_key=True),
        sa.Column(
            "signal_id",
            UUID,
            sa.ForeignKey("ai_analysis_signals.signal_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("entry_reached", sa.Boolean(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('ACTIVE','COMPLETED','STALE','EXPIRED','STOPPED',"
            "'TARGET_HIT','STOP_HIT','SUPERSEDED')",
            name="ck_ai_analysis_signal_outcome_status",
        ),
    )
    op.create_index(
        "ix_ai_analysis_signal_outcomes_signal_id",
        "ai_analysis_signal_outcomes",
        ["signal_id"],
        unique=True,
    )
    for name, columns in (
        ("ix_ai_analysis_signal_outcomes_status", ["status"]),
        ("ix_ai_analysis_signal_outcomes_entry_reached", ["entry_reached"]),
        ("ix_ai_analysis_signal_outcomes_evaluated_at", ["evaluated_at"]),
        ("ix_ai_analysis_signal_outcomes_completed_at", ["completed_at"]),
    ):
        op.create_index(name, "ai_analysis_signal_outcomes", columns)


def downgrade() -> None:
    for name in (
        "ix_ai_analysis_signal_outcomes_completed_at",
        "ix_ai_analysis_signal_outcomes_evaluated_at",
        "ix_ai_analysis_signal_outcomes_entry_reached",
        "ix_ai_analysis_signal_outcomes_status",
        "ix_ai_analysis_signal_outcomes_signal_id",
    ):
        op.drop_index(name, table_name="ai_analysis_signal_outcomes")
    op.drop_table("ai_analysis_signal_outcomes")
