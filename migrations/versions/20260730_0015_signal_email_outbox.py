"""Add the durable signal email notification outbox.

Revision ID: 20260730_0015
Revises: 20260729_0014
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260730_0015"
down_revision = "20260729_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signal_email_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("signal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "decision_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("signal_decisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("recipient", sa.String(320), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_message_id", sa.String(256), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING','PROCESSING','SENT','FAILED')",
            name="ck_signal_email_outbox_status",
        ),
    )
    op.create_index(
        "ux_signal_email_outbox_signal_id",
        "signal_email_outbox",
        ["signal_id"],
        unique=True,
    )
    op.create_index(
        "ix_signal_email_outbox_decision_id",
        "signal_email_outbox",
        ["decision_id"],
    )
    op.create_index(
        "ix_signal_email_outbox_status",
        "signal_email_outbox",
        ["status"],
    )
    op.create_index(
        "ix_signal_email_outbox_claim",
        "signal_email_outbox",
        ["status", "next_retry_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_signal_email_outbox_claim", table_name="signal_email_outbox")
    op.drop_index("ix_signal_email_outbox_status", table_name="signal_email_outbox")
    op.drop_index("ix_signal_email_outbox_decision_id", table_name="signal_email_outbox")
    op.drop_index("ux_signal_email_outbox_signal_id", table_name="signal_email_outbox")
    op.drop_table("signal_email_outbox")
