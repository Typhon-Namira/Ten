"""Add durable ten-minute AI reasoning idempotency locks.

Revision ID: 20260724_0008
Revises: 20260724_0007
Create Date: 2026-07-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260724_0008"
down_revision: str | None = "20260724_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_reasoning_window_locks",
        sa.Column("idempotency_key", sa.String(64), primary_key=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("ten_minute_bucket", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_state_version", sa.String(32), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ai_reasoning_requests.request_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "forecast_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ai_market_forecasts.forecast_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('claimed','completed','failed')",
            name="ck_ai_reasoning_window_lock_status",
        ),
    )
    op.create_index(
        "ix_ai_reasoning_window_instrument_bucket",
        "ai_reasoning_window_locks",
        ["instrument", "ten_minute_bucket"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ai_reasoning_window_instrument_bucket",
        table_name="ai_reasoning_window_locks",
    )
    op.drop_table("ai_reasoning_window_locks")
