"""Convert ten-minute reasoning windows to synchronized-cycle claims.

Revision ID: 20260727_0009
Revises: 20260724_0008
Create Date: 2026-07-27

Existing rows are retained as historical claims. New code derives keys from the UMS
boundary and provider contract rather than a wall-clock bucket.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260727_0009"
down_revision: str | None = "20260724_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.rename_table("ai_reasoning_window_locks", "ai_reasoning_cycle_locks")
    op.drop_index(
        "ix_ai_reasoning_window_instrument_bucket",
        table_name="ai_reasoning_cycle_locks",
    )
    op.alter_column(
        "ai_reasoning_cycle_locks",
        "ten_minute_bucket",
        new_column_name="ums_boundary",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    op.alter_column(
        "ai_reasoning_cycle_locks",
        "market_state_version",
        new_column_name="cycle_version",
        existing_type=sa.String(length=32),
        nullable=False,
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column(
            "provider_contract_version",
            sa.String(length=128),
            nullable=False,
            server_default="legacy-ten-minute-contract",
        ),
    )
    op.alter_column(
        "ai_reasoning_cycle_locks",
        "provider_contract_version",
        existing_type=sa.String(length=128),
        server_default=None,
    )
    op.create_index(
        "ix_ai_reasoning_cycle_instrument_boundary",
        "ai_reasoning_cycle_locks",
        ["instrument", "ums_boundary"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ai_reasoning_cycle_instrument_boundary",
        table_name="ai_reasoning_cycle_locks",
    )
    op.drop_column("ai_reasoning_cycle_locks", "provider_contract_version")
    op.alter_column(
        "ai_reasoning_cycle_locks",
        "cycle_version",
        new_column_name="market_state_version",
        existing_type=sa.String(length=32),
        nullable=False,
    )
    op.alter_column(
        "ai_reasoning_cycle_locks",
        "ums_boundary",
        new_column_name="ten_minute_bucket",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    op.create_index(
        "ix_ai_reasoning_window_instrument_bucket",
        "ai_reasoning_cycle_locks",
        ["instrument", "ten_minute_bucket"],
    )
    op.rename_table("ai_reasoning_cycle_locks", "ai_reasoning_window_locks")
