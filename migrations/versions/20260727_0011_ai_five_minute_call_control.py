"""Add durable five-minute AI analysis call control.

Revision ID: 20260727_0011
Revises: 20260727_0010
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260727_0011"
down_revision = "20260727_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("analysis_timeframe", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column(
            "five_minute_window_start",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column("market_state_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column(
            "analysis_contract_version",
            sa.String(length=128),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_ai_reasoning_cycle_locks_five_minute_window_start",
        "ai_reasoning_cycle_locks",
        ["five_minute_window_start"],
    )
    # Legacy rows intentionally retain NULL in the new columns. PostgreSQL unique
    # indexes permit those historical rows while enforcing exactly one claim for
    # every new five-minute analysis cycle and every immutable market state.
    op.create_index(
        "ux_ai_reasoning_five_minute_cycle",
        "ai_reasoning_cycle_locks",
        [
            "instrument",
            "analysis_timeframe",
            "five_minute_window_start",
            "analysis_contract_version",
        ],
        unique=True,
    )
    op.create_index(
        "ux_ai_reasoning_market_state_contract",
        "ai_reasoning_cycle_locks",
        ["instrument", "market_state_hash", "analysis_contract_version"],
        unique=True,
    )
    op.drop_constraint(
        "ck_ai_reasoning_window_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.create_check_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('claimed','completed','failed','skipped')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.create_check_constraint(
        "ck_ai_reasoning_window_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('claimed','completed','failed')",
    )
    op.drop_index(
        "ux_ai_reasoning_market_state_contract",
        table_name="ai_reasoning_cycle_locks",
    )
    op.drop_index(
        "ux_ai_reasoning_five_minute_cycle",
        table_name="ai_reasoning_cycle_locks",
    )
    op.drop_index(
        "ix_ai_reasoning_cycle_locks_five_minute_window_start",
        table_name="ai_reasoning_cycle_locks",
    )
    op.drop_column("ai_reasoning_cycle_locks", "analysis_contract_version")
    op.drop_column("ai_reasoning_cycle_locks", "market_state_hash")
    op.drop_column("ai_reasoning_cycle_locks", "five_minute_window_start")
    op.drop_column("ai_reasoning_cycle_locks", "analysis_timeframe")
