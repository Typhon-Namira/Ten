"""Persist deterministic signals derived from validated AI analyses.

Revision ID: 20260728_0012
Revises: 20260727_0011
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260728_0012"
down_revision = "20260727_0011"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "ai_analysis_signals",
        sa.Column("signal_id", UUID, primary_key=True),
        sa.Column(
            "analysis_id",
            UUID,
            sa.ForeignKey("ai_market_analyses.analysis_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cycle_id", UUID, nullable=False),
        sa.Column(
            "snapshot_id",
            UUID,
            sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("signal", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("strength", sa.String(24), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "signal IN ('BUY','SELL','HOLD')",
            name="ck_ai_analysis_signal_action",
        ),
        sa.CheckConstraint(
            "confidence BETWEEN 0 AND 100",
            name="ck_ai_analysis_signal_confidence",
        ),
    )
    op.create_index(
        "ix_ai_analysis_signals_analysis_id",
        "ai_analysis_signals",
        ["analysis_id"],
        unique=True,
    )
    for name, columns in (
        ("ix_ai_analysis_signals_cycle_id", ["cycle_id"]),
        ("ix_ai_analysis_signals_snapshot_id", ["snapshot_id"]),
        ("ix_ai_analysis_signals_instrument", ["instrument"]),
        ("ix_ai_analysis_signals_timeframe", ["timeframe"]),
        ("ix_ai_analysis_signals_signal", ["signal"]),
        ("ix_ai_analysis_signals_strength", ["strength"]),
        ("ix_ai_analysis_signals_generated_at", ["generated_at"]),
    ):
        op.create_index(name, "ai_analysis_signals", columns)
    op.create_index(
        "ux_ai_analysis_signal_cycle",
        "ai_analysis_signals",
        ["instrument", "timeframe", "cycle_id", "schema_version"],
        unique=True,
    )
    op.create_index(
        "ix_ai_analysis_signal_latest",
        "ai_analysis_signals",
        ["instrument", "timeframe", "generated_at"],
    )
    op.drop_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.create_check_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('claimed','completed','failed','skipped','COMPLETED',"
        "'FAILED_PROVIDER','FAILED_SCHEMA','FAILED_PERSISTENCE','TIMED_OUT',"
        "'SKIPPED_WITH_REASON')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        type_="check",
    )
    op.create_check_constraint(
        "ck_ai_reasoning_cycle_lock_status",
        "ai_reasoning_cycle_locks",
        "status IN ('claimed','completed','failed','skipped')",
    )
    op.drop_index("ix_ai_analysis_signal_latest", table_name="ai_analysis_signals")
    op.drop_index("ux_ai_analysis_signal_cycle", table_name="ai_analysis_signals")
    for name in (
        "ix_ai_analysis_signals_generated_at",
        "ix_ai_analysis_signals_strength",
        "ix_ai_analysis_signals_signal",
        "ix_ai_analysis_signals_timeframe",
        "ix_ai_analysis_signals_instrument",
        "ix_ai_analysis_signals_snapshot_id",
        "ix_ai_analysis_signals_cycle_id",
        "ix_ai_analysis_signals_analysis_id",
    ):
        op.drop_index(name, table_name="ai_analysis_signals")
    op.drop_table("ai_analysis_signals")
