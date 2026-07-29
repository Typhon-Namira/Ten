"""Persist independent M5/M15 and combined analytical signals.

Revision ID: 20260729_0014
Revises: 20260728_0013
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260729_0014"
down_revision = "20260728_0013"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "multi_timeframe_signal_sets",
        sa.Column("synthesis_id", UUID, primary_key=True),
        sa.Column("cycle_id", UUID, nullable=False),
        sa.Column(
            "market_state_id",
            UUID,
            sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "analysis_id",
            UUID,
            sa.ForeignKey("ai_market_analyses.analysis_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "quantitative_forecast_id",
            UUID,
            sa.ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("combined_direction", sa.String(16), nullable=False),
        sa.Column("combined_confidence", sa.Float(), nullable=False),
        sa.Column("execution_status", sa.String(16), nullable=False),
        sa.Column("schema_version", sa.String(16), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("market_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "combined_direction IN ('BUY','SELL')",
            name="ck_multi_timeframe_signal_direction",
        ),
        sa.CheckConstraint(
            "combined_confidence BETWEEN 0 AND 100",
            name="ck_multi_timeframe_signal_confidence",
        ),
        sa.CheckConstraint(
            "execution_status IN ('READY','BLOCKED')",
            name="ck_multi_timeframe_signal_execution_status",
        ),
    )
    for name, columns, unique in (
        ("ix_multi_timeframe_signal_sets_cycle_id", ["cycle_id"], False),
        ("ux_multi_timeframe_signal_sets_market_state_id", ["market_state_id"], True),
        ("ux_multi_timeframe_signal_sets_analysis_id", ["analysis_id"], True),
        ("ix_multi_timeframe_signal_sets_quantitative_forecast_id", ["quantitative_forecast_id"], False),
        ("ix_multi_timeframe_signal_sets_instrument", ["instrument"], False),
        ("ix_multi_timeframe_signal_sets_combined_direction", ["combined_direction"], False),
        ("ix_multi_timeframe_signal_sets_execution_status", ["execution_status"], False),
        ("ix_multi_timeframe_signal_sets_market_timestamp", ["market_timestamp"], False),
        ("ix_multi_timeframe_signal_sets_created_at", ["created_at"], False),
    ):
        op.create_index(name, "multi_timeframe_signal_sets", columns, unique=unique)

    op.create_table(
        "timeframe_analytical_signals",
        sa.Column("signal_id", UUID, primary_key=True),
        sa.Column(
            "synthesis_id",
            UUID,
            sa.ForeignKey("multi_timeframe_signal_sets.synthesis_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("analytical_direction", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("strength", sa.String(24), nullable=False),
        sa.Column("execution_status", sa.String(16), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "timeframe IN ('M5','M15','COMBINED')",
            name="ck_timeframe_analytical_signal_timeframe",
        ),
        sa.CheckConstraint(
            "analytical_direction IN ('BUY','SELL')",
            name="ck_timeframe_analytical_signal_direction",
        ),
        sa.CheckConstraint(
            "confidence BETWEEN 0 AND 100",
            name="ck_timeframe_analytical_signal_confidence",
        ),
        sa.CheckConstraint(
            "execution_status IN ('READY','BLOCKED')",
            name="ck_timeframe_analytical_signal_execution_status",
        ),
    )
    op.create_index(
        "ux_timeframe_analytical_signal_scope",
        "timeframe_analytical_signals",
        ["synthesis_id", "timeframe"],
        unique=True,
    )
    for name, columns in (
        ("ix_timeframe_analytical_signals_synthesis_id", ["synthesis_id"]),
        ("ix_timeframe_analytical_signals_instrument", ["instrument"]),
        ("ix_timeframe_analytical_signals_timeframe", ["timeframe"]),
        ("ix_timeframe_analytical_signals_analytical_direction", ["analytical_direction"]),
        ("ix_timeframe_analytical_signals_strength", ["strength"]),
        ("ix_timeframe_analytical_signals_execution_status", ["execution_status"]),
        ("ix_timeframe_analytical_signals_completed_at", ["completed_at"]),
    ):
        op.create_index(name, "timeframe_analytical_signals", columns)
    op.create_index(
        "ix_timeframe_analytical_signal_latest",
        "timeframe_analytical_signals",
        ["instrument", "timeframe", "completed_at"],
    )


def downgrade() -> None:
    for name in (
        "ix_timeframe_analytical_signal_latest",
        "ix_timeframe_analytical_signals_completed_at",
        "ix_timeframe_analytical_signals_execution_status",
        "ix_timeframe_analytical_signals_strength",
        "ix_timeframe_analytical_signals_analytical_direction",
        "ix_timeframe_analytical_signals_timeframe",
        "ix_timeframe_analytical_signals_instrument",
        "ix_timeframe_analytical_signals_synthesis_id",
        "ux_timeframe_analytical_signal_scope",
    ):
        op.drop_index(name, table_name="timeframe_analytical_signals")
    op.drop_table("timeframe_analytical_signals")
    for name in (
        "ix_multi_timeframe_signal_sets_created_at",
        "ix_multi_timeframe_signal_sets_market_timestamp",
        "ix_multi_timeframe_signal_sets_execution_status",
        "ix_multi_timeframe_signal_sets_combined_direction",
        "ix_multi_timeframe_signal_sets_instrument",
        "ix_multi_timeframe_signal_sets_quantitative_forecast_id",
        "ux_multi_timeframe_signal_sets_analysis_id",
        "ux_multi_timeframe_signal_sets_market_state_id",
        "ix_multi_timeframe_signal_sets_cycle_id",
    ):
        op.drop_index(name, table_name="multi_timeframe_signal_sets")
    op.drop_table("multi_timeframe_signal_sets")
