"""Add analysis-only AI persistence and temporal Signal Engine lineage.

Revision ID: 20260727_0010
Revises: 20260727_0009
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260727_0010"
down_revision = "20260727_0009"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "ai_market_analyses",
        sa.Column("analysis_id", UUID, primary_key=True),
        sa.Column(
            "request_id",
            UUID,
            sa.ForeignKey("ai_reasoning_requests.request_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cycle_id", UUID, nullable=False),
        sa.Column(
            "market_snapshot_id",
            UUID,
            sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "quantitative_forecast_id",
            UUID,
            sa.ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("analysis_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("validation_passed", sa.Boolean(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for name, columns in (
        ("ix_ai_market_analyses_cycle_id", ["cycle_id"]),
        ("ix_ai_market_analyses_market_snapshot_id", ["market_snapshot_id"]),
        ("ix_ai_market_analyses_quantitative_forecast_id", ["quantitative_forecast_id"]),
        ("ix_ai_market_analyses_symbol", ["symbol"]),
        ("ix_ai_market_analyses_timeframe", ["timeframe"]),
        ("ix_ai_market_analyses_analysis_timestamp", ["analysis_timestamp"]),
        ("ix_ai_market_analyses_status", ["status"]),
        ("ix_ai_market_analyses_schema_version", ["schema_version"]),
        ("ix_ai_market_analyses_provider", ["provider"]),
        ("ix_ai_market_analyses_validation_passed", ["validation_passed"]),
        ("ix_ai_market_analyses_created_at", ["created_at"]),
    ):
        op.create_index(name, "ai_market_analyses", columns)
    op.create_index(
        "ix_ai_market_analyses_request_id",
        "ai_market_analyses",
        ["request_id"],
        unique=True,
    )
    op.create_index(
        "ux_ai_market_analysis_cycle",
        "ai_market_analyses",
        ["symbol", "timeframe", "cycle_id", "schema_version"],
        unique=True,
    )
    op.create_index(
        "ix_ai_market_analysis_history",
        "ai_market_analyses",
        ["symbol", "timeframe", "analysis_timestamp"],
    )
    op.add_column(
        "ai_reasoning_cycle_locks",
        sa.Column(
            "analysis_id",
            UUID,
            sa.ForeignKey("ai_market_analyses.analysis_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_ai_reasoning_cycle_locks_analysis_id",
        "ai_reasoning_cycle_locks",
        ["analysis_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ai_reasoning_cycle_locks_analysis_id",
        table_name="ai_reasoning_cycle_locks",
    )
    op.drop_column("ai_reasoning_cycle_locks", "analysis_id")
    op.drop_index("ix_ai_market_analysis_history", table_name="ai_market_analyses")
    op.drop_index("ux_ai_market_analysis_cycle", table_name="ai_market_analyses")
    for name in (
        "ix_ai_market_analyses_created_at",
        "ix_ai_market_analyses_validation_passed",
        "ix_ai_market_analyses_provider",
        "ix_ai_market_analyses_schema_version",
        "ix_ai_market_analyses_status",
        "ix_ai_market_analyses_analysis_timestamp",
        "ix_ai_market_analyses_timeframe",
        "ix_ai_market_analyses_symbol",
        "ix_ai_market_analyses_quantitative_forecast_id",
        "ix_ai_market_analyses_market_snapshot_id",
        "ix_ai_market_analyses_cycle_id",
        "ix_ai_market_analyses_request_id",
    ):
        op.drop_index(name, table_name="ai_market_analyses")
    op.drop_table("ai_market_analyses")
