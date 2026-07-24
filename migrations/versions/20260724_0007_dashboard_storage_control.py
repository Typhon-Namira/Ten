"""Add compact current-state pointers and bounded storage-control metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260724_0007"
down_revision: str | None = "20260723_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "unified_market_state_current",
        sa.Column("instrument", sa.String(32), primary_key=True),
        sa.Column(
            "state_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("state_hash", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_unified_market_state_current_state",
        "unified_market_state_current",
        ["state_id"],
    )
    op.create_table(
        "pipeline_stage_current",
        sa.Column("instrument", sa.String(32), primary_key=True),
        sa.Column("stage", sa.String(64), primary_key=True),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("reason", sa.String(160), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("record_id", sa.String(64), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.CheckConstraint(
            "status IN ('healthy','running','degraded','failed','disabled','blocked','stale','no_data')",
            name="ck_pipeline_stage_current_status",
        ),
    )
    op.create_table(
        "pipeline_stage_history",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("reason", sa.String(160), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("record_id", sa.String(64), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.UniqueConstraint(
            "instrument", "stage", "fingerprint",
            name="ux_pipeline_stage_history_meaningful_change",
        ),
    )
    op.create_index(
        "ix_pipeline_stage_history_lookup",
        "pipeline_stage_history",
        ["instrument", "observed_at"],
    )
    op.create_table(
        "storage_retention_policies",
        sa.Column("relation_name", sa.String(96), primary_key=True),
        sa.Column("retention_days", sa.Integer(), nullable=False),
        sa.Column("cleanup_batch_size", sa.Integer(), nullable=False),
        sa.Column("protected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("description", sa.String(256), nullable=False),
        sa.CheckConstraint("retention_days >= 1", name="ck_storage_retention_days_positive"),
        sa.CheckConstraint("cleanup_batch_size BETWEEN 1 AND 5000", name="ck_storage_cleanup_batch_bounded"),
    )
    policy = sa.table(
        "storage_retention_policies",
        sa.column("relation_name", sa.String),
        sa.column("retention_days", sa.Integer),
        sa.column("cleanup_batch_size", sa.Integer),
        sa.column("protected", sa.Boolean),
        sa.column("description", sa.String),
    )
    op.bulk_insert(
        policy,
        [
            {
                "relation_name": "realtime_candles",
                "retention_days": 7,
                "cleanup_batch_size": 1000,
                "protected": False,
                "description": "Transient observations; canonical historical candle remains.",
            },
            {
                "relation_name": "pipeline_stage_history",
                "retention_days": 30,
                "cleanup_batch_size": 1000,
                "protected": False,
                "description": "Operational transitions only; current state is retained separately.",
            },
            {
                "relation_name": "unified_market_states",
                "retention_days": 30,
                "cleanup_batch_size": 250,
                "protected": True,
                "description": "Immutable analytical audit history; cleanup requires archive confirmation.",
            },
            {
                "relation_name": "evidence_items",
                "retention_days": 30,
                "cleanup_batch_size": 1000,
                "protected": True,
                "description": "Immutable evidence links; cascades only with archived market states.",
            },
        ],
    )


def downgrade() -> None:
    op.drop_table("storage_retention_policies")
    op.drop_index("ix_pipeline_stage_history_lookup", table_name="pipeline_stage_history")
    op.drop_table("pipeline_stage_history")
    op.drop_table("pipeline_stage_current")
    op.drop_index("ix_unified_market_state_current_state", table_name="unified_market_state_current")
    op.drop_table("unified_market_state_current")
