"""Persist authoritative AI reasoning gate decisions.

Revision ID: 20260731_0019
Revises: 20260731_0018
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "20260731_0019"
down_revision = "20260731_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_reasoning_gate_decisions",
        sa.Column("decision_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("trigger_timeframe", sa.String(16)),
        sa.Column("attempted_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("analysis_lookup_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "market_state_id",
            UUID(as_uuid=True),
            sa.ForeignKey("unified_market_states.state_id", ondelete="SET NULL"),
        ),
        sa.Column("snapshot_id", UUID(as_uuid=True)),
        sa.Column("gate_decision", sa.String(32), nullable=False),
        sa.Column("gate_skip_reason", sa.String(96)),
        sa.Column(
            "existing_analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("ai_market_analyses.analysis_id", ondelete="SET NULL"),
        ),
        sa.Column("analysis_created_at", sa.DateTime(timezone=True)),
        sa.Column("analysis_market_cutoff", sa.DateTime(timezone=True)),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "gate_decision IN ('PROCEED','SKIPPED','REUSED')",
            name="ck_ai_reasoning_gate_decision",
        ),
    )
    for name, columns in (
        ("ix_ai_reasoning_gate_decisions_instrument", ["instrument"]),
        ("ix_ai_reasoning_gate_decisions_trigger_timeframe", ["trigger_timeframe"]),
        ("ix_ai_reasoning_gate_decisions_attempted_cutoff", ["attempted_cutoff"]),
        ("ix_ai_reasoning_gate_decisions_market_state_id", ["market_state_id"]),
        ("ix_ai_reasoning_gate_decisions_snapshot_id", ["snapshot_id"]),
        ("ix_ai_reasoning_gate_decisions_gate_decision", ["gate_decision"]),
        ("ix_ai_reasoning_gate_decisions_gate_skip_reason", ["gate_skip_reason"]),
        ("ix_ai_reasoning_gate_decisions_existing_analysis_id", ["existing_analysis_id"]),
        ("ix_ai_reasoning_gate_decisions_created_at", ["created_at"]),
        (
            "ix_ai_reasoning_gate_decision_boundary",
            ["instrument", "attempted_cutoff"],
        ),
    ):
        op.create_index(name, "ai_reasoning_gate_decisions", columns)


def downgrade() -> None:
    op.drop_table("ai_reasoning_gate_decisions")
