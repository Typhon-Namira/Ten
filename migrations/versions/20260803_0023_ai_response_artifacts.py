"""Persist authoritative AI provider-response lifecycle stages.

Revision ID: 20260803_0023
Revises: 20260731_0022
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision = "20260803_0023"
down_revision = "20260731_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_response_artifacts",
        sa.Column(
            "request_id",
            UUID(as_uuid=True),
            sa.ForeignKey("ai_reasoning_requests.request_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("cycle_id", UUID(as_uuid=True), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("market_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("provider_mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(48), nullable=False),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("ai_market_analyses.analysis_id", ondelete="SET NULL"),
        ),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('PROVIDER_RESPONSE_RECEIVED', 'NORMALIZED', 'VALIDATED', "
            "'COMMITTED', 'TERMINAL_SCHEMA_FAILURE')",
            name="ck_ai_response_artifact_status",
        ),
    )
    for name, columns in (
        ("ix_ai_response_artifacts_cycle_id", ["cycle_id"]),
        ("ix_ai_response_artifacts_instrument", ["instrument"]),
        ("ix_ai_response_artifacts_market_cutoff", ["market_cutoff"]),
        ("ix_ai_response_artifacts_provider", ["provider"]),
        ("ix_ai_response_artifacts_provider_mode", ["provider_mode"]),
        ("ix_ai_response_artifacts_status", ["status"]),
        ("ix_ai_response_artifacts_analysis_id", ["analysis_id"]),
        ("ix_ai_response_artifacts_updated_at", ["updated_at"]),
    ):
        op.create_index(name, "ai_response_artifacts", columns)


def downgrade() -> None:
    op.drop_table("ai_response_artifacts")
