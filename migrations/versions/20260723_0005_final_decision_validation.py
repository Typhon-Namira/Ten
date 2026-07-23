"""Deterministic final decisions, analytical publication, validation, and readiness."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260723_0005"
down_revision: str | None = "20260723_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSON = postgresql.JSONB()


def upgrade() -> None:
    op.create_table(
        "hard_gate_versions",
        sa.Column("gate_id", sa.String(96), primary_key=True),
        sa.Column("gate_version", sa.String(32), primary_key=True),
        sa.Column("registry_version", sa.String(64), nullable=False, index=True),
        sa.Column("category", sa.String(32), nullable=False, index=True),
        sa.Column("payload", JSON, nullable=False),
    )
    op.create_table(
        "final_system_actions",
        sa.Column("final_action_id", UUID, primary_key=True),
        sa.Column("ai_proposal_id", UUID, sa.ForeignKey("ai_signal_proposals.proposal_id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("managed_signal_id", UUID, sa.ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("market_state_id", UUID, sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("quantitative_forecast_id", UUID, sa.ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("ai_forecast_id", UUID, sa.ForeignKey("ai_market_forecasts.forecast_id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("action", sa.String(48), nullable=False, index=True),
        sa.Column("approval_state", sa.String(32), nullable=False, index=True),
        sa.Column("publication_state", sa.String(32), nullable=False, index=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )
    op.create_index("ix_final_system_actions_signal_created", "final_system_actions", ["managed_signal_id", "created_at"])
    op.create_table(
        "guardrail_evaluations",
        sa.Column("evaluation_id", UUID, primary_key=True),
        sa.Column("final_action_id", UUID, sa.ForeignKey("final_system_actions.final_action_id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("gate_id", sa.String(96), nullable=False, index=True),
        sa.Column("gate_version", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, index=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.UniqueConstraint("final_action_id", "gate_id", name="ux_guardrail_evaluation_action_gate"),
    )
    op.create_table(
        "published_analytical_signals",
        sa.Column("publication_id", UUID, primary_key=True),
        sa.Column("signal_id", UUID, sa.ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), nullable=False, unique=True, index=True),
        sa.Column("final_action_id", UUID, sa.ForeignKey("final_system_actions.final_action_id", ondelete="RESTRICT"), nullable=False, unique=True, index=True),
        sa.Column("proposal_id", UUID, sa.ForeignKey("ai_signal_proposals.proposal_id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("instrument", sa.String(32), nullable=False, index=True),
        sa.Column("direction", sa.String(16), nullable=False, index=True),
        sa.Column("setup_family", sa.String(64), nullable=False, index=True),
        sa.Column("lifecycle_state", sa.String(32), nullable=False, index=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )
    op.create_table(
        "llm_usage_metrics",
        sa.Column("metric_id", UUID, primary_key=True),
        sa.Column("usage_date", sa.String(10), nullable=False, index=True),
        sa.Column("request_hash", sa.String(64), nullable=False, index=True),
        sa.Column("market_state_hash", sa.String(64), nullable=False, index=True),
        sa.Column("model_identifier", sa.String(128), nullable=False, index=True),
        sa.Column("success", sa.Boolean(), nullable=False, index=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )
    op.create_index("ix_llm_usage_metrics_date_model", "llm_usage_metrics", ["usage_date", "model_identifier"])
    op.create_table(
        "detailed_signal_outcomes",
        sa.Column("outcome_id", UUID, primary_key=True),
        sa.Column("signal_id", UUID, sa.ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), nullable=False, unique=True, index=True),
        sa.Column("status", sa.String(32), nullable=False, index=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )
    op.create_table(
        "ai_performance_reports",
        sa.Column("report_id", UUID, primary_key=True),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )
    op.create_table(
        "ai_production_readiness_reports",
        sa.Column("report_id", UUID, primary_key=True),
        sa.Column("status", sa.String(32), nullable=False, index=True),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    for table in (
        "ai_production_readiness_reports",
        "ai_performance_reports",
        "detailed_signal_outcomes",
        "llm_usage_metrics",
        "published_analytical_signals",
        "guardrail_evaluations",
        "final_system_actions",
        "hard_gate_versions",
    ):
        op.drop_table(table)
