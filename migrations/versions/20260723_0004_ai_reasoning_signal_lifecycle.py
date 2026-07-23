"""add Phase 3/4 AI reasoning and signal lifecycle

Revision ID: 20260723_0004
Revises: 20260723_0003
Create Date: 2026-07-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260723_0004"
down_revision: str | None = "20260723_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
JSON = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "ai_setup_family_versions",
        sa.Column("setup_family_id", sa.String(64), primary_key=True),
        sa.Column("version", sa.String(32), primary_key=True),
        sa.Column("registry_version", sa.String(64), nullable=False),
        sa.Column("payload", JSON, nullable=False),
    )
    op.create_table(
        "ai_reasoning_requests",
        sa.Column("request_id", sa.UUID(), primary_key=True),
        sa.Column("cycle_id", sa.UUID(), nullable=False),
        sa.Column("market_state_id", sa.UUID(), sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), nullable=False),
        sa.Column("quantitative_forecast_id", sa.UUID(), sa.ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("analysis_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("model_identifier", sa.String(128), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "ai_market_forecasts",
        sa.Column("forecast_id", sa.UUID(), primary_key=True),
        sa.Column("request_id", sa.UUID(), sa.ForeignKey("ai_reasoning_requests.request_id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("market_state_id", sa.UUID(), sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), nullable=False),
        sa.Column("quantitative_forecast_id", sa.UUID(), sa.ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("dominant_direction", sa.String(16), nullable=True),
        sa.Column("selected_setup_family", sa.String(64), nullable=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "ai_forecast_scenarios",
        sa.Column("forecast_id", sa.UUID(), sa.ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
        sa.Column("scenario_name", sa.String(128), nullable=False),
        sa.Column("payload", JSON, nullable=False),
    )
    op.create_table(
        "ai_forecast_evidence_links",
        sa.Column("forecast_id", sa.UUID(), sa.ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("evidence_id", sa.UUID(), sa.ForeignKey("evidence_items.evidence_id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("role", sa.String(32), primary_key=True),
    )
    op.create_table(
        "ai_signal_proposals",
        sa.Column("proposal_id", sa.UUID(), primary_key=True),
        sa.Column("forecast_id", sa.UUID(), sa.ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), nullable=False),
        sa.Column("market_state_id", sa.UUID(), sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), nullable=False),
        sa.Column("structural_opportunity_key", sa.String(64), nullable=False),
        sa.Column("recommended_action", sa.String(48), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ai_signal_proposals_opportunity_created", "ai_signal_proposals", ["structural_opportunity_key", "created_at"])
    op.create_table(
        "managed_signals",
        sa.Column("signal_id", sa.UUID(), primary_key=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("structural_opportunity_key", sa.String(64), nullable=False, unique=True),
        sa.Column("setup_family", sa.String(64), nullable=False),
        sa.Column("direction", sa.String(16), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("current_proposal_id", sa.UUID(), sa.ForeignKey("ai_signal_proposals.proposal_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "signal_state_transitions",
        sa.Column("transition_id", sa.UUID(), primary_key=True),
        sa.Column("signal_id", sa.UUID(), sa.ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), nullable=False),
        sa.Column("previous_state", sa.String(32), nullable=False),
        sa.Column("new_state", sa.String(32), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "signal_level_revisions",
        sa.Column("revision_id", sa.UUID(), primary_key=True),
        sa.Column("signal_id", sa.UUID(), sa.ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), nullable=False),
        sa.Column("level_type", sa.String(32), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "signal_monitoring_evaluations",
        sa.Column("evaluation_id", sa.UUID(), primary_key=True),
        sa.Column("signal_id", sa.UUID(), sa.ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), nullable=False),
        sa.Column("forecast_id", sa.UUID(), sa.ForeignKey("ai_market_forecasts.forecast_id", ondelete="CASCADE"), nullable=False),
        sa.Column("thesis_valid", sa.Boolean(), nullable=False),
        sa.Column("recommended_action", sa.String(48), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "managed_signal_outcomes",
        sa.Column("outcome_id", sa.UUID(), primary_key=True),
        sa.Column("signal_id", sa.UUID(), sa.ForeignKey("managed_signals.signal_id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("final_state", sa.String(32), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "market_memory_entries",
        sa.Column("entry_id", sa.UUID(), primary_key=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("cycle_id", sa.UUID(), nullable=False),
        sa.Column("market_state_id", sa.UUID(), sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), nullable=False),
        sa.Column("category", sa.String(48), nullable=False),
        sa.Column("opportunity_key", sa.String(64), nullable=True),
        sa.Column("signal_id", sa.UUID(), sa.ForeignKey("managed_signals.signal_id", ondelete="SET NULL"), nullable=True),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_market_memory_entries_series_time", "market_memory_entries", ["instrument", "occurred_at"])
    op.create_table(
        "llm_structured_output_failures",
        sa.Column("failure_id", sa.UUID(), primary_key=True),
        sa.Column("request_id", sa.UUID(), sa.ForeignKey("ai_reasoning_requests.request_id", ondelete="CASCADE"), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("model_identifier", sa.String(128), nullable=False),
        sa.Column("failure_state", sa.String(64), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "llm_structured_output_failures",
        "market_memory_entries",
        "managed_signal_outcomes",
        "signal_monitoring_evaluations",
        "signal_level_revisions",
        "signal_state_transitions",
        "managed_signals",
        "ai_signal_proposals",
        "ai_forecast_evidence_links",
        "ai_forecast_scenarios",
        "ai_market_forecasts",
        "ai_reasoning_requests",
        "ai_setup_family_versions",
    ):
        op.drop_table(table)
