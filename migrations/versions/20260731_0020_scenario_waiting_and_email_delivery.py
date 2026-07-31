"""Add recoverable scenario waiting and explicit email delivery states.

Revision ID: 20260731_0020
Revises: 20260731_0019
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "20260731_0020"
down_revision = "20260731_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_ai_reasoning_gate_decision",
        "ai_reasoning_gate_decisions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_ai_reasoning_gate_decision",
        "ai_reasoning_gate_decisions",
        "gate_decision IN ('PROCEED','SKIPPED','REUSED','COMMITTED')",
    )
    op.drop_constraint(
        "ck_authoritative_simulation_attempt_status",
        "authoritative_simulation_attempts",
        type_="check",
    )
    op.create_check_constraint(
        "ck_authoritative_simulation_attempt_status",
        "authoritative_simulation_attempts",
        "status IN ('SCHEDULED','WAITING_FOR_AI_ANALYSIS','RUNNING','SUCCESS',"
        "'NO_SIGNAL','ANALYTICAL_ONLY','BLOCKED','FAILED','SKIPPED')",
    )
    for column in (
        sa.Column("correlation_id", UUID(as_uuid=True)),
        sa.Column(
            "market_state_id",
            UUID(as_uuid=True),
            sa.ForeignKey("unified_market_states.state_id", ondelete="SET NULL"),
        ),
        sa.Column("snapshot_id", UUID(as_uuid=True)),
        sa.Column(
            "quantitative_forecast_id",
            UUID(as_uuid=True),
            sa.ForeignKey("quantitative_forecasts.result_id", ondelete="SET NULL"),
        ),
        sa.Column(
            "ai_analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("ai_market_analyses.analysis_id", ondelete="SET NULL"),
        ),
        sa.Column("ai_analysis_cutoff", sa.DateTime(timezone=True)),
        sa.Column("ai_analysis_committed_at", sa.DateTime(timezone=True)),
        sa.Column("dependency_lookup_result", sa.String(64)),
    ):
        op.add_column("authoritative_simulation_attempts", column)
    for name, columns in (
        ("ix_authoritative_simulation_attempts_correlation_id", ["correlation_id"]),
        ("ix_authoritative_simulation_attempts_market_state_id", ["market_state_id"]),
        ("ix_authoritative_simulation_attempts_snapshot_id", ["snapshot_id"]),
        (
            "ix_authoritative_simulation_attempts_quantitative_forecast_id",
            ["quantitative_forecast_id"],
        ),
        ("ix_authoritative_simulation_attempts_ai_analysis_id", ["ai_analysis_id"]),
    ):
        op.create_index(name, "authoritative_simulation_attempts", columns)

    op.drop_constraint(
        "ck_signal_email_outbox_status",
        "signal_email_outbox",
        type_="check",
    )
    op.alter_column(
        "signal_email_outbox",
        "status",
        existing_type=sa.String(16),
        type_=sa.String(24),
        existing_nullable=False,
    )
    op.create_check_constraint(
        "ck_signal_email_outbox_status",
        "signal_email_outbox",
        "status IN ('PENDING','PROCESSING','SENT','FAILED','PERMANENTLY_FAILED')",
    )


def downgrade() -> None:
    op.execute(
        "UPDATE ai_reasoning_gate_decisions SET gate_decision = 'REUSED' "
        "WHERE gate_decision = 'COMMITTED'"
    )
    op.drop_constraint(
        "ck_ai_reasoning_gate_decision",
        "ai_reasoning_gate_decisions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_ai_reasoning_gate_decision",
        "ai_reasoning_gate_decisions",
        "gate_decision IN ('PROCEED','SKIPPED','REUSED')",
    )
    op.execute(
        "UPDATE signal_email_outbox SET status = 'FAILED' "
        "WHERE status = 'PERMANENTLY_FAILED'"
    )
    op.drop_constraint(
        "ck_signal_email_outbox_status",
        "signal_email_outbox",
        type_="check",
    )
    op.create_check_constraint(
        "ck_signal_email_outbox_status",
        "signal_email_outbox",
        "status IN ('PENDING','PROCESSING','SENT','FAILED')",
    )
    op.alter_column(
        "signal_email_outbox",
        "status",
        existing_type=sa.String(24),
        type_=sa.String(16),
        existing_nullable=False,
    )

    for name in (
        "ix_authoritative_simulation_attempts_ai_analysis_id",
        "ix_authoritative_simulation_attempts_quantitative_forecast_id",
        "ix_authoritative_simulation_attempts_snapshot_id",
        "ix_authoritative_simulation_attempts_market_state_id",
        "ix_authoritative_simulation_attempts_correlation_id",
    ):
        op.drop_index(name, table_name="authoritative_simulation_attempts")
    for column in (
        "dependency_lookup_result",
        "ai_analysis_committed_at",
        "ai_analysis_cutoff",
        "ai_analysis_id",
        "quantitative_forecast_id",
        "snapshot_id",
        "market_state_id",
        "correlation_id",
    ):
        op.drop_column("authoritative_simulation_attempts", column)
    op.drop_constraint(
        "ck_authoritative_simulation_attempt_status",
        "authoritative_simulation_attempts",
        type_="check",
    )
    op.create_check_constraint(
        "ck_authoritative_simulation_attempt_status",
        "authoritative_simulation_attempts",
        "status IN ('SCHEDULED','RUNNING','SUCCESS','NO_SIGNAL',"
        "'ANALYTICAL_ONLY','BLOCKED','FAILED','SKIPPED')",
    )
