"""Add immutable forward market scenarios and post-expiry outcomes.

Revision ID: 20260730_0016
Revises: 20260730_0015
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260730_0016"
down_revision = "20260730_0015"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "forward_market_scenarios",
        sa.Column("scenario_id", UUID, primary_key=True),
        sa.Column("cycle_id", UUID, nullable=False),
        sa.Column(
            "market_state_id",
            UUID,
            sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "synthesis_id",
            UUID,
            sa.ForeignKey("multi_timeframe_signal_sets.synthesis_id", ondelete="RESTRICT"),
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
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("primary_direction", sa.String(16), nullable=False),
        sa.Column("scenario_validity", sa.String(16), nullable=False),
        sa.Column("execution_geometry_validity", sa.String(24), nullable=False),
        sa.Column("market_cutoff_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expiry", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("timeframe IN ('M5','M15')", name="ck_forward_scenario_timeframe"),
        sa.CheckConstraint(
            "primary_direction IN ('BULLISH','BEARISH','RANGE','INCONCLUSIVE')",
            name="ck_forward_scenario_direction",
        ),
    )
    for name, columns, unique in (
        ("ix_forward_market_scenarios_cycle_id", ["cycle_id"], False),
        ("ix_forward_market_scenarios_market_state_id", ["market_state_id"], False),
        ("ix_forward_market_scenarios_synthesis_id", ["synthesis_id"], False),
        ("ix_forward_market_scenarios_analysis_id", ["analysis_id"], False),
        ("ix_forward_market_scenarios_quantitative_forecast_id", ["quantitative_forecast_id"], False),
        ("ix_forward_market_scenarios_instrument", ["instrument"], False),
        ("ix_forward_market_scenarios_timeframe", ["timeframe"], False),
        ("ix_forward_market_scenarios_primary_direction", ["primary_direction"], False),
        ("ix_forward_market_scenarios_scenario_validity", ["scenario_validity"], False),
        ("ix_forward_market_scenarios_geometry_validity", ["execution_geometry_validity"], False),
        ("ix_forward_market_scenarios_market_cutoff_time", ["market_cutoff_time"], False),
        ("ix_forward_market_scenarios_expiry", ["expiry"], False),
        ("ix_forward_market_scenarios_created_at", ["created_at"], False),
        ("ux_forward_market_scenario_boundary", ["instrument", "timeframe", "market_cutoff_time"], True),
        ("ix_forward_market_scenario_latest", ["instrument", "timeframe", "market_cutoff_time"], False),
    ):
        op.create_index(name, "forward_market_scenarios", columns, unique=unique)

    op.create_table(
        "combined_forward_scenarios",
        sa.Column("combined_scenario_id", UUID, primary_key=True),
        sa.Column("cycle_id", UUID, nullable=False),
        sa.Column(
            "market_state_id",
            UUID,
            sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "m5_scenario_id",
            UUID,
            sa.ForeignKey("forward_market_scenarios.scenario_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "m15_scenario_id",
            UUID,
            sa.ForeignKey("forward_market_scenarios.scenario_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("agreement", sa.String(24), nullable=False),
        sa.Column("combined_direction", sa.String(16), nullable=False),
        sa.Column("execution_geometry_validity", sa.String(24), nullable=False),
        sa.Column("market_cutoff_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expiry", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for name, columns, unique in (
        ("ix_combined_forward_scenarios_cycle_id", ["cycle_id"], False),
        ("ix_combined_forward_scenarios_market_state_id", ["market_state_id"], True),
        ("ix_combined_forward_scenarios_m5_scenario_id", ["m5_scenario_id"], False),
        ("ix_combined_forward_scenarios_m15_scenario_id", ["m15_scenario_id"], False),
        ("ix_combined_forward_scenarios_instrument", ["instrument"], False),
        ("ix_combined_forward_scenarios_agreement", ["agreement"], False),
        ("ix_combined_forward_scenarios_combined_direction", ["combined_direction"], False),
        ("ix_combined_forward_scenarios_geometry_validity", ["execution_geometry_validity"], False),
        ("ix_combined_forward_scenarios_market_cutoff_time", ["market_cutoff_time"], False),
        ("ix_combined_forward_scenarios_expiry", ["expiry"], False),
        ("ix_combined_forward_scenarios_created_at", ["created_at"], False),
    ):
        op.create_index(name, "combined_forward_scenarios", columns, unique=unique)

    op.create_table(
        "scenario_outcomes",
        sa.Column("outcome_id", UUID, primary_key=True),
        sa.Column(
            "scenario_id",
            UUID,
            sa.ForeignKey("forward_market_scenarios.scenario_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("calibration_bucket", sa.String(16), nullable=False),
        sa.Column("directional_accuracy", sa.Float(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "directional_accuracy BETWEEN 0 AND 1",
            name="ck_scenario_outcome_directional_accuracy",
        ),
    )
    for name, columns, unique in (
        ("ux_scenario_outcomes_scenario_id", ["scenario_id"], True),
        ("ix_scenario_outcomes_status", ["status"], False),
        ("ix_scenario_outcomes_calibration_bucket", ["calibration_bucket"], False),
        ("ix_scenario_outcomes_evaluated_at", ["evaluated_at"], False),
        ("ix_scenario_outcomes_completed_at", ["completed_at"], False),
    ):
        op.create_index(name, "scenario_outcomes", columns, unique=unique)


def downgrade() -> None:
    op.drop_table("scenario_outcomes")
    op.drop_table("combined_forward_scenarios")
    op.drop_table("forward_market_scenarios")
