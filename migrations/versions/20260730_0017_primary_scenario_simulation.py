"""Add normalized authoritative market simulations and Primary Scenarios.

Revision ID: 20260730_0017
Revises: 20260730_0016
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260730_0017"
down_revision = "20260730_0016"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "market_simulation_cycles",
        sa.Column("simulation_cycle_id", UUID, primary_key=True),
        sa.Column("cycle_id", UUID, nullable=False),
        sa.Column("market_state_id", UUID, sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("synthesis_id", UUID, sa.ForeignKey("multi_timeframe_signal_sets.synthesis_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("analysis_id", UUID, sa.ForeignKey("ai_market_analyses.analysis_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("quantitative_forecast_id", UUID, sa.ForeignKey("quantitative_forecasts.result_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("market_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("engine_version", sa.String(32), nullable=False),
        sa.Column("configuration_version", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("candidate_count BETWEEN 5 AND 10", name="ck_market_simulation_candidate_count"),
    )
    for name, columns, unique in (
        ("ix_market_simulation_cycles_cycle_id", ["cycle_id"], False),
        ("ix_market_simulation_cycles_market_state_id", ["market_state_id"], True),
        ("ix_market_simulation_cycles_synthesis_id", ["synthesis_id"], False),
        ("ix_market_simulation_cycles_analysis_id", ["analysis_id"], False),
        ("ix_market_simulation_cycles_quantitative_forecast_id", ["quantitative_forecast_id"], False),
        ("ix_market_simulation_cycles_instrument", ["instrument"], False),
        ("ix_market_simulation_cycles_market_cutoff", ["market_cutoff"], False),
        ("ix_market_simulation_cycles_created_at", ["created_at"], False),
        ("ux_market_simulation_boundary", ["instrument", "market_cutoff"], True),
    ):
        op.create_index(name, "market_simulation_cycles", columns, unique=unique)

    op.create_table(
        "candidate_market_scenarios",
        sa.Column("candidate_id", UUID, primary_key=True),
        sa.Column("simulation_cycle_id", UUID, sa.ForeignKey("market_simulation_cycles.simulation_cycle_id", ondelete="CASCADE"), nullable=False),
        sa.Column("direction", sa.String(16), nullable=False),
        sa.Column("scenario_type", sa.String(96), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("final_scenario_score", sa.Float(), nullable=False),
        sa.Column("scenario_validity", sa.String(16), nullable=False),
        sa.Column("geometry_validity", sa.String(24), nullable=False),
        sa.Column("diversity_key", sa.String(64), nullable=False),
        sa.Column("expiry", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.CheckConstraint("rank BETWEEN 1 AND 10", name="ck_candidate_scenario_rank"),
        sa.CheckConstraint("final_scenario_score BETWEEN 0 AND 100", name="ck_candidate_scenario_score"),
    )
    for name, columns, unique in (
        ("ix_candidate_market_scenarios_simulation_cycle_id", ["simulation_cycle_id"], False),
        ("ix_candidate_market_scenarios_direction", ["direction"], False),
        ("ix_candidate_market_scenarios_scenario_type", ["scenario_type"], False),
        ("ix_candidate_market_scenarios_scenario_validity", ["scenario_validity"], False),
        ("ix_candidate_market_scenarios_geometry_validity", ["geometry_validity"], False),
        ("ix_candidate_market_scenarios_expiry", ["expiry"], False),
        ("ux_candidate_market_scenario_rank", ["simulation_cycle_id", "rank"], True),
        ("ux_candidate_market_scenario_diversity", ["simulation_cycle_id", "diversity_key"], True),
    ):
        op.create_index(name, "candidate_market_scenarios", columns, unique=unique)

    op.create_table(
        "scenario_path_stages",
        sa.Column("stage_id", UUID, primary_key=True),
        sa.Column("candidate_id", UUID, sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.UniqueConstraint("candidate_id", "sequence", name="ux_scenario_path_stage_sequence"),
    )
    op.create_index("ix_scenario_path_stages_candidate_id", "scenario_path_stages", ["candidate_id"])

    op.create_table(
        "scenario_score_components",
        sa.Column("candidate_id", UUID, sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("contribution", sa.Float(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
    )

    op.create_table(
        "primary_scenario_selections",
        sa.Column("selection_id", UUID, primary_key=True),
        sa.Column("simulation_cycle_id", UUID, sa.ForeignKey("market_simulation_cycles.simulation_cycle_id", ondelete="CASCADE"), nullable=False),
        sa.Column("market_state_id", UUID, sa.ForeignKey("unified_market_states.state_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("primary_candidate_id", UUID, sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="RESTRICT")),
        sa.Column("alternative_candidate_id", UUID, sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="RESTRICT")),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("authoritative_action", sa.String(16), nullable=False),
        sa.Column("signal_eligible", sa.Boolean(), nullable=False),
        sa.Column("market_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("selected_at", sa.DateTime(timezone=True), nullable=False),
    )
    for name, columns, unique in (
        ("ix_primary_scenario_selections_simulation_cycle_id", ["simulation_cycle_id"], True),
        ("ix_primary_scenario_selections_market_state_id", ["market_state_id"], True),
        ("ix_primary_scenario_selections_primary_candidate_id", ["primary_candidate_id"], False),
        ("ix_primary_scenario_selections_alternative_candidate_id", ["alternative_candidate_id"], False),
        ("ix_primary_scenario_selections_instrument", ["instrument"], False),
        ("ix_primary_scenario_selections_status", ["status"], False),
        ("ix_primary_scenario_selections_authoritative_action", ["authoritative_action"], False),
        ("ix_primary_scenario_selections_signal_eligible", ["signal_eligible"], False),
        ("ix_primary_scenario_selections_market_cutoff", ["market_cutoff"], False),
        ("ix_primary_scenario_selections_selected_at", ["selected_at"], False),
    ):
        op.create_index(name, "primary_scenario_selections", columns, unique=unique)

    op.create_table(
        "primary_scenario_geometries",
        sa.Column("selection_id", UUID, sa.ForeignKey("primary_scenario_selections.selection_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("candidate_id", UUID, sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="RESTRICT"), nullable=False, unique=True),
        sa.Column("entry_type", sa.String(32), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
    )
    op.create_table(
        "scenario_lifecycle_transitions",
        sa.Column("transition_id", UUID, primary_key=True),
        sa.Column("selection_id", UUID, sa.ForeignKey("primary_scenario_selections.selection_id", ondelete="CASCADE"), nullable=False),
        sa.Column("previous_status", sa.String(32)),
        sa.Column("new_status", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(128), nullable=False),
        sa.Column("transitioned_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_scenario_lifecycle_selection", "scenario_lifecycle_transitions", ["selection_id"])
    op.create_index("ix_scenario_lifecycle_status", "scenario_lifecycle_transitions", ["new_status"])
    op.create_index("ix_scenario_lifecycle_time", "scenario_lifecycle_transitions", ["transitioned_at"])
    op.create_table(
        "scenario_calibration_metrics",
        sa.Column("metric_id", UUID, primary_key=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("scenario_type", sa.String(96), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("calibrated_probability", sa.Float()),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_scenario_calibration_instrument", "scenario_calibration_metrics", ["instrument"])
    op.create_index("ix_scenario_calibration_type", "scenario_calibration_metrics", ["scenario_type"])
    op.create_index("ix_scenario_calibration_time", "scenario_calibration_metrics", ["calculated_at"])
    op.create_table(
        "candidate_scenario_outcomes",
        sa.Column("outcome_id", UUID, primary_key=True),
        sa.Column("candidate_id", UUID, sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="CASCADE"), nullable=False),
        sa.Column("selection_id", UUID, sa.ForeignKey("primary_scenario_selections.selection_id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("directional_accuracy", sa.Float(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("directional_accuracy BETWEEN 0 AND 1", name="ck_candidate_outcome_directional_accuracy"),
    )
    op.create_index("ux_candidate_scenario_outcome_candidate", "candidate_scenario_outcomes", ["candidate_id"], unique=True)
    op.create_index("ix_candidate_scenario_outcome_selection", "candidate_scenario_outcomes", ["selection_id"])
    op.create_index("ix_candidate_scenario_outcome_status", "candidate_scenario_outcomes", ["status"])
    op.create_index("ix_candidate_scenario_outcome_completed", "candidate_scenario_outcomes", ["completed_at"])
    op.add_column("signal_email_outbox", sa.Column("primary_scenario_id", UUID))
    op.create_foreign_key(
        "fk_signal_email_outbox_primary_scenario",
        "signal_email_outbox",
        "candidate_market_scenarios",
        ["primary_scenario_id"],
        ["candidate_id"],
        ondelete="SET NULL",
    )
    op.add_column("signal_email_outbox", sa.Column("deduplication_key", sa.String(64)))
    op.create_index(
        "ux_signal_email_outbox_deduplication_key",
        "signal_email_outbox",
        ["deduplication_key"],
        unique=True,
    )
    op.create_index(
        "ix_signal_email_outbox_primary_scenario_id",
        "signal_email_outbox",
        ["primary_scenario_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_signal_email_outbox_primary_scenario_id", table_name="signal_email_outbox")
    op.drop_index("ux_signal_email_outbox_deduplication_key", table_name="signal_email_outbox")
    op.drop_column("signal_email_outbox", "deduplication_key")
    op.drop_constraint(
        "fk_signal_email_outbox_primary_scenario",
        "signal_email_outbox",
        type_="foreignkey",
    )
    op.drop_column("signal_email_outbox", "primary_scenario_id")
    op.drop_table("candidate_scenario_outcomes")
    op.drop_table("scenario_calibration_metrics")
    op.drop_table("scenario_lifecycle_transitions")
    op.drop_table("primary_scenario_geometries")
    op.drop_table("primary_scenario_selections")
    op.drop_table("scenario_score_components")
    op.drop_table("scenario_path_stages")
    op.drop_table("candidate_market_scenarios")
    op.drop_table("market_simulation_cycles")
