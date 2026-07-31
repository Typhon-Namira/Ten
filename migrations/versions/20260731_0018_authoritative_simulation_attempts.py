"""Add durable authoritative M15 simulation attempt lifecycle.

Revision ID: 20260731_0018
Revises: 20260730_0017
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "20260731_0018"
down_revision = "20260730_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ux_market_simulation_boundary", table_name="market_simulation_cycles")
    op.create_index(
        "ux_market_simulation_boundary",
        "market_simulation_cycles",
        ["instrument", "market_cutoff", "configuration_version"],
        unique=True,
    )
    op.drop_index(
        "ix_market_simulation_cycles_market_state_id",
        table_name="market_simulation_cycles",
    )
    op.create_index(
        "ix_market_simulation_cycles_market_state_id",
        "market_simulation_cycles",
        ["market_state_id"],
        unique=False,
    )
    op.drop_index(
        "ix_primary_scenario_selections_market_state_id",
        table_name="primary_scenario_selections",
    )
    op.create_index(
        "ix_primary_scenario_selections_market_state_id",
        "primary_scenario_selections",
        ["market_state_id"],
        unique=False,
    )
    op.create_table(
        "authoritative_simulation_attempts",
        sa.Column("attempt_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("timeframe", sa.String(8), nullable=False),
        sa.Column("market_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("simulation_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("candidate_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "simulation_cycle_id",
            UUID(as_uuid=True),
            sa.ForeignKey("market_simulation_cycles.simulation_cycle_id", ondelete="SET NULL"),
        ),
        sa.Column(
            "primary_scenario_id",
            UUID(as_uuid=True),
            sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="SET NULL"),
        ),
        sa.Column(
            "alternative_scenario_id",
            UUID(as_uuid=True),
            sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="SET NULL"),
        ),
        sa.Column("failure_stage", sa.String(96)),
        sa.Column("failure_type", sa.String(128)),
        sa.Column("failure_message", sa.String(1000)),
        sa.Column("skip_reason", sa.String(128)),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint(
            "status IN ('SCHEDULED','RUNNING','SUCCESS','NO_SIGNAL','ANALYTICAL_ONLY','BLOCKED','FAILED','SKIPPED')",
            name="ck_authoritative_simulation_attempt_status",
        ),
        sa.CheckConstraint(
            "candidate_count BETWEEN 0 AND 10",
            name="ck_authoritative_simulation_attempt_candidate_count",
        ),
    )
    for name, columns, unique in (
        ("ix_authoritative_simulation_attempts_instrument", ["instrument"], False),
        ("ix_authoritative_simulation_attempts_timeframe", ["timeframe"], False),
        ("ix_authoritative_simulation_attempts_market_cutoff", ["market_cutoff"], False),
        ("ix_authoritative_simulation_attempts_status", ["status"], False),
        ("ix_authoritative_simulation_attempts_scheduled_at", ["scheduled_at"], False),
        ("ix_authoritative_simulation_attempts_completed_at", ["completed_at"], False),
        ("ix_authoritative_simulation_attempts_simulation_cycle_id", ["simulation_cycle_id"], False),
        ("ix_authoritative_simulation_attempts_primary_scenario_id", ["primary_scenario_id"], False),
        ("ix_authoritative_simulation_attempts_alternative_scenario_id", ["alternative_scenario_id"], False),
        (
            "ux_authoritative_simulation_attempt_boundary",
            ["instrument", "timeframe", "market_cutoff", "simulation_version"],
            True,
        ),
    ):
        op.create_index(name, "authoritative_simulation_attempts", columns, unique=unique)


def downgrade() -> None:
    op.drop_table("authoritative_simulation_attempts")
    op.drop_index(
        "ix_primary_scenario_selections_market_state_id",
        table_name="primary_scenario_selections",
    )
    op.create_index(
        "ix_primary_scenario_selections_market_state_id",
        "primary_scenario_selections",
        ["market_state_id"],
        unique=True,
    )
    op.drop_index(
        "ix_market_simulation_cycles_market_state_id",
        table_name="market_simulation_cycles",
    )
    op.create_index(
        "ix_market_simulation_cycles_market_state_id",
        "market_simulation_cycles",
        ["market_state_id"],
        unique=True,
    )
    op.drop_index("ux_market_simulation_boundary", table_name="market_simulation_cycles")
    op.create_index(
        "ux_market_simulation_boundary",
        "market_simulation_cycles",
        ["instrument", "market_cutoff"],
        unique=True,
    )
