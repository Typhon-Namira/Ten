"""Make Primary Scenario publication and notification lifecycle durable.

Revision ID: 20260804_0024
Revises: 20260803_0023
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision = "20260804_0024"
down_revision = "20260803_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "signal_email_outbox_decision_id_fkey",
        "signal_email_outbox",
        type_="foreignkey",
    )
    op.alter_column(
        "signal_email_outbox",
        "decision_id",
        existing_type=UUID(as_uuid=True),
        nullable=True,
    )
    op.create_foreign_key(
        "signal_email_outbox_decision_id_fkey",
        "signal_email_outbox",
        "signal_decisions",
        ["decision_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_table(
        "primary_scenario_publications",
        sa.Column(
            "selection_id",
            UUID(as_uuid=True),
            sa.ForeignKey("primary_scenario_selections.selection_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "primary_scenario_id",
            UUID(as_uuid=True),
            sa.ForeignKey("candidate_market_scenarios.candidate_id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "decision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("signal_decisions.id", ondelete="SET NULL"),
        ),
        sa.Column("publication_status", sa.String(24), nullable=False),
        sa.Column("publication_reason", sa.String(128), nullable=False),
        sa.Column("email_status", sa.String(24), nullable=False),
        sa.Column("email_reason", sa.String(128), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "publication_status IN ('ELIGIBLE','INELIGIBLE')",
            name="ck_primary_scenario_publication_status",
        ),
        sa.CheckConstraint(
            "email_status IN ('ELIGIBLE','NOT_ELIGIBLE','ENQUEUED')",
            name="ck_primary_scenario_email_status",
        ),
    )
    for name, columns in (
        ("ix_primary_scenario_publications_primary_scenario_id", ["primary_scenario_id"]),
        ("ix_primary_scenario_publications_decision_id", ["decision_id"]),
        ("ix_primary_scenario_publications_publication_status", ["publication_status"]),
        ("ix_primary_scenario_publications_email_status", ["email_status"]),
        ("ix_primary_scenario_publications_evaluated_at", ["evaluated_at"]),
    ):
        op.create_index(name, "primary_scenario_publications", columns)


def downgrade() -> None:
    op.drop_constraint(
        "signal_email_outbox_decision_id_fkey",
        "signal_email_outbox",
        type_="foreignkey",
    )
    op.execute(
        "UPDATE signal_email_outbox AS email SET decision_id = publication.decision_id "
        "FROM primary_scenario_publications AS publication "
        "WHERE email.primary_scenario_id = publication.primary_scenario_id "
        "AND email.decision_id IS NULL AND publication.decision_id IS NOT NULL"
    )
    op.alter_column(
        "signal_email_outbox",
        "decision_id",
        existing_type=UUID(as_uuid=True),
        nullable=False,
    )
    op.create_foreign_key(
        "signal_email_outbox_decision_id_fkey",
        "signal_email_outbox",
        "signal_decisions",
        ["decision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_table("primary_scenario_publications")
